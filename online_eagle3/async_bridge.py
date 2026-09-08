# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import atexit
import contextlib
import logging
import queue
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import torch

from .qwen3_trainer import Qwen3Eagle3CpuTrainer
from .weights import TrainableWeightSnapshot

logger = logging.getLogger(__name__)

Qwen3Eagle3StepFn = Callable[
    [Qwen3Eagle3CpuTrainer, Sequence["TrainObservation"]], None
]


@dataclass(slots=True)
class TrainObservation:
    request_id: str
    step_id: int
    payload: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass(slots=True)
class ResetRequest:
    request_id: str


class Qwen3Eagle3AsyncBridge:
    """Single-worker queue bridge for request-local CPU updates."""

    def __init__(
        self,
        trainer: Qwen3Eagle3CpuTrainer,
        step_fn: Qwen3Eagle3StepFn,
        *,
        update_interval: int = 1,
    ) -> None:
        if update_interval < 1:
            raise ValueError("update_interval must be >= 1")
        self.trainer = trainer
        self._step_fn = step_fn
        self._update_interval = update_interval
        self._baseline_snapshot = trainer.snapshot()
        self._inbox: queue.SimpleQueue[TrainObservation | ResetRequest | None]
        self._inbox = queue.SimpleQueue()
        self._latest_snapshot: TrainableWeightSnapshot | None = None
        self._outbox_lock = threading.Lock()
        self._reset_request_ids: set[str] = set()
        self._reset_lock = threading.Lock()
        self._closed = threading.Event()
        self._shutdown_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._worker,
            name="online_eagle3_cpu_worker",
            daemon=True,
        )
        self._thread.start()
        atexit.register(self.shutdown)

    def observe_step(
        self,
        request_id: str,
        step_id: int,
        payload: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        if self._closed.is_set():
            return
        with torch.profiler.record_function("online_eagle3.bridge_enqueue_observation"):
            self._inbox.put(
                TrainObservation(
                    request_id=request_id,
                    step_id=step_id,
                    payload=dict(payload or {}),
                )
            )

    def reset_request(self, request_id: str) -> None:
        if self._closed.is_set():
            return
        with torch.profiler.record_function("online_eagle3.bridge_enqueue_reset"):
            with self._reset_lock:
                self._reset_request_ids.add(request_id)
            with self._outbox_lock:
                self._latest_snapshot = None
            self._inbox.put(ResetRequest(request_id=request_id))

    def maybe_apply_pending_weights(self) -> TrainableWeightSnapshot | None:
        with (
            torch.profiler.record_function("online_eagle3.bridge_poll_snapshot"),
            self._outbox_lock,
        ):
            latest = self._latest_snapshot
            self._latest_snapshot = None
        return latest

    def shutdown(self) -> None:
        with self._shutdown_lock:
            if self._closed.is_set():
                return
            self._closed.set()
            self._inbox.put(None)

        if threading.current_thread() is not self._thread:
            self._thread.join()
        with contextlib.suppress(ValueError):
            atexit.unregister(self.shutdown)

    def _worker(self) -> None:
        pending_observations: list[TrainObservation] = []
        while True:
            item = self._inbox.get()
            if item is None or self._closed.is_set():
                return
            if isinstance(item, ResetRequest):
                with torch.profiler.record_function("online_eagle3.cpu_reset_request"):
                    pending_observations = [
                        observation
                        for observation in pending_observations
                        if observation.request_id != item.request_id
                    ]
                    try:
                        self.trainer.restore_snapshot(self._baseline_snapshot)
                        self._publish_snapshot(self.trainer.snapshot())
                    except Exception:
                        logger.exception("Failed to reset online EAGLE3 draft weights")
                    finally:
                        with self._reset_lock:
                            self._reset_request_ids.discard(item.request_id)
                continue

            if self._is_reset_requested(item.request_id):
                pending_observations = [
                    observation
                    for observation in pending_observations
                    if observation.request_id != item.request_id
                ]
                continue

            pending_observations.append(item)
            pending_observations = [
                observation
                for observation in pending_observations
                if not self._is_reset_requested(observation.request_id)
            ]
            if len(pending_observations) < self._update_interval:
                continue

            step_observations = tuple(pending_observations)
            try:
                with torch.profiler.record_function("online_eagle3.cpu_update_step"):
                    self._step_fn(self.trainer, step_observations)
            except Exception:
                logger.exception("Failed to update online EAGLE3 draft weights")
                self.trainer.zero_grad()
                pending_observations.clear()
                continue
            pending_observations.clear()
            if not self._closed.is_set() and not self._has_reset_requested(
                step_observations
            ):
                with torch.profiler.record_function(
                    "online_eagle3.cpu_publish_snapshot"
                ):
                    self._publish_snapshot(self.trainer.snapshot())

    def _publish_snapshot(self, snapshot: TrainableWeightSnapshot) -> None:
        with self._outbox_lock:
            self._latest_snapshot = snapshot

    def _is_reset_requested(self, request_id: str) -> bool:
        with self._reset_lock:
            return request_id in self._reset_request_ids

    def _has_reset_requested(self, observations: Sequence[TrainObservation]) -> bool:
        with self._reset_lock:
            return any(
                observation.request_id in self._reset_request_ids
                for observation in observations
            )


class Qwen3Eagle3LazyBridge:
    """Defers CPU draft loading until the first trainable observation."""

    def __init__(self, bridge_factory: Callable[[], Qwen3Eagle3AsyncBridge]) -> None:
        self._bridge_factory = bridge_factory
        self._bridge: Qwen3Eagle3AsyncBridge | None = None
        self._lock = threading.Lock()
        self._closed = threading.Event()

    @property
    def is_loaded(self) -> bool:
        return self._bridge is not None

    @property
    def trainer(self) -> Qwen3Eagle3CpuTrainer:
        bridge = self._get_bridge(create=True)
        if bridge is None:
            raise RuntimeError("online EAGLE3 bridge is closed")
        return bridge.trainer

    def observe_step(
        self,
        request_id: str,
        step_id: int,
        payload: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        bridge = self._get_bridge(create=True)
        if bridge is None:
            return
        bridge.observe_step(request_id, step_id, payload)

    def reset_request(self, request_id: str) -> None:
        bridge = self._get_bridge(create=False)
        if bridge is None:
            return
        bridge.reset_request(request_id)

    def maybe_apply_pending_weights(self) -> TrainableWeightSnapshot | None:
        bridge = self._get_bridge(create=False)
        if bridge is None:
            return None
        return bridge.maybe_apply_pending_weights()

    def shutdown(self) -> None:
        self._closed.set()
        bridge = self._get_bridge(create=False)
        if bridge is not None:
            bridge.shutdown()

    def _get_bridge(self, *, create: bool) -> Qwen3Eagle3AsyncBridge | None:
        if self._closed.is_set():
            return None
        bridge = self._bridge
        if bridge is not None or not create:
            return bridge
        with self._lock:
            if self._closed.is_set():
                return None
            if self._bridge is None:
                with torch.profiler.record_function(
                    "online_eagle3.cpu_lazy_load_bridge"
                ):
                    self._bridge = self._bridge_factory()
            return self._bridge
