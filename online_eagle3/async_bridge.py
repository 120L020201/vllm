# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import atexit
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
        self._outbox: queue.SimpleQueue[TrainableWeightSnapshot]
        self._inbox = queue.SimpleQueue()
        self._outbox = queue.SimpleQueue()
        self._closed = threading.Event()
        self._shutdown_lock = threading.Lock()
        self._thread = threading.Thread(target=self._worker, daemon=True)
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
        self._inbox.put(ResetRequest(request_id=request_id))

    def maybe_apply_pending_weights(self) -> TrainableWeightSnapshot | None:
        latest: TrainableWeightSnapshot | None = None
        while True:
            try:
                latest = self._outbox.get_nowait()
            except queue.Empty:
                break
        return latest

    def shutdown(self) -> None:
        with self._shutdown_lock:
            if self._closed.is_set():
                return
            self._closed.set()
            self._inbox.put(None)

        if threading.current_thread() is not self._thread:
            self._thread.join()
        try:
            atexit.unregister(self.shutdown)
        except ValueError:
            pass

    def _worker(self) -> None:
        pending_observations: list[TrainObservation] = []
        while True:
            item = self._inbox.get()
            if item is None or self._closed.is_set():
                return
            if isinstance(item, ResetRequest):
                pending_observations.clear()
                try:
                    self.trainer.restore_snapshot(self._baseline_snapshot)
                    self._outbox.put(self.trainer.snapshot())
                except Exception:
                    logger.exception("Failed to reset online EAGLE3 draft weights")
                continue

            pending_observations.append(item)
            if len(pending_observations) < self._update_interval:
                continue

            try:
                self._step_fn(self.trainer, tuple(pending_observations))
            except Exception:
                logger.exception("Failed to update online EAGLE3 draft weights")
                self.trainer.zero_grad()
                pending_observations.clear()
                continue
            pending_observations.clear()
            if not self._closed.is_set():
                self._outbox.put(self.trainer.snapshot())
