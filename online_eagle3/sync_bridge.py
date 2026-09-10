# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping

import torch

from .data import Qwen3Eagle3StepFn, TrainObservation
from .qwen3_trainer import Qwen3Eagle3CpuTrainer
from .weights import TrainableWeightSnapshot

logger = logging.getLogger(__name__)


class Qwen3Eagle3SyncBridge:
    """Synchronous bridge for request-local CPU EAGLE3 updates."""

    def __init__(
        self,
        trainer: Qwen3Eagle3CpuTrainer,
        step_fn: Qwen3Eagle3StepFn,
        *,
        update_interval: int = 1,
        fail_on_error: bool = False,
    ) -> None:
        if update_interval < 1:
            raise ValueError("update_interval must be >= 1")
        self.trainer = trainer
        self._step_fn = step_fn
        self._update_interval = update_interval
        self._fail_on_error = fail_on_error
        self._baseline_snapshot = trainer.snapshot()
        self._pending_observations: list[TrainObservation] = []
        self._latest_snapshot: TrainableWeightSnapshot | None = None
        self._closed = False
        self._logged_first_update = False

    def observe_step(
        self,
        request_id: str,
        step_id: int,
        payload: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        if self._closed:
            return

        self._pending_observations.append(
            TrainObservation(
                request_id=request_id,
                step_id=step_id,
                payload=dict(payload or {}),
            )
        )
        if len(self._pending_observations) < self._update_interval:
            return

        observations = tuple(self._pending_observations)
        self._pending_observations.clear()
        version_before = self.trainer.version
        try:
            with (
                torch.profiler.record_function("online_eagle3.cpu_update_step"),
                torch.inference_mode(False),
                torch.enable_grad(),
            ):
                self._step_fn(self.trainer, observations)
        except Exception:
            logger.exception("Failed to update online EAGLE3 CPU draft")
            self.trainer.zero_grad()
            if self._fail_on_error:
                raise
            return

        if self.trainer.version != version_before:
            self._latest_snapshot = self.trainer.snapshot()
            if not self._logged_first_update:
                logger.info(
                    "Completed first synchronous online EAGLE3 CPU update: "
                    "observations=%d, version=%d",
                    len(observations),
                    self.trainer.version,
                )
                self._logged_first_update = True

    def maybe_apply_pending_weights(self) -> TrainableWeightSnapshot | None:
        latest = self._latest_snapshot
        self._latest_snapshot = None
        return latest

    def reset_request(self, request_id: str) -> None:
        if self._closed:
            return
        self._pending_observations = [
            observation
            for observation in self._pending_observations
            if observation.request_id != request_id
        ]
        try:
            with (
                torch.profiler.record_function("online_eagle3.cpu_reset"),
                torch.inference_mode(False),
                torch.enable_grad(),
            ):
                self.trainer.restore_snapshot(self._baseline_snapshot)
                self._latest_snapshot = self.trainer.snapshot()
            logger.info("Reset online EAGLE3 CPU draft for request %s", request_id)
        except Exception:
            logger.exception("Failed to reset online EAGLE3 CPU draft")
            self.trainer.zero_grad()
            if self._fail_on_error:
                raise

    def shutdown(self) -> None:
        self._closed = True
        self._pending_observations.clear()
        self._latest_snapshot = None
        self.trainer.clear_request_state()


class Qwen3Eagle3LazySyncBridge:
    """Lazy wrapper that defers CPU draft loading until the first update."""

    def __init__(
        self,
        bridge_factory: Callable[[], Qwen3Eagle3SyncBridge],
    ) -> None:
        self._bridge_factory = bridge_factory
        self._bridge: Qwen3Eagle3SyncBridge | None = None
        self._closed = False

    @property
    def is_loaded(self) -> bool:
        return self._bridge is not None

    @property
    def trainer(self) -> Qwen3Eagle3CpuTrainer:
        bridge = self._get_bridge()
        if bridge is None:
            raise RuntimeError("online EAGLE3 sync bridge is closed")
        return bridge.trainer

    def observe_step(
        self,
        request_id: str,
        step_id: int,
        payload: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        bridge = self._get_bridge()
        if bridge is None:
            return
        bridge.observe_step(request_id, step_id, payload)

    def maybe_apply_pending_weights(self) -> TrainableWeightSnapshot | None:
        if self._bridge is None:
            return None
        return self._bridge.maybe_apply_pending_weights()

    def reset_request(self, request_id: str) -> None:
        if self._bridge is None:
            return
        self._bridge.reset_request(request_id)

    def shutdown(self) -> None:
        self._closed = True
        if self._bridge is not None:
            self._bridge.shutdown()
        self._bridge = None

    def _get_bridge(self) -> Qwen3Eagle3SyncBridge | None:
        if self._closed:
            return None
        if self._bridge is None:
            self._bridge = self._bridge_factory()
        return self._bridge
