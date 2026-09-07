# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import torch

from .qwen3_trainer import Qwen3Eagle3CpuTrainer
from .weights import TrainableWeightSnapshot


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
        step_fn: Callable[[Qwen3Eagle3CpuTrainer, TrainObservation], None],
    ) -> None:
        self.trainer = trainer
        self._step_fn = step_fn
        self._baseline_snapshot = trainer.snapshot()
        self._inbox: queue.SimpleQueue[TrainObservation | ResetRequest | None]
        self._outbox: queue.SimpleQueue[TrainableWeightSnapshot]
        self._inbox = queue.SimpleQueue()
        self._outbox = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def observe_step(
        self,
        request_id: str,
        step_id: int,
        payload: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        self._inbox.put(
            TrainObservation(
                request_id=request_id,
                step_id=step_id,
                payload=dict(payload or {}),
            )
        )

    def reset_request(self, request_id: str) -> None:
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
        self._inbox.put(None)
        self._thread.join(timeout=5.0)

    def _worker(self) -> None:
        while True:
            item = self._inbox.get()
            if item is None:
                return
            if isinstance(item, ResetRequest):
                self.trainer.restore_snapshot(self._baseline_snapshot)
                self._outbox.put(self.trainer.snapshot())
                continue

            self._step_fn(self.trainer, item)
            self._outbox.put(self.trainer.snapshot())
