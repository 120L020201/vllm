# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from queue import Queue
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Any

import torch

from online_draft.runtime import profile_range
from online_draft.runtime.capture_runtime import PreparedRound
from online_draft.runtime.cpu_worker import CPUWorkerConfig
from online_draft.runtime.weight_snapshot import DraftWeightSnapshot
from online_draft.training.eagle3.cache import PersistentEagle3KVCache
from online_draft.training.eagle3.window import (
    Eagle3TrainingWindow,
    Eagle3WindowMode,
    Eagle3WindowResult,
    train_eagle3_window,
)
from online_draft.training.trainer import DraftTrainer

TrainWindow = Callable[
    [
        DraftTrainer,
        Eagle3TrainingWindow,
        PersistentEagle3KVCache,
        Eagle3WindowMode,
        bool,
    ],
    tuple[PersistentEagle3KVCache, Eagle3WindowResult],
]


@dataclass(frozen=True, slots=True)
class _RoundEvent:
    round: PreparedRound


@dataclass(frozen=True, slots=True)
class _CloseEpochEvent:
    request_id: str


@dataclass(frozen=True, slots=True)
class _FinishRequestEvent:
    request_id: str
    done: Event


class AsyncDraftPipeline:
    """Train draft weights behind an ordered round queue."""

    def __init__(
        self,
        *,
        trainer: DraftTrainer,
        on_queue_size: Callable[[int], None] | None = None,
        train_rounds: Callable,
        profile_timeline: bool = False,
        cpu_config: CPUWorkerConfig | None = None,
    ) -> None:
        self._trainer = trainer
        self._on_queue_size = on_queue_size
        self._train_rounds = train_rounds
        self._profile_timeline = profile_timeline
        self._cpu_config = cpu_config

        self._queue: Queue[object] = Queue()
        self._stop = object()
        self._lock = Lock()
        self._closed = False
        self._failure: BaseException | None = None

        self._request_id: str | None = None
        self._bootstrap_done = False
        self._epoch_rounds: list[PreparedRound] = []
        self._persistent_cache: Any = None
        self._snapshot_pending = False
        self._ready_snapshot: DraftWeightSnapshot | None = None
        self.update_stats: list[dict] = []
        self.finished_request: dict | None = None
        parameters = dict(trainer.model.named_parameters())
        self._snapshot_parameters = tuple(
            (name, parameters[name]) for name in trainer.trainable_parameter_names
        )
        # The next epoch starts only after H2D commits, so one pinned buffer is safe.
        with torch.inference_mode(False):
            self._snapshot_tensors = DraftWeightSnapshot.from_named_tensors(
                self._snapshot_parameters, version=trainer.version
            ).tensors

        self._worker = Thread(
            target=self._run,
            name="online-draft-trainer",
            daemon=True,
        )
        self._worker.start()

    @property
    def failed(self) -> BaseException | None:
        with self._lock:
            return self._failure

    def submit_round(self, prepared_round: PreparedRound) -> int:
        return self._submit(_RoundEvent(prepared_round))

    def close_epoch(self, request_id: str) -> int:
        return self._submit(_CloseEpochEvent(request_id))

    def finish_request(self, request_id: str) -> None:
        event = _FinishRequestEvent(
            request_id=request_id,
            done=Event(),
        )

        with self._lock:
            if self._closed:
                raise RuntimeError("pipeline is closed")
            self._queue.put_nowait(event)

        event.done.wait()
        self.raise_if_failed()

    def poll_snapshot(self) -> DraftWeightSnapshot | None:
        with self._lock:
            snapshot = self._ready_snapshot
            self._ready_snapshot = None
            return snapshot

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._queue.put(self._stop)

        self._worker.join()

    def raise_if_failed(self) -> None:
        failure = self.failed
        if failure is not None:
            raise failure

    def _submit(self, event: object) -> int:
        with self._lock:
            if self._closed or self._failure is not None:
                return self._queue.qsize()

            self._queue.put_nowait(event)

        queue_size = self._queue.qsize()
        if self._on_queue_size is not None:
            self._on_queue_size(queue_size)
        return queue_size

    def _run(self) -> None:
        while True:
            event = self._queue.get()
            try:
                if event is self._stop:
                    return

                if self._cpu_config is not None:
                    config, self._cpu_config = self._cpu_config, None
                    config.apply()

                if isinstance(event, _FinishRequestEvent):
                    self._handle_finish_request(event.request_id)
                    continue

                if self.failed is not None:
                    continue

                if isinstance(event, _RoundEvent):
                    self._handle_round(event.round)
                elif isinstance(event, _CloseEpochEvent):
                    self._handle_close_epoch(event.request_id)
                else:
                    raise RuntimeError("unknown draft pipeline event")
            except Exception as error:
                self._set_failure(error)
            finally:
                if isinstance(event, _FinishRequestEvent):
                    event.done.set()
                self._queue.task_done()

    def _handle_round(self, prepared_round: PreparedRound) -> None:
        self._set_request(prepared_round.request_id)

        if not self._bootstrap_done:
            self._train((prepared_round,))
            self._bootstrap_done = True
            return

        self._epoch_rounds.append(prepared_round)

    def _handle_close_epoch(self, request_id: str) -> None:
        if self._request_id != request_id:
            raise RuntimeError("closed epoch does not match active request")
        if not self._snapshot_pending:
            raise RuntimeError("closed epoch has no pending snapshot")

        self._snapshot_pending = False
        rounds = tuple(self._epoch_rounds)
        self._epoch_rounds.clear()

        if rounds:
            self._train(rounds)

    def _handle_finish_request(self, request_id: str) -> None:
        if self._request_id is not None and self._request_id != request_id:
            raise RuntimeError("finished request does not match active request")

        self.finished_request = {
            "request_id": request_id,
            "cpu_version": self._trainer.version,
            "updates": self.update_stats,
        }
        self._trainer.reset()
        self._request_id = None
        self._bootstrap_done = False
        self._epoch_rounds.clear()
        self._persistent_cache = None
        self.update_stats = []

        with self._lock:
            self._snapshot_pending = False
            self._ready_snapshot = None

    def _train(
        self,
        rounds: tuple[PreparedRound, ...],
    ) -> None:
        name = (
            f"online_draft.cpu_train[version={self._trainer.version + 1},"
            f"rounds={len(rounds)}]"
        )
        start = perf_counter()
        cpu_version = self._trainer.version
        with profile_range(name, self._profile_timeline):
            updated_cache, result = self._train_rounds(
                self._trainer,
                tuple(prepared.batch for prepared in rounds),
                self._persistent_cache,
                self._profile_timeline,
            )
        self._persistent_cache = updated_cache
        self.update_stats.append(
            {
                "cpu_start_version": cpu_version,
                "cpu_version": result.model_version,
                "start_step": rounds[0].step_id,
                "end_step": rounds[-1].step_id,
                "rounds": len(rounds),
                "trained_rounds": result.round_count,
                "target_count": result.target_count,
                "source_versions": sorted({r.source_weight_version for r in rounds}),
                "train_seconds": perf_counter() - start,
                "loss": self._trainer.last_loss,
            }
        )
        self._publish_snapshot(result.model_version)

    def _publish_snapshot(self, version: int) -> None:
        name = f"online_draft.cpu_snapshot[version={version}]"
        with profile_range(name, self._profile_timeline):
            for (_, destination), (_, parameter) in zip(
                self._snapshot_tensors, self._snapshot_parameters, strict=True
            ):
                destination.copy_(parameter.detach())
            snapshot = DraftWeightSnapshot(version, self._snapshot_tensors)

        with self._lock:
            if self._ready_snapshot is not None:
                raise RuntimeError("a snapshot is already ready")
            self._ready_snapshot = snapshot
            self._snapshot_pending = True

    def _set_request(self, request_id: str) -> None:
        if self._request_id is None:
            self._request_id = request_id
        elif self._request_id != request_id:
            raise RuntimeError("pipeline received multiple active requests")

    def _set_failure(self, error: BaseException) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = error


class AsyncEagle3Pipeline(AsyncDraftPipeline):
    """Run the EAGLE3 recipe using the shared asynchronous scheduler."""

    def __init__(
        self,
        *,
        trainer: DraftTrainer,
        mode: Eagle3WindowMode = Eagle3WindowMode.CONFIRMED_PATH,
        on_queue_size: Callable[[int], None] | None = None,
        train_window: TrainWindow = train_eagle3_window,
        profile_timeline: bool = False,
        cpu_config: CPUWorkerConfig | None = None,
    ) -> None:
        def train_rounds(trainer, rounds, cache, profile):
            return train_window(
                trainer, Eagle3TrainingWindow(rounds), cache, mode, profile
            )

        super().__init__(
            trainer=trainer,
            train_rounds=train_rounds,
            on_queue_size=on_queue_size,
            profile_timeline=profile_timeline,
            cpu_config=cpu_config,
        )
