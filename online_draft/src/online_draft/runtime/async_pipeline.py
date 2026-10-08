# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from queue import Queue
from threading import Event, Lock, Thread

from online_draft.runtime.capture_runtime import Eagle3PreparedRound
from online_draft.runtime.weight_snapshot import DraftWeightSnapshot
from online_draft.training.eagle3_cache import PersistentEagle3KVCache
from online_draft.training.eagle3_window import (
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
    ],
    tuple[PersistentEagle3KVCache, Eagle3WindowResult],
]


@dataclass(frozen=True, slots=True)
class _RoundEvent:
    round: Eagle3PreparedRound


@dataclass(frozen=True, slots=True)
class _CloseEpochEvent:
    request_id: str


@dataclass(frozen=True, slots=True)
class _FinishRequestEvent:
    request_id: str
    done: Event


class AsyncEagle3Pipeline:
    """Train CPU EAGLE3 weights behind an ordered round queue."""

    def __init__(
        self,
        *,
        trainer: DraftTrainer,
        mode: Eagle3WindowMode = Eagle3WindowMode.CONFIRMED_PATH,
        on_queue_size: Callable[[int], None] | None = None,
        train_window: TrainWindow = train_eagle3_window,
    ) -> None:
        self._trainer = trainer
        self._mode = mode
        self._on_queue_size = on_queue_size
        self._train_window = train_window

        self._queue: Queue[object] = Queue()
        self._stop = object()
        self._lock = Lock()
        self._closed = False
        self._failure: BaseException | None = None

        self._request_id: str | None = None
        self._bootstrap_done = False
        self._epoch_rounds: list[Eagle3PreparedRound] = []
        self._persistent_cache: PersistentEagle3KVCache = None
        self._snapshot_pending = False
        self._ready_snapshot: DraftWeightSnapshot | None = None

        self._worker = Thread(
            target=self._run,
            name="eagle3-trainer",
            daemon=True,
        )
        self._worker.start()

    @property
    def failed(self) -> BaseException | None:
        with self._lock:
            return self._failure

    def submit_round(self, prepared_round: Eagle3PreparedRound) -> int:
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
        self.raise_if_failed()

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
                    raise RuntimeError("unknown EAGLE3 pipeline event")
            except Exception as error:
                self._set_failure(error)
            finally:
                if isinstance(event, _FinishRequestEvent):
                    event.done.set()
                self._queue.task_done()

    def _handle_round(self, prepared_round: Eagle3PreparedRound) -> None:
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

        self._trainer.reset()
        self._request_id = None
        self._bootstrap_done = False
        self._epoch_rounds.clear()
        self._persistent_cache = None

        with self._lock:
            self._snapshot_pending = False
            self._ready_snapshot = None

    def _train(
        self,
        rounds: tuple[Eagle3PreparedRound, ...],
    ) -> None:
        window = Eagle3TrainingWindow(
            rounds=tuple(prepared_round.batch for prepared_round in rounds)
        )
        updated_cache, result = self._train_window(
            self._trainer,
            window,
            self._persistent_cache,
            self._mode,
        )
        self._persistent_cache = updated_cache
        self._publish_snapshot(result.model_version)

    def _publish_snapshot(self, version: int) -> None:
        named_parameters = dict(self._trainer.model.named_parameters())
        tensors = tuple(
            (name, named_parameters[name])
            for name in self._trainer.trainable_parameter_names
        )
        snapshot = DraftWeightSnapshot.from_named_tensors(
            tensors,
            version=version,
        )

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
