# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Request-ordered CPU training bridge for experimental policies."""

import atexit
import contextlib
import logging
import os
import queue
import threading
from dataclasses import dataclass

import torch
from online_draft.models.qwen3_eagle3 import Eagle3KVCache
from online_draft.training.eagle3_batch import Eagle3DistillationBatch
from online_draft.training.eagle3_cache import (
    append_confirmed_window_to_persistent_cache,
)
from online_draft.training.eagle3_window import Eagle3TrainingWindow
from online_draft.training.trainer import DraftTrainer

from methods.ospec_common.ensemble import (
    ChunkEnsemble,
    WeightSnapshot,
    export_trainable_state,
)
from methods.random_sampling.gating import BernoulliGate
from methods.tts_common.step import train_batches
from methods.utils.stats import WorkerStats

logger = logging.getLogger(__name__)
_DEFAULT_TORCH_THREADS = min(
    20,
    len(os.sched_getaffinity(0))
    if hasattr(os, "sched_getaffinity")
    else os.cpu_count() or 1,
)


@dataclass(frozen=True, slots=True)
class TrainObservation:
    request_id: str
    step_id: int
    batch: Eagle3DistillationBatch


@dataclass(frozen=True, slots=True)
class ResetRequest:
    request_id: str
    done: threading.Event


def _load_trainable_state(
    trainer: DraftTrainer,
    state_dict: dict[str, torch.Tensor],
) -> None:
    current = trainer.model.state_dict()
    expected = set(trainer.trainable_parameter_names)
    if set(state_dict) != expected:
        raise KeyError("trainable snapshot does not match the trainer")
    with torch.no_grad():
        for name, source in state_dict.items():
            current[name].copy_(source)
    trainer.clear_optimizer_state()
    trainer.last_loss = None


class PolicyBridge:
    def __init__(self, trainer: DraftTrainer, *, method: str) -> None:
        if method not in {"tts", "random_sampling", "ospec"}:
            raise ValueError(f"unsupported method: {method}")
        self.trainer = trainer
        self.method = method
        self.torch_threads = int(
            os.environ.get("OSD_TORCH_THREADS", str(_DEFAULT_TORCH_THREADS))
        )
        self.stride = int(os.environ.get("OSD_UPDATE_STRIDE", "1"))
        self.chunk_size = int(os.environ.get("OSD_CHUNK_SIZE", "5"))
        if self.stride < 1 or self.chunk_size < 1:
            raise ValueError("stride and chunk size must be positive")
        self.gate = None
        if method == "random_sampling":
            self.gate = BernoulliGate(
                float(os.environ.get("OSD_PROBABILITY", "0.5")),
                int(os.environ.get("OSD_SEED", "0")),
                self.stride,
            )
        self.ensemble = None
        if method == "ospec":
            rates = tuple(
                float(value)
                for value in os.environ.get("OSD_ENSEMBLE_LRS", "1e-5,2e-5,3e-5").split(
                    ","
                )
            )
            self.ensemble = ChunkEnsemble(
                trainer,
                rates,
                float(os.environ.get("OSD_EPSILON", "0.1")),
            )
        self._baseline = WeightSnapshot(
            version=trainer.version,
            state_dict=export_trainable_state(trainer),
        )
        self._inbox: queue.Queue[TrainObservation | ResetRequest | None] = queue.Queue()
        self._lock = threading.Lock()
        self._latest: WeightSnapshot | None = None
        self._error: Exception | None = None
        self._closed = False
        self._request_id: str | None = None
        self._next_step_id = 0
        self._rounds = 0
        self._persistent_cache: Eagle3KVCache | None = None
        self._current_request: list[Eagle3DistillationBatch] = []
        self._chunk_requests: list[tuple[Eagle3DistillationBatch, ...]] = []
        self.stats = WorkerStats()
        self._thread = threading.Thread(
            target=self._work,
            daemon=True,
            name=f"{method}_cpu_worker",
        )
        self._thread.start()
        atexit.register(self.shutdown)

    def _check_error(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"{self.method} CPU training failed") from self._error

    def observe_step(
        self,
        request_id: str,
        step_id: int,
        batch: Eagle3DistillationBatch,
    ) -> None:
        self._check_error()
        if not self._closed:
            self.stats.enqueue()
            self._inbox.put(TrainObservation(request_id, step_id, batch))

    def reset_request(self, request_id: str) -> None:
        self._check_error()
        if self._closed:
            return
        done = threading.Event()
        self._inbox.put(ResetRequest(request_id, done))
        done.wait()
        self._check_error()

    def maybe_apply_pending_weights(self) -> WeightSnapshot | None:
        self._check_error()
        with self._lock:
            result = self._latest
            self._latest = None
        return result

    def _publish(self, snapshot: WeightSnapshot) -> None:
        with self._lock:
            self._latest = snapshot

    def _work(self) -> None:
        while True:
            item = self._inbox.get()
            if item is None:
                return
            completed_updates = 0
            try:
                if torch.get_num_threads() != self.torch_threads:
                    torch.set_num_threads(self.torch_threads)
                if isinstance(item, TrainObservation):
                    completed_updates = self._observe(item)
                else:
                    self._reset(item.request_id)
            except Exception as error:
                logger.exception("%s CPU training failed", self.method)
                self._error = error
            finally:
                if isinstance(item, TrainObservation):
                    self.stats.complete(completed_updates)
                if isinstance(item, ResetRequest):
                    item.done.set()

    def _observe(self, observation: TrainObservation) -> int:
        completed_updates = 0
        if self._request_id is None:
            self._request_id = observation.request_id
        if observation.request_id != self._request_id:
            raise ValueError("methods support one active request per worker")
        if observation.step_id != self._next_step_id:
            raise ValueError(
                f"expected step {self._next_step_id}, got {observation.step_id}"
            )
        self._next_step_id += 1
        self._rounds += 1

        if self.ensemble is not None:
            self._current_request.append(observation.batch)
            return 0

        should_train = self._rounds % self.stride == 0
        if self.gate is not None:
            should_train = self.gate.select(observation.request_id)
        if should_train:
            self._persistent_cache, _ = train_batches(
                self.trainer,
                (observation.batch,),
                persistent_cache=self._persistent_cache,
            )
            completed_updates = 1
            self._publish(
                WeightSnapshot(
                    version=self.trainer.version,
                    state_dict=export_trainable_state(self.trainer),
                )
            )
        else:
            self._persistent_cache = append_confirmed_window_to_persistent_cache(
                self.trainer.model,
                Eagle3TrainingWindow(rounds=(observation.batch,)),
                self._persistent_cache,
            )
        return completed_updates

    def _reset(self, request_id: str) -> None:
        if self._request_id is not None and request_id != self._request_id:
            return
        if self.gate is not None:
            self.gate.reset(request_id)
        if self.ensemble is not None:
            if self._current_request:
                self._chunk_requests.append(tuple(self._current_request))
            self._current_request.clear()
            if len(self._chunk_requests) >= self.chunk_size:
                snapshot = self.ensemble.update(tuple(self._chunk_requests))
                if snapshot is not None:
                    self._publish(snapshot)
                    self.stats.updates += self.ensemble.last_update_steps
                    self.stats.write()
                self._chunk_requests.clear()
        elif os.environ.get("OSD_KEEP_WEIGHTS", "0") != "1":
            _load_trainable_state(
                self.trainer,
                self._baseline.state_dict,
            )
            self._publish(self._baseline)
        self._request_id = None
        self._next_step_id = 0
        self._rounds = 0
        self._persistent_cache = None

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._inbox.put(None)
        if threading.current_thread() is not self._thread:
            self._thread.join()
        with contextlib.suppress(ValueError):
            atexit.unregister(self.shutdown)
