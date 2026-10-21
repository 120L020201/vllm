# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import dataclass
from threading import Event, Thread
from types import SimpleNamespace

import pytest
import torch
from online_draft.runtime.async_pipeline import AsyncDraftPipeline
from online_draft.runtime.contracts import PreparedRound
from online_draft.runtime.cpu_worker import CPUWorkerConfig
from torch import nn


def test_cpu_worker_settings_do_not_change_inference_thread():
    """Training parallelism and affinity must stay local to its worker thread."""
    inference_threads = torch.get_num_threads()
    inference_cpus = os.sched_getaffinity(0)
    training_cpu = min(inference_cpus)
    config = CPUWorkerConfig(threads=1, cpus=(training_cpu,))
    observed = []

    def run():
        config.apply()
        observed.append((torch.get_num_threads(), os.sched_getaffinity(0)))

    worker = Thread(target=run)
    worker.start()
    worker.join()
    assert observed == [(1, {training_cpu})]
    assert torch.get_num_threads() == inference_threads
    assert os.sched_getaffinity(0) == inference_cpus


@dataclass
class _FakeTrainer:
    model: nn.Module
    version: int = 0
    reset_count: int = 0
    last_loss: float | None = None

    @property
    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.model.named_parameters())

    def reset(self) -> None:
        self.version = 0
        self.reset_count += 1


def _make_round(
    step_id: int,
    *,
    request_id: str = "request-0",
) -> PreparedRound[int]:
    return PreparedRound(
        request_id=request_id,
        step_id=step_id,
        source_weight_version=0,
        batch=step_id,
    )


def _wait_for_snapshot(pipeline: AsyncDraftPipeline):
    for _ in range(500):
        snapshot = pipeline.poll_snapshot()
        if snapshot is not None:
            return snapshot
        Event().wait(0.01)
    return None


def test_bootstrap_and_epoch_close_publish_delayed_snapshots() -> None:
    trainer = _FakeTrainer(nn.Linear(1, 1))
    calls: list[tuple[tuple[int, ...], object]] = []
    trained = Event()

    def train_rounds(trainer, rounds, persistent_cache, profile_timeline):
        assert not profile_timeline
        calls.append((rounds, persistent_cache))
        trainer.version += 1
        trained.set()
        return f"cache-{trainer.version}", SimpleNamespace(
            model_version=trainer.version,
            round_count=len(rounds),
            target_count=1,
        )

    pipeline = AsyncDraftPipeline(
        trainer=trainer,
        train_rounds=train_rounds,
    )

    try:
        pipeline.submit_round(_make_round(0))
        assert trained.wait(timeout=5)

        first_snapshot = _wait_for_snapshot(pipeline)
        assert first_snapshot is not None
        assert first_snapshot.version == 1
        assert calls == [((0,), None)]

        trained.clear()
        pipeline.submit_round(_make_round(1))
        pipeline.submit_round(_make_round(2))
        assert not trained.wait(timeout=0.1)

        pipeline.close_epoch("request-0")
        assert trained.wait(timeout=5)

        second_snapshot = _wait_for_snapshot(pipeline)
        assert second_snapshot is not None
        assert second_snapshot.version == 2
        assert calls == [((0,), None), ((1, 2), "cache-1")]
        assert pipeline.update_stats[-1]["rounds"] == 2
    finally:
        pipeline.close()


def test_queue_size_is_reported_while_training_is_busy() -> None:
    trainer = _FakeTrainer(nn.Linear(1, 1))
    entered = Event()
    release = Event()
    queue_sizes: list[int] = []

    def train_rounds(trainer, rounds, persistent_cache, profile_timeline):
        assert not profile_timeline
        entered.set()
        release.wait(timeout=5)
        trainer.version += 1
        return persistent_cache, SimpleNamespace(
            model_version=trainer.version,
            round_count=len(rounds),
            target_count=1,
        )

    pipeline = AsyncDraftPipeline(
        trainer=trainer,
        on_queue_size=queue_sizes.append,
        train_rounds=train_rounds,
    )

    try:
        pipeline.submit_round(_make_round(0))
        assert entered.wait(timeout=5)

        queue_size = pipeline.submit_round(_make_round(1))
        assert queue_size >= 1
        assert queue_sizes[-1] == queue_size
    finally:
        release.set()
        pipeline.close()


def test_finish_resets_pipeline_for_next_request() -> None:
    trainer = _FakeTrainer(nn.Linear(1, 1))
    cache_was_empty: list[bool] = []

    def train_rounds(trainer, rounds, persistent_cache, profile_timeline):
        assert not profile_timeline
        cache_was_empty.append(persistent_cache is None)
        trainer.version += 1
        return f"cache-{trainer.version}", SimpleNamespace(
            model_version=trainer.version,
            round_count=len(rounds),
            target_count=1,
        )

    pipeline = AsyncDraftPipeline(
        trainer=trainer,
        train_rounds=train_rounds,
    )

    try:
        pipeline.submit_round(_make_round(0))
        pipeline.submit_round(_make_round(1))

        pipeline.finish_request("request-0")

        assert trainer.version == 0
        assert trainer.reset_count == 1
        assert cache_was_empty == [True]
        assert pipeline.poll_snapshot() is None

        pipeline.submit_round(_make_round(0, request_id="request-1"))

        snapshot = _wait_for_snapshot(pipeline)

        assert snapshot is not None
        assert snapshot.version == 1
        assert trainer.version == 1
        assert cache_was_empty == [True, True]
    finally:
        pipeline.close()


def test_training_failure_stops_pipeline_and_rejects_new_rounds() -> None:
    trainer = _FakeTrainer(nn.Linear(1, 1))
    calls = 0

    def train_rounds(trainer, rounds, persistent_cache, profile_timeline):
        assert not profile_timeline
        nonlocal calls
        calls += 1
        raise RuntimeError("training failed")

    pipeline = AsyncDraftPipeline(
        trainer=trainer,
        train_rounds=train_rounds,
    )

    pipeline.submit_round(_make_round(0))

    try:
        for _ in range(50):
            if pipeline.failed is not None:
                break
            Event().wait(0.01)

        assert isinstance(pipeline.failed, RuntimeError)
        pipeline.submit_round(_make_round(1))
        assert calls == 1
        with pytest.raises(RuntimeError, match="training failed"):
            pipeline.raise_if_failed()
        with pytest.raises(RuntimeError, match="training failed"):
            pipeline.finish_request("request-0")
        assert trainer.reset_count == 1
    finally:
        pipeline.close()
