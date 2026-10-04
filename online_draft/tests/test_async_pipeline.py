# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from threading import Event
from types import SimpleNamespace

import pytest
import torch
from online_draft.runtime.async_pipeline import AsyncEagle3Pipeline
from online_draft.runtime.capture_runtime import Eagle3PreparedRound
from online_draft.training.eagle3_batch import Eagle3DistillationBatch
from online_draft.training.eagle3_window import Eagle3WindowMode
from torch import nn

HIDDEN_SIZE = 2
AUX_HIDDEN_STATES = 1
DRAFT_VOCAB_SIZE = 3


@dataclass
class _FakeTrainer:
    model: nn.Module
    version: int = 0

    @property
    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.model.named_parameters())


def _make_round(step_id: int, anchor: int, *, prompt: bool) -> Eagle3PreparedRound:
    prefill_positions = (
        torch.tensor([0, anchor], dtype=torch.long)
        if prompt
        else torch.tensor([anchor], dtype=torch.long)
    )
    prefill_length = prefill_positions.shape[0]

    batch = Eagle3DistillationBatch(
        prefill_positions=prefill_positions,
        prefill_input_embeds=torch.zeros(prefill_length, HIDDEN_SIZE),
        prefill_aux_hidden_states=torch.zeros(
            prefill_length,
            HIDDEN_SIZE * AUX_HIDDEN_STATES,
        ),
        proposal_positions=torch.tensor([anchor, anchor + 1]),
        draft_token_input_embeds=torch.zeros(1, HIDDEN_SIZE),
        draft_recurrent_hidden_states=torch.zeros(1, HIDDEN_SIZE),
        teacher_probabilities=torch.full((2, DRAFT_VOCAB_SIZE), 1.0 / DRAFT_VOCAB_SIZE),
        confirmed_positions=torch.tensor([anchor + 1]),
        confirmed_input_embeds=torch.zeros(1, HIDDEN_SIZE),
        confirmed_aux_hidden_states=torch.zeros(
            1,
            HIDDEN_SIZE * AUX_HIDDEN_STATES,
        ),
        rejection_position=0,
    )
    return Eagle3PreparedRound(
        request_id="request-0",
        step_id=step_id,
        source_weight_version=0,
        batch=batch,
    )


def _wait_for_snapshot(pipeline: AsyncEagle3Pipeline):
    for _ in range(500):
        snapshot = pipeline.poll_snapshot()
        if snapshot is not None:
            return snapshot
        Event().wait(0.01)
    return None


def test_bootstrap_and_epoch_close_publish_delayed_snapshots() -> None:
    trainer = _FakeTrainer(nn.Linear(1, 1))
    calls: list[tuple[int, bool, Eagle3WindowMode]] = []
    prefill_positions: list[list[int]] = []
    trained = Event()

    def train_window(trainer, window, persistent_cache, mode):
        calls.append((window.round_count, persistent_cache is None, mode))
        prefill_positions.append(window.rounds[0].prefill_positions.tolist())
        trainer.version += 1
        trained.set()
        return f"cache-{trainer.version}", SimpleNamespace(
            model_version=trainer.version
        )

    pipeline = AsyncEagle3Pipeline(
        trainer=trainer,
        train_window=train_window,
    )

    try:
        pipeline.submit_round(_make_round(0, 1, prompt=True))
        assert trained.wait(timeout=5)

        first_snapshot = _wait_for_snapshot(pipeline)
        assert first_snapshot is not None
        assert first_snapshot.version == 1
        assert calls == [(1, True, Eagle3WindowMode.CONFIRMED_PATH)]
        assert prefill_positions == [[0, 1]]

        trained.clear()
        pipeline.submit_round(_make_round(1, 2, prompt=False))
        assert not trained.wait(timeout=0.1)

        pipeline.close_epoch("request-0")
        assert trained.wait(timeout=5)

        second_snapshot = _wait_for_snapshot(pipeline)
        assert second_snapshot is not None
        assert second_snapshot.version == 2
        assert calls == [
            (1, True, Eagle3WindowMode.CONFIRMED_PATH),
            (1, False, Eagle3WindowMode.CONFIRMED_PATH),
        ]
        assert prefill_positions == [[0, 1], [2]]
    finally:
        pipeline.close()


def test_queue_size_is_reported_while_training_is_busy() -> None:
    trainer = _FakeTrainer(nn.Linear(1, 1))
    entered = Event()
    release = Event()
    queue_sizes: list[int] = []

    def train_window(trainer, window, persistent_cache, mode):
        entered.set()
        release.wait(timeout=5)
        trainer.version += 1
        return persistent_cache, SimpleNamespace(model_version=trainer.version)

    pipeline = AsyncEagle3Pipeline(
        trainer=trainer,
        on_queue_size=queue_sizes.append,
        train_window=train_window,
    )

    try:
        pipeline.submit_round(_make_round(0, 1, prompt=True))
        assert entered.wait(timeout=5)

        queue_size = pipeline.submit_round(_make_round(1, 2, prompt=False))
        assert queue_size >= 1
        assert queue_sizes[-1] == queue_size
    finally:
        release.set()
        pipeline.close()


def test_training_failure_stops_pipeline_and_rejects_new_rounds() -> None:
    trainer = _FakeTrainer(nn.Linear(1, 1))
    calls = 0

    def train_window(trainer, window, persistent_cache, mode):
        nonlocal calls
        calls += 1
        raise RuntimeError("training failed")

    pipeline = AsyncEagle3Pipeline(
        trainer=trainer,
        train_window=train_window,
    )

    pipeline.submit_round(_make_round(0, 1, prompt=True))

    try:
        for _ in range(50):
            if pipeline.failed is not None:
                break
            Event().wait(0.01)

        assert isinstance(pipeline.failed, RuntimeError)
        pipeline.submit_round(_make_round(1, 2, prompt=False))
        assert calls == 1
    finally:
        with pytest.raises(RuntimeError, match="training failed"):
            pipeline.close()
