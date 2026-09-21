# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from online_draft.models.qwen3_eagle3 import Eagle3KVCache
from online_draft.runtime.contracts import (
    Eagle3Observation,
    Eagle3TrainStatus,
    Eagle3TrainTask,
)
from online_draft.runtime.executor import DirectEagle3Executor
from online_draft.training.eagle3_batch import Eagle3DistillationBatch
from online_draft.training.eagle3_window import (
    Eagle3TrainingWindow,
    Eagle3WindowMode,
    Eagle3WindowResult,
)
from online_draft.training.trainer import DraftTrainer, TrainerConfig


def _make_task() -> Eagle3TrainTask:
    batch = Eagle3DistillationBatch(
        prefill_positions=torch.tensor([0]),
        prefill_input_embeds=torch.randn(1, 1),
        prefill_aux_hidden_states=torch.randn(1, 1),
        proposal_positions=torch.tensor([0]),
        draft_token_input_embeds=torch.empty(0, 1),
        draft_recurrent_hidden_states=torch.empty(0, 1),
        teacher_probabilities=torch.ones(1, 1),
        confirmed_positions=torch.tensor([1]),
        confirmed_input_embeds=torch.randn(1, 1),
        confirmed_aux_hidden_states=torch.randn(1, 1),
        rejection_position=0,
    )
    observation = Eagle3Observation(
        request_id="request-1",
        request_generation=0,
        step_id=0,
        source_weight_version=4,
        batch=batch,
    )

    return Eagle3TrainTask(
        task_id=0,
        observations=(observation,),
        mode=Eagle3WindowMode.CONFIRMED_PATH,
    )


def _make_trainer() -> DraftTrainer:
    return DraftTrainer(
        torch.nn.Linear(1, 1, bias=False),
        TrainerConfig(learning_rate=1e-2),
    )


def _make_cache() -> Eagle3KVCache:
    return (
        (
            torch.zeros(1, 2, 1),
            torch.zeros(1, 2, 1),
        ),
    )


def _update_trainer_once(trainer: DraftTrainer) -> float:
    parameter = next(trainer.model.parameters())
    return trainer.backward_and_step(parameter.square().sum())


def test_direct_executor_returns_success_after_one_cpu_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _make_trainer()
    executor = DirectEagle3Executor(trainer)
    task = _make_task()
    cache = _make_cache()

    def train_once(
        *,
        trainer: DraftTrainer,
        window: Eagle3TrainingWindow,
        persistent_cache: Eagle3KVCache | None,
        mode: Eagle3WindowMode,
    ) -> tuple[Eagle3KVCache, Eagle3WindowResult]:
        assert window is task.window
        assert persistent_cache is task.persistent_cache
        assert mode is task.mode
        loss = _update_trainer_once(trainer)
        return cache, Eagle3WindowResult(
            loss=loss,
            model_version=trainer.version,
            mode=mode,
            round_count=1,
            input_length=1,
            target_count=1,
            confirmed_length=1,
            persistent_cache_length=2,
        )

    monkeypatch.setattr(
        "online_draft.runtime.executor.train_eagle3_window",
        train_once,
    )

    result = executor.execute(task)

    assert result.status is Eagle3TrainStatus.SUCCEEDED
    assert result.cpu_start_version == 0
    assert result.cpu_end_version == 1
    assert result.updated_persistent_cache is cache
    assert executor.model_version == 1


def test_direct_executor_returns_failure_before_cpu_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = DirectEagle3Executor(_make_trainer())
    task = _make_task()

    def fail_before_update(**_kwargs: object) -> None:
        raise ValueError("invalid training input")

    monkeypatch.setattr(
        "online_draft.runtime.executor.train_eagle3_window",
        fail_before_update,
    )

    result = executor.execute(task)

    assert result.status is Eagle3TrainStatus.FAILED
    assert result.cpu_start_version == 0
    assert result.cpu_end_version == 0
    assert result.updated_persistent_cache is None
    assert result.error == "ValueError: invalid training input"


def test_direct_executor_does_not_publish_cache_after_post_update_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _make_trainer()
    executor = DirectEagle3Executor(trainer)
    task = _make_task()

    def fail_after_update(
        *,
        trainer: DraftTrainer,
        **_kwargs: object,
    ) -> None:
        _update_trainer_once(trainer)
        raise RuntimeError("cache rebuild failed")

    monkeypatch.setattr(
        "online_draft.runtime.executor.train_eagle3_window",
        fail_after_update,
    )

    result = executor.execute(task)

    assert result.status is Eagle3TrainStatus.FAILED
    assert result.cpu_start_version == 0
    assert result.cpu_end_version == 1
    assert result.training_result is None
    assert result.updated_persistent_cache is None
    assert result.error == "RuntimeError: cache rebuild failed"
