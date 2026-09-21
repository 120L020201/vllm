# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import pytest
import torch
from online_draft.models.qwen3_eagle3 import Eagle3KVCache
from online_draft.runtime.contracts import (
    Eagle3Observation,
    Eagle3TrainResult,
    Eagle3TrainStatus,
    Eagle3TrainTask,
)
from online_draft.training.eagle3_batch import Eagle3DistillationBatch
from online_draft.training.eagle3_window import (
    Eagle3WindowMode,
    Eagle3WindowResult,
)

HIDDEN_SIZE = 4
NUM_AUX_HIDDEN_STATES = 3
DRAFT_VOCAB_SIZE = 5


def _make_batch(
    *,
    anchor_position: int,
    draft_length: int = 2,
    rejection_position: int | None = 0,
    prefill_length: int = 1,
) -> Eagle3DistillationBatch:
    confirmed_length = (
        draft_length + 1 if rejection_position is None else rejection_position + 1
    )
    prefill_positions = torch.arange(
        anchor_position - prefill_length + 1,
        anchor_position + 1,
        dtype=torch.long,
    )

    return Eagle3DistillationBatch(
        prefill_positions=prefill_positions,
        prefill_input_embeds=torch.randn(prefill_length, HIDDEN_SIZE),
        prefill_aux_hidden_states=torch.randn(
            prefill_length,
            HIDDEN_SIZE * NUM_AUX_HIDDEN_STATES,
        ),
        proposal_positions=torch.arange(
            anchor_position,
            anchor_position + draft_length,
            dtype=torch.long,
        ),
        draft_token_input_embeds=torch.randn(
            draft_length - 1,
            HIDDEN_SIZE,
        ),
        draft_recurrent_hidden_states=torch.randn(
            draft_length - 1,
            HIDDEN_SIZE,
        ),
        teacher_probabilities=torch.randn(
            draft_length,
            DRAFT_VOCAB_SIZE,
        ).softmax(dim=-1),
        confirmed_positions=torch.arange(
            anchor_position + 1,
            anchor_position + 1 + confirmed_length,
            dtype=torch.long,
        ),
        confirmed_input_embeds=torch.randn(
            confirmed_length,
            HIDDEN_SIZE,
        ),
        confirmed_aux_hidden_states=torch.randn(
            confirmed_length,
            HIDDEN_SIZE * NUM_AUX_HIDDEN_STATES,
        ),
        rejection_position=rejection_position,
    )


def _make_observation(
    *,
    step_id: int,
    anchor_position: int,
    request_id: str = "request-1",
    request_generation: int = 2,
    source_weight_version: int = 7,
    prefill_length: int = 1,
) -> Eagle3Observation:
    return Eagle3Observation(
        request_id=request_id,
        request_generation=request_generation,
        step_id=step_id,
        source_weight_version=source_weight_version,
        batch=_make_batch(
            anchor_position=anchor_position,
            prefill_length=prefill_length,
        ),
    )


def _make_first_task() -> Eagle3TrainTask:
    return Eagle3TrainTask(
        task_id=3,
        observations=(
            _make_observation(
                step_id=4,
                anchor_position=1,
                source_weight_version=7,
                prefill_length=2,
            ),
            _make_observation(
                step_id=5,
                anchor_position=2,
                source_weight_version=8,
            ),
        ),
        mode=Eagle3WindowMode.CONFIRMED_PATH,
    )


def _make_cache(length: int) -> Eagle3KVCache:
    return (
        (
            torch.zeros(2, length, 4),
            torch.zeros(2, length, 4),
        ),
    )


def _make_window_result(
    *,
    model_version: int = 8,
    persistent_cache_length: int = 4,
) -> Eagle3WindowResult:
    return Eagle3WindowResult(
        loss=1.5,
        model_version=model_version,
        mode=Eagle3WindowMode.CONFIRMED_PATH,
        round_count=2,
        input_length=2,
        target_count=2,
        confirmed_length=2,
        persistent_cache_length=persistent_cache_length,
    )


def test_observation_exposes_round_metadata_without_copying_payload() -> None:
    batch = _make_batch(anchor_position=1)
    observation = Eagle3Observation(
        request_id="request-1",
        request_generation=2,
        step_id=4,
        source_weight_version=7,
        batch=batch,
    )

    assert observation.batch is batch
    assert observation.anchor_position == 1
    assert observation.end_confirmed_position == 2


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("request_id", ""),
        ("request_generation", -1),
        ("step_id", True),
        ("source_weight_version", -1),
    ],
)
def test_observation_rejects_invalid_metadata(
    field_name: str,
    invalid_value: object,
) -> None:
    values = {
        "request_id": "request-1",
        "request_generation": 2,
        "step_id": 4,
        "source_weight_version": 7,
        "batch": _make_batch(anchor_position=1),
    }
    values[field_name] = invalid_value

    with pytest.raises(ValueError):
        Eagle3Observation(**values)  # type: ignore[arg-type]


def test_task_builds_one_consecutive_training_window() -> None:
    task = _make_first_task()

    assert task.request_id == "request-1"
    assert task.request_generation == 2
    assert task.start_step_id == 4
    assert task.end_step_id == 5
    assert task.source_weight_versions == (7, 8)
    assert task.is_initial
    assert task.prompt_prefix_length == 1
    assert task.window.rounds == tuple(
        observation.batch for observation in task.observations
    )


def test_task_rejects_nonconsecutive_steps() -> None:
    first = _make_observation(step_id=4, anchor_position=1)
    second = _make_observation(step_id=6, anchor_position=2)

    with pytest.raises(
        ValueError,
        match="step_ids must be consecutive",
    ):
        Eagle3TrainTask(
            task_id=3,
            observations=(first, second),
            mode=Eagle3WindowMode.CONFIRMED_PATH,
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("request_id", "request-2"),
        ("request_generation", 3),
    ],
)
def test_task_rejects_mixed_request_scope(
    field_name: str,
    invalid_value: object,
) -> None:
    first = _make_observation(step_id=4, anchor_position=1)
    second = _make_observation(step_id=5, anchor_position=2)
    second = replace(second, **{field_name: invalid_value})

    with pytest.raises(ValueError, match="must share"):
        Eagle3TrainTask(
            task_id=3,
            observations=(first, second),
            mode=Eagle3WindowMode.CONFIRMED_PATH,
        )


def test_task_rejects_position_gap_between_rounds() -> None:
    first = _make_observation(step_id=4, anchor_position=1)
    second = _make_observation(step_id=5, anchor_position=3)

    with pytest.raises(
        ValueError,
        match="anchor must equal",
    ):
        Eagle3TrainTask(
            task_id=3,
            observations=(first, second),
            mode=Eagle3WindowMode.CONFIRMED_PATH,
        )


def test_later_task_requires_cache_through_first_anchor() -> None:
    observation = _make_observation(
        step_id=6,
        anchor_position=3,
    )

    with pytest.raises(
        ValueError,
        match="must end at the first task anchor",
    ):
        Eagle3TrainTask(
            task_id=4,
            observations=(observation,),
            mode=Eagle3WindowMode.PROPOSAL_CANVAS,
            persistent_cache=_make_cache(3),
        )


def test_initial_task_rejects_repeated_prefill_after_first_round() -> None:
    first = _make_observation(
        step_id=0,
        anchor_position=1,
        prefill_length=2,
    )
    second = _make_observation(
        step_id=1,
        anchor_position=2,
        prefill_length=2,
    )

    with pytest.raises(
        ValueError,
        match="exactly one prefill anchor",
    ):
        Eagle3TrainTask(
            task_id=0,
            observations=(first, second),
            mode=Eagle3WindowMode.CONFIRMED_PATH,
        )


def test_later_task_accepts_only_anchor_prefill() -> None:
    task = Eagle3TrainTask(
        task_id=1,
        observations=(
            _make_observation(
                step_id=2,
                anchor_position=3,
            ),
        ),
        mode=Eagle3WindowMode.CONFIRMED_PATH,
        persistent_cache=_make_cache(4),
    )

    assert not task.is_initial
    assert task.prompt_prefix_length == 0


def test_later_task_rejects_repeated_prefill_history() -> None:
    with pytest.raises(
        ValueError,
        match="exactly one prefill anchor",
    ):
        Eagle3TrainTask(
            task_id=1,
            observations=(
                _make_observation(
                    step_id=2,
                    anchor_position=3,
                    prefill_length=2,
                ),
            ),
            mode=Eagle3WindowMode.CONFIRMED_PATH,
            persistent_cache=_make_cache(4),
        )


def test_success_result_publishes_updated_cache_and_versions() -> None:
    task = _make_first_task()
    cache = _make_cache(4)
    result = Eagle3TrainResult.succeeded(
        task=task,
        cpu_start_version=7,
        updated_persistent_cache=cache,
        training_result=_make_window_result(),
    )

    assert result.status is Eagle3TrainStatus.SUCCEEDED
    assert result.cpu_start_version == 7
    assert result.cpu_end_version == 8
    assert result.updated_persistent_cache is cache
    assert result.error is None


def test_failure_result_does_not_publish_partial_outputs() -> None:
    task = _make_first_task()
    result = Eagle3TrainResult.failed(
        task=task,
        cpu_start_version=7,
        cpu_end_version=8,
        error="cache rebuild failed",
    )

    assert result.status is Eagle3TrainStatus.FAILED
    assert result.cpu_end_version == 8
    assert result.training_result is None
    assert result.updated_persistent_cache is None
    assert result.error == "cache rebuild failed"


def test_success_result_rejects_cache_length_mismatch() -> None:
    task = _make_first_task()

    with pytest.raises(
        ValueError,
        match="cache length must match",
    ):
        Eagle3TrainResult.succeeded(
            task=task,
            cpu_start_version=7,
            updated_persistent_cache=_make_cache(5),
            training_result=_make_window_result(),
        )
