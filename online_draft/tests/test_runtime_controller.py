# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from online_draft.models.qwen3_eagle3 import (
    Eagle3KVCache,
    Qwen3Eagle3Config,
    Qwen3Eagle3ForCausalLM,
)
from online_draft.runtime.contracts import (
    Eagle3Observation,
    Eagle3TrainResult,
    Eagle3TrainStatus,
    Eagle3TrainTask,
)
from online_draft.runtime.controller import (
    Eagle3ControllerConfig,
    Eagle3SyncController,
)
from online_draft.runtime.executor import DirectEagle3Executor
from online_draft.training.eagle3_batch import Eagle3DistillationBatch
from online_draft.training.eagle3_window import (
    Eagle3WindowResult,
)
from online_draft.training.trainer import DraftTrainer, TrainerConfig

HIDDEN_SIZE = 2
DRAFT_VOCAB_SIZE = 3


def _make_observation(
    *,
    step_id: int,
    anchor_position: int,
    full_prompt: bool = False,
    request_id: str = "request-1",
    request_generation: int = 0,
    source_weight_version: int = 4,
) -> Eagle3Observation:
    if full_prompt:
        prefill_positions = torch.arange(
            anchor_position + 1,
            dtype=torch.long,
        )
    else:
        prefill_positions = torch.tensor([anchor_position])
    prefill_length = prefill_positions.numel()

    batch = Eagle3DistillationBatch(
        prefill_positions=prefill_positions,
        prefill_input_embeds=torch.randn(prefill_length, HIDDEN_SIZE),
        prefill_aux_hidden_states=torch.randn(prefill_length, HIDDEN_SIZE),
        proposal_positions=torch.tensor([anchor_position]),
        draft_token_input_embeds=torch.empty(0, HIDDEN_SIZE),
        draft_recurrent_hidden_states=torch.empty(0, HIDDEN_SIZE),
        teacher_probabilities=torch.full(
            (1, DRAFT_VOCAB_SIZE),
            1.0 / DRAFT_VOCAB_SIZE,
        ),
        confirmed_positions=torch.tensor([anchor_position + 1]),
        confirmed_input_embeds=torch.randn(1, HIDDEN_SIZE),
        confirmed_aux_hidden_states=torch.randn(1, HIDDEN_SIZE),
        rejection_position=0,
    )
    return Eagle3Observation(
        request_id=request_id,
        request_generation=request_generation,
        step_id=step_id,
        source_weight_version=source_weight_version,
        batch=batch,
    )


def _make_cache(length: int) -> Eagle3KVCache:
    return (
        (
            torch.zeros(1, length, 1),
            torch.zeros(1, length, 1),
        ),
    )


class _StubExecutor:
    def __init__(
        self,
        *,
        fail: bool = False,
        advance_before_failure: bool = False,
        published_cache_length: int | None = None,
        raise_error: bool = False,
    ) -> None:
        self.fail = fail
        self.advance_before_failure = advance_before_failure
        self.published_cache_length = published_cache_length
        self.raise_error = raise_error
        self.version = 0
        self.tasks: list[Eagle3TrainTask] = []

    @property
    def model_version(self) -> int:
        return self.version

    def execute(self, task: Eagle3TrainTask) -> Eagle3TrainResult:
        self.tasks.append(task)
        start_version = self.version

        if self.raise_error:
            raise RuntimeError("unexpected executor error")

        if self.fail:
            if self.advance_before_failure:
                self.version += 1
            return Eagle3TrainResult.failed(
                task=task,
                cpu_start_version=start_version,
                cpu_end_version=self.version,
                error="injected executor failure",
            )

        self.version += 1
        cache_length = (
            task.window.end_confirmed_position + 1
            if self.published_cache_length is None
            else self.published_cache_length
        )
        cache = _make_cache(cache_length)
        training_result = Eagle3WindowResult(
            loss=1.0,
            model_version=self.version,
            mode=task.mode,
            round_count=task.window.round_count,
            input_length=task.window.round_count,
            target_count=task.window.round_count,
            confirmed_length=sum(
                batch.confirmed_length for batch in task.window.rounds
            ),
            persistent_cache_length=cache_length,
        )
        return Eagle3TrainResult.succeeded(
            task=task,
            cpu_start_version=start_version,
            updated_persistent_cache=cache,
            training_result=training_result,
        )


def test_controller_commits_one_fixed_training_window() -> None:
    executor = _StubExecutor()
    controller = Eagle3SyncController(
        executor,
        Eagle3ControllerConfig(window_size=2),
    )
    controller.start_request(
        request_id="request-1",
        request_generation=0,
    )

    first_result = controller.observe(
        _make_observation(
            step_id=0,
            anchor_position=1,
            full_prompt=True,
        )
    )

    assert first_result is None
    assert controller.pending_count == 1
    assert controller.request_state is not None
    assert controller.request_state.cache_length == 0

    result = controller.observe(
        _make_observation(
            step_id=1,
            anchor_position=2,
        )
    )

    assert result is not None
    assert result.status is Eagle3TrainStatus.SUCCEEDED
    assert controller.pending_count == 0
    assert controller.request_state.last_consumed_step_id == 1
    assert controller.request_state.cache_length == 4
    assert len(executor.tasks) == 1


def test_next_window_uses_previous_committed_cache() -> None:
    executor = _StubExecutor()
    controller = Eagle3SyncController(
        executor,
        Eagle3ControllerConfig(window_size=2),
    )
    controller.start_request(
        request_id="request-1",
        request_generation=0,
    )

    controller.observe(
        _make_observation(
            step_id=0,
            anchor_position=1,
            full_prompt=True,
        )
    )
    controller.observe(_make_observation(step_id=1, anchor_position=2))
    assert controller.request_state is not None
    first_cache = controller.request_state.persistent_cache

    controller.observe(_make_observation(step_id=2, anchor_position=3))
    controller.observe(_make_observation(step_id=3, anchor_position=4))

    assert len(executor.tasks) == 2
    assert executor.tasks[1].persistent_cache is first_cache
    assert executor.tasks[1].start_step_id == 2
    assert executor.tasks[1].end_step_id == 3
    assert controller.request_state.last_consumed_step_id == 3
    assert controller.request_state.cache_length == 6


def test_invalid_observation_does_not_modify_pending_window() -> None:
    controller = Eagle3SyncController(
        _StubExecutor(),
        Eagle3ControllerConfig(window_size=2),
    )
    controller.start_request(
        request_id="request-1",
        request_generation=0,
    )
    controller.observe(
        _make_observation(
            step_id=0,
            anchor_position=1,
            full_prompt=True,
        )
    )

    with pytest.raises(
        ValueError,
        match="next expected step 1",
    ):
        controller.observe(_make_observation(step_id=2, anchor_position=2))

    assert controller.pending_count == 1
    assert controller.request_state is not None
    assert controller.request_state.last_consumed_step_id is None


def test_failed_window_does_not_commit_request_state() -> None:
    controller = Eagle3SyncController(
        _StubExecutor(
            fail=True,
            advance_before_failure=True,
        )
    )
    controller.start_request(
        request_id="request-1",
        request_generation=0,
    )

    result = controller.observe(
        _make_observation(
            step_id=0,
            anchor_position=1,
            full_prompt=True,
        )
    )

    assert result is not None
    assert result.status is Eagle3TrainStatus.FAILED
    assert result.cpu_end_version == 1
    assert controller.failure is result
    assert controller.request_state is not None
    assert controller.request_state.cache_length == 0
    assert controller.request_state.last_consumed_step_id is None

    with pytest.raises(RuntimeError, match="failed request"):
        controller.observe(_make_observation(step_id=0, anchor_position=1))


def test_invalid_published_cache_fails_without_state_commit() -> None:
    controller = Eagle3SyncController(_StubExecutor(published_cache_length=0))
    controller.start_request(
        request_id="request-1",
        request_generation=0,
    )

    result = controller.observe(
        _make_observation(
            step_id=0,
            anchor_position=1,
            full_prompt=True,
        )
    )

    assert result is not None
    assert result.status is Eagle3TrainStatus.FAILED
    assert "must be longer" in result.error
    assert controller.request_state is not None
    assert controller.request_state.cache_length == 0
    assert controller.request_state.last_consumed_step_id is None


def test_executor_exception_is_converted_to_terminal_failure() -> None:
    controller = Eagle3SyncController(_StubExecutor(raise_error=True))
    controller.start_request(
        request_id="request-1",
        request_generation=0,
    )

    result = controller.observe(
        _make_observation(
            step_id=0,
            anchor_position=1,
            full_prompt=True,
        )
    )

    assert result is not None
    assert result.status is Eagle3TrainStatus.FAILED
    assert result.error == "RuntimeError: unexpected executor error"
    assert controller.failure is result

    with pytest.raises(RuntimeError, match="failed request"):
        controller.observe(_make_observation(step_id=0, anchor_position=1))


def test_finish_discards_tail_and_reused_id_requires_new_generation() -> None:
    controller = Eagle3SyncController(
        _StubExecutor(),
        Eagle3ControllerConfig(window_size=2),
    )
    controller.start_request(
        request_id="request-1",
        request_generation=0,
    )
    controller.observe(
        _make_observation(
            step_id=0,
            anchor_position=1,
            full_prompt=True,
        )
    )

    discarded_count = controller.finish_request(
        request_id="request-1",
        request_generation=0,
    )

    assert discarded_count == 1
    assert controller.request_state is None
    assert controller.pending_count == 0

    with pytest.raises(ValueError, match="advance request_generation"):
        controller.start_request(
            request_id="request-1",
            request_generation=0,
        )

    controller.start_request(
        request_id="request-1",
        request_generation=1,
    )
    assert controller.request_state is not None
    assert controller.request_state.request_generation == 1


def test_controller_executes_real_training_stack() -> None:
    model = Qwen3Eagle3ForCausalLM(
        Qwen3Eagle3Config(
            hidden_size=HIDDEN_SIZE,
            intermediate_size=4,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=HIDDEN_SIZE,
            num_hidden_layers=1,
            target_vocab_size=4,
            draft_vocab_size=DRAFT_VOCAB_SIZE,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            num_aux_hidden_states=1,
        )
    )
    executor = DirectEagle3Executor(
        DraftTrainer(
            model,
            TrainerConfig(learning_rate=1e-2),
        )
    )
    controller = Eagle3SyncController(
        executor,
        Eagle3ControllerConfig(window_size=2),
    )
    controller.start_request(
        request_id="request-1",
        request_generation=0,
    )

    first_result = controller.observe(
        _make_observation(
            step_id=0,
            anchor_position=1,
            full_prompt=True,
        )
    )
    result = controller.observe(
        _make_observation(
            step_id=1,
            anchor_position=2,
        )
    )

    assert first_result is None
    assert result is not None
    assert result.status is Eagle3TrainStatus.SUCCEEDED
    assert result.cpu_start_version == 0
    assert result.cpu_end_version == 1
    assert controller.request_state is not None
    assert controller.request_state.last_consumed_step_id == 1
    assert controller.request_state.cache_length == 4
