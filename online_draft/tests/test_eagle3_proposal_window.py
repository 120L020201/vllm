# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import math
from dataclasses import replace

import pytest
import torch
from online_draft.models.qwen3_eagle3 import (
    Eagle3AttentionBackend,
    Eagle3KVCache,
    Qwen3Eagle3Config,
    Qwen3Eagle3ForCausalLM,
)
from online_draft.training.eagle3_batch import (
    Eagle3DistillationBatch,
)
from online_draft.training.eagle3_cache import (
    append_confirmed_to_persistent_cache,
    persistent_cache_length,
    prepare_training_window_spine,
)
from online_draft.training.eagle3_distill import (
    window_forward_kl_loss,
)
from online_draft.training.eagle3_inputs import (
    prepare_eagle3_confirmed_path_inputs,
    prepare_eagle3_proposal_window_inputs,
    prepare_eagle3_training_inputs,
)
from online_draft.training.eagle3_window import (
    Eagle3TrainingWindow,
    Eagle3WindowMode,
    train_eagle3_window,
)
from online_draft.training.trainer import (
    DraftTrainer,
    TrainerConfig,
)

HIDDEN_SIZE = 4
NUM_AUX_HIDDEN_STATES = 3
DRAFT_VOCAB_SIZE = 5
TARGET_VOCAB_SIZE = 8


def _make_model(
    dtype: torch.dtype = torch.float32,
    attention_backend: Eagle3AttentionBackend = Eagle3AttentionBackend.EAGER,
) -> Qwen3Eagle3ForCausalLM:
    config = Qwen3Eagle3Config(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        num_hidden_layers=2,
        target_vocab_size=TARGET_VOCAB_SIZE,
        draft_vocab_size=DRAFT_VOCAB_SIZE,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        num_aux_hidden_states=NUM_AUX_HIDDEN_STATES,
        attention_backend=attention_backend,
    )

    return Qwen3Eagle3ForCausalLM(config).to(dtype=dtype)


def _make_batch(
    *,
    anchor_position: int,
    draft_length: int,
    rejection_position: int | None,
    feature_dtype: torch.dtype = torch.float32,
) -> Eagle3DistillationBatch:
    confirmed_length = (
        draft_length + 1 if rejection_position is None else rejection_position + 1
    )
    prefill_length = 2

    return Eagle3DistillationBatch(
        prefill_positions=torch.arange(
            anchor_position - prefill_length + 1,
            anchor_position + 1,
            dtype=torch.long,
        ),
        prefill_input_embeds=torch.randn(
            prefill_length,
            HIDDEN_SIZE,
            dtype=feature_dtype,
        ),
        prefill_aux_hidden_states=torch.randn(
            prefill_length,
            HIDDEN_SIZE * NUM_AUX_HIDDEN_STATES,
            dtype=feature_dtype,
        ),
        proposal_positions=torch.arange(
            anchor_position,
            anchor_position + draft_length,
            dtype=torch.long,
        ),
        draft_token_input_embeds=torch.randn(
            draft_length - 1,
            HIDDEN_SIZE,
            dtype=feature_dtype,
        ),
        draft_recurrent_hidden_states=torch.randn(
            draft_length - 1,
            HIDDEN_SIZE,
            dtype=feature_dtype,
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
            dtype=feature_dtype,
        ),
        confirmed_aux_hidden_states=torch.randn(
            confirmed_length,
            HIDDEN_SIZE * NUM_AUX_HIDDEN_STATES,
            dtype=feature_dtype,
        ),
        rejection_position=rejection_position,
    )


def _align_next_anchor(
    previous: Eagle3DistillationBatch,
    current: Eagle3DistillationBatch,
) -> Eagle3DistillationBatch:
    input_embeds = current.prefill_input_embeds.clone()
    input_embeds[-1] = previous.confirmed_input_embeds[-1]
    auxiliary_hidden_states = current.prefill_aux_hidden_states.clone()
    auxiliary_hidden_states[-1] = previous.confirmed_aux_hidden_states[-1]

    return replace(
        current,
        prefill_input_embeds=input_embeds,
        prefill_aux_hidden_states=auxiliary_hidden_states,
    )


def _make_two_round_window(
    feature_dtype: torch.dtype = torch.float32,
) -> Eagle3TrainingWindow:
    first = _make_batch(
        anchor_position=1,
        draft_length=3,
        rejection_position=None,
        feature_dtype=feature_dtype,
    )
    second = _make_batch(
        anchor_position=5,
        draft_length=3,
        rejection_position=1,
        feature_dtype=feature_dtype,
    )
    second = _align_next_anchor(first, second)

    return Eagle3TrainingWindow(rounds=(first, second))


def _slice_cache_before_position(
    cache: Eagle3KVCache,
    position: int,
) -> Eagle3KVCache | None:
    if position == 0:
        return None

    return tuple(
        (
            key[:, :position].detach(),
            value[:, :position].detach(),
        )
        for key, value in cache
    )


def _assert_cache_detached(cache: Eagle3KVCache) -> None:
    for key, value in cache:
        assert not key.requires_grad
        assert not value.requires_grad
        assert key.grad_fn is None
        assert value.grad_fn is None
        assert not torch.is_inference(key)
        assert not torch.is_inference(value)


def _assert_cache_close(
    actual: Eagle3KVCache,
    expected: Eagle3KVCache,
) -> None:
    for actual_layer, expected_layer in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_layer[0], expected_layer[0])
        torch.testing.assert_close(actual_layer[1], expected_layer[1])


def _forward_full_history(
    model: Qwen3Eagle3ForCausalLM,
    window: Eagle3TrainingWindow,
) -> Eagle3KVCache:
    first = window.rounds[0]
    positions = torch.cat(
        (
            first.prefill_positions,
            *(batch.confirmed_positions for batch in window.rounds),
        )
    )
    input_embeds = torch.cat(
        (
            first.prefill_input_embeds,
            *(batch.confirmed_input_embeds for batch in window.rounds),
        )
    )
    auxiliary_hidden_states = torch.cat(
        (
            first.prefill_aux_hidden_states,
            *(batch.confirmed_aux_hidden_states for batch in window.rounds),
        )
    )

    with torch.no_grad():
        hidden_states = model.combine_hidden_states(auxiliary_hidden_states)
        output = model(
            positions=positions,
            input_embeds=input_embeds,
            hidden_states=hidden_states,
        )

    return output.past_key_values


def test_explicit_causal_attention_mask_matches_default() -> None:
    torch.manual_seed(20)

    model = _make_model().eval()
    positions = torch.arange(5)
    input_embeds = torch.randn(5, HIDDEN_SIZE)
    hidden_states = torch.randn(5, HIDDEN_SIZE)
    attention_mask = torch.triu(
        torch.ones(5, 5, dtype=torch.bool),
        diagonal=1,
    )

    with torch.no_grad():
        default_output = model(
            positions=positions,
            input_embeds=input_embeds,
            hidden_states=hidden_states,
        )
        masked_output = model(
            positions=positions,
            input_embeds=input_embeds,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

    torch.testing.assert_close(
        masked_output.hidden_states,
        default_output.hidden_states,
    )
    torch.testing.assert_close(
        masked_output.recurrent_hidden_states,
        default_output.recurrent_hidden_states,
    )


def test_proposal_window_mask_isolates_round_branches() -> None:
    torch.manual_seed(21)

    model = _make_model()
    first = _make_batch(
        anchor_position=1,
        draft_length=3,
        rejection_position=0,
    )
    second = _make_batch(
        anchor_position=2,
        draft_length=2,
        rejection_position=None,
    )
    second = _align_next_anchor(first, second)
    window = Eagle3TrainingWindow(rounds=(first, second))

    inputs = prepare_eagle3_proposal_window_inputs(
        model,
        window,
        spine_length=6,
    )

    expected_mask = torch.ones(5, 11, dtype=torch.bool)
    expected_mask[0, [0, 6]] = False
    expected_mask[1, [0, 6, 7]] = False
    expected_mask[2, [0, 6, 7, 8]] = False
    expected_mask[3, [0, 1, 9]] = False
    expected_mask[4, [0, 1, 9, 10]] = False

    assert inputs.positions.tolist() == [1, 2, 3, 2, 3]
    assert inputs.input_length == 5
    assert inputs.target_count == 5
    assert inputs.rejection_target_mask.tolist() == [
        True,
        False,
        False,
        False,
        False,
    ]
    assert torch.equal(inputs.attention_mask, expected_mask)


def test_packed_proposal_forward_matches_independent_rounds() -> None:
    torch.manual_seed(22)

    packed_model = _make_model()
    independent_model = copy.deepcopy(packed_model)
    window = _make_two_round_window()

    packed_spine = prepare_training_window_spine(
        packed_model,
        window,
        persistent_cache=None,
    )
    _assert_cache_detached(packed_spine)
    packed_inputs = prepare_eagle3_proposal_window_inputs(
        packed_model,
        window,
        spine_length=persistent_cache_length(packed_spine),
    )
    packed_output = packed_model(
        positions=packed_inputs.positions,
        input_embeds=packed_inputs.input_embeds,
        hidden_states=packed_inputs.hidden_states,
        past_key_values=packed_spine,
        attention_mask=packed_inputs.attention_mask,
    )
    packed_logits = packed_model.compute_draft_logits(packed_output.hidden_states)
    packed_loss = window_forward_kl_loss(
        student_logits=packed_logits,
        teacher_probabilities=packed_inputs.teacher_probabilities,
        rejection_target_mask=packed_inputs.rejection_target_mask,
    )
    packed_loss.backward()

    independent_spine = prepare_training_window_spine(
        independent_model,
        window,
        persistent_cache=None,
    )
    independent_logits_parts: list[torch.Tensor] = []
    rejection_mask_parts: list[torch.Tensor] = []

    for round_batch in window.rounds:
        round_inputs = prepare_eagle3_training_inputs(
            independent_model,
            round_batch,
        )
        anchor_position = int(round_inputs.positions[0].item())
        round_output = independent_model(
            positions=round_inputs.positions,
            input_embeds=round_inputs.input_embeds,
            hidden_states=round_inputs.hidden_states,
            past_key_values=_slice_cache_before_position(
                independent_spine,
                anchor_position,
            ),
        )
        independent_logits_parts.append(
            independent_model.compute_draft_logits(round_output.hidden_states)
        )

        rejection_mask = torch.zeros(
            round_batch.draft_length,
            dtype=torch.bool,
        )
        if round_batch.rejection_position is not None:
            rejection_mask[round_batch.rejection_position] = True
        rejection_mask_parts.append(rejection_mask)

    independent_logits = torch.cat(independent_logits_parts, dim=0)
    independent_loss = window_forward_kl_loss(
        student_logits=independent_logits,
        teacher_probabilities=packed_inputs.teacher_probabilities,
        rejection_target_mask=torch.cat(rejection_mask_parts, dim=0),
    )
    independent_loss.backward()

    torch.testing.assert_close(packed_logits, independent_logits)
    torch.testing.assert_close(packed_loss, independent_loss)

    independent_parameters = dict(independent_model.named_parameters())
    gradient_count = 0
    for name, packed_parameter in packed_model.named_parameters():
        independent_parameter = independent_parameters[name]
        assert (packed_parameter.grad is None) == (independent_parameter.grad is None)
        if packed_parameter.grad is not None:
            gradient_count += 1
            torch.testing.assert_close(
                packed_parameter.grad,
                independent_parameter.grad,
                rtol=1e-5,
                atol=1e-6,
            )

    assert gradient_count > 0


def test_packed_confirmed_forward_matches_independent_rounds() -> None:
    torch.manual_seed(28)

    packed_model = _make_model()
    independent_model = copy.deepcopy(packed_model)
    window = _make_two_round_window()

    packed_spine = prepare_training_window_spine(
        packed_model,
        window,
        persistent_cache=None,
    )
    packed_inputs = prepare_eagle3_confirmed_path_inputs(
        packed_model,
        window,
        spine_length=persistent_cache_length(packed_spine),
    )
    packed_output = packed_model(
        positions=packed_inputs.positions,
        input_embeds=packed_inputs.input_embeds,
        hidden_states=packed_inputs.hidden_states,
        past_key_values=packed_spine,
        attention_mask=packed_inputs.attention_mask,
    )
    packed_logits = packed_model.compute_draft_logits(packed_output.hidden_states)
    packed_loss = window_forward_kl_loss(
        student_logits=packed_logits,
        teacher_probabilities=packed_inputs.teacher_probabilities,
        rejection_target_mask=packed_inputs.rejection_target_mask,
    )
    packed_loss.backward()

    independent_spine = prepare_training_window_spine(
        independent_model,
        window,
        persistent_cache=None,
    )
    independent_logits_parts: list[torch.Tensor] = []
    rejection_mask_parts: list[torch.Tensor] = []

    for round_batch in window.rounds:
        round_inputs = prepare_eagle3_training_inputs(
            independent_model,
            round_batch,
        )
        target_count = (
            round_batch.draft_length
            if round_batch.rejection_position is None
            else round_batch.confirmed_length
        )
        anchor_position = int(round_inputs.positions[0].item())
        round_output = independent_model(
            positions=round_inputs.positions[:target_count],
            input_embeds=round_inputs.input_embeds[:target_count],
            hidden_states=round_inputs.hidden_states[:target_count],
            past_key_values=_slice_cache_before_position(
                independent_spine,
                anchor_position,
            ),
        )
        independent_logits_parts.append(
            independent_model.compute_draft_logits(round_output.hidden_states)
        )

        rejection_mask = torch.zeros(target_count, dtype=torch.bool)
        if round_batch.rejection_position is not None:
            rejection_mask[round_batch.rejection_position] = True
        rejection_mask_parts.append(rejection_mask)

    independent_logits = torch.cat(independent_logits_parts, dim=0)
    independent_loss = window_forward_kl_loss(
        student_logits=independent_logits,
        teacher_probabilities=packed_inputs.teacher_probabilities,
        rejection_target_mask=torch.cat(rejection_mask_parts, dim=0),
    )
    independent_loss.backward()

    torch.testing.assert_close(packed_logits, independent_logits)
    torch.testing.assert_close(packed_loss, independent_loss)

    independent_parameters = dict(independent_model.named_parameters())
    gradient_count = 0
    for name, packed_parameter in packed_model.named_parameters():
        independent_parameter = independent_parameters[name]
        assert (packed_parameter.grad is None) == (independent_parameter.grad is None)
        if packed_parameter.grad is not None:
            gradient_count += 1
            torch.testing.assert_close(
                packed_parameter.grad,
                independent_parameter.grad,
                rtol=1e-5,
                atol=1e-6,
            )

    assert gradient_count > 0


def test_future_spine_and_other_branch_do_not_leak() -> None:
    torch.manual_seed(23)

    model = _make_model().eval()
    original_window = _make_two_round_window()
    first, second = original_window.rounds
    perturbed_first = replace(
        first,
        confirmed_input_embeds=first.confirmed_input_embeds + 100.0,
        confirmed_aux_hidden_states=(first.confirmed_aux_hidden_states - 100.0),
    )
    perturbed_second = replace(
        second,
        prefill_input_embeds=second.prefill_input_embeds + 50.0,
        prefill_aux_hidden_states=second.prefill_aux_hidden_states - 50.0,
        draft_token_input_embeds=(second.draft_token_input_embeds + 75.0),
        draft_recurrent_hidden_states=(second.draft_recurrent_hidden_states - 75.0),
        confirmed_input_embeds=second.confirmed_input_embeds + 25.0,
        confirmed_aux_hidden_states=(second.confirmed_aux_hidden_states - 25.0),
    )
    perturbed_window = Eagle3TrainingWindow(rounds=(perturbed_first, perturbed_second))

    def first_branch_logits(window: Eagle3TrainingWindow) -> torch.Tensor:
        spine = prepare_training_window_spine(
            model,
            window,
            persistent_cache=None,
        )
        inputs = prepare_eagle3_proposal_window_inputs(
            model,
            window,
            spine_length=persistent_cache_length(spine),
        )
        with torch.no_grad():
            output = model(
                positions=inputs.positions,
                input_embeds=inputs.input_embeds,
                hidden_states=inputs.hidden_states,
                past_key_values=spine,
                attention_mask=inputs.attention_mask,
            )
            logits = model.compute_draft_logits(output.hidden_states)

        return logits[: first.draft_length]

    torch.testing.assert_close(
        first_branch_logits(perturbed_window),
        first_branch_logits(original_window),
    )


def test_proposal_window_trains_once_and_rebuilds_confirmed_cache() -> None:
    torch.manual_seed(24)

    model = _make_model()
    trainer = DraftTrainer(
        model,
        TrainerConfig(learning_rate=1e-2),
    )
    window = _make_two_round_window()
    forward_records: list[tuple[list[int], bool, bool]] = []
    lm_head_token_counts: list[int] = []

    def record_model_forward(
        _module: torch.nn.Module,
        _args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        positions = kwargs["positions"]
        assert isinstance(positions, torch.Tensor)
        forward_records.append(
            (
                positions.tolist(),
                torch.is_grad_enabled(),
                kwargs.get("attention_mask") is not None,
            )
        )

    def record_lm_head_forward(
        _module: torch.nn.Module,
        args: tuple[object, ...],
    ) -> None:
        hidden_states = args[0]
        assert isinstance(hidden_states, torch.Tensor)
        lm_head_token_counts.append(hidden_states.shape[0])

    model_handle = model.register_forward_pre_hook(
        record_model_forward,
        with_kwargs=True,
    )
    lm_head_handle = model.lm_head.register_forward_pre_hook(
        record_lm_head_forward,
    )

    try:
        with torch.inference_mode():
            persistent_cache, result = train_eagle3_window(
                trainer=trainer,
                window=window,
                persistent_cache=None,
                mode=Eagle3WindowMode.PROPOSAL_CANVAS,
            )
    finally:
        model_handle.remove()
        lm_head_handle.remove()

    assert forward_records == [
        ([0, 1, 2, 3, 4, 5, 6, 7], False, False),
        ([1, 2, 3, 5, 6, 7], True, True),
        ([0, 1, 2, 3, 4, 5, 6, 7], False, False),
    ]
    assert lm_head_token_counts == [6]

    assert trainer.version == 1
    assert result.model_version == 1
    assert result.mode is Eagle3WindowMode.PROPOSAL_CANVAS
    assert result.round_count == 2
    assert result.input_length == 6
    assert result.target_count == 6
    assert result.confirmed_length == 6
    assert result.persistent_cache_length == 8
    assert math.isfinite(result.loss)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(
        int(state["step"].item()) == 1 for state in trainer.optimizer.state.values()
    )

    assert persistent_cache_length(persistent_cache) == 8
    _assert_cache_detached(persistent_cache)
    _assert_cache_close(
        persistent_cache,
        _forward_full_history(model, window),
    )


def test_later_proposal_window_preserves_existing_history() -> None:
    torch.manual_seed(25)

    model = _make_model()
    trainer = DraftTrainer(
        model,
        TrainerConfig(learning_rate=1e-2),
    )
    history_batch = _make_batch(
        anchor_position=1,
        draft_length=2,
        rejection_position=0,
    )
    persistent_cache = append_confirmed_to_persistent_cache(
        model,
        history_batch,
        persistent_cache=None,
    )
    old_cache = tuple(
        (
            key.clone(),
            value.clone(),
        )
        for key, value in persistent_cache
    )

    first = _make_batch(
        anchor_position=2,
        draft_length=2,
        rejection_position=None,
    )
    second = _make_batch(
        anchor_position=5,
        draft_length=2,
        rejection_position=0,
    )
    second = _align_next_anchor(first, second)
    window = Eagle3TrainingWindow(rounds=(first, second))

    updated_cache, result = train_eagle3_window(
        trainer=trainer,
        window=window,
        persistent_cache=persistent_cache,
        mode=Eagle3WindowMode.PROPOSAL_CANVAS,
    )

    assert result.input_length == 4
    assert result.target_count == 4
    assert result.confirmed_length == 4
    assert result.persistent_cache_length == 7

    old_length = persistent_cache_length(persistent_cache)
    assert old_length == 3
    for old_layer, updated_layer in zip(
        old_cache,
        updated_cache,
        strict=True,
    ):
        assert torch.equal(
            old_layer[0],
            updated_layer[0][:, :old_length],
        )
        assert torch.equal(
            old_layer[1],
            updated_layer[1][:, :old_length],
        )


def test_bfloat16_proposal_window_training() -> None:
    torch.manual_seed(26)

    model = _make_model(
        dtype=torch.bfloat16,
        attention_backend=Eagle3AttentionBackend.FLASH_ATTENTION,
    )
    trainer = DraftTrainer(
        model,
        TrainerConfig(
            learning_rate=1e-2,
            fused=True,
        ),
    )
    window = _make_two_round_window(feature_dtype=torch.bfloat16)

    persistent_cache, result = train_eagle3_window(
        trainer=trainer,
        window=window,
        persistent_cache=None,
        mode=Eagle3WindowMode.PROPOSAL_CANVAS,
    )

    assert math.isfinite(result.loss)
    assert result.target_count == window.proposal_target_count
    for key, value in persistent_cache:
        assert key.dtype == torch.bfloat16
        assert value.dtype == torch.bfloat16

    for state in trainer.optimizer.state.values():
        assert state["exp_avg"].dtype == torch.bfloat16
        assert state["exp_avg_sq"].dtype == torch.bfloat16


@pytest.mark.parametrize("mode", tuple(Eagle3WindowMode))
def test_window_modes_support_fused_optimizer_and_flash_attention(
    mode: Eagle3WindowMode,
) -> None:
    torch.manual_seed(27)

    model = _make_model(
        attention_backend=Eagle3AttentionBackend.FLASH_ATTENTION,
    )
    trainer = DraftTrainer(
        model,
        TrainerConfig(
            learning_rate=1e-2,
            fused=True,
        ),
    )

    persistent_cache, result = train_eagle3_window(
        trainer=trainer,
        window=_make_two_round_window(),
        persistent_cache=None,
        mode=mode,
    )

    assert model.config.attention_backend is Eagle3AttentionBackend.FLASH_ATTENTION
    assert trainer.optimizer.defaults["fused"] is True
    assert trainer.version == 1
    assert result.mode is mode
    assert math.isfinite(result.loss)
    _assert_cache_detached(persistent_cache)
