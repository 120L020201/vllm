# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from online_draft.models.qwen3_eagle3 import (
    Qwen3Eagle3Config,
    Qwen3Eagle3ForCausalLM,
)
from online_draft.training.eagle3_batch import (
    Eagle3DistillationBatch,
)
from online_draft.training.eagle3_cache import (
    persistent_cache_length,
    prepare_training_window_spine,
)
from online_draft.training.eagle3_distill import (
    forward_kl_loss,
    window_forward_kl_loss,
)
from online_draft.training.eagle3_inputs import (
    prepare_eagle3_confirmed_path_inputs,
    prepare_eagle3_training_inputs,
)
from online_draft.training.eagle3_window import (
    Eagle3TrainingWindow,
)

HIDDEN_SIZE = 4
NUM_AUX_HIDDEN_STATES = 3
DRAFT_VOCAB_SIZE = 6
TARGET_VOCAB_SIZE = 10
ANCHOR_POSITION = 5
PREFILL_LENGTH = 3


def _make_model(
    dtype: torch.dtype = torch.float32,
) -> Qwen3Eagle3ForCausalLM:
    config = Qwen3Eagle3Config(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        num_hidden_layers=1,
        target_vocab_size=TARGET_VOCAB_SIZE,
        draft_vocab_size=DRAFT_VOCAB_SIZE,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        num_aux_hidden_states=NUM_AUX_HIDDEN_STATES,
    )

    return Qwen3Eagle3ForCausalLM(config).to(dtype=dtype)


def _make_batch(
    *,
    anchor_position: int = ANCHOR_POSITION,
    feature_dtype: torch.dtype = torch.float32,
    draft_length: int = 4,
    rejection_position: int | None = 2,
) -> Eagle3DistillationBatch:
    confirmed_length = (
        draft_length + 1 if rejection_position is None else rejection_position + 1
    )

    return Eagle3DistillationBatch(
        prefill_positions=torch.arange(
            anchor_position - PREFILL_LENGTH + 1,
            anchor_position + 1,
            dtype=torch.long,
        ),
        prefill_input_embeds=torch.randn(
            PREFILL_LENGTH,
            HIDDEN_SIZE,
            dtype=feature_dtype,
        ),
        prefill_aux_hidden_states=torch.randn(
            PREFILL_LENGTH,
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
            dtype=torch.float32,
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


def test_prepare_uses_anchor_and_gpu_generated_later_inputs() -> None:
    torch.manual_seed(0)

    model = _make_model()
    batch = _make_batch()

    training_inputs = prepare_eagle3_training_inputs(
        model,
        batch,
    )

    with torch.no_grad():
        expected_anchor_hidden = model.combine_hidden_states(
            batch.prefill_aux_hidden_states[-1:]
        )

    assert training_inputs.draft_length == batch.draft_length
    assert torch.equal(
        training_inputs.positions,
        batch.proposal_positions,
    )

    torch.testing.assert_close(
        training_inputs.input_embeds[0],
        batch.prefill_input_embeds[-1],
    )
    torch.testing.assert_close(
        training_inputs.input_embeds[1:],
        batch.draft_token_input_embeds,
    )

    torch.testing.assert_close(
        training_inputs.hidden_states[0],
        expected_anchor_hidden[0],
    )
    torch.testing.assert_close(
        training_inputs.hidden_states[1:],
        batch.draft_recurrent_hidden_states,
    )

    assert training_inputs.hidden_states.requires_grad
    assert not batch.draft_recurrent_hidden_states.requires_grad


def test_canvas_uses_one_parallel_forward_and_backward() -> None:
    torch.manual_seed(1)

    model = _make_model()
    batch = _make_batch()
    training_inputs = prepare_eagle3_training_inputs(
        model,
        batch,
    )

    forward_token_counts: list[int] = []

    def record_forward(
        _module: torch.nn.Module,
        _args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        positions = kwargs["positions"]
        assert isinstance(positions, torch.Tensor)
        forward_token_counts.append(positions.numel())

    handle = model.register_forward_pre_hook(
        record_forward,
        with_kwargs=True,
    )

    try:
        output = model(
            positions=training_inputs.positions,
            input_embeds=training_inputs.input_embeds,
            hidden_states=training_inputs.hidden_states,
        )
    finally:
        handle.remove()

    logits = model.compute_draft_logits(output.hidden_states)
    loss = forward_kl_loss(
        student_logits=logits,
        teacher_probabilities=(training_inputs.teacher_probabilities),
        rejection_position=(training_inputs.rejection_position),
    )
    loss.backward()

    assert forward_token_counts == [
        batch.draft_length,
    ]

    fc_gradient = model.model.fc.weight.grad
    assert fc_gradient is not None
    assert torch.isfinite(fc_gradient).all()
    assert fc_gradient.abs().sum().item() > 0

    assert batch.draft_recurrent_hidden_states.grad is None


def test_draft_length_one_only_uses_anchor() -> None:
    torch.manual_seed(2)

    model = _make_model()
    batch = _make_batch(
        draft_length=1,
        rejection_position=None,
    )

    training_inputs = prepare_eagle3_training_inputs(
        model,
        batch,
    )

    assert training_inputs.positions.shape == (1,)
    assert training_inputs.input_embeds.shape == (
        1,
        HIDDEN_SIZE,
    )
    assert training_inputs.hidden_states.shape == (
        1,
        HIDDEN_SIZE,
    )
    assert batch.draft_token_input_embeds.shape == (
        0,
        HIDDEN_SIZE,
    )
    assert batch.draft_recurrent_hidden_states.shape == (
        0,
        HIDDEN_SIZE,
    )


def test_bfloat16_canvas_backward() -> None:
    torch.manual_seed(3)

    model = _make_model(dtype=torch.bfloat16)
    batch = _make_batch(
        feature_dtype=torch.bfloat16,
    )
    training_inputs = prepare_eagle3_training_inputs(
        model,
        batch,
    )

    output = model(
        positions=training_inputs.positions,
        input_embeds=training_inputs.input_embeds,
        hidden_states=training_inputs.hidden_states,
    )
    logits = model.compute_draft_logits(output.hidden_states)
    loss = forward_kl_loss(
        student_logits=logits,
        teacher_probabilities=(training_inputs.teacher_probabilities),
        rejection_position=(training_inputs.rejection_position),
    )
    loss.backward()

    assert training_inputs.input_embeds.dtype == torch.bfloat16
    assert training_inputs.hidden_states.dtype == torch.bfloat16
    assert logits.dtype == torch.bfloat16
    assert loss.dtype == torch.float32

    fc_gradient = model.model.fc.weight.grad
    assert fc_gradient is not None
    assert fc_gradient.dtype == torch.bfloat16
    assert torch.isfinite(fc_gradient).all()


def test_model_and_batch_dtypes_must_match() -> None:
    model = _make_model(dtype=torch.float32)
    batch = _make_batch(
        feature_dtype=torch.bfloat16,
    )

    with pytest.raises(
        ValueError,
        match="batch feature dtype must match",
    ):
        prepare_eagle3_training_inputs(
            model,
            batch,
        )


def test_confirmed_path_packs_only_confirmed_queries() -> None:
    torch.manual_seed(4)

    model = _make_model()
    first = _make_batch(
        anchor_position=5,
        draft_length=3,
        rejection_position=None,
    )
    second = _make_batch(
        anchor_position=9,
        draft_length=3,
        rejection_position=1,
    )
    window = Eagle3TrainingWindow(
        rounds=(
            first,
            second,
        )
    )

    fc_input_shapes: list[torch.Size] = []

    def record_fc_input(
        _module: torch.nn.Module,
        args: tuple[object, ...],
    ) -> None:
        hidden_states = args[0]
        assert isinstance(hidden_states, torch.Tensor)
        fc_input_shapes.append(hidden_states.shape)

    handle = model.model.fc.register_forward_pre_hook(record_fc_input)
    try:
        inputs = prepare_eagle3_confirmed_path_inputs(
            model,
            window,
            spine_length=12,
        )
    finally:
        handle.remove()

    expected_input_embeds = torch.cat(
        (
            first.prefill_input_embeds[-1:],
            first.draft_token_input_embeds,
            second.prefill_input_embeds[-1:],
            second.draft_token_input_embeds[:1],
        ),
        dim=0,
    )

    with torch.no_grad():
        expected_hidden_states = torch.cat(
            (
                model.combine_hidden_states(first.prefill_aux_hidden_states[-1:]),
                first.draft_recurrent_hidden_states,
                model.combine_hidden_states(second.prefill_aux_hidden_states[-1:]),
                second.draft_recurrent_hidden_states[:1],
            ),
            dim=0,
        )

    expected_mask = torch.ones(5, 17, dtype=torch.bool)
    expected_mask[0, [0, 1, 2, 3, 4, 12]] = False
    expected_mask[1, [0, 1, 2, 3, 4, 12, 13]] = False
    expected_mask[2, [0, 1, 2, 3, 4, 12, 13, 14]] = False
    expected_mask[3, [0, 1, 2, 3, 4, 5, 6, 7, 8, 15]] = False
    expected_mask[4, [0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16]] = False

    assert inputs.positions.tolist() == [5, 6, 7, 9, 10]
    assert inputs.rejection_target_mask.tolist() == [
        False,
        False,
        False,
        False,
        True,
    ]
    assert inputs.input_length == 5
    assert inputs.target_count == 5
    assert fc_input_shapes == [torch.Size((1, 12)), torch.Size((1, 12))]
    assert torch.equal(inputs.attention_mask, expected_mask)

    torch.testing.assert_close(
        inputs.input_embeds,
        expected_input_embeds,
    )
    torch.testing.assert_close(
        inputs.hidden_states,
        expected_hidden_states,
    )
    torch.testing.assert_close(
        inputs.teacher_probabilities,
        torch.cat(
            (
                first.teacher_probabilities,
                second.teacher_probabilities[:2],
            ),
            dim=0,
        ),
    )
    assert inputs.hidden_states.requires_grad


def test_confirmed_path_trims_final_bonus_prediction_row() -> None:
    torch.manual_seed(5)

    model = _make_model()
    round_batch = _make_batch(
        anchor_position=5,
        draft_length=3,
        rejection_position=None,
    )
    window = Eagle3TrainingWindow(rounds=(round_batch,))

    inputs = prepare_eagle3_confirmed_path_inputs(
        model,
        window,
        spine_length=10,
    )

    assert inputs.positions.tolist() == [5, 6, 7]
    assert inputs.rejection_target_mask.tolist() == [
        False,
        False,
        False,
    ]
    assert inputs.input_length == 3
    assert inputs.target_count == 3

    torch.testing.assert_close(
        inputs.input_embeds,
        torch.cat(
            (
                round_batch.prefill_input_embeds[-1:],
                round_batch.draft_token_input_embeds,
            ),
            dim=0,
        ),
    )
    torch.testing.assert_close(
        inputs.teacher_probabilities,
        round_batch.teacher_probabilities,
    )


def test_confirmed_path_forwards_only_loss_rows() -> None:
    torch.manual_seed(6)

    model = _make_model()
    first = _make_batch(
        anchor_position=2,
        draft_length=3,
        rejection_position=None,
    )
    second = _make_batch(
        anchor_position=6,
        draft_length=3,
        rejection_position=1,
    )
    window = Eagle3TrainingWindow(
        rounds=(
            first,
            second,
        )
    )
    spine = prepare_training_window_spine(
        model,
        window,
        persistent_cache=None,
    )
    inputs = prepare_eagle3_confirmed_path_inputs(
        model,
        window,
        spine_length=persistent_cache_length(spine),
    )

    output = model(
        positions=inputs.positions,
        input_embeds=inputs.input_embeds,
        hidden_states=inputs.hidden_states,
        past_key_values=spine,
        attention_mask=inputs.attention_mask,
    )
    student_logits = model.compute_draft_logits(output.hidden_states)
    loss = window_forward_kl_loss(
        student_logits=student_logits,
        teacher_probabilities=inputs.teacher_probabilities,
        rejection_target_mask=inputs.rejection_target_mask,
    )

    loss.backward()

    assert output.hidden_states.shape[0] == 5
    assert student_logits.shape == (
        inputs.target_count,
        DRAFT_VOCAB_SIZE,
    )
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)

    fc_gradient = model.model.fc.weight.grad
    assert fc_gradient is not None
    assert torch.isfinite(fc_gradient).all()
    assert fc_gradient.abs().sum().item() > 0
