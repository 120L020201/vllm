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
from online_draft.training.eagle3_distill import (
    forward_kl_loss,
)
from online_draft.training.eagle3_inputs import (
    prepare_eagle3_training_inputs,
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
    feature_dtype: torch.dtype = torch.float32,
    draft_length: int = 4,
    rejection_position: int | None = 2,
) -> Eagle3DistillationBatch:
    confirmed_length = (
        draft_length + 1 if rejection_position is None else rejection_position + 1
    )

    return Eagle3DistillationBatch(
        prefill_positions=torch.arange(
            ANCHOR_POSITION - PREFILL_LENGTH + 1,
            ANCHOR_POSITION + 1,
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
            ANCHOR_POSITION,
            ANCHOR_POSITION + draft_length,
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
            ANCHOR_POSITION + 1,
            ANCHOR_POSITION + 1 + confirmed_length,
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
