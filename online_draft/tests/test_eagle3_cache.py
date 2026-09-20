# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from online_draft.models.qwen3_eagle3 import (
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
    prepare_training_prefix,
)
from online_draft.training.eagle3_inputs import (
    prepare_eagle3_training_inputs,
)

HIDDEN_SIZE = 4
NUM_AUX_HIDDEN_STATES = 3
DRAFT_VOCAB_SIZE = 6


def _make_model() -> Qwen3Eagle3ForCausalLM:
    config = Qwen3Eagle3Config(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        num_hidden_layers=1,
        target_vocab_size=10,
        draft_vocab_size=DRAFT_VOCAB_SIZE,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        num_aux_hidden_states=NUM_AUX_HIDDEN_STATES,
    )

    return Qwen3Eagle3ForCausalLM(config)


def _make_batch(
    *,
    prefill_start: int,
    prefill_length: int,
    draft_length: int = 3,
    rejection_position: int | None = 1,
) -> Eagle3DistillationBatch:
    anchor_position = prefill_start + prefill_length - 1
    confirmed_length = (
        draft_length + 1 if rejection_position is None else rejection_position + 1
    )

    return Eagle3DistillationBatch(
        prefill_positions=torch.arange(
            prefill_start,
            prefill_start + prefill_length,
            dtype=torch.long,
        ),
        prefill_input_embeds=torch.randn(
            prefill_length,
            HIDDEN_SIZE,
        ),
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


def _assert_cache_detached(
    cache: Eagle3KVCache,
) -> None:
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
    for (
        actual_key,
        actual_value,
    ), (
        expected_key,
        expected_value,
    ) in zip(
        actual,
        expected,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_key,
            expected_key,
        )
        torch.testing.assert_close(
            actual_value,
            expected_value,
        )


def test_first_prefix_excludes_anchor_and_is_detached() -> None:
    torch.manual_seed(0)

    model = _make_model()
    batch = _make_batch(
        prefill_start=0,
        prefill_length=3,
    )

    prefix_cache = prepare_training_prefix(
        model,
        batch,
        persistent_cache=None,
    )

    assert prefix_cache is not None
    assert persistent_cache_length(prefix_cache) == 2
    _assert_cache_detached(prefix_cache)

    with torch.no_grad():
        expected_hidden_states = model.combine_hidden_states(
            batch.prefill_aux_hidden_states[:-1]
        )
        expected_output = model(
            positions=batch.prefill_positions[:-1],
            input_embeds=batch.prefill_input_embeds[:-1],
            hidden_states=expected_hidden_states,
        )

    _assert_cache_close(
        prefix_cache,
        expected_output.past_key_values,
    )


def test_first_append_builds_prefill_and_confirmed_history() -> None:
    torch.manual_seed(1)

    model = _make_model()
    batch = _make_batch(
        prefill_start=0,
        prefill_length=3,
        rejection_position=None,
    )

    with torch.inference_mode():
        persistent_cache = append_confirmed_to_persistent_cache(
            model,
            batch,
            persistent_cache=None,
        )

    expected_length = int(batch.confirmed_positions[-1].item()) + 1

    assert persistent_cache_length(persistent_cache) == expected_length
    _assert_cache_detached(persistent_cache)

    all_positions = torch.cat(
        (
            batch.prefill_positions,
            batch.confirmed_positions,
        )
    )
    all_input_embeds = torch.cat(
        (
            batch.prefill_input_embeds,
            batch.confirmed_input_embeds,
        )
    )
    all_aux_hidden_states = torch.cat(
        (
            batch.prefill_aux_hidden_states,
            batch.confirmed_aux_hidden_states,
        )
    )

    with torch.no_grad():
        expected_hidden_states = model.combine_hidden_states(all_aux_hidden_states)
        expected_output = model(
            positions=all_positions,
            input_embeds=all_input_embeds,
            hidden_states=expected_hidden_states,
        )

    _assert_cache_close(
        persistent_cache,
        expected_output.past_key_values,
    )


def test_later_prefix_does_not_modify_persistent_cache() -> None:
    torch.manual_seed(2)

    model = _make_model()
    first_batch = _make_batch(
        prefill_start=0,
        prefill_length=3,
        rejection_position=1,
    )
    persistent_cache = append_confirmed_to_persistent_cache(
        model,
        first_batch,
        persistent_cache=None,
    )

    assert persistent_cache_length(persistent_cache) == 5

    original_cache = tuple(
        (
            key.clone(),
            value.clone(),
        )
        for key, value in persistent_cache
    )

    next_batch = _make_batch(
        prefill_start=3,
        prefill_length=2,
        rejection_position=0,
    )

    forward_count = 0

    def record_forward(
        _module: torch.nn.Module,
        _args: tuple[object, ...],
        _kwargs: dict[str, object],
    ) -> None:
        nonlocal forward_count
        forward_count += 1

    handle = model.register_forward_pre_hook(
        record_forward,
        with_kwargs=True,
    )

    try:
        prefix_cache = prepare_training_prefix(
            model,
            next_batch,
            persistent_cache,
        )
    finally:
        handle.remove()

    assert prefix_cache is not None
    assert forward_count == 0
    assert persistent_cache_length(prefix_cache) == 4
    assert persistent_cache_length(persistent_cache) == 5

    for (
        prefix_key,
        prefix_value,
    ), (
        persistent_key,
        persistent_value,
    ) in zip(
        prefix_cache,
        persistent_cache,
        strict=True,
    ):
        torch.testing.assert_close(
            prefix_key,
            persistent_key[:, :4],
        )
        torch.testing.assert_close(
            prefix_value,
            persistent_value[:, :4],
        )

    _assert_cache_close(
        persistent_cache,
        original_cache,
    )


def test_differentiable_canvas_cache_is_not_persistent() -> None:
    torch.manual_seed(3)

    model = _make_model()
    batch = _make_batch(
        prefill_start=0,
        prefill_length=3,
    )

    prefix_cache = prepare_training_prefix(
        model,
        batch,
        persistent_cache=None,
    )
    training_inputs = prepare_eagle3_training_inputs(
        model,
        batch,
    )

    output = model(
        positions=training_inputs.positions,
        input_embeds=training_inputs.input_embeds,
        hidden_states=training_inputs.hidden_states,
        past_key_values=prefix_cache,
    )

    assert any(
        key.requires_grad or value.requires_grad
        for key, value in output.past_key_values
    )

    with pytest.raises(
        ValueError,
        match="persistent KV cache must be detached",
    ):
        persistent_cache_length(output.past_key_values)


def test_later_append_only_forwards_new_confirmed_tokens() -> None:
    torch.manual_seed(4)

    model = _make_model()
    first_batch = _make_batch(
        prefill_start=0,
        prefill_length=3,
        rejection_position=1,
    )
    persistent_cache = append_confirmed_to_persistent_cache(
        model,
        first_batch,
        persistent_cache=None,
    )

    old_length = persistent_cache_length(persistent_cache)
    old_cache = tuple(
        (
            key.clone(),
            value.clone(),
        )
        for key, value in persistent_cache
    )

    next_batch = _make_batch(
        prefill_start=3,
        prefill_length=2,
        rejection_position=0,
    )

    with torch.no_grad():
        model.model.fc.weight.add_(0.05)

    forwarded_positions: list[list[int]] = []

    def record_forward(
        _module: torch.nn.Module,
        _args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        positions = kwargs["positions"]
        assert isinstance(positions, torch.Tensor)
        forwarded_positions.append(positions.tolist())

    handle = model.register_forward_pre_hook(
        record_forward,
        with_kwargs=True,
    )

    try:
        updated_cache = append_confirmed_to_persistent_cache(
            model,
            next_batch,
            persistent_cache,
        )
    finally:
        handle.remove()

    assert forwarded_positions == [[5]]
    assert old_length == 5
    assert persistent_cache_length(updated_cache) == 6
    _assert_cache_detached(updated_cache)

    _assert_cache_close(
        persistent_cache,
        old_cache,
    )

    for (
        old_key,
        old_value,
    ), (
        new_key,
        new_value,
    ) in zip(
        old_cache,
        updated_cache,
        strict=True,
    ):
        assert torch.equal(
            old_key,
            new_key[:, :old_length],
        )
        assert torch.equal(
            old_value,
            new_value[:, :old_length],
        )


def test_first_observation_must_start_at_zero() -> None:
    model = _make_model()
    batch = _make_batch(
        prefill_start=2,
        prefill_length=3,
    )

    with pytest.raises(
        ValueError,
        match="first observation must contain the full prompt",
    ):
        prepare_training_prefix(
            model,
            batch,
            persistent_cache=None,
        )
