# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math

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
    persistent_cache_length,
)
from online_draft.training.eagle3_step import (
    train_eagle3_round,
)
from online_draft.training.trainer import (
    DraftTrainer,
    TrainerConfig,
)

HIDDEN_SIZE = 4
NUM_AUX_HIDDEN_STATES = 3
DRAFT_VOCAB_SIZE = 6


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
        target_vocab_size=10,
        draft_vocab_size=DRAFT_VOCAB_SIZE,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        num_aux_hidden_states=NUM_AUX_HIDDEN_STATES,
    )

    return Qwen3Eagle3ForCausalLM(config).to(dtype=dtype)


def _make_batch(
    *,
    prefill_start: int,
    prefill_length: int,
    feature_dtype: torch.dtype = torch.float32,
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


def _forward_full_confirmed_history(
    model: Qwen3Eagle3ForCausalLM,
    batch: Eagle3DistillationBatch,
) -> Eagle3KVCache:
    positions = torch.cat(
        (
            batch.prefill_positions,
            batch.confirmed_positions,
        )
    )
    input_embeds = torch.cat(
        (
            batch.prefill_input_embeds,
            batch.confirmed_input_embeds,
        )
    )
    auxiliary_hidden_states = torch.cat(
        (
            batch.prefill_aux_hidden_states,
            batch.confirmed_aux_hidden_states,
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


def test_first_round_trains_and_appends_updated_cache() -> None:
    torch.manual_seed(0)

    model = _make_model()
    trainer = DraftTrainer(
        model,
        TrainerConfig(learning_rate=1e-2),
    )
    batch = _make_batch(
        prefill_start=0,
        prefill_length=3,
        rejection_position=1,
    )

    old_fc_weight = model.model.fc.weight.detach().clone()
    forward_records: list[tuple[int, bool]] = []

    def record_forward(
        _module: torch.nn.Module,
        _args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        positions = kwargs["positions"]
        assert isinstance(positions, torch.Tensor)

        forward_records.append(
            (
                positions.numel(),
                torch.is_grad_enabled(),
            )
        )

    handle = model.register_forward_pre_hook(
        record_forward,
        with_kwargs=True,
    )

    try:
        with torch.inference_mode():
            persistent_cache, result = train_eagle3_round(
                trainer,
                batch,
                persistent_cache=None,
            )
    finally:
        handle.remove()

    assert forward_records == [
        (2, False),
        (3, True),
        (3, False),
        (2, False),
    ]

    assert trainer.version == 1
    assert result.model_version == 1
    assert result.draft_length == 3
    assert result.confirmed_length == 2
    assert result.persistent_cache_length == 5
    assert math.isfinite(result.loss)

    assert not torch.equal(
        model.model.fc.weight,
        old_fc_weight,
    )
    assert all(parameter.grad is None for parameter in model.parameters())

    assert persistent_cache_length(persistent_cache) == 5
    _assert_cache_detached(persistent_cache)

    expected_cache = _forward_full_confirmed_history(
        model,
        batch,
    )
    _assert_cache_close(
        persistent_cache,
        expected_cache,
    )


def test_second_round_reuses_history_and_only_appends_confirmed() -> None:
    torch.manual_seed(1)

    model = _make_model()
    trainer = DraftTrainer(
        model,
        TrainerConfig(learning_rate=1e-2),
    )

    first_batch = _make_batch(
        prefill_start=0,
        prefill_length=3,
        rejection_position=1,
    )
    persistent_cache, _ = train_eagle3_round(
        trainer,
        first_batch,
        persistent_cache=None,
    )

    old_cache = tuple(
        (
            key.clone(),
            value.clone(),
        )
        for key, value in persistent_cache
    )
    old_length = persistent_cache_length(persistent_cache)

    second_batch = _make_batch(
        prefill_start=3,
        prefill_length=2,
        rejection_position=0,
    )

    forward_records: list[tuple[list[int], bool]] = []

    def record_forward(
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
            )
        )

    handle = model.register_forward_pre_hook(
        record_forward,
        with_kwargs=True,
    )

    try:
        updated_cache, result = train_eagle3_round(
            trainer,
            second_batch,
            persistent_cache,
        )
    finally:
        handle.remove()

    assert forward_records == [
        ([4, 5, 6], True),
        ([5], False),
    ]

    assert trainer.version == 2
    assert result.model_version == 2
    assert result.confirmed_length == 1
    assert result.persistent_cache_length == 6
    assert old_length == 5
    assert persistent_cache_length(updated_cache) == 6

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


def test_invalid_teacher_does_not_update_model() -> None:
    torch.manual_seed(2)

    model = _make_model()
    trainer = DraftTrainer(
        model,
        TrainerConfig(learning_rate=1e-2),
    )
    batch = _make_batch(
        prefill_start=0,
        prefill_length=3,
    )

    batch.teacher_probabilities[0, 0] = float("nan")

    old_parameters = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }

    with pytest.raises(
        ValueError,
        match="teacher probabilities must be finite",
    ):
        train_eagle3_round(
            trainer,
            batch,
            persistent_cache=None,
        )

    assert trainer.version == 0
    assert trainer.last_loss is None
    assert not trainer.optimizer.state

    for name, parameter in model.named_parameters():
        assert torch.equal(
            parameter,
            old_parameters[name],
        )


def test_bfloat16_training_round() -> None:
    torch.manual_seed(3)

    model = _make_model(
        dtype=torch.bfloat16,
    )
    trainer = DraftTrainer(
        model,
        TrainerConfig(learning_rate=1e-2),
    )
    batch = _make_batch(
        prefill_start=0,
        prefill_length=3,
        feature_dtype=torch.bfloat16,
        rejection_position=None,
    )

    persistent_cache, result = train_eagle3_round(
        trainer,
        batch,
        persistent_cache=None,
    )

    assert trainer.version == 1
    assert result.model_version == 1
    assert result.confirmed_length == 4
    assert result.persistent_cache_length == 7
    assert math.isfinite(result.loss)

    _assert_cache_detached(persistent_cache)

    for key, value in persistent_cache:
        assert key.dtype == torch.bfloat16
        assert value.dtype == torch.bfloat16

    for state in trainer.optimizer.state.values():
        assert state["exp_avg"].dtype == torch.bfloat16
        assert state["exp_avg_sq"].dtype == torch.bfloat16
        assert state["step"].dtype == torch.float32
