# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import replace

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


def _make_model() -> Qwen3Eagle3ForCausalLM:
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

    return Qwen3Eagle3ForCausalLM(config)


def _make_batch(
    *,
    anchor_position: int,
    draft_length: int = 3,
    rejection_position: int | None = 1,
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
        teacher_probabilities=torch.full(
            (
                draft_length,
                DRAFT_VOCAB_SIZE,
            ),
            1.0 / DRAFT_VOCAB_SIZE,
        ),
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
    for actual_layer, expected_layer in zip(
        actual,
        expected,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_layer[0],
            expected_layer[0],
        )
        torch.testing.assert_close(
            actual_layer[1],
            expected_layer[1],
        )


def _forward_full_window_history(
    model: Qwen3Eagle3ForCausalLM,
    window: Eagle3TrainingWindow,
) -> Eagle3KVCache:
    first_round = window.rounds[0]
    positions = torch.cat(
        (
            first_round.prefill_positions,
            *(batch.confirmed_positions for batch in window.rounds),
        )
    )
    input_embeds = torch.cat(
        (
            first_round.prefill_input_embeds,
            *(batch.confirmed_input_embeds for batch in window.rounds),
        )
    )
    auxiliary_hidden_states = torch.cat(
        (
            first_round.prefill_aux_hidden_states,
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


def test_window_modes_have_stable_values() -> None:
    assert Eagle3WindowMode.CONFIRMED_PATH == "confirmed_path"
    assert Eagle3WindowMode.PROPOSAL_CANVAS == "proposal_canvas"


def test_consecutive_rounds_form_one_window() -> None:
    first = _make_batch(
        anchor_position=4,
        draft_length=3,
        rejection_position=1,
    )
    second = _make_batch(
        anchor_position=6,
        draft_length=4,
        rejection_position=None,
    )

    window = Eagle3TrainingWindow(
        rounds=(
            first,
            second,
        )
    )

    assert window.round_count == 2
    assert window.start_anchor_position == 4
    assert window.end_confirmed_position == 11
    assert window.proposal_target_count == 7

    # The first round trains one accepted position plus the correction.
    # The second trains four accepted positions and excludes its bonus.
    assert window.confirmed_path_target_count == 6


def test_single_round_is_a_valid_window() -> None:
    round_batch = _make_batch(
        anchor_position=7,
        rejection_position=0,
    )

    window = Eagle3TrainingWindow(rounds=(round_batch,))

    assert window.round_count == 1
    assert window.start_anchor_position == 7
    assert window.end_confirmed_position == 8
    assert window.proposal_target_count == 3
    assert window.confirmed_path_target_count == 1


def test_empty_window_is_invalid() -> None:
    with pytest.raises(
        ValueError,
        match="at least one round",
    ):
        Eagle3TrainingWindow(rounds=())


def test_rounds_must_be_a_tuple() -> None:
    round_batch = _make_batch(anchor_position=4)

    with pytest.raises(
        TypeError,
        match="rounds must be a tuple",
    ):
        Eagle3TrainingWindow(
            rounds=[round_batch],  # type: ignore[arg-type]
        )


def test_window_entries_must_be_batches() -> None:
    with pytest.raises(
        TypeError,
        match="Eagle3DistillationBatch",
    ):
        Eagle3TrainingWindow(
            rounds=(object(),),  # type: ignore[arg-type]
        )


def test_round_anchors_must_follow_confirmed_history() -> None:
    first = _make_batch(
        anchor_position=4,
        rejection_position=1,
    )
    second = _make_batch(
        anchor_position=7,
        rejection_position=0,
    )

    with pytest.raises(
        ValueError,
        match="anchor must equal",
    ):
        Eagle3TrainingWindow(
            rounds=(
                first,
                second,
            )
        )


def test_round_must_have_proposal_positions() -> None:
    round_batch = _make_batch(anchor_position=4)
    invalid_batch = replace(
        round_batch,
        proposal_positions=torch.empty(
            0,
            dtype=torch.long,
        ),
    )

    with pytest.raises(
        ValueError,
        match="proposal positions",
    ):
        Eagle3TrainingWindow(rounds=(invalid_batch,))


def test_round_must_have_confirmed_positions() -> None:
    round_batch = _make_batch(anchor_position=4)
    invalid_batch = replace(
        round_batch,
        confirmed_positions=torch.empty(
            0,
            dtype=torch.long,
        ),
    )

    with pytest.raises(
        ValueError,
        match="confirmed positions",
    ):
        Eagle3TrainingWindow(rounds=(invalid_batch,))


def test_confirmed_path_window_uses_one_optimizer_step() -> None:
    torch.manual_seed(10)

    model = _make_model()
    trainer = DraftTrainer(
        model,
        TrainerConfig(learning_rate=1e-2),
    )
    first = _make_batch(
        anchor_position=1,
        draft_length=3,
        rejection_position=None,
    )
    second = _make_batch(
        anchor_position=5,
        draft_length=3,
        rejection_position=1,
    )
    window = Eagle3TrainingWindow(
        rounds=(
            first,
            second,
        )
    )

    old_fc_weight = model.model.fc.weight.detach().clone()
    forward_records: list[tuple[list[int], bool]] = []
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
                mode=Eagle3WindowMode.CONFIRMED_PATH,
            )
    finally:
        model_handle.remove()
        lm_head_handle.remove()

    assert forward_records == [
        ([0, 1, 2, 3, 4, 5, 6, 7], False),
        ([1, 2, 3, 5, 6], True),
        ([0, 1, 2, 3, 4, 5, 6, 7], False),
    ]
    assert lm_head_token_counts == [5]

    assert trainer.version == 1
    assert result.model_version == 1
    assert result.mode is Eagle3WindowMode.CONFIRMED_PATH
    assert result.round_count == 2
    assert result.input_length == 5
    assert result.target_count == 5
    assert result.confirmed_length == 6
    assert result.persistent_cache_length == 8
    assert math.isfinite(result.loss)

    assert not torch.equal(
        model.model.fc.weight,
        old_fc_weight,
    )
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(
        int(state["step"].item()) == 1 for state in trainer.optimizer.state.values()
    )

    assert persistent_cache_length(persistent_cache) == 8
    _assert_cache_detached(persistent_cache)
    _assert_cache_close(
        persistent_cache,
        _forward_full_window_history(model, window),
    )


def test_later_confirmed_path_window_preserves_fixed_history() -> None:
    torch.manual_seed(11)

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
    window = Eagle3TrainingWindow(
        rounds=(
            first,
            second,
        )
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
        updated_cache, result = train_eagle3_window(
            trainer=trainer,
            window=window,
            persistent_cache=persistent_cache,
            mode=Eagle3WindowMode.CONFIRMED_PATH,
        )
    finally:
        handle.remove()

    assert forward_records == [
        ([3, 4, 5, 6], False),
        ([2, 3, 5], True),
        ([3, 4, 5, 6], False),
    ]
    assert trainer.version == 1
    assert result.round_count == 2
    assert result.input_length == 3
    assert result.target_count == 3
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
