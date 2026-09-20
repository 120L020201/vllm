# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace

import pytest
import torch
from online_draft.training.eagle3_batch import (
    Eagle3DistillationBatch,
)

HIDDEN_SIZE = 4
NUM_AUX_HIDDEN_STATES = 3
DRAFT_VOCAB_SIZE = 5
PREFILL_LENGTH = 2
ANCHOR_POSITION = 7


def _make_batch(
    *,
    draft_length: int = 3,
    rejection_position: int | None = 1,
) -> Eagle3DistillationBatch:
    confirmed_length = (
        draft_length + 1 if rejection_position is None else rejection_position + 1
    )

    prefill_positions = torch.arange(
        ANCHOR_POSITION - PREFILL_LENGTH + 1,
        ANCHOR_POSITION + 1,
        dtype=torch.long,
    )
    proposal_positions = torch.arange(
        ANCHOR_POSITION,
        ANCHOR_POSITION + draft_length,
        dtype=torch.long,
    )
    confirmed_positions = torch.arange(
        ANCHOR_POSITION + 1,
        ANCHOR_POSITION + 1 + confirmed_length,
        dtype=torch.long,
    )

    return Eagle3DistillationBatch(
        prefill_positions=prefill_positions,
        prefill_input_embeds=torch.randn(
            PREFILL_LENGTH,
            HIDDEN_SIZE,
            dtype=torch.float32,
        ),
        prefill_aux_hidden_states=torch.randn(
            PREFILL_LENGTH,
            HIDDEN_SIZE * NUM_AUX_HIDDEN_STATES,
            dtype=torch.float32,
        ),
        proposal_positions=proposal_positions,
        draft_token_input_embeds=torch.randn(
            draft_length - 1,
            HIDDEN_SIZE,
            dtype=torch.float32,
        ),
        draft_recurrent_hidden_states=torch.randn(
            draft_length - 1,
            HIDDEN_SIZE,
            dtype=torch.float32,
        ),
        teacher_probabilities=torch.full(
            (
                draft_length,
                DRAFT_VOCAB_SIZE,
            ),
            1.0 / DRAFT_VOCAB_SIZE,
            dtype=torch.float32,
        ),
        confirmed_positions=confirmed_positions,
        confirmed_input_embeds=torch.randn(
            confirmed_length,
            HIDDEN_SIZE,
            dtype=torch.float32,
        ),
        confirmed_aux_hidden_states=torch.randn(
            confirmed_length,
            HIDDEN_SIZE * NUM_AUX_HIDDEN_STATES,
            dtype=torch.float32,
        ),
        rejection_position=rejection_position,
    )


def _validate(batch: Eagle3DistillationBatch) -> None:
    batch.validate(
        hidden_size=HIDDEN_SIZE,
        num_aux_hidden_states=NUM_AUX_HIDDEN_STATES,
        draft_vocab_size=DRAFT_VOCAB_SIZE,
        feature_dtype=torch.float32,
    )


def test_valid_batch_uses_prefill_anchor() -> None:
    batch = _make_batch()

    _validate(batch)

    assert batch.proposal_positions[0].item() == (batch.prefill_positions[-1].item())

    proposal_input_embeds = torch.cat(
        (
            batch.prefill_input_embeds[-1:],
            batch.draft_token_input_embeds,
        ),
        dim=0,
    )

    assert proposal_input_embeds.shape == (
        batch.draft_length,
        HIDDEN_SIZE,
    )
    torch.testing.assert_close(
        proposal_input_embeds[0],
        batch.prefill_input_embeds[-1],
    )
    torch.testing.assert_close(
        proposal_input_embeds[1:],
        batch.draft_token_input_embeds,
    )


def test_draft_length_one_has_no_later_draft_inputs() -> None:
    batch = _make_batch(
        draft_length=1,
        rejection_position=None,
    )

    _validate(batch)

    assert batch.draft_token_input_embeds.shape == (
        0,
        HIDDEN_SIZE,
    )
    assert batch.draft_recurrent_hidden_states.shape == (
        0,
        HIDDEN_SIZE,
    )
    assert batch.confirmed_length == 2


def test_rejection_at_first_position_keeps_one_confirmed_token() -> None:
    batch = _make_batch(
        draft_length=3,
        rejection_position=0,
    )

    _validate(batch)

    assert batch.confirmed_length == 1
    assert batch.confirmed_positions.tolist() == [
        ANCHOR_POSITION + 1,
    ]


def test_all_accepted_keeps_bonus_position() -> None:
    batch = _make_batch(
        draft_length=3,
        rejection_position=None,
    )

    _validate(batch)

    assert batch.confirmed_length == 4
    assert batch.confirmed_positions.tolist() == [8, 9, 10, 11]


def test_draft_token_embedding_must_exclude_anchor() -> None:
    batch = _make_batch(draft_length=3)

    invalid_batch = replace(
        batch,
        draft_token_input_embeds=torch.randn(
            batch.draft_length,
            HIDDEN_SIZE,
        ),
    )

    with pytest.raises(
        ValueError,
        match="draft_token_input_embeds",
    ):
        _validate(invalid_batch)


def test_proposal_positions_must_start_at_anchor() -> None:
    batch = _make_batch()

    invalid_positions = batch.proposal_positions + 1
    invalid_batch = replace(
        batch,
        proposal_positions=invalid_positions,
    )

    with pytest.raises(
        ValueError,
        match="proposal positions must start at the prefill anchor",
    ):
        _validate(invalid_batch)


def test_inference_tensor_is_rejected() -> None:
    batch = _make_batch()

    with torch.inference_mode():
        inference_embeds = batch.prefill_input_embeds.clone()

    invalid_batch = replace(
        batch,
        prefill_input_embeds=inference_embeds,
    )

    with pytest.raises(
        ValueError,
        match="must not be an inference tensor",
    ):
        _validate(invalid_batch)


def test_batch_feature_must_be_detached() -> None:
    batch = _make_batch()
    invalid_batch = replace(
        batch,
        prefill_input_embeds=batch.prefill_input_embeds.requires_grad_(),
    )

    with pytest.raises(
        ValueError,
        match="prefill_input_embeds must be detached",
    ):
        _validate(invalid_batch)
