from __future__ import annotations

import torch

from online_eagle3.observations import (
    IGNORE_LABEL,
    build_target_to_draft_map,
    build_verified_target_labels,
    map_target_to_draft_labels,
)


def test_build_verified_target_labels_all_draft_tokens_active() -> None:
    sampled_token_ids = torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.int32)
    num_sampled = torch.tensor([5], dtype=torch.int32)

    labels = build_verified_target_labels(
        sampled_token_ids,
        num_sampled,
        num_speculative_tokens=4,
    )

    assert labels.target_token_ids.tolist() == [10, 11, 12, 13]
    assert labels.loss_mask.tolist() == [True, True, True, True]


def test_build_verified_target_labels_masks_after_first_rejected_token() -> None:
    sampled_token_ids = torch.tensor([[10, 99, -1, -1, -1]], dtype=torch.int32)
    num_sampled = torch.tensor([2], dtype=torch.int32)

    labels = build_verified_target_labels(
        sampled_token_ids,
        num_sampled,
        num_speculative_tokens=4,
    )

    assert labels.target_token_ids.tolist() == [10, 99, IGNORE_LABEL, IGNORE_LABEL]
    assert labels.loss_mask.tolist() == [True, True, False, False]


def test_build_verified_target_labels_handles_empty_verify_result() -> None:
    sampled_token_ids = torch.tensor([[-1, -1]], dtype=torch.int32)
    num_sampled = torch.tensor([0], dtype=torch.int32)

    labels = build_verified_target_labels(
        sampled_token_ids,
        num_sampled,
        num_speculative_tokens=2,
    )

    assert labels.target_token_ids.tolist() == [IGNORE_LABEL, IGNORE_LABEL]
    assert labels.loss_mask.tolist() == [False, False]


def test_map_target_to_draft_labels_uses_d2t_offsets() -> None:
    # draft 0 -> target 0, draft 1 -> target 10, draft 2 -> target 12.
    d2t = torch.tensor([0, 9, 10], dtype=torch.long)
    target_to_draft = build_target_to_draft_map(d2t, target_vocab_size=13)

    labels = map_target_to_draft_labels(
        torch.tensor([0, 10, 5, 12]),
        target_to_draft,
    )

    assert labels.draft_token_ids.tolist() == [0, 1, IGNORE_LABEL, 2]
    assert labels.loss_mask.tolist() == [True, True, False, True]


def test_map_target_to_draft_labels_preserves_existing_mask() -> None:
    d2t = torch.tensor([0, 9, 10], dtype=torch.long)
    target_to_draft = build_target_to_draft_map(d2t, target_vocab_size=13)

    labels = map_target_to_draft_labels(
        torch.tensor([0, 10, 12]),
        target_to_draft,
        torch.tensor([True, False, True]),
    )

    assert labels.draft_token_ids.tolist() == [0, IGNORE_LABEL, 2]
    assert labels.loss_mask.tolist() == [True, False, True]
