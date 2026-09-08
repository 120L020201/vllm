# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

import torch

IGNORE_LABEL = -100


@dataclass(slots=True)
class DraftVerifyLabels:
    target_token_ids: torch.Tensor
    loss_mask: torch.Tensor


@dataclass(slots=True)
class DraftTrainLabels:
    draft_token_ids: torch.Tensor
    loss_mask: torch.Tensor


def build_verified_target_labels(
    sampled_token_ids: torch.Tensor,
    num_sampled: torch.Tensor,
    num_speculative_tokens: int,
    *,
    ignore_index: int = IGNORE_LABEL,
) -> DraftVerifyLabels:
    """Build per-draft-step labels from one request's verify result."""
    if num_speculative_tokens < 1:
        raise ValueError("num_speculative_tokens must be >= 1")
    if sampled_token_ids.ndim != 2 or sampled_token_ids.shape[0] != 1:
        raise ValueError("sampled_token_ids must have shape [1, num_tokens]")

    labels = sampled_token_ids.new_full(
        (num_speculative_tokens,), ignore_index, dtype=torch.long
    )
    loss_mask = torch.zeros(
        num_speculative_tokens, dtype=torch.bool, device=sampled_token_ids.device
    )

    active_tokens = max(0, int(num_sampled.reshape(-1)[0].item()))
    active_tokens = min(
        active_tokens,
        num_speculative_tokens,
        sampled_token_ids.shape[1],
    )
    if active_tokens == 0:
        return DraftVerifyLabels(target_token_ids=labels, loss_mask=loss_mask)

    active_labels = sampled_token_ids[0, :active_tokens].to(torch.long)
    labels[:active_tokens] = active_labels
    loss_mask[:active_tokens] = active_labels >= 0
    return DraftVerifyLabels(target_token_ids=labels, loss_mask=loss_mask)


def build_target_to_draft_map(
    draft_id_to_target_id: torch.Tensor,
    target_vocab_size: int,
) -> torch.Tensor:
    """Invert EAGLE's draft-id to target-id offset mapping."""
    if target_vocab_size < 1:
        raise ValueError("target_vocab_size must be >= 1")
    d2t = draft_id_to_target_id.to(torch.long)
    draft_ids = torch.arange(d2t.shape[0], dtype=torch.long, device=d2t.device)
    target_ids = draft_ids + d2t
    target_to_draft = torch.full(
        (target_vocab_size,), -1, dtype=torch.long, device=d2t.device
    )
    valid = (target_ids >= 0) & (target_ids < target_vocab_size)
    target_to_draft[target_ids[valid]] = draft_ids[valid]
    return target_to_draft


def map_target_to_draft_labels(
    target_token_ids: torch.Tensor,
    target_to_draft: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    *,
    ignore_index: int = IGNORE_LABEL,
) -> DraftTrainLabels:
    """Map target-vocab CE labels into the draft vocabulary."""
    target_ids = target_token_ids.to(torch.long)
    if loss_mask is None:
        active_mask = torch.ones_like(target_ids, dtype=torch.bool)
    else:
        active_mask = loss_mask.to(torch.bool).clone()

    active_mask &= target_ids >= 0
    active_mask &= target_ids < target_to_draft.shape[0]

    draft_labels = torch.full_like(target_ids, ignore_index)
    final_mask = torch.zeros_like(active_mask)

    if active_mask.any():
        active_indices = active_mask.nonzero(as_tuple=True)
        mapped = target_to_draft[target_ids[active_indices]]
        mapped_mask = mapped >= 0
        if mapped_mask.any():
            kept_indices = tuple(index[mapped_mask] for index in active_indices)
            draft_labels[kept_indices] = mapped[mapped_mask]
            final_mask[kept_indices] = True

    return DraftTrainLabels(draft_token_ids=draft_labels, loss_mask=final_mask)
