# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F

# Per-round depth decay of the EAGLE3 and MTP online losses.
_POSITION_DECAY = 0.9


def project_teacher_distribution(
    teacher_logits: torch.Tensor,
    target_token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project target-model logits onto the draft vocabulary.

    Args:
        teacher_logits: Target logits shaped [tokens, target_vocab_size].
        target_token_ids: Target token ID for each draft token ID.

    Returns:
        Conditional teacher probabilities and retained probability mass.

    Raises:
        ValueError: If tensor shapes, dtypes, or devices are invalid.
    """
    if teacher_logits.ndim != 2 or teacher_logits.numel() == 0:
        raise ValueError(
            "teacher_logits must be a nonempty [tokens, vocabulary] tensor"
        )
    if target_token_ids.ndim != 1 or target_token_ids.numel() == 0:
        raise ValueError("target_token_ids must be a nonempty one-dimensional tensor")
    if not teacher_logits.is_floating_point():
        raise ValueError("teacher_logits must be floating point")
    if target_token_ids.dtype != torch.long:
        raise ValueError("target_token_ids must use torch.long")
    if target_token_ids.device != teacher_logits.device:
        raise ValueError("teacher_logits and target_token_ids must use the same device")

    full_logits = teacher_logits.float()
    selected_logits = full_logits.index_select(
        dim=-1,
        index=target_token_ids,
    )

    teacher_probabilities = selected_logits.softmax(dim=-1)
    coverage = (selected_logits.logsumexp(dim=-1) - full_logits.logsumexp(dim=-1)).exp()

    return teacher_probabilities.detach(), coverage.detach()


def position_forward_kl_loss(
    student_logits: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    proposal_depths: torch.Tensor,
) -> torch.Tensor:
    """Compute forward KL weighted by ``0.9 ** depth`` within each round.

    Depth restarts at zero on every round's anchor row. The weighted sum is
    divided by the row count, so later rows count less without raising the
    weight of earlier ones.

    Args:
        student_logits: Selected draft logits shaped
            [target_count, draft_vocab_size].
        teacher_probabilities: Teacher probabilities with the same shape.
        proposal_depths: Draft depth of each row shaped [target_count].

    Returns:
        A scalar FP32 position-weighted forward-KL loss.
    """
    per_position_kl = F.kl_div(
        F.log_softmax(student_logits.float(), dim=-1),
        teacher_probabilities.detach().float(),
        reduction="none",
    ).sum(dim=-1)
    weights = _POSITION_DECAY ** proposal_depths.to(per_position_kl).float()
    return (per_position_kl * weights).sum() / per_position_kl.numel()
