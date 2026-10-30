# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F

_POSITION_WEIGHT = 1.0
_REJECTION_POSITION_WEIGHT = 2.0


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


def window_forward_kl_loss(
    student_logits: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    rejection_target_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute forward KL for selected targets from multiple rounds.

    The caller selects supervised hidden-state rows before the LM head,
    so student and teacher rows are already aligned. Multiple positions
    may be marked as rejected because one window may contain many rounds.

    Args:
        student_logits: Selected draft logits shaped
            [target_count, draft_vocab_size].
        teacher_probabilities: Teacher probabilities with the same shape.
        rejection_target_mask: Boolean rejection markers shaped
            [target_count].

    Returns:
        A scalar FP32 rejection-weighted forward-KL loss.
    """
    position_weights = _make_masked_position_weights(rejection_target_mask)
    per_position_kl = F.kl_div(
        F.log_softmax(student_logits.float(), dim=-1),
        teacher_probabilities.detach().float(),
        reduction="none",
    ).sum(dim=-1)
    return (per_position_kl * position_weights).sum() / position_weights.sum()


def _make_masked_position_weights(
    rejection_target_mask: torch.Tensor,
) -> torch.Tensor:
    weights = torch.full(
        rejection_target_mask.shape,
        _POSITION_WEIGHT,
        dtype=torch.float32,
        device=rejection_target_mask.device,
    )
    weights[rejection_target_mask] = _REJECTION_POSITION_WEIGHT

    return weights
