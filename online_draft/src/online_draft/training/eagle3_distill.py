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


def forward_kl_loss(
    student_logits: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    rejection_position: int | None,
) -> torch.Tensor:
    """Compute rejection-weighted forward KL.

    Args:
        student_logits: Draft logits shaped [draft_length, draft_vocab_size].
        teacher_probabilities: Teacher probabilities with the same shape.
        rejection_position: Zero-based rejected draft position, or None when
            every draft token was accepted.

    Returns:
        A scalar FP32 forward-KL loss.

    Raises:
        TypeError: If rejection_position is not an integer or None.
        ValueError: If inputs or rejection_position are invalid.
    """
    _validate_distillation_inputs(
        student_logits,
        teacher_probabilities,
    )

    draft_length = student_logits.shape[0]
    position_weights = _make_position_weights(
        draft_length=draft_length,
        rejection_position=rejection_position,
        device=student_logits.device,
    )

    return _compute_weighted_forward_kl(
        student_logits=student_logits,
        teacher_probabilities=teacher_probabilities,
        position_weights=position_weights,
    )


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

    Raises:
        ValueError: If tensor shapes, dtypes, or devices are invalid.
    """
    _validate_distillation_inputs(
        student_logits,
        teacher_probabilities,
    )

    if rejection_target_mask.shape != student_logits.shape[:1]:
        raise ValueError("rejection_target_mask must match the student token count")
    if rejection_target_mask.dtype != torch.bool:
        raise ValueError("rejection_target_mask must use torch.bool")
    if rejection_target_mask.device != student_logits.device:
        raise ValueError("rejection mask and student logits must use the same device")

    position_weights = _make_masked_position_weights(rejection_target_mask)

    return _compute_weighted_forward_kl(
        student_logits=student_logits,
        teacher_probabilities=teacher_probabilities,
        position_weights=position_weights,
    )


def _validate_distillation_inputs(
    student_logits: torch.Tensor,
    teacher_probabilities: torch.Tensor,
) -> None:
    if (
        student_logits.ndim != 2
        or student_logits.shape != teacher_probabilities.shape
        or student_logits.shape[0] == 0
    ):
        raise ValueError(
            "student and teacher must have matching nonempty "
            "[tokens, vocabulary] shapes"
        )
    if not student_logits.is_floating_point():
        raise ValueError("student_logits must be floating point")
    if not teacher_probabilities.is_floating_point():
        raise ValueError("teacher_probabilities must be floating point")
    if student_logits.device != teacher_probabilities.device:
        raise ValueError(
            "student logits and teacher probabilities must use the same device"
        )


def _compute_weighted_forward_kl(
    *,
    student_logits: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    position_weights: torch.Tensor,
) -> torch.Tensor:
    teacher = teacher_probabilities.detach().float()

    if not torch.isfinite(teacher).all():
        raise ValueError("teacher probabilities must be finite")
    if (teacher < 0).any():
        raise ValueError("teacher probabilities must be nonnegative")

    probability_sums = teacher.sum(dim=-1)
    if not torch.allclose(
        probability_sums,
        torch.ones_like(probability_sums),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("teacher probabilities must sum to one")

    elementwise_kl = F.kl_div(
        F.log_softmax(student_logits.float(), dim=-1),
        teacher,
        reduction="none",
    )
    per_position_kl = elementwise_kl.sum(dim=-1)

    loss = (per_position_kl * position_weights).sum() / position_weights.sum()

    if not torch.isfinite(loss):
        raise ValueError("distillation loss must be finite")

    return loss


def _make_position_weights(
    draft_length: int,
    rejection_position: int | None,
    device: torch.device,
) -> torch.Tensor:
    if rejection_position is not None:
        if isinstance(rejection_position, bool) or not isinstance(
            rejection_position, int
        ):
            raise TypeError("rejection_position must be an integer or None")
        if not 0 <= rejection_position < draft_length:
            raise ValueError("rejection_position must be within the draft sequence")

    rejection_target_mask = torch.zeros(
        (draft_length,),
        dtype=torch.bool,
        device=device,
    )

    if rejection_position is not None:
        rejection_target_mask[rejection_position] = True

    return _make_masked_position_weights(rejection_target_mask)


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
