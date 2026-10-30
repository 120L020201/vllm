# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch
import torch.nn.functional as F

LOSS_MODES = ("uniform", "position", "entropy")
POSITION_DECAY_RHO = 0.9
ENTROPY_GAMMA = 2.0


def forward_kl_loss(
    student_logits: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    *,
    mode: str,
    proposal_depths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute one of the supported online-draft KL objectives."""
    if mode not in LOSS_MODES:
        raise ValueError(f"loss mode must be one of {LOSS_MODES}")
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
        raise ValueError("student logits must be floating point")
    if not teacher_probabilities.is_floating_point():
        raise ValueError("teacher probabilities must be floating point")
    if student_logits.device != teacher_probabilities.device:
        raise ValueError("student and teacher must use the same device")

    teacher = teacher_probabilities.detach().float()
    per_position = F.kl_div(
        F.log_softmax(student_logits.float(), dim=-1),
        teacher,
        reduction="none",
    ).sum(dim=-1)

    if mode == "uniform":
        weights = torch.ones_like(per_position)
    elif mode == "position":
        if proposal_depths is None or proposal_depths.shape != per_position.shape:
            raise ValueError("position loss requires one proposal depth per token")
        if proposal_depths.dtype != torch.long:
            raise ValueError("proposal depths must use torch.long")
        if proposal_depths.device != student_logits.device:
            raise ValueError("proposal depths and student logits must share a device")
        if (proposal_depths < 0).any():
            raise ValueError("proposal depths must be nonnegative")
        weights = POSITION_DECAY_RHO ** proposal_depths.float()
    else:
        if teacher.shape[-1] <= 1:
            raise ValueError("entropy loss requires multiple vocabulary entries")
        normalized_entropy = torch.special.entr(teacher).sum(dim=-1)
        normalized_entropy /= math.log(teacher.shape[-1])
        weights = torch.exp(-ENTROPY_GAMMA * normalized_entropy.clamp(min=0.0, max=1.0))

    return (per_position * weights).sum() / per_position.numel()
