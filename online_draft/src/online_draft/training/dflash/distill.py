# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F
from online_draft.training.dflash.batch import DFlashBatch

# DFlash Eq. 4 decay; part of the loss definition, not a runtime setting.
_LOSS_DECAY_GAMMA = 7.0


def position_weights(rounds: tuple[DFlashBatch, ...]) -> torch.Tensor:
    """DFlash Eq. 4: first predicted position has weight one in every block."""
    offsets = torch.cat([torch.arange(batch.target_count) for batch in rounds])
    return torch.exp(-offsets.float() / _LOSS_DECAY_GAMMA)


def dflash_kl_loss(
    logits: torch.Tensor,
    teacher: torch.Tensor,
    weights: torch.Tensor,
    *,
    temperature: float = 0.0,
) -> torch.Tensor:
    """Forward KL normalized by the supervised position weights.

    Args:
        logits: Draft logits shaped [rows, vocab_size].
        teacher: Teacher probabilities at T=1 with the same shape.
        weights: Position weights shaped [rows].
        temperature: Zero keeps the T=1 KL. A positive value sharpens both
            distributions to it; the teacher stays dense (no top-k/top-p
            truncation) so every vocabulary entry gets a gradient.

    Returns:
        A scalar FP32 weighted forward-KL loss.
    """
    student_logits = logits.float()
    teacher = teacher.detach()
    if temperature > 0:
        teacher = (teacher.float().clamp_min(1e-12).log() / temperature).softmax(dim=-1)
        student_logits = student_logits / temperature
    per_position = F.kl_div(
        F.log_softmax(student_logits, dim=-1), teacher, reduction="none"
    ).sum(dim=-1)
    return (per_position * weights).sum() / weights.sum()
