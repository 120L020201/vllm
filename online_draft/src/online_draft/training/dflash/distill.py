# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F
from online_draft.training.dflash.batch import DFlashBatch


def position_weights(
    rounds: tuple[DFlashBatch, ...], gamma: float = 7.0
) -> torch.Tensor:
    """DFlash Eq. 4: first predicted position has weight one in every block."""
    offsets = torch.cat([torch.arange(batch.target_count) for batch in rounds])
    return torch.exp(-offsets.float() / gamma)


def dflash_kl_loss(
    logits: torch.Tensor, teacher: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Forward KL at T=1, normalized by the supervised position weights."""
    per_position = F.kl_div(
        F.log_softmax(logits.float(), dim=-1), teacher.detach(), reduction="none"
    ).sum(dim=-1)
    return (per_position * weights).sum() / weights.sum()
