# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F


def project_teacher_distribution(
    logits: torch.Tensor, d2t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Condition the teacher on the draft vocabulary at temperature one."""
    ids = torch.arange(d2t.numel(), device=d2t.device) + d2t
    selected = logits.float().index_select(-1, ids)
    coverage = (selected.logsumexp(-1) - logits.float().logsumexp(-1)).exp()
    return selected.softmax(-1).detach(), coverage.detach()


def distillation_loss(logits: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Uniform-depth forward KL over the complete verified canvas."""
    if logits.shape != teacher.shape or logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError(
            "Student and teacher must have matching nonempty [K, V] shapes"
        )
    if not torch.isfinite(teacher).all() or (teacher < 0).any():
        raise ValueError("Teacher probabilities must be finite and nonnegative")
    if not torch.allclose(teacher.sum(-1), torch.ones_like(teacher[:, 0]), atol=1e-5):
        raise ValueError("Teacher probabilities must sum to one")
    return F.kl_div(
        F.log_softmax(logits.float(), dim=-1),
        teacher.detach().float(),
        reduction="batchmean",
    )
