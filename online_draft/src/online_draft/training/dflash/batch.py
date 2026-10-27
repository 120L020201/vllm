# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DFlashBatch:
    """One verified DFlash query block.

    Block row 0 holds the anchor token and row j predicts the token j
    positions later. The first ``target_count`` predictions are supervised:
    accepted drafts plus the correction, or every draft when all are
    accepted. ``confirmed_fc`` holds the frozen FC output of the verified
    rows starting at the anchor; it becomes context for the next round.
    """

    input_ids: torch.Tensor
    positions: torch.Tensor
    # Whole-prompt FC output for the first round, else None.
    prompt_fc: torch.Tensor | None
    confirmed_fc: torch.Tensor
    teacher_probabilities: torch.Tensor
    rejection_position: int | None

    @property
    def anchor(self) -> int:
        return int(self.positions[0])

    @property
    def target_count(self) -> int:
        return self.teacher_probabilities.shape[0]
