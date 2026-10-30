# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility batch for the local GPU training adapters."""

import torch

from methods.shared.eagle3.batch import (
    Eagle3DistillationBatch as _Eagle3DistillationBatch,
)


class Eagle3DistillationBatch(_Eagle3DistillationBatch):
    """Map captured GPU fields onto the current standalone training contract."""

    __slots__ = ("teacher_coverage",)

    def __init__(
        self,
        *,
        prefill_positions: torch.Tensor,
        prefill_input_embeds: torch.Tensor,
        prefill_aux_hidden_states: torch.Tensor,
        proposal_positions: torch.Tensor,
        draft_token_input_embeds: torch.Tensor,
        draft_recurrent_hidden_states: torch.Tensor,
        teacher_probabilities: torch.Tensor,
        teacher_coverage: torch.Tensor,
        confirmed_positions: torch.Tensor,
        confirmed_input_embeds: torch.Tensor,
        confirmed_aux_hidden_states: torch.Tensor,
        rejection_position: int | None,
    ) -> None:
        super().__init__(
            prefix_positions=prefill_positions,
            prefix_input_embeds=prefill_input_embeds,
            prefix_aux_hidden_states=prefill_aux_hidden_states,
            proposal_positions=proposal_positions,
            draft_token_input_embeds=draft_token_input_embeds,
            draft_input_hidden_states=draft_recurrent_hidden_states,
            teacher_probabilities=teacher_probabilities,
            confirmed_positions=confirmed_positions,
            confirmed_input_embeds=confirmed_input_embeds,
            confirmed_aux_hidden_states=confirmed_aux_hidden_states,
            rejection_position=rejection_position,
        )
        self.teacher_coverage = teacher_coverage

    @property
    def prefill_positions(self) -> torch.Tensor:
        return self.prefix_positions

    @property
    def prefill_input_embeds(self) -> torch.Tensor:
        return self.prefix_input_embeds

    @property
    def prefill_aux_hidden_states(self) -> torch.Tensor:
        return self.prefix_aux_hidden_states

    @property
    def draft_recurrent_hidden_states(self) -> torch.Tensor:
        return self.draft_input_hidden_states
