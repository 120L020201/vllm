# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from methods.tts_common.training.eagle3.draft_model import AutoregressiveDraftModel

if TYPE_CHECKING:
    from methods.tts_common.training.eagle3.window import (
        Eagle3TrainingWindow,
    )


@dataclass(frozen=True, slots=True)
class Eagle3WindowInputs:
    """Packed query rows of every round and their isolation mask.

    Each round's first row is its anchor, recomputed through the trainable
    FC. Later rows use the fixed hidden inputs captured during GPU draft
    generation. Every row has a distillation target. In the boolean
    attention mask, True blocks a query-key pair.
    """

    positions: torch.Tensor
    input_embeds: torch.Tensor
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor
    teacher_probabilities: torch.Tensor
    rejection_target_mask: torch.Tensor

    @property
    def input_length(self) -> int:
        """Return the number of packed query rows."""
        return self.positions.numel()

    @property
    def target_count(self) -> int:
        """Return the number of distillation targets."""
        return self.teacher_probabilities.shape[0]


def prepare_eagle3_confirmed_path_inputs(
    model: AutoregressiveDraftModel,
    window: "Eagle3TrainingWindow",
    *,
    spine_length: int,
) -> Eagle3WindowInputs:
    """Pack only query rows whose targets are confirmed.

    A rejected round keeps rows through the rejection position, including
    the row trained against the correction target. A fully accepted round
    keeps all draft rows; its bonus has no prediction row or direct loss.

    Args:
        model: CPU EAGLE3 training model.
        window: Consecutive verification rounds.
        spine_length: Length of the temporary full confirmed spine.

    Returns:
        Packed confirmed query branches, targets, and isolation mask.
    """
    target_counts = tuple(
        (
            round_batch.draft_length
            if round_batch.rejection_position is None
            else round_batch.confirmed_length
        )
        for round_batch in window.rounds
    )
    return _pack_window_inputs(
        model,
        window,
        spine_length=spine_length,
        target_counts=target_counts,
    )


def prepare_eagle3_proposal_window_inputs(
    model: AutoregressiveDraftModel,
    window: "Eagle3TrainingWindow",
    *,
    spine_length: int,
) -> Eagle3WindowInputs:
    """Pack all proposal branches for one differentiable forward.

    Every branch sees confirmed spine positions strictly before its anchor
    and a causal prefix of only its own proposal rows. Other branches and
    future confirmed positions remain blocked.

    Args:
        model: CPU EAGLE3 training model.
        window: Consecutive verification rounds.
        spine_length: Length of the temporary full confirmed spine.

    Returns:
        Packed proposal inputs, targets, and branch-isolation mask.
    """
    target_counts = tuple(round_batch.draft_length for round_batch in window.rounds)
    return _pack_window_inputs(
        model,
        window,
        spine_length=spine_length,
        target_counts=target_counts,
    )


def _pack_window_inputs(
    model: AutoregressiveDraftModel,
    window: "Eagle3TrainingWindow",
    *,
    spine_length: int,
    target_counts: tuple[int, ...],
) -> Eagle3WindowInputs:
    anchor_aux_hidden_states = torch.cat(
        tuple(
            round_batch.prefix_aux_hidden_states[-1:] for round_batch in window.rounds
        ),
        dim=0,
    )
    anchor_hidden_states = model.combine_hidden_states(anchor_aux_hidden_states)

    position_parts: list[torch.Tensor] = []
    input_embed_parts: list[torch.Tensor] = []
    hidden_state_parts: list[torch.Tensor] = []
    teacher_parts: list[torch.Tensor] = []
    for index, (round_batch, target_count) in enumerate(
        zip(window.rounds, target_counts, strict=True)
    ):
        # The anchor is the final prefix row; later rows were generated on
        # the GPU and are fixed inputs rather than CPU model outputs.
        input_embeds = torch.cat(
            (
                round_batch.prefix_input_embeds[-1:],
                round_batch.draft_token_input_embeds,
            ),
            dim=0,
        )
        hidden_states = torch.cat(
            (
                anchor_hidden_states[index : index + 1],
                round_batch.draft_input_hidden_states,
            ),
            dim=0,
        )
        position_parts.append(round_batch.proposal_positions[:target_count])
        input_embed_parts.append(input_embeds[:target_count])
        hidden_state_parts.append(hidden_states[:target_count])
        teacher_parts.append(round_batch.teacher_probabilities[:target_count])

    positions = torch.cat(position_parts, dim=0)
    total_query_length = positions.numel()

    attention_mask = torch.ones(
        (
            total_query_length,
            spine_length + total_query_length,
        ),
        dtype=torch.bool,
        device=positions.device,
    )
    rejection_target_mask = torch.zeros(
        total_query_length,
        dtype=torch.bool,
        device=positions.device,
    )

    query_start = 0
    for round_batch, target_count in zip(window.rounds, target_counts, strict=True):
        query_end = query_start + target_count
        anchor_position = int(round_batch.proposal_positions[0].item())

        # The detached spine already contains the anchor. Block that copy
        # and later spine entries so this branch uses its trainable anchor.
        attention_mask[query_start:query_end, :anchor_position] = False

        branch_causal_mask = torch.triu(
            torch.ones(
                target_count,
                target_count,
                dtype=torch.bool,
                device=positions.device,
            ),
            diagonal=1,
        )
        branch_key_start = spine_length + query_start
        branch_key_end = spine_length + query_end
        attention_mask[
            query_start:query_end,
            branch_key_start:branch_key_end,
        ] = branch_causal_mask

        rejection_position = round_batch.rejection_position
        if rejection_position is not None and rejection_position < target_count:
            rejection_target_mask[query_start + rejection_position] = True

        query_start = query_end

    return Eagle3WindowInputs(
        positions=positions,
        input_embeds=torch.cat(input_embed_parts, dim=0),
        hidden_states=torch.cat(hidden_state_parts, dim=0),
        attention_mask=attention_mask,
        teacher_probabilities=torch.cat(teacher_parts, dim=0),
        rejection_target_mask=rejection_target_mask,
    )
