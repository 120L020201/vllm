# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import torch

from online_draft.runtime.capture.base import (
    CaptureAdapterBase,
    CaptureBuffers,
    ConfirmedRows,
    TargetHistoryBuilder,
    TargetPacket,
    teacher_probabilities,
)
from online_draft.runtime.contracts import PreparedRound
from online_draft.runtime.cpu_worker import CPUWorkerConfig
from online_draft.runtime.transport import PinnedStaging
from online_draft.training.eagle3.batch import Eagle3DistillationBatch

Eagle3PreparedRound = PreparedRound[Eagle3DistillationBatch]
# Draft prefill input embeddings and concatenated target aux hidden states.
Eagle3Rows = tuple[torch.Tensor, torch.Tensor]


class Eagle3CaptureBuffers(CaptureBuffers):
    """EAGLE3 packet layout in the shared pinned staging."""

    def __init__(
        self,
        max_tokens: int,
        num_speculative_tokens: int,
        hidden_size: int,
        num_aux_hidden_states: int,
        draft_vocab_size: int,
        *,
        dtype: torch.dtype,
        teacher_dtype: torch.dtype,
    ) -> None:
        proposal_shape = (max(num_speculative_tokens - 1, 0), hidden_size)
        super().__init__(
            max_tokens=max_tokens,
            teacher_shape=(num_speculative_tokens + 1, draft_vocab_size),
            teacher_dtype=teacher_dtype,
            layouts={
                "proposal_token_input_embeds": (proposal_shape, dtype),
                "proposal_recurrent_hidden_states": (proposal_shape, dtype),
                "input_embeds": ((max_tokens, hidden_size), dtype),
                "aux_hidden_states": (
                    (max_tokens, num_aux_hidden_states * hidden_size),
                    dtype,
                ),
            },
        )


@dataclass(slots=True)
class Eagle3Proposal:
    """Draft rows after the anchor from one completed GPU proposal."""

    token_input_embeds: torch.Tensor
    recurrent_hidden_states: torch.Tensor
    source_weight_version: int

    def to_cpu(
        self,
        buffers: PinnedStaging,
        *,
        non_blocking: bool = False,
    ) -> Eagle3Proposal:
        return replace(
            self,
            token_input_embeds=buffers.copy(
                "proposal_token_input_embeds",
                self.token_input_embeds,
                non_blocking,
            ),
            recurrent_hidden_states=buffers.copy(
                "proposal_recurrent_hidden_states",
                self.recurrent_hidden_states,
                non_blocking,
            ),
        )


@dataclass(slots=True, kw_only=True)
class Eagle3CapturePacket(TargetPacket[Eagle3Proposal]):
    """Draft prefill rows: input embeddings and target aux hidden states."""

    input_embeds: torch.Tensor
    aux_hidden_states: torch.Tensor


class _Eagle3BatchBuilder(
    TargetHistoryBuilder[Eagle3CapturePacket, Eagle3Rows, Eagle3DistillationBatch]
):
    """Assemble EAGLE3 rounds and keep the last confirmed token."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_aux_hidden_states: int,
        draft_vocab_size: int,
        feature_dtype: torch.dtype,
        on_round: Callable[[Eagle3PreparedRound], object],
        on_epoch_close: Callable[[str], object],
        on_finish: Callable[[str], object],
    ) -> None:
        super().__init__(
            on_round=on_round,
            on_epoch_close=on_epoch_close,
            on_finish=on_finish,
        )
        self._hidden_size = hidden_size
        self._num_aux_hidden_states = num_aux_hidden_states
        self._draft_vocab_size = draft_vocab_size
        self._feature_dtype = feature_dtype
        self._last_confirmed: ConfirmedRows[Eagle3Rows] | None = None

    def _own_rows(self, packet: Eagle3CapturePacket, count: int) -> Eagle3Rows:
        return (
            packet.input_embeds[:count].clone(),
            packet.aux_hidden_states[:count].clone(),
        )

    def _build_batch(
        self,
        proposal: Eagle3Proposal,
        teacher_logits: torch.Tensor,
        confirmed: ConfirmedRows[Eagle3Rows],
        prompt: list[ConfirmedRows[Eagle3Rows]],
    ) -> Eagle3DistillationBatch:
        # The first round's prefill is the whole prompt; later rounds start
        # from the previous round's last confirmed token.
        if prompt:
            prefill_positions = torch.cat([rows.positions for rows in prompt])
            prefill_input_embeds = torch.cat([rows.features[0] for rows in prompt])
            prefill_aux = torch.cat([rows.features[1] for rows in prompt])
        elif self._last_confirmed is not None:
            prefill_positions = self._last_confirmed.positions
            prefill_input_embeds, prefill_aux = self._last_confirmed.features
        else:
            raise RuntimeError("continuation has no previous confirmed token")

        confirmed_input_embeds, confirmed_aux = confirmed.features
        confirmed_length = confirmed.positions.numel()
        draft_length = proposal.token_input_embeds.shape[0] + 1
        anchor_position = int(prefill_positions[-1].item())

        batch = Eagle3DistillationBatch(
            prefill_positions=prefill_positions,
            prefill_input_embeds=prefill_input_embeds,
            prefill_aux_hidden_states=prefill_aux,
            proposal_positions=torch.arange(
                anchor_position,
                anchor_position + draft_length,
                dtype=torch.long,
            ),
            draft_token_input_embeds=proposal.token_input_embeds.clone(),
            draft_recurrent_hidden_states=proposal.recurrent_hidden_states.clone(),
            teacher_probabilities=teacher_probabilities(teacher_logits[:draft_length]),
            confirmed_positions=confirmed.positions,
            confirmed_input_embeds=confirmed_input_embeds,
            confirmed_aux_hidden_states=confirmed_aux,
            rejection_position=(
                None if confirmed_length == draft_length + 1 else confirmed_length - 1
            ),
        )
        batch.validate(
            hidden_size=self._hidden_size,
            num_aux_hidden_states=self._num_aux_hidden_states,
            draft_vocab_size=self._draft_vocab_size,
            feature_dtype=self._feature_dtype,
        )

        self._last_confirmed = ConfirmedRows(
            confirmed.positions[-1:].clone(),
            (confirmed_input_embeds[-1:].clone(), confirmed_aux[-1:].clone()),
        )
        return batch

    def _reset_history(self) -> None:
        self._last_confirmed = None


class Eagle3CaptureAdapter(CaptureAdapterBase[Eagle3Proposal]):
    """Capture EAGLE3 draft prefill rows and proposals."""

    _packet_type = Eagle3CapturePacket

    def __init__(
        self,
        *,
        buffers: Eagle3CaptureBuffers,
        target_token_ids: torch.Tensor,
        source_weight_version: Callable[[], int],
        hidden_size: int,
        num_aux_hidden_states: int,
        draft_vocab_size: int,
        feature_dtype: torch.dtype,
        on_round: Callable[[Eagle3PreparedRound], object],
        on_epoch_close: Callable[[str], object],
        on_finish: Callable[[str], object],
        on_queue_size: Callable[[int], None] | None = None,
        profile_timeline: bool = False,
        cpu_config: CPUWorkerConfig | None = None,
    ) -> None:
        super().__init__(
            buffers=buffers,
            builder=_Eagle3BatchBuilder(
                hidden_size=hidden_size,
                num_aux_hidden_states=num_aux_hidden_states,
                draft_vocab_size=draft_vocab_size,
                feature_dtype=feature_dtype,
                on_round=on_round,
                on_epoch_close=on_epoch_close,
                on_finish=on_finish,
            ),
            source_weight_version=source_weight_version,
            target_token_ids=target_token_ids,
            on_queue_size=on_queue_size,
            profile_timeline=profile_timeline,
            cpu_config=cpu_config,
        )

    def capture_target(
        self,
        *,
        request_id: str,
        final_prefill: bool,
        input_embeds: torch.Tensor,
        positions: torch.Tensor,
        aux_hidden_states: Sequence[torch.Tensor],
        confirmed_length: torch.Tensor,
        num_sampled: torch.Tensor,
    ) -> None:
        """Submit one draft prefill and the pending proposal it verifies.

        Args:
            request_id: Active request.
            final_prefill: Whether this step completes the prompt.
            input_embeds: Draft prefill input embeddings for scheduled tokens.
            positions: Draft prefill positions for scheduled tokens.
            aux_hidden_states: Target auxiliary hidden states, row-aligned
                with input_embeds.
            confirmed_length: Device scalar; leading rows that are prompt
                tokens or accepted tokens.
            num_sampled: Device scalar sampled token count for this step.
        """
        with self._profile("online_draft.capture_target"):
            self._submit_target(
                request_id=request_id,
                final_prefill=final_prefill,
                positions=positions,
                confirmed_length=confirmed_length,
                num_sampled=num_sampled,
                input_embeds=input_embeds.detach().clone(),
                aux_hidden_states=torch.cat(
                    tuple(tensor.detach() for tensor in aux_hidden_states),
                    dim=-1,
                ),
            )

    def capture_proposal(
        self,
        *,
        token_input_embeds: torch.Tensor,
        recurrent_hidden_states: torch.Tensor,
    ) -> None:
        self._save_proposal(
            lambda version: Eagle3Proposal(
                token_input_embeds=token_input_embeds.detach().clone(),
                recurrent_hidden_states=recurrent_hidden_states.detach().clone(),
                source_weight_version=version,
            )
        )
