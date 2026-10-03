# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from online_draft.runtime.transport import (
    Eagle3CapturePacket,
    Eagle3CaptureQueue,
    Eagle3PromptContext,
)
from online_draft.training.eagle3_batch import Eagle3DistillationBatch


@dataclass(frozen=True, slots=True)
class Eagle3PreparedRound:
    """One CPU batch with request and weight metadata."""

    request_id: str
    step_id: int
    source_weight_version: int
    batch: Eagle3DistillationBatch


class Eagle3CaptureRuntime:
    """Transfer GPU chunks and schedule prepared CPU EAGLE3 rounds."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_aux_hidden_states: int,
        draft_vocab_size: int,
        feature_dtype: torch.dtype,
        transfer_chunk_size: int,
        train_window_size: int | None,
        on_window: Callable[
            [tuple[Eagle3PreparedRound, ...]],
            None,
        ],
    ) -> None:
        self._hidden_size = hidden_size
        self._num_aux_hidden_states = num_aux_hidden_states
        self._draft_vocab_size = draft_vocab_size
        self._feature_dtype = feature_dtype
        self._on_window = on_window

        if train_window_size is not None and train_window_size <= 0:
            raise ValueError("train_window_size must be positive or None")

        self._train_window_size = train_window_size
        self._pending_rounds: list[Eagle3PreparedRound] = []

        self._request_id: str | None = None
        self._last_confirmed_input_embeds: torch.Tensor | None = None
        self._last_confirmed_aux_hidden_states: torch.Tensor | None = None
        self._last_confirmed_position: int | None = None

        self._queue = Eagle3CaptureQueue(
            transfer_chunk_size=transfer_chunk_size,
            consume_chunk=self._consume_chunk,
            on_finish=self._reset_request,
            on_release=self._release_pending,
        )

    def submit(self, packet: Eagle3CapturePacket) -> None:
        self._queue.submit(packet)

    def finish_request(self, request_id: str) -> None:
        self._queue.finish_request(request_id)

    def notify_install_complete(self, request_id: str) -> None:
        self._queue.release_pending(request_id)

    def close(self) -> None:
        self._queue.close()

    def raise_if_failed(self) -> None:
        self._queue.raise_if_failed()

    def _consume_chunk(
        self,
        packets: tuple[Eagle3CapturePacket, ...],
    ) -> None:
        rounds = tuple(self._build_round(packet) for packet in packets)
        self._pending_rounds.extend(rounds)

        if self._train_window_size is None:
            return

        while len(self._pending_rounds) >= self._train_window_size:
            window = tuple(self._pending_rounds[: self._train_window_size])
            del self._pending_rounds[: self._train_window_size]
            self._on_window(window)

    def _release_pending(self, request_id: str) -> None:
        if self._request_id != request_id:
            return

        if self._train_window_size is not None:
            return

        if not self._pending_rounds:
            return

        rounds = tuple(self._pending_rounds)
        self._pending_rounds.clear()
        self._on_window(rounds)

    def _build_round(
        self,
        packet: Eagle3CapturePacket,
    ) -> Eagle3PreparedRound:
        self._set_request(packet.request_id)

        if packet.prompt_context is not None:
            prefill_positions, prefill_input_embeds, prefill_aux = (
                self._build_initial_prefill(packet.prompt_context)
            )
        else:
            prefill_positions, prefill_input_embeds, prefill_aux = (
                self._build_continuation_prefill()
            )

        confirmed_length = int(packet.confirmed_prefill_length.item())
        sampled = int(packet.num_sampled.item())

        if confirmed_length != sampled:
            raise ValueError("confirmed prefill length must equal num_sampled")

        confirmed_input_embeds = packet.confirmed_prefill_input_embeds[
            :confirmed_length
        ]
        confirmed_positions = packet.confirmed_prefill_positions[:confirmed_length]
        verify_rows = packet.verify_row_indices[:confirmed_length]

        confirmed_aux = torch.cat(
            tuple(
                hidden_states.index_select(0, verify_rows)
                for hidden_states in packet.target_aux_hidden_states
            ),
            dim=-1,
        )

        draft_length = packet.proposal_token_input_embeds.shape[0] + 1
        anchor_position = int(prefill_positions[-1].item())

        proposal_positions = torch.arange(
            anchor_position,
            anchor_position + draft_length,
            dtype=torch.long,
        )

        teacher_probabilities = torch.softmax(
            packet.teacher_logits[:draft_length].float(),
            dim=-1,
        )
        teacher_probabilities = teacher_probabilities / teacher_probabilities.sum(
            dim=-1,
            keepdim=True,
            dtype=torch.float64,
        ).to(teacher_probabilities.dtype)

        rejection_position = (
            None if confirmed_length == draft_length + 1 else confirmed_length - 1
        )

        batch = Eagle3DistillationBatch(
            prefill_positions=prefill_positions,
            prefill_input_embeds=prefill_input_embeds,
            prefill_aux_hidden_states=prefill_aux,
            proposal_positions=proposal_positions,
            draft_token_input_embeds=packet.proposal_token_input_embeds,
            draft_recurrent_hidden_states=(packet.proposal_recurrent_hidden_states),
            teacher_probabilities=teacher_probabilities,
            confirmed_positions=confirmed_positions,
            confirmed_input_embeds=confirmed_input_embeds,
            confirmed_aux_hidden_states=confirmed_aux,
            rejection_position=rejection_position,
        )

        batch.validate(
            hidden_size=self._hidden_size,
            num_aux_hidden_states=self._num_aux_hidden_states,
            draft_vocab_size=self._draft_vocab_size,
            feature_dtype=self._feature_dtype,
        )

        self._last_confirmed_input_embeds = confirmed_input_embeds[-1:].clone()
        self._last_confirmed_aux_hidden_states = confirmed_aux[-1:].clone()
        self._last_confirmed_position = int(confirmed_positions[-1].item())

        return Eagle3PreparedRound(
            request_id=packet.request_id,
            step_id=packet.step_id,
            source_weight_version=packet.source_weight_version,
            batch=batch,
        )

    def _build_initial_prefill(
        self,
        context: Eagle3PromptContext,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefill_length = int(context.prefill_length.item())

        positions = context.prefill_positions[:prefill_length]
        input_embeds = context.prefill_input_embeds[:prefill_length]
        aux_rows = context.prefill_aux_row_indices[:prefill_length]

        aux_hidden_states = torch.cat(
            tuple(
                hidden_states.index_select(0, aux_rows)
                for hidden_states in context.prefill_aux_hidden_states
            ),
            dim=-1,
        )

        return positions, input_embeds, aux_hidden_states

    def _build_continuation_prefill(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._last_confirmed_input_embeds is None:
            raise RuntimeError("continuation has no previous confirmed token")

        return (
            torch.tensor(
                [self._last_confirmed_position],
                dtype=torch.long,
            ),
            self._last_confirmed_input_embeds,
            self._last_confirmed_aux_hidden_states,
        )

    def _set_request(self, request_id: str) -> None:
        if self._request_id is None:
            self._request_id = request_id
        elif self._request_id != request_id:
            raise RuntimeError("capture runtime received multiple active requests")

    def _reset_request(self, request_id: str) -> None:
        if self._request_id != request_id:
            raise RuntimeError("finished request does not match active request")

        self._pending_rounds.clear()
        self._request_id = None
        self._last_confirmed_input_embeds = None
        self._last_confirmed_aux_hidden_states = None
        self._last_confirmed_position = None
