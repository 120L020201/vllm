# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from online_draft.runtime.contracts import PreparedRound
from online_draft.runtime.cpu_worker import CPUWorkerConfig
from online_draft.runtime.transport import CaptureQueue, copy_to_cpu
from online_draft.training.eagle3.batch import Eagle3DistillationBatch

Eagle3PreparedRound = PreparedRound[Eagle3DistillationBatch]


@dataclass(slots=True)
class Eagle3PromptContext:
    """Prompt data transferred once for one request."""

    prefill_input_embeds: torch.Tensor
    prefill_positions: torch.Tensor
    prefill_length: torch.Tensor
    prefill_aux_hidden_states: tuple[torch.Tensor, ...]
    prefill_aux_row_indices: torch.Tensor

    def to_cpu(
        self,
        *,
        non_blocking: bool = False,
    ) -> Eagle3PromptContext:
        return replace(
            self,
            prefill_input_embeds=copy_to_cpu(
                self.prefill_input_embeds,
                non_blocking=non_blocking,
            ),
            prefill_positions=copy_to_cpu(
                self.prefill_positions,
                non_blocking=non_blocking,
            ),
            prefill_length=copy_to_cpu(
                self.prefill_length,
                non_blocking=non_blocking,
            ),
            prefill_aux_hidden_states=tuple(
                copy_to_cpu(
                    tensor,
                    non_blocking=non_blocking,
                )
                for tensor in self.prefill_aux_hidden_states
            ),
            prefill_aux_row_indices=copy_to_cpu(
                self.prefill_aux_row_indices,
                non_blocking=non_blocking,
            ),
        )


@dataclass(slots=True)
class Eagle3CapturePacket:
    """One GPU-resident EAGLE3 verification round."""

    request_id: str
    step_id: int
    source_weight_version: int

    prompt_context: Eagle3PromptContext | None

    proposal_token_input_embeds: torch.Tensor
    proposal_recurrent_hidden_states: torch.Tensor

    teacher_logits: torch.Tensor
    verify_row_indices: torch.Tensor
    target_aux_hidden_states: tuple[torch.Tensor, ...]
    num_sampled: torch.Tensor

    confirmed_prefill_input_embeds: torch.Tensor
    confirmed_prefill_positions: torch.Tensor
    confirmed_prefill_length: torch.Tensor

    ready: torch.cuda.Event | None = None

    @property
    def device(self) -> torch.device:
        return self.proposal_token_input_embeds.device

    def to_cpu(
        self,
        *,
        non_blocking: bool = False,
    ) -> Eagle3CapturePacket:
        prompt_context = (
            None
            if self.prompt_context is None
            else self.prompt_context.to_cpu(
                non_blocking=non_blocking,
            )
        )

        return replace(
            self,
            prompt_context=prompt_context,
            proposal_token_input_embeds=copy_to_cpu(
                self.proposal_token_input_embeds,
                non_blocking=non_blocking,
            ),
            proposal_recurrent_hidden_states=copy_to_cpu(
                self.proposal_recurrent_hidden_states,
                non_blocking=non_blocking,
            ),
            teacher_logits=copy_to_cpu(
                self.teacher_logits,
                non_blocking=non_blocking,
            ),
            verify_row_indices=copy_to_cpu(
                self.verify_row_indices,
                non_blocking=non_blocking,
            ),
            target_aux_hidden_states=tuple(
                copy_to_cpu(
                    tensor,
                    non_blocking=non_blocking,
                )
                for tensor in self.target_aux_hidden_states
            ),
            num_sampled=copy_to_cpu(
                self.num_sampled,
                non_blocking=non_blocking,
            ),
            confirmed_prefill_input_embeds=copy_to_cpu(
                self.confirmed_prefill_input_embeds,
                non_blocking=non_blocking,
            ),
            confirmed_prefill_positions=copy_to_cpu(
                self.confirmed_prefill_positions,
                non_blocking=non_blocking,
            ),
            confirmed_prefill_length=copy_to_cpu(
                self.confirmed_prefill_length,
                non_blocking=non_blocking,
            ),
            ready=None,
        )


class _Eagle3BatchBuilder:
    """Build CPU rounds and own confirmed history on the capture worker."""

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
        self._hidden_size = hidden_size
        self._num_aux_hidden_states = num_aux_hidden_states
        self._draft_vocab_size = draft_vocab_size
        self._feature_dtype = feature_dtype
        self._on_round = on_round
        self._on_epoch_close_callback = on_epoch_close
        self._on_finish_callback = on_finish

        self._request_id: str | None = None
        self._last_confirmed_input_embeds: torch.Tensor | None = None
        self._last_confirmed_aux_hidden_states: torch.Tensor | None = None
        self._last_confirmed_position: int | None = None

    def consume(
        self,
        packets: tuple[Eagle3CapturePacket, ...],
    ) -> None:
        for packet in packets:
            self._on_round(self._build_round(packet))

    def close_epoch(self, request_id: str) -> None:
        if self._request_id != request_id:
            raise RuntimeError("closed epoch does not match active request")

        self._on_epoch_close_callback(request_id)

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

        return PreparedRound(
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

    def finish_request(self, request_id: str) -> None:
        if self._request_id is not None and self._request_id != request_id:
            raise RuntimeError("finished request does not match active request")

        self._request_id = None
        self._last_confirmed_input_embeds = None
        self._last_confirmed_aux_hidden_states = None
        self._last_confirmed_position = None

        self._on_finish_callback(request_id)


@dataclass(slots=True)
class _GpuProposal:
    prefill_input_embeds: torch.Tensor
    prefill_positions: torch.Tensor
    prefill_length: torch.Tensor
    prefill_aux_hidden_states: tuple[torch.Tensor, ...]
    prefill_aux_row_indices: torch.Tensor
    token_input_embeds: torch.Tensor
    recurrent_hidden_states: torch.Tensor
    source_weight_version: int


class Eagle3CaptureAdapter:
    """Convert vLLM EAGLE3 callbacks into GPU capture packets."""

    def __init__(
        self,
        *,
        target_token_ids: torch.Tensor,
        source_weight_version: Callable[[], int],
        hidden_size: int,
        num_aux_hidden_states: int,
        draft_vocab_size: int,
        feature_dtype: torch.dtype,
        on_round: Callable[[Eagle3PreparedRound], object],
        on_epoch_close: Callable[[str], object],
        on_finish: Callable[[str], object],
        transfer_chunk_size: int = 1,
        on_queue_size: Callable[[int], None] | None = None,
        profile_timeline: bool = False,
        cpu_config: CPUWorkerConfig | None = None,
    ) -> None:
        self._target_token_ids = target_token_ids.detach()
        self._source_weight_version = source_weight_version

        self._request_id: str | None = None
        self._step_id = 0
        self._proposal: _GpuProposal | None = None
        self._teacher_logits: torch.Tensor | None = None

        self._batch_builder = _Eagle3BatchBuilder(
            hidden_size=hidden_size,
            num_aux_hidden_states=num_aux_hidden_states,
            draft_vocab_size=draft_vocab_size,
            feature_dtype=feature_dtype,
            on_round=on_round,
            on_epoch_close=on_epoch_close,
            on_finish=on_finish,
        )
        self._queue = CaptureQueue(
            transfer_chunk_size=transfer_chunk_size,
            consume_chunk=self._batch_builder.consume,
            on_finish=self._batch_builder.finish_request,
            on_close_epoch=self._batch_builder.close_epoch,
            on_queue_size=on_queue_size,
            profile_timeline=profile_timeline,
            cpu_config=cpu_config,
        )

    @property
    def has_proposal(self) -> bool:
        return self._proposal is not None

    @property
    def completed_rounds(self) -> int:
        return self._step_id

    def capture_teacher_logits(self, logits: torch.Tensor) -> None:
        """Project teacher logits before sampling modifies them."""

        if self._proposal is None:
            return

        self._teacher_logits = (
            logits.index_select(
                dim=-1,
                index=self._target_token_ids,
            )
            .detach()
            .clone()
        )

    def capture_proposal(
        self,
        *,
        request_id: str,
        verify_row_indices: torch.Tensor,
        target_aux_hidden_states: Sequence[torch.Tensor],
        proposal: Any,
    ) -> None:
        """Save a completed proposal for a later verification round."""

        if self._proposal is None:
            self._request_id = request_id
            aux_row_indices = self._initial_aux_rows(target_aux_hidden_states)
        else:
            if request_id != self._request_id:
                raise RuntimeError("another request is already being captured")
            aux_row_indices = verify_row_indices

        self._proposal = self._save_proposal(
            proposal=proposal,
            target_aux_hidden_states=target_aux_hidden_states,
            aux_row_indices=aux_row_indices,
        )

    def capture_prefill(
        self,
        *,
        request_id: str,
        verify_row_indices: torch.Tensor,
        target_aux_hidden_states: Sequence[torch.Tensor],
        num_sampled: torch.Tensor | int,
        prefill_input_embeds: torch.Tensor,
        prefill_positions: torch.Tensor,
        prefill_length: torch.Tensor,
    ) -> None:
        """Submit the previous proposal when the next prefill is ready."""

        if self._proposal is None:
            return

        if request_id != self._request_id:
            raise RuntimeError("another request is already being captured")

        if self._teacher_logits is None:
            raise RuntimeError("teacher logits are missing")

        device = prefill_input_embeds.device
        sampled = self._as_device_scalar(num_sampled, device)

        prompt_context = None
        if self._step_id == 0:
            prompt_context = Eagle3PromptContext(
                prefill_input_embeds=(self._proposal.prefill_input_embeds),
                prefill_positions=self._proposal.prefill_positions,
                prefill_length=self._proposal.prefill_length,
                prefill_aux_hidden_states=(self._proposal.prefill_aux_hidden_states),
                prefill_aux_row_indices=(self._proposal.prefill_aux_row_indices),
            )

        captured_verify_row_indices = verify_row_indices.detach().clone()
        captured_target_aux_hidden_states = tuple(
            tensor.detach().clone() for tensor in target_aux_hidden_states
        )
        confirmed_input_embeds = prefill_input_embeds.detach().clone()
        confirmed_positions = prefill_positions.detach().clone()
        confirmed_length = prefill_length.detach().clone()

        ready = None
        if device.type == "cuda":
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(device))

        packet = Eagle3CapturePacket(
            request_id=request_id,
            step_id=self._step_id,
            source_weight_version=self._proposal.source_weight_version,
            prompt_context=prompt_context,
            proposal_token_input_embeds=(self._proposal.token_input_embeds),
            proposal_recurrent_hidden_states=(self._proposal.recurrent_hidden_states),
            teacher_logits=self._teacher_logits,
            verify_row_indices=captured_verify_row_indices,
            target_aux_hidden_states=captured_target_aux_hidden_states,
            num_sampled=sampled,
            confirmed_prefill_input_embeds=confirmed_input_embeds,
            confirmed_prefill_positions=confirmed_positions,
            confirmed_prefill_length=confirmed_length,
            ready=ready,
        )

        self._queue.submit(packet)

        self._teacher_logits = None
        self._step_id += 1

    def finish_request(self, request_id: str) -> int:
        if self._request_id is not None and request_id != self._request_id:
            raise RuntimeError("request does not match active capture")

        self._queue.finish_request(request_id)

        self._request_id = None
        self._proposal = None
        self._teacher_logits = None
        self._step_id = 0
        return 0

    def close_epoch(self, request_id: str) -> None:
        if request_id != self._request_id:
            raise RuntimeError("request does not match active capture")

        self._queue.close_epoch(request_id)

    def close(self) -> None:
        self._queue.close()

    def raise_if_failed(self) -> None:
        self._queue.raise_if_failed()

    def _save_proposal(
        self,
        *,
        proposal: Any,
        target_aux_hidden_states: Sequence[torch.Tensor],
        aux_row_indices: torch.Tensor,
    ) -> _GpuProposal:
        if proposal.prefill_length is None:
            raise ValueError("capture requires a device-side proposal prefill length")

        return _GpuProposal(
            prefill_input_embeds=(proposal.prefill_input_embeds.detach().clone()),
            prefill_positions=proposal.prefill_positions.detach().clone(),
            prefill_length=proposal.prefill_length.detach().clone(),
            prefill_aux_hidden_states=tuple(
                tensor.detach().clone() for tensor in target_aux_hidden_states
            ),
            prefill_aux_row_indices=aux_row_indices.detach().clone(),
            token_input_embeds=(proposal.draft_token_input_embeds.detach().clone()),
            recurrent_hidden_states=(
                proposal.draft_recurrent_hidden_states.detach().clone()
            ),
            source_weight_version=self._source_weight_version(),
        )

    @staticmethod
    def _initial_aux_rows(
        target_aux_hidden_states: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        return torch.arange(
            target_aux_hidden_states[0].shape[0],
            dtype=torch.long,
            device=target_aux_hidden_states[0].device,
        )

    @staticmethod
    def _as_device_scalar(
        value: torch.Tensor | int,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().clone()

        return torch.tensor(
            value,
            dtype=torch.long,
            device=device,
        )
