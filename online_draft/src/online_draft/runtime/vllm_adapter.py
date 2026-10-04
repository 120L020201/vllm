# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from online_draft.runtime.capture_runtime import Eagle3CaptureRuntime
from online_draft.runtime.transport import (
    Eagle3CapturePacket,
    Eagle3PromptContext,
)


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


class Eagle3VllmCaptureAdapter:
    """Convert vLLM EAGLE3 callbacks into GPU capture packets."""

    def __init__(
        self,
        *,
        runtime: Eagle3CaptureRuntime,
        target_token_ids: torch.Tensor,
        source_weight_version: Callable[[], int],
    ) -> None:
        self._runtime = runtime
        self._target_token_ids = target_token_ids.detach()
        self._source_weight_version = source_weight_version

        self._request_id: str | None = None
        self._step_id = 0
        self._proposal: _GpuProposal | None = None
        self._teacher_logits: torch.Tensor | None = None

    @property
    def has_proposal(self) -> bool:
        return self._proposal is not None

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
        num_sampled: torch.Tensor | int,
        proposal: Any,
    ) -> None:
        """Capture the first proposal or complete one verified round."""

        if self._proposal is None:
            self._request_id = request_id
            self._proposal = self._save_proposal(
                proposal=proposal,
                target_aux_hidden_states=target_aux_hidden_states,
                aux_row_indices=self._initial_aux_rows(target_aux_hidden_states),
            )
            return

        if request_id != self._request_id:
            raise RuntimeError("another request is already being captured")

        if self._teacher_logits is None:
            raise RuntimeError("teacher logits are missing")

        next_proposal = self._save_proposal(
            proposal=proposal,
            target_aux_hidden_states=target_aux_hidden_states,
            aux_row_indices=verify_row_indices,
        )

        device = next_proposal.prefill_input_embeds.device
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
            confirmed_prefill_input_embeds=(next_proposal.prefill_input_embeds),
            confirmed_prefill_positions=(next_proposal.prefill_positions),
            confirmed_prefill_length=next_proposal.prefill_length,
            ready=ready,
        )

        self._runtime.submit(packet)

        self._proposal = next_proposal
        self._teacher_logits = None
        self._step_id += 1

    def finish_request(self, request_id: str) -> int:
        if request_id != self._request_id:
            raise RuntimeError("request does not match active capture")

        self._runtime.finish_request(request_id)

        self._request_id = None
        self._proposal = None
        self._teacher_logits = None
        self._step_id = 0
        return 0

    def close_epoch(self, request_id: str) -> None:
        if request_id != self._request_id:
            raise RuntimeError("request does not match active capture")

        self._runtime.close_epoch(request_id)

    def close(self) -> None:
        self._runtime.close()

    def raise_if_failed(self) -> None:
        self._runtime.raise_if_failed()

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
