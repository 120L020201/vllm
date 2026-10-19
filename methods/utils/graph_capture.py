# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CUDA-Graph-safe proposal capture for GPU online draft training."""

from typing import Any

import torch

_INSTALLED = False


def install_graph_safe_capture() -> None:
    """Extend the autoregressive speculator without changing vLLM sources."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as ar

    speculator_type = ar.AutoRegressiveSpeculator
    original_load_model = speculator_type.load_model
    original_get_proposal_snapshot = speculator_type.get_proposal_snapshot
    original_prepare_input_embeds = speculator_type._prepare_input_embeds
    original_prefill_begin = speculator_type.on_prefill_begin
    original_prefill_end = speculator_type.on_prefill_end
    original_decode_begin = speculator_type.on_multi_step_decode_begin
    original_decode_end = speculator_type.on_multi_step_decode_end

    def load_model(self, target_model):
        result = original_load_model(self, target_model)
        if not getattr(self, "_methods_capture_enabled", False):
            return result
        if self.supports_mm_inputs:
            raise ValueError("online GPU training does not support multimodal drafts")

        self.inputs_embeds = torch.empty(
            self.max_num_tokens,
            self.hidden_size,
            dtype=self.dtype,
            device=self.device,
        )
        self._methods_prefill_input_embeds = torch.empty_like(self.inputs_embeds)
        self._methods_prefill_positions = torch.empty(
            self.max_num_tokens,
            dtype=torch.long,
            device=self.device,
        )
        self._methods_prefill_length = torch.zeros(
            (),
            dtype=torch.long,
            device=self.device,
        )
        capture_shape = (
            (self.num_speculative_steps - 1) * self.max_num_reqs,
            self.hidden_size,
        )
        self._methods_draft_input_embeds = torch.empty(
            capture_shape,
            dtype=self.dtype,
            device=self.device,
        )
        self._methods_draft_hidden_states = torch.empty_like(
            self._methods_draft_input_embeds
        )
        self._methods_capture_request_indices = torch.arange(
            self.max_num_reqs,
            dtype=torch.long,
            device=self.device,
        )
        self._methods_capture_phase = None
        self._methods_capture_num_reqs = 0
        self._methods_capture_result_num_reqs = 0
        self._methods_capture_steps = 0
        return result

    def on_prefill_begin(self, num_reqs: int) -> None:
        if getattr(self, "_methods_capture_enabled", False):
            self._methods_capture_phase = "prefill"
            self._methods_capture_num_reqs = num_reqs
        original_prefill_begin(self, num_reqs)

    def on_prefill_end(self, num_reqs: int) -> None:
        original_prefill_end(self, num_reqs)
        if getattr(self, "_methods_capture_enabled", False):
            self._methods_capture_phase = None

    def on_decode_begin(self, num_reqs: int) -> None:
        if getattr(self, "_methods_capture_enabled", False):
            self._methods_capture_phase = "decode"
            self._methods_capture_num_reqs = num_reqs
        original_decode_begin(self, num_reqs)

    def on_decode_end(self, num_reqs: int) -> None:
        original_decode_end(self, num_reqs)
        if getattr(self, "_methods_capture_enabled", False):
            self._methods_capture_phase = None

    def prepare_input_embeds(
        self,
        num_tokens: int,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        if not getattr(self, "_methods_capture_enabled", False):
            return original_prepare_input_embeds(self, num_tokens, mm_inputs)
        if mm_inputs is not None:
            raise ValueError("online GPU training does not support multimodal inputs")

        assert self.inputs_embeds is not None
        input_ids = self.input_buffers.input_ids[:num_tokens]
        self.inputs_embeds[:num_tokens].copy_(self.model.embed_input_ids(input_ids))
        self.last_input_embeds = self.inputs_embeds[:num_tokens]

        if self._methods_capture_phase == "prefill":
            self._methods_prefill_input_embeds[:num_tokens].copy_(
                self.last_input_embeds
            )
            self._methods_prefill_positions[:num_tokens].copy_(
                self.input_buffers.positions[:num_tokens]
            )
            request_index = self._methods_capture_num_reqs - 1
            self._methods_prefill_length.copy_(
                self.last_token_indices[request_index] + 1
            )
        return self.last_input_embeds

    def generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        self._prepare_eplb_forward(num_reqs)

        idx_mapping = self.idx_mapping[:num_reqs]
        positions = self.input_buffers.positions[:num_reqs]
        last_hidden_states, hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )

        if getattr(self, "_methods_capture_enabled", False):
            assert self.last_input_embeds is not None
            row_indices = (
                torch.clamp_min(self.current_draft_step - 1, 0) * self.max_num_reqs
                + self._methods_capture_request_indices[:num_reqs]
            )
            self._methods_draft_input_embeds.index_copy_(
                0,
                row_indices,
                self.last_input_embeds[:num_reqs],
            )
            self._methods_draft_hidden_states.index_copy_(
                0,
                row_indices,
                hidden_states[:num_reqs],
            )

        last_hidden_states = last_hidden_states[:num_reqs]
        sample_positions = positions
        if not self.advance_draft_positions:
            sample_positions = positions + self.current_draft_step
        draft_tokens = self.sample_draft(
            last_hidden_states,
            sample_positions,
            idx_mapping,
            self.temperature,
            self.seeds,
            self.current_draft_step,
            self.draft_logits,
        )
        ar.update_draft_inputs(
            draft_tokens,
            self.current_draft_step,
            hidden_states,
            self.draft_tokens,
            self.hidden_states,
            self.input_buffers,
            num_reqs,
            self.max_model_len,
            self.num_speculative_steps,
            advance_draft_positions=self.advance_draft_positions,
        )

    def get_proposal_snapshot(self):
        if not getattr(self, "_methods_capture_enabled", False):
            return original_get_proposal_snapshot(self)
        num_reqs = self._methods_capture_result_num_reqs
        steps = self._methods_capture_steps
        capture_shape = (
            self.num_speculative_steps - 1,
            self.max_num_reqs,
            self.hidden_size,
        )
        draft_input_embeds = self._methods_draft_input_embeds.view(capture_shape)[
            :steps, :num_reqs
        ].flatten(0, 1)
        draft_hidden_states = self._methods_draft_hidden_states.view(capture_shape)[
            :steps, :num_reqs
        ].flatten(0, 1)
        return ar.DraftProposalSnapshot(
            prefill_input_embeds=self._methods_prefill_input_embeds,
            prefill_positions=self._methods_prefill_positions,
            prefill_length=self._methods_prefill_length,
            draft_token_input_embeds=draft_input_embeds,
            draft_recurrent_hidden_states=draft_hidden_states,
        )

    speculator_type.load_model = load_model
    speculator_type.on_prefill_begin = on_prefill_begin
    speculator_type.on_prefill_end = on_prefill_end
    speculator_type.on_multi_step_decode_begin = on_decode_begin
    speculator_type.on_multi_step_decode_end = on_decode_end
    speculator_type._prepare_input_embeds = prepare_input_embeds
    speculator_type._generate_draft = generate_draft
    speculator_type.get_proposal_snapshot = get_proposal_snapshot
