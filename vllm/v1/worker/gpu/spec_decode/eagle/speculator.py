# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.nn as nn
from online_draft.runtime.draft_weight_installer import DraftWeightInstaller

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model
from vllm.v1.worker.gpu.spec_decode.eagle.weight_slots import (
    Eagle3WeightSlots,
)


class EagleSpeculator(AutoRegressiveSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self.online_draft_training_enabled = envs.VLLM_ONLINE_DRAFT_TRAIN
        self.explicit_input_embeds = self.online_draft_training_enabled
        self.draft_weight_slots: Eagle3WeightSlots | None = None
        self.capture_adapter: Any | None = None

    def set_capture_adapter(self, adapter: Any) -> None:
        self.capture_adapter = adapter

    def on_target_logits_ready(self, logits: torch.Tensor) -> None:
        if self.capture_adapter is not None:
            self.capture_adapter.capture_teacher_logits(logits)

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        eagle_model = load_eagle_model(target_model, self.vllm_config)

        if self.online_draft_training_enabled:
            slots = Eagle3WeightSlots.from_models(
                target_model=target_model,
                draft_model=eagle_model,
            )
            self.draft_weight_slots = slots
            self.draft_weight_installer = DraftWeightInstaller(
                slots=slots.slots,
                mutable_names=slots.owned_names,
            )

        return eagle_model

    def on_prefill_input_embeds_ready(
        self,
        num_reqs: int,
        input_batch: InputBatch,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        capture_enabled: bool,
    ) -> None:
        if capture_enabled and self.capture_adapter is not None:
            assert num_reqs == 1
            assert aux_hidden_states is not None
            assert self.last_prefill_input_embeds is not None
            assert self.last_prefill_positions is not None
            assert self.last_prefill_length is not None

            self.capture_adapter.capture_prefill(
                request_id=input_batch.req_ids[0],
                verify_row_indices=input_batch.logits_indices,
                target_aux_hidden_states=aux_hidden_states,
                num_sampled=num_sampled[0],
                prefill_input_embeds=self.last_prefill_input_embeds,
                prefill_positions=self.last_prefill_positions,
                prefill_length=num_sampled[0],
            )

        if self.draft_weight_installer is None:
            return

        installed_version = self.draft_weight_installer.commit_if_ready()
        if installed_version is not None:
            assert self.draft_weight_slots is not None
            self.draft_weight_slots.bind(self.draft_weight_installer.active_weights)

            if capture_enabled and self.capture_adapter is not None:
                self.capture_adapter.close_epoch(input_batch.req_ids[0])

    def on_proposal_ready(
        self,
        input_batch: InputBatch,
        aux_hidden_states: list[torch.Tensor] | None,
        capture_enabled: bool,
    ) -> None:
        if not capture_enabled or self.capture_adapter is None:
            return

        assert input_batch.num_reqs == 1
        assert aux_hidden_states is not None

        self.capture_adapter.capture_proposal(
            request_id=input_batch.req_ids[0],
            verify_row_indices=input_batch.logits_indices,
            target_aux_hidden_states=aux_hidden_states,
            proposal=self.get_proposal_snapshot(),
        )
