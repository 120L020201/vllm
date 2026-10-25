# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model
from vllm.v1.worker.gpu.spec_decode.online import OnlineDraftSpeculator


class EagleSpeculator(OnlineDraftSpeculator, AutoRegressiveSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self._init_online_draft()
        self.explicit_input_embeds = self.online_draft_training_enabled

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        self._keep_online_target(target_model)
        return load_eagle_model(target_model, self.vllm_config)

    def on_prefill_input_embeds_ready(
        self,
        num_reqs: int,
        input_batch: InputBatch,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        capture_enabled: bool,
    ) -> None:
        # Capture and install before FC so the trainable FC uses new weights.
        if not capture_enabled or self.capture_adapter is None:
            return

        assert aux_hidden_states is not None
        assert self.last_prefill_input_embeds is not None
        assert self.last_prefill_positions is not None
        assert self.last_prefill_length is not None

        # Draft prefill row i pairs target row i with the next token.
        num_tokens = input_batch.num_tokens
        self._capture_target_and_install(
            input_batch,
            input_embeds=self.last_prefill_input_embeds[:num_tokens],
            positions=self.last_prefill_positions[:num_tokens],
            aux_hidden_states=[
                hidden_states[:num_tokens] for hidden_states in aux_hidden_states
            ],
            confirmed_length=self.last_prefill_length,
            num_sampled=num_sampled[0],
        )

    def on_proposal_ready(
        self,
        input_batch: InputBatch,
        aux_hidden_states: list[torch.Tensor] | None,
        capture_enabled: bool,
    ) -> None:
        if not capture_enabled or self.capture_adapter is None:
            return

        proposal = self.get_proposal_snapshot()
        self.capture_adapter.capture_proposal(
            token_input_embeds=proposal.draft_token_input_embeds,
            input_hidden_states=proposal.draft_input_hidden_states,
        )
