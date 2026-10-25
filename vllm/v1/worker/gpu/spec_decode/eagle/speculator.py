# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.spec_decode.autoregressive.online import (
    AutoRegressiveOnlineDraft,
)
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model


class EagleSpeculator(AutoRegressiveOnlineDraft, AutoRegressiveSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self._init_autoregressive_online_draft()

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        self._keep_online_target(target_model)
        return load_eagle_model(target_model, self.vllm_config)

    def _online_target_features(
        self,
        aux_hidden_states: list[torch.Tensor] | None,
        target_hidden_states: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        # Captured before FC so the trainable FC uses newly installed weights.
        assert aux_hidden_states is not None
        return aux_hidden_states
