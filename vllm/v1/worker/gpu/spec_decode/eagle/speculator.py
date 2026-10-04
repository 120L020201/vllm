# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig
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
        self.draft_weight_slots: Eagle3WeightSlots | None = None

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        eagle_model = load_eagle_model(target_model, self.vllm_config)

        if self.online_draft_training_enabled:
            self.draft_weight_slots = Eagle3WeightSlots.from_models(
                target_model=target_model,
                draft_model=eagle_model,
            )

        return eagle_model
