# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Mapping

import torch

from vllm.v1.worker.gpu.spec_decode.weight_slots import DraftWeightSlots


class DFlashWeightSlots:
    """Keep context-projection copies in the same versioned slot as parameters."""

    def __init__(self, target_model, draft_model, mutable_names: Iterable[str]):
        self.weights = DraftWeightSlots.from_models(
            target_model, draft_model, mutable_names
        )
        self.model = draft_model.model
        self.parameters = dict(draft_model.named_parameters())
        self.slots = self.weights.slots
        self.owned_names = self.weights.owned_names
        self.derived = tuple(
            {
                "_fused_kv_weight": self.model._fused_kv_weight.clone(),
                "_k_norm_weights": self.model._k_norm_weights.clone(),
            }
            for _ in range(2)
        )

    @torch.no_grad()
    def prepare(self, slot: int, weights: Mapping[str, torch.Tensor]) -> None:
        kv_weights, norms = [], []
        for index, layer in enumerate(self.model.layers):
            prefix = f"model.layers.{index}.self_attn"
            qkv_name = f"{prefix}.qkv_proj.weight"
            norm_name = f"{prefix}.k_norm.weight"
            qkv = weights.get(qkv_name, self.parameters[qkv_name])
            kv_weights.append(qkv[layer.self_attn.q_size :])
            norms.append(weights.get(norm_name, self.parameters[norm_name]))
        torch.cat(kv_weights, dim=0, out=self.derived[slot]["_fused_kv_weight"])
        torch.stack(norms, dim=0, out=self.derived[slot]["_k_norm_weights"])

    def bind(self, weights: Mapping[str, torch.Tensor]) -> None:
        self.weights.bind(weights)
        slot = 0 if weights is self.slots[0] else 1
        self.model._fused_kv_weight = self.derived[slot]["_fused_kv_weight"]
        self.model._k_norm_weights = self.derived[slot]["_k_norm_weights"]
        self.model._hidden_norm_weight = self.model.hidden_norm.weight.data
