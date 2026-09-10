# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import torch

from .torch_eagle3 import TorchEagle3Config, TorchEagle3ForCausalLM


def load_torch_eagle3_model(
    model_path: str | Path,
    *,
    dtype: torch.dtype = torch.float32,
) -> TorchEagle3ForCausalLM:
    path = Path(model_path)
    with (path / "config.json").open() as config_file:
        config = TorchEagle3Config.from_dict(json.load(config_file))

    model = TorchEagle3ForCausalLM(config).to(dtype=dtype)
    state = torch.load(
        path / "pytorch_model.bin",
        map_location="cpu",
        weights_only=True,
    )
    if "state_dict" in state:
        state = state["state_dict"]
    converted = convert_eagle3_checkpoint_state(state)
    model.load_state_dict(converted, strict=True)
    return model


def convert_eagle3_checkpoint_state(
    checkpoint_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    state = dict(checkpoint_state)
    converted: dict[str, torch.Tensor] = {
        "draft_id_to_target_id": state["d2t"],
        "model.layers.0.self_attn.o_proj.weight": state[
            "midlayer.self_attn.o_proj.weight"
        ],
        "model.layers.0.mlp.down_proj.weight": state["midlayer.mlp.down_proj.weight"],
        "model.layers.0.hidden_norm.weight": state["midlayer.hidden_norm.weight"],
        "model.layers.0.input_layernorm.weight": state[
            "midlayer.input_layernorm.weight"
        ],
        "model.layers.0.post_attention_layernorm.weight": state[
            "midlayer.post_attention_layernorm.weight"
        ],
        "model.norm.weight": state["norm.weight"],
        "model.fc.weight": state["fc.weight"],
        "lm_head.weight": state["lm_head.weight"],
    }
    converted["model.layers.0.self_attn.qkv_proj.weight"] = torch.cat(
        [
            state["midlayer.self_attn.q_proj.weight"],
            state["midlayer.self_attn.k_proj.weight"],
            state["midlayer.self_attn.v_proj.weight"],
        ],
        dim=0,
    )
    converted["model.layers.0.mlp.gate_up_proj.weight"] = torch.cat(
        [
            state["midlayer.mlp.gate_proj.weight"],
            state["midlayer.mlp.up_proj.weight"],
        ],
        dim=0,
    )
    return converted
