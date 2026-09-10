# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json

import torch

from online_eagle3.checkpoint import (
    convert_eagle3_checkpoint_state,
    load_torch_eagle3_model,
)
from online_eagle3.torch_eagle3 import (
    TorchEagle3Config,
    TorchEagle3ForCausalLM,
)


def _make_config_dict() -> dict[str, int | float | bool | str | list[str]]:
    return {
        "architectures": ["LlamaForCausalLMEagle3"],
        "model_type": "llama",
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "num_hidden_layers": 1,
        "vocab_size": 10,
        "draft_vocab_size": 6,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "attention_bias": False,
    }


def _make_checkpoint_state() -> dict[str, torch.Tensor]:
    return {
        "d2t": torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.long),
        "t2d": torch.ones(10, dtype=torch.bool),
        "midlayer.self_attn.q_proj.weight": torch.randn(4, 8),
        "midlayer.self_attn.k_proj.weight": torch.randn(2, 8),
        "midlayer.self_attn.v_proj.weight": torch.randn(2, 8),
        "midlayer.self_attn.o_proj.weight": torch.randn(4, 4),
        "midlayer.mlp.gate_proj.weight": torch.randn(8, 4),
        "midlayer.mlp.up_proj.weight": torch.randn(8, 4),
        "midlayer.mlp.down_proj.weight": torch.randn(4, 8),
        "midlayer.hidden_norm.weight": torch.randn(4),
        "midlayer.input_layernorm.weight": torch.randn(4),
        "midlayer.post_attention_layernorm.weight": torch.randn(4),
        "norm.weight": torch.randn(4),
        "fc.weight": torch.randn(4, 12),
        "lm_head.weight": torch.randn(6, 4),
    }


def test_convert_eagle3_checkpoint_state_matches_torch_model_keys() -> None:
    config = TorchEagle3Config.from_dict(_make_config_dict())
    model = TorchEagle3ForCausalLM(config)

    converted = convert_eagle3_checkpoint_state(_make_checkpoint_state())

    assert set(converted) == set(model.state_dict())
    assert converted["model.layers.0.self_attn.qkv_proj.weight"].shape == (8, 8)
    assert converted["model.layers.0.mlp.gate_up_proj.weight"].shape == (16, 4)


def test_load_torch_eagle3_model_and_forward(tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_make_config_dict()))
    torch.save(_make_checkpoint_state(), tmp_path / "pytorch_model.bin")

    model = load_torch_eagle3_model(tmp_path)

    input_ids = torch.tensor([1, 2], dtype=torch.long)
    positions = torch.tensor([4, 5], dtype=torch.long)
    hidden_states = torch.randn(2, 4)
    input_embeds = torch.randn(2, 4)

    output, aux_output = model(
        input_ids=input_ids,
        positions=positions,
        hidden_states=hidden_states,
        inputs_embeds=input_embeds,
    )
    logits = model.compute_draft_logits(output)
    full_logits = model.compute_logits(output)

    assert output.shape == (2, 4)
    assert aux_output.shape == (2, 4)
    assert logits.shape == (2, 6)
    assert full_logits.shape == (2, 10)
