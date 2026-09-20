# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from pathlib import Path

import torch
from online_draft.models.qwen3_eagle3 import (
    Qwen3Eagle3Config,
    Qwen3Eagle3ForCausalLM,
    convert_angelslim_eagle3_state_dict,
    load_qwen3_eagle3_checkpoint,
)


def _make_config() -> Qwen3Eagle3Config:
    return Qwen3Eagle3Config(
        hidden_size=4,
        intermediate_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        num_hidden_layers=1,
        target_vocab_size=10,
        draft_vocab_size=6,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )


def _make_source_state_dict() -> dict[str, torch.Tensor]:
    return {
        "d2t": torch.tensor([0, 1, 2, 3, 4, 4]),
        "t2d": torch.zeros(10, dtype=torch.bool),
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


def test_config_from_dict() -> None:
    config = Qwen3Eagle3Config.from_dict(
        {
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 1,
            "vocab_size": 151936,
            "draft_vocab_size": 32000,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
        }
    )

    assert config.hidden_size == 4096
    assert config.head_dim == 128
    assert config.num_aux_hidden_states == 3
    assert config.target_vocab_size == 151936
    assert config.draft_vocab_size == 32000


def test_checkpoint_conversion() -> None:
    source = _make_source_state_dict()
    converted = convert_angelslim_eagle3_state_dict(source)
    model = Qwen3Eagle3ForCausalLM(_make_config())

    assert set(converted) == set(model.state_dict())

    expected_qkv = torch.cat(
        (
            source["midlayer.self_attn.q_proj.weight"],
            source["midlayer.self_attn.k_proj.weight"],
            source["midlayer.self_attn.v_proj.weight"],
        ),
        dim=0,
    )
    expected_gate_up = torch.cat(
        (
            source["midlayer.mlp.gate_proj.weight"],
            source["midlayer.mlp.up_proj.weight"],
        ),
        dim=0,
    )

    torch.testing.assert_close(
        converted["model.layers.0.self_attn.qkv_proj.weight"],
        expected_qkv,
    )
    torch.testing.assert_close(
        converted["model.layers.0.mlp.gate_up_proj.weight"],
        expected_gate_up,
    )

    model.load_state_dict(converted, strict=True)


def test_cached_forward_matches_full_forward() -> None:
    torch.manual_seed(0)

    model = Qwen3Eagle3ForCausalLM(_make_config())
    model.eval()

    positions = torch.arange(5)
    input_embeds = torch.randn(5, 4)
    hidden_states = torch.randn(5, 4)

    with torch.no_grad():
        full_output = model(
            positions=positions,
            input_embeds=input_embeds,
            hidden_states=hidden_states,
        )

        prefix_output = model(
            positions=positions[:3],
            input_embeds=input_embeds[:3],
            hidden_states=hidden_states[:3],
        )

        suffix_output = model(
            positions=positions[3:],
            input_embeds=input_embeds[3:],
            hidden_states=hidden_states[3:],
            past_key_values=prefix_output.past_key_values,
        )

    torch.testing.assert_close(
        suffix_output.hidden_states,
        full_output.hidden_states[3:],
    )
    torch.testing.assert_close(
        suffix_output.recurrent_hidden_states,
        full_output.recurrent_hidden_states[3:],
    )

    prefix_key, prefix_value = prefix_output.past_key_values[0]
    suffix_key, suffix_value = suffix_output.past_key_values[0]

    assert prefix_key.shape == (1, 3, 2)
    assert prefix_value.shape == (1, 3, 2)
    assert suffix_key.shape == (1, 5, 2)
    assert suffix_value.shape == (1, 5, 2)


def test_gradients_reach_all_trainable_blocks() -> None:
    torch.manual_seed(1)

    model = Qwen3Eagle3ForCausalLM(_make_config())
    model.train()

    positions = torch.arange(4)
    input_embeds = torch.randn(4, 4)
    auxiliary_hidden_states = torch.randn(
        4,
        12,
        requires_grad=True,
    )

    combined_hidden_states = model.combine_hidden_states(auxiliary_hidden_states)
    output = model(
        positions=positions,
        input_embeds=input_embeds,
        hidden_states=combined_hidden_states,
    )
    logits = model.compute_draft_logits(output.hidden_states)
    loss = logits.float().square().mean()

    loss.backward()

    parameters = dict(model.named_parameters())
    expected_gradient_names = (
        "model.fc.weight",
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "lm_head.weight",
    )

    for name in expected_gradient_names:
        gradient = parameters[name].grad

        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum().item() > 0

    assert auxiliary_hidden_states.grad is not None
    assert torch.isfinite(auxiliary_hidden_states.grad).all()
    assert model.draft_id_to_target_id.grad is None


def test_bfloat16_training_step() -> None:
    torch.manual_seed(2)

    model = Qwen3Eagle3ForCausalLM(_make_config())
    model.to(dtype=torch.bfloat16)
    model.train()

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=1e-3,
    )

    positions = torch.arange(4)
    input_embeds = torch.randn(
        4,
        4,
        dtype=torch.bfloat16,
    )
    auxiliary_hidden_states = torch.randn(
        4,
        12,
        dtype=torch.bfloat16,
    )

    combined_hidden_states = model.combine_hidden_states(auxiliary_hidden_states)
    output = model(
        positions=positions,
        input_embeds=input_embeds,
        hidden_states=combined_hidden_states,
    )
    logits = model.compute_draft_logits(output.hidden_states)

    loss = logits.float().square().mean()
    loss.backward()

    assert all(parameter.dtype == torch.bfloat16 for parameter in trainable_parameters)
    assert all(
        parameter.grad is not None and parameter.grad.dtype == torch.bfloat16
        for parameter in trainable_parameters
    )

    optimizer.step()

    for parameter in trainable_parameters:
        state = optimizer.state[parameter]

        assert state["exp_avg"].dtype == torch.bfloat16
        assert state["exp_avg_sq"].dtype == torch.bfloat16
        assert state["step"].dtype == torch.float32


def test_load_checkpoint_from_directory(tmp_path: Path) -> None:
    torch.manual_seed(3)

    raw_config = {
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
    }
    source_state_dict = _make_source_state_dict()

    (tmp_path / "config.json").write_text(
        json.dumps(raw_config),
        encoding="utf-8",
    )
    torch.save(
        source_state_dict,
        tmp_path / "pytorch_model.bin",
    )

    model = load_qwen3_eagle3_checkpoint(tmp_path)
    expected_state_dict = convert_angelslim_eagle3_state_dict(source_state_dict)
    loaded_state_dict = model.state_dict()

    assert model.config == _make_config()
    assert set(loaded_state_dict) == set(expected_state_dict)

    for name, expected_tensor in expected_state_dict.items():
        torch.testing.assert_close(
            loaded_state_dict[name],
            expected_tensor,
        )
