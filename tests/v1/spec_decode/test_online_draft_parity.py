# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CPU training draft must compute what the vLLM draft layers compute.

The online trainer updates a separate PyTorch implementation of the draft and
installs its weights into the GPU model, so any numerical drift between the
two trains a different model than the one serving. These tests compare the
CPU building blocks against vLLM's native layer implementations, and check
that loading from a vLLM draft rejects settings the CPU model cannot match.
"""

from types import SimpleNamespace

import pytest
import torch
from online_draft.models.qwen3_eagle3 import (
    Qwen3Eagle3Config,
    Qwen3Eagle3ForCausalLM,
    Qwen3Eagle3MLP,
    Qwen3Eagle3RMSNorm,
    _apply_rotary_embedding,
    load_qwen3_eagle3_from_vllm,
)

from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import get_rope

DTYPES = [torch.float32, torch.bfloat16]
ROPE_THETA = 1_000_000.0


def _rope(head_dim: int, dtype: torch.dtype = torch.float32, **parameters):
    return get_rope(
        head_dim,
        max_position=4096,
        is_neox_style=True,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": ROPE_THETA,
            **parameters,
        },
        dtype=dtype,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("with_residual", [False, True])
def test_rms_norm_matches_vllm(default_vllm_config, dtype, with_residual) -> None:
    torch.manual_seed(0)
    hidden = torch.randn(7, 64, dtype=dtype)
    residual = torch.randn_like(hidden) if with_residual else None
    weight = torch.randn(64, dtype=dtype)

    vllm_norm = RMSNorm(64, eps=1e-6, dtype=dtype)
    vllm_norm.weight.data.copy_(weight)
    cpu_norm = Qwen3Eagle3RMSNorm(64, 1e-6)
    cpu_norm.weight.data = weight.clone()

    # vLLM's fused residual path may update its inputs in place.
    expected = vllm_norm.forward_native(
        hidden.clone(), None if residual is None else residual.clone()
    )
    actual = cpu_norm(hidden, residual)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", DTYPES)
def test_rotary_matches_vllm(default_vllm_config, dtype) -> None:
    torch.manual_seed(0)
    heads, kv_heads, head_dim = 4, 2, 64
    positions = torch.tensor([0, 1, 7, 100, 2047, 4095])
    query = torch.randn(positions.numel(), heads * head_dim, dtype=dtype)
    key = torch.randn(positions.numel(), kv_heads * head_dim, dtype=dtype)

    expected_query, expected_key = _rope(head_dim, dtype).forward_native(
        positions, query.clone(), key.clone()
    )

    def by_head(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(positions.numel(), -1, head_dim).transpose(0, 1)

    actual_query, actual_key = _apply_rotary_embedding(
        by_head(query), by_head(key), positions, ROPE_THETA
    )

    torch.testing.assert_close(by_head(expected_query), actual_query)
    torch.testing.assert_close(by_head(expected_key), actual_key)


def test_mlp_activation_matches_vllm(default_vllm_config) -> None:
    torch.manual_seed(0)
    config = Qwen3Eagle3Config(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_hidden_layers=1,
        target_vocab_size=10,
        draft_vocab_size=6,
        rms_norm_eps=1e-6,
        rope_theta=ROPE_THETA,
    )
    mlp = Qwen3Eagle3MLP(config)
    hidden = torch.randn(5, 16)

    expected = mlp.down_proj(SiluAndMul.forward_native(mlp.gate_up_proj(hidden)))

    torch.testing.assert_close(mlp(hidden), expected)


def _config(head_dim: int = 8) -> Qwen3Eagle3Config:
    return Qwen3Eagle3Config(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=head_dim,
        num_hidden_layers=1,
        target_vocab_size=10,
        draft_vocab_size=6,
        rms_norm_eps=1e-5,
        rope_theta=ROPE_THETA,
    )


def _vllm_like_draft(source: Qwen3Eagle3ForCausalLM, rotary, **overrides):
    """Mirror the Eagle3LlamaForCausalLM attributes the loader reads."""
    config = source.config
    attention = SimpleNamespace(
        total_num_heads=config.num_attention_heads,
        total_num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        scaling=config.head_dim**-0.5,
        qkv_proj=SimpleNamespace(bias=None),
        rotary_emb=rotary,
    )
    layer = SimpleNamespace(
        self_attn=attention,
        input_layernorm=SimpleNamespace(variance_epsilon=config.rms_norm_eps),
    )
    state = dict(source.state_dict())
    state["model.embed_tokens.weight"] = torch.zeros(config.target_vocab_size, 16)
    for scale in ("_q_scale", "_k_scale", "_v_scale", "_prob_scale"):
        state[f"model.layers.0.self_attn.attn.{scale}"] = torch.tensor(1.0)
    state.update(overrides.pop("extra_weights", {}))
    return SimpleNamespace(
        config=SimpleNamespace(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            vocab_size=config.target_vocab_size,
            draft_vocab_size=config.draft_vocab_size,
        ),
        model=SimpleNamespace(
            layers=[layer],
            use_aux_hidden_state=True,
            norm_before_fc=False,
            norm_output=False,
            num_aux_hidden_states=config.num_aux_hidden_states,
            fc_norm=None,
        ),
        logits_processor=SimpleNamespace(scale=1.0),
        use_parallel_drafting=False,
        draft_id_to_target_id=source.draft_id_to_target_id,
        state_dict=lambda: state,
        **overrides,
    )


def test_load_copies_draft_weights_and_settings(default_vllm_config) -> None:
    torch.manual_seed(0)
    source = Qwen3Eagle3ForCausalLM(_config())

    model = load_qwen3_eagle3_from_vllm(_vllm_like_draft(source, _rope(8)))

    assert model.config == source.config
    for name, tensor in source.state_dict().items():
        assert torch.equal(model.state_dict()[name], tensor), name


@pytest.mark.parametrize(
    ("rope_parameters", "message"),
    [
        ({"partial_rotary_factor": 0.5}, "partial rotary"),
        (
            {
                "rope_type": "llama3",
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 64,
            },
            "RoPE does not match",
        ),
    ],
)
def test_load_rejects_rope_the_cpu_model_cannot_match(
    default_vllm_config, rope_parameters, message
) -> None:
    source = Qwen3Eagle3ForCausalLM(_config())

    with pytest.raises(ValueError, match=message):
        load_qwen3_eagle3_from_vllm(
            _vllm_like_draft(source, _rope(8, **rope_parameters))
        )


def test_load_rejects_weights_outside_the_cpu_layout(default_vllm_config) -> None:
    source = Qwen3Eagle3ForCausalLM(_config())
    draft = _vllm_like_draft(
        source,
        _rope(8),
        extra_weights={"model.layers.0.self_attn.o_proj.bias": torch.zeros(16)},
    )

    with pytest.raises(ValueError, match="o_proj.bias"):
        load_qwen3_eagle3_from_vllm(draft)
