# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Trainable CPU copy of the Qwen3.5 MTP draft layer.

The MTP layer fuses the next token's embedding with the target's final hidden
state through ``fc``, runs one gated full-attention decoder layer, and feeds
its normalized output both to the shared target head and to the next draft
step. It satisfies the EAGLE3-style training interface with one target
feature row and the full target vocabulary.
"""

from dataclasses import dataclass

import torch
from torch import nn

from online_draft.models.layers import (
    VLLM_ATTENTION_STATE_SUFFIXES,
    AttentionBackend,
    DraftForwardOutput,
    GatedMLP,
    LayerKV,
    RMSNorm,
    apply_rotary_embedding,
    cached_attention,
    validate_vllm_rotary,
)


@dataclass(frozen=True, slots=True)
class Qwen3_5MTPConfig:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rotary_dim: int
    rope_theta: float
    rms_norm_eps: float
    vocab_size: int
    attention_backend: AttentionBackend = AttentionBackend.EAGER

    @property
    def num_aux_hidden_states(self) -> int:
        return 1

    @property
    def draft_vocab_size(self) -> int:
        return self.vocab_size


class Qwen3_5MTPAttention(nn.Module):
    """Gated attention; ``qkv_proj`` interleaves ``[q | gate]`` per head."""

    def __init__(self, config: Qwen3_5MTPConfig) -> None:
        super().__init__()
        self.config = config
        self.q_size = config.num_attention_heads * config.head_dim
        self.kv_size = config.num_key_value_heads * config.head_dim
        self.qkv_proj = nn.Linear(
            config.hidden_size, 2 * self.q_size + 2 * self.kv_size, bias=False
        )
        self.o_proj = nn.Linear(self.q_size, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps, unit_offset=True)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps, unit_offset=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        past_key_value: LayerKV | None,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, LayerKV]:
        config = self.config
        tokens = hidden_states.shape[0]
        query_gate, key, value = self.qkv_proj(hidden_states).split(
            (2 * self.q_size, self.kv_size, self.kv_size), dim=-1
        )
        query, gate = query_gate.view(
            tokens, config.num_attention_heads, 2 * config.head_dim
        ).chunk(2, dim=-1)
        query = self.q_norm(query).transpose(0, 1)
        key = self.k_norm(
            key.view(tokens, config.num_key_value_heads, config.head_dim)
        ).transpose(0, 1)
        value = value.view(
            tokens, config.num_key_value_heads, config.head_dim
        ).transpose(0, 1)

        query, key = apply_rotary_embedding(
            query, key, positions, config.rope_theta, config.rotary_dim
        )
        output, present = cached_attention(
            query, key, value, past_key_value, attention_mask, config.attention_backend
        )
        output = output.transpose(0, 1).reshape(tokens, self.q_size)
        output = output * torch.sigmoid(gate.reshape(tokens, self.q_size))
        return self.o_proj(output), present


class Qwen3_5MTPLayer(nn.Module):
    def __init__(self, config: Qwen3_5MTPConfig) -> None:
        super().__init__()
        eps = config.rms_norm_eps
        self.input_layernorm = RMSNorm(config.hidden_size, eps, unit_offset=True)
        self.self_attn = Qwen3_5MTPAttention(config)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps, unit_offset=True
        )
        self.mlp = GatedMLP(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        past_key_value: LayerKV | None,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, LayerKV]:
        residual = hidden_states
        hidden_states, present = self.self_attn(
            self.input_layernorm(hidden_states),
            positions,
            past_key_value,
            attention_mask,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual, present


class Qwen3_5MTPModel(nn.Module):
    def __init__(self, config: Qwen3_5MTPConfig) -> None:
        super().__init__()
        eps = config.rms_norm_eps
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.pre_fc_norm_embedding = RMSNorm(config.hidden_size, eps, unit_offset=True)
        self.pre_fc_norm_hidden = RMSNorm(config.hidden_size, eps, unit_offset=True)
        self.layers = nn.ModuleList([Qwen3_5MTPLayer(config)])
        self.norm = RMSNorm(config.hidden_size, eps, unit_offset=True)


class Qwen3_5MTPForCausalLM(nn.Module):
    def __init__(self, config: Qwen3_5MTPConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3_5MTPModel(config)
        # Shared with the target, so never trained or installed.
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.requires_grad_(False)

    def combine_hidden_states(
        self, auxiliary_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """MTP consumes the target's final hidden state as captured."""
        return auxiliary_hidden_states

    def forward(
        self,
        positions: torch.Tensor,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_values: tuple[LayerKV, ...] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> DraftForwardOutput:
        model = self.model
        hidden_states = model.fc(
            torch.cat(
                (
                    model.pre_fc_norm_embedding(input_embeds),
                    model.pre_fc_norm_hidden(hidden_states),
                ),
                dim=-1,
            )
        )
        hidden_states, residual, present = model.layers[0](
            hidden_states,
            positions,
            None if past_key_values is None else past_key_values[0],
            attention_mask,
        )
        output, _ = model.norm(hidden_states, residual)
        return DraftForwardOutput(
            hidden_states=output,
            recurrent_hidden_states=output,
            past_key_values=(present,),
        )

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


def load_qwen3_5_mtp_from_vllm(
    gpu_model: nn.Module,
    *,
    attention_backend: AttentionBackend | str | None = None,
) -> Qwen3_5MTPForCausalLM:
    """Copy the loaded vLLM ``Qwen3_5MTP`` draft into a trainable CPU model.

    Sizes, norms, and RoPE come from the GPU modules; settings this CPU model
    does not implement are rejected.

    Raises:
        ValueError: If the draft uses an unsupported setting, or its weights do
            not match the CPU parameter layout.
    """
    config = gpu_model.config
    layers = gpu_model.model.layers
    layer = layers[0]
    attention = layer.self_attn
    rotary = attention.rotary_emb
    unsupported = {
        "multiple MTP layers": len(layers) != 1,
        "layer scale": layer.layer_scale,
        "mixture-of-experts MLP": not hasattr(layer.mlp, "gate_up_proj"),
        "ungated attention": not attention.attn_output_gate,
        "qkv bias": attention.qkv_proj.bias is not None,
        "non-NeoX rotary": not rotary.is_neox_style,
        "attention scaling": attention.scaling != attention.head_dim**-0.5,
        "logit scale": gpu_model.logits_processor.scale != 1.0,
    }
    if any(unsupported.values()):
        names = ", ".join(name for name, found in unsupported.items() if found)
        raise ValueError(f"online MTP CPU model does not support: {names}")

    cpu_config = Qwen3_5MTPConfig(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_attention_heads=attention.total_num_heads,
        num_key_value_heads=attention.total_num_kv_heads,
        head_dim=attention.head_dim,
        rotary_dim=rotary.rotary_dim,
        rope_theta=float(rotary.base),
        rms_norm_eps=layer.input_layernorm.variance_epsilon,
        vocab_size=config.vocab_size,
        attention_backend=AttentionBackend(attention_backend or "eager"),
    )
    validate_vllm_rotary(rotary, cpu_config.rotary_dim, cpu_config.rope_theta, "MTP")

    with torch.inference_mode(False), torch.device("meta"):
        model = Qwen3_5MTPForCausalLM(cpu_config)

    source = gpu_model.state_dict()
    # The embedding is shared with the target and never read on the CPU.
    expected = {
        name
        for name in source
        if name != "model.embed_tokens.weight"
        and not name.endswith(VLLM_ATTENTION_STATE_SUFFIXES)
    }
    if expected != set(model.state_dict()):
        missing = sorted(set(model.state_dict()) - expected)
        extra = sorted(expected - set(model.state_dict()))
        raise ValueError(
            f"MTP draft weights do not match the CPU layout: "
            f"missing={missing}, unsupported={extra}"
        )

    with torch.inference_mode(False):
        weights = {
            name: source[name].detach().to(device="cpu", copy=True)
            for name in model.state_dict()
        }
        model.load_state_dict(weights, assign=True)
    return model
