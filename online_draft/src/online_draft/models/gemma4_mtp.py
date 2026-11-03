# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Trainable CPU copy of the Gemma4 MTP assistant.

Every assistant layer is Q-only: it attends the K/V that the frozen target
wrote for its last sliding and last full attention layer. The CPU model
takes that K/V as fixed history, so only the query side is differentiable.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from online_draft.models.layers import GatedMLP, RMSNorm, RotaryEmbedding

# Shared target KV per layer: [tokens, kv_heads, 2 * head_dim], K then V.
TargetKV = tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class Gemma4MTPLayerConfig:
    head_dim: int
    num_key_value_heads: int
    sliding_window: int | None
    # Index of the shared target KV this layer reads.
    kv_index: int
    # Only used without a copied vLLM table; Gemma4 full layers need the table.
    rope_theta: float = 10000.0


@dataclass(frozen=True)
class Gemma4MTPConfig:
    backbone_hidden_size: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    vocab_size: int
    rms_norm_eps: float
    layers: tuple[Gemma4MTPLayerConfig, ...]

    @property
    def kv_shapes(self) -> tuple[tuple[int, int], ...]:
        """Per shared target layer: (kv_heads, 2 * head_dim)."""
        shapes = {
            layer.kv_index: (layer.num_key_value_heads, 2 * layer.head_dim)
            for layer in self.layers
        }
        return tuple(shapes[index] for index in sorted(shapes))


class Gemma4MLP(GatedMLP):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(hidden_states).split(
            self.intermediate_size, dim=-1
        )
        return self.down_proj(F.gelu(gate, approximate="tanh") * up)


class Gemma4MTPAttention(nn.Module):
    def __init__(self, config: Gemma4MTPConfig, layer: Gemma4MTPLayerConfig):
        super().__init__()
        self.layer = layer
        self.num_heads = config.num_attention_heads
        q_size = self.num_heads * layer.head_dim
        self.q_proj = nn.Linear(config.hidden_size, q_size, bias=False)
        self.o_proj = nn.Linear(q_size, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(layer.head_dim, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(layer.head_dim, layer.rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv: torch.Tensor,
        visible: torch.Tensor,
    ) -> torch.Tensor:
        tokens = hidden_states.shape[0]
        head_dim = self.layer.head_dim
        query = self.q_norm(
            self.q_proj(hidden_states).view(tokens, self.num_heads, head_dim)
        ).transpose(0, 1)
        query, _ = self.rotary_emb(query, query, positions)
        key, value = kv.transpose(0, 1).split(head_dim, dim=-1)
        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=visible, scale=1.0, enable_gqa=True
        )
        return self.o_proj(output.transpose(0, 1).reshape(tokens, -1))


class Gemma4MTPLayer(nn.Module):
    def __init__(self, config: Gemma4MTPConfig, layer: Gemma4MTPLayerConfig):
        super().__init__()
        eps = config.rms_norm_eps
        self.self_attn = Gemma4MTPAttention(config, layer)
        self.mlp = Gemma4MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv: torch.Tensor,
        visible: torch.Tensor,
    ) -> torch.Tensor:
        attention = self.self_attn(
            self.input_layernorm(hidden_states), positions, kv, visible
        )
        hidden_states = hidden_states + self.post_attention_layernorm(attention)
        mlp = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        hidden_states = hidden_states + self.post_feedforward_layernorm(mlp)
        return hidden_states * self.layer_scalar


class Gemma4MTPModel(nn.Module):
    def __init__(self, config: Gemma4MTPConfig):
        super().__init__()
        self.pre_projection = nn.Linear(
            2 * config.backbone_hidden_size, config.hidden_size, bias=False
        )
        self.post_projection = nn.Linear(
            config.hidden_size, config.backbone_hidden_size, bias=False
        )
        self.layers = nn.ModuleList(
            Gemma4MTPLayer(config, layer) for layer in config.layers
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)


class Gemma4MTPForCausalLM(nn.Module):
    def __init__(self, config: Gemma4MTPConfig):
        super().__init__()
        self.config = config
        self.model = Gemma4MTPModel(config)
        # Tied to the assistant's own embedding; kept frozen.
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.requires_grad_(False)
        # Only feeds later draft steps, whose inputs are captured, so it never
        # receives a gradient; freezing it also keeps it out of each install.
        self.model.post_projection.requires_grad_(False)

    def forward(
        self,
        positions: torch.Tensor,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        history: TargetKV,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return draft-dim hidden for logits and backbone hidden for feedback.

        Each query row sees history positions up to its own position, limited
        to the sliding window on sliding layers. History starts at position 0.
        """
        keys = torch.arange(history[0].shape[0])
        offsets = positions[:, None] - keys[None, :]
        visible = {None: offsets >= 0}
        hidden = self.model.pre_projection(torch.cat((input_embeds, hidden_states), -1))
        for layer in self.model.layers:
            window = layer.self_attn.layer.sliding_window
            if window not in visible:
                visible[window] = (offsets >= 0) & (offsets < window)
            hidden = layer(
                hidden,
                positions,
                history[layer.self_attn.layer.kv_index],
                visible[window],
            )
        hidden = self.model.norm(hidden)
        return hidden, self.model.post_projection(hidden)


def load_gemma4_mtp_from_vllm(gpu_model: nn.Module) -> Gemma4MTPForCausalLM:
    """Copy the loaded vLLM ``Gemma4MTP`` assistant into a CPU model.

    Draft layers are grouped by attention type: all sliding layers read one
    target KV (index 0) and all full layers read another (index 1).
    """
    predictor = gpu_model.model
    attentions = [layer.self_attn for layer in predictor.layers]
    if getattr(predictor.config, "attn_logit_softcapping", None):
        raise ValueError("online Gemma4 MTP does not support attention softcap")

    config = Gemma4MTPConfig(
        backbone_hidden_size=predictor.backbone_hidden_size,
        hidden_size=predictor.hidden_size,
        intermediate_size=predictor.layers[0].mlp.down_proj.input_size,
        num_attention_heads=attentions[0].total_num_heads,
        vocab_size=predictor.vocab_size,
        rms_norm_eps=predictor.norm.variance_epsilon,
        layers=tuple(
            Gemma4MTPLayerConfig(
                head_dim=attention.head_dim,
                num_key_value_heads=attention.total_num_kv_heads,
                sliding_window=attention.attn.sliding_window,
                kv_index=0 if attention.is_sliding else 1,
                rope_theta=float(attention.rotary_emb.base),
            )
            for attention in attentions
        ),
    )
    with torch.inference_mode(False), torch.device("meta"):
        model = Gemma4MTPForCausalLM(config)
    source = gpu_model.state_dict()
    with torch.inference_mode(False):
        weights = {
            name: source[name].detach().to(device="cpu", copy=True)
            for name in model.state_dict()
        }
        model.load_state_dict(weights, assign=True)
        tables: dict[int, torch.Tensor] = {}
        for cpu_attention, attention in zip(
            (layer.self_attn for layer in model.model.layers), attentions, strict=True
        ):
            table = attention.rotary_emb.cos_sin_cache
            if table.data_ptr() not in tables:
                tables[table.data_ptr()] = table.detach().to("cpu", copy=True)
            cpu_attention.rotary_emb.cos_sin_cache = tables[table.data_ptr()]
    return model
