# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from online_draft.models.cpu_attention import (
    context_attention,
    load_attention_backward,
)
from online_draft.models.layers import (
    GatedMLP,
    LayerKV,
    RMSNorm,
    apply_rotary_embedding,
    validate_vllm_rotary,
)

ContextKV = tuple[LayerKV, ...]


@dataclass(frozen=True)
class DFlashLayerAttention:
    """Which keys one draft layer's queries see, as configured in vLLM.

    A causal layer hides later block rows; a sliding window of W limits keys
    to the W positions ending at the query, matching vLLM's ``(W - 1, 0)``.
    """

    sliding_window: int | None = None
    causal: bool = False


@dataclass(frozen=True)
class Qwen3DFlashConfig:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_hidden_layers: int
    vocab_size: int
    num_aux_hidden_states: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    attention_blocks_per_chunk: int = 64
    grouped_query_attention: bool = False
    selective_last_layer: bool = False
    # One entry per layer; empty means every layer is full and non-causal.
    layer_attention: tuple[DFlashLayerAttention, ...] = ()

    def attention(self, layer_index: int) -> DFlashLayerAttention:
        if not self.layer_attention:
            return DFlashLayerAttention()
        return self.layer_attention[layer_index]


class DFlashAttention(nn.Module):
    def __init__(
        self,
        config: Qwen3DFlashConfig,
        layer_attention: DFlashLayerAttention | None = None,
    ):
        super().__init__()
        load_attention_backward()
        self.config = config
        self.layer_attention = layer_attention or DFlashLayerAttention()
        self.q_size = config.num_attention_heads * config.head_dim
        self.kv_size = config.num_key_value_heads * config.head_dim
        self.qkv_proj = nn.Linear(
            config.hidden_size, self.q_size + 2 * self.kv_size, bias=False
        )
        self.o_proj = nn.Linear(self.q_size, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)

    def _heads(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(tensor.shape[0], -1, self.config.head_dim).transpose(0, 1)

    def context_kv(self, hidden: torch.Tensor, positions: torch.Tensor) -> LayerKV:
        kv = F.linear(hidden, self.qkv_proj.weight[self.q_size :])
        key, value = kv.split(self.kv_size, dim=-1)
        key = self.k_norm(self._heads(key))
        key, _ = apply_rotary_embedding(key, key, positions, self.config.rope_theta)
        return key, self._heads(value)

    def _restrict(
        self,
        mask: torch.Tensor | None,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
    ) -> torch.Tensor | None:
        """Apply this layer's causal and sliding-window limits to a visible mask."""
        causal = self.layer_attention.causal
        window = self.layer_attention.sliding_window
        if not causal and window is None:
            return mask
        offset = query_positions[:, None] - key_positions[None, :]
        visible = offset >= 0
        if window is not None:
            visible &= offset < window
        return visible if mask is None else mask & visible

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        context: LayerKV,
        attention_mask: torch.Tensor | None = None,
        block_size: int | None = None,
        output_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query, key, value = self.qkv_proj(hidden).split(
            (self.q_size, self.kv_size, self.kv_size), dim=-1
        )
        query = self.q_norm(self._heads(query))
        key = self.k_norm(self._heads(key))
        query, key = apply_rotary_embedding(
            query, key, positions, self.config.rope_theta
        )
        value = self._heads(value)
        repeats = self.config.num_attention_heads // self.config.num_key_value_heads
        chunk = (
            block_size * self.config.attention_blocks_per_chunk
            if block_size and self.config.attention_blocks_per_chunk
            else hidden.shape[0]
        )
        window = self.layer_attention.sliding_window
        context_length = context[0].shape[1]
        outputs = []
        for start in range(0, hidden.shape[0], chunk):
            end = min(start + chunk, hidden.shape[0])
            context_start, context_end = 0, context_length
            mask = attention_mask
            if block_size is not None:
                # Packed training windows always pass their block mask.
                attention_mask = cast(torch.Tensor, attention_mask)
                # Later blocks start later, so the chunk's last anchor bounds
                # its context and its first row bounds a sliding window.
                context_end = int(positions[end - block_size])
                if window is not None:
                    context_start = min(
                        context_end, max(0, int(positions[start]) - window + 1)
                    )
                mask = torch.cat(
                    (
                        attention_mask[start:end, context_start:context_end],
                        attention_mask[
                            start:end,
                            context_length + start : context_length + end,
                        ],
                    ),
                    dim=1,
                )
            mask = self._restrict(
                mask,
                positions[start:end],
                torch.cat(
                    (
                        torch.arange(
                            context_start, context_end, device=positions.device
                        ),
                        positions[start:end],
                    )
                ),
            )
            context_key, context_value = (
                tensor[:, context_start:context_end] for tensor in context
            )
            keys, values = key[:, start:end], value[:, start:end]
            queries = query[:, start:end]
            if output_rows is not None:
                rows = output_rows[(output_rows >= start) & (output_rows < end)]
                queries = query[:, rows]
                if mask is not None:
                    mask = mask[rows - start]
            output_shape = queries.shape
            if self.config.grouped_query_attention:
                queries = queries.reshape(keys.shape[0], -1, self.config.head_dim)
                if mask is not None:
                    mask = mask.repeat(repeats, 1)
            else:
                context_key = context_key.repeat_interleave(repeats, dim=0)
                context_value = context_value.repeat_interleave(repeats, dim=0)
                keys = keys.repeat_interleave(repeats, dim=0)
                values = values.repeat_interleave(repeats, dim=0)
            output = context_attention(
                queries, context_key, context_value, keys, values, mask
            )
            outputs.append(output.reshape(output_shape))
        output = torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0]
        return self.o_proj(output.transpose(0, 1).reshape(output.shape[1], -1))


class DFlashLayer(nn.Module):
    def __init__(self, config: Qwen3DFlashConfig, layer_index: int = 0):
        super().__init__()
        self.self_attn = DFlashAttention(config, config.attention(layer_index))
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = GatedMLP(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        context: LayerKV,
        attention_mask: torch.Tensor | None = None,
        block_size: int | None = None,
        output_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden if output_rows is None else hidden[output_rows]
        hidden = residual + self.self_attn(
            self.input_layernorm(hidden),
            positions,
            context,
            attention_mask,
            block_size,
            output_rows,
        )
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class DFlashModel(nn.Module):
    def __init__(self, config: Qwen3DFlashConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_tokens.requires_grad_(False)
        self.fc = nn.Linear(
            config.hidden_size * config.num_aux_hidden_states,
            config.hidden_size,
            bias=False,
        )
        self.fc.requires_grad_(False)
        self.hidden_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hidden_norm.requires_grad_(False)
        self.layers = nn.ModuleList(
            DFlashLayer(config, index) for index in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)


class Qwen3DFlashForCausalLM(nn.Module):
    """Differentiable Qwen3 DFlash with vLLM-compatible packed weights."""

    def __init__(self, config: Qwen3DFlashConfig):
        super().__init__()
        self.config = config
        self.model = DFlashModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.requires_grad_(False)

    @torch.no_grad()
    def project_context(
        self,
        fc_output: torch.Tensor,
        positions: torch.Tensor,
    ) -> ContextKV:
        """Build detached context KV from captured FC output with current weights."""
        hidden = self.model.hidden_norm(fc_output)
        return tuple(
            layer.self_attn.context_kv(hidden, positions) for layer in self.model.layers
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        context: ContextKV,
        attention_mask: torch.Tensor | None = None,
        block_size: int | None = None,
        output_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run query blocks over context KV.

        ``attention_mask`` is the boolean [queries, context + queries] mask of
        visible pairs shared by all layers; each layer adds its own causal and
        sliding-window limits.
        """
        hidden = self.model.embed_tokens(input_ids)
        for index, (layer, layer_context) in enumerate(
            zip(self.model.layers, context, strict=True)
        ):
            hidden = layer(
                hidden,
                positions,
                layer_context,
                attention_mask,
                block_size,
                output_rows if index == len(self.model.layers) - 1 else None,
            )
        return self.model.norm(hidden)


def load_qwen3_dflash_from_vllm(
    gpu_model: nn.Module,
    *,
    grouped_query_attention: bool = False,
    selective_last_layer: bool = False,
) -> Qwen3DFlashForCausalLM:
    """Copy the loaded draft, including its frozen shared target head, at startup.

    Each layer's causal flag and sliding window are read from the vLLM layers
    so the CPU masks follow what the GPU draft computes.
    """
    config = gpu_model.config
    layers = gpu_model.model.layers
    rotary = layers[0].self_attn.rotary_emb
    causal = gpu_model.get_draft_attn_causal()
    unsupported = {
        "attention bias": config.attention_bias,
        "separate mask embedding": gpu_model.model.has_separate_mask_embedding,
        "draft-to-target map": gpu_model.draft_id_to_target_id is not None,
        "non-NeoX rotary": not rotary.is_neox_style,
        "partial rotary": rotary.rotary_dim != config.head_dim,
        "attention sinks": any(
            layer.self_attn.attention_sink_bias is not None for layer in layers
        ),
        "non-causal sliding window": any(
            layer.self_attn.sliding_window is not None and not layer_causal
            for layer, layer_causal in zip(layers, causal, strict=True)
        ),
    }
    if any(unsupported.values()):
        names = ", ".join(name for name, found in unsupported.items() if found)
        raise ValueError(f"online DFlash CPU model does not support: {names}")

    rope_theta = config.rope_parameters["rope_theta"]
    validate_vllm_rotary(rotary, rotary.rotary_dim, rope_theta, "DFlash")
    cpu_config = Qwen3DFlashConfig(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        num_hidden_layers=config.num_hidden_layers,
        vocab_size=config.vocab_size,
        num_aux_hidden_states=len(config.dflash_config["target_layer_ids"]),
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=rope_theta,
        grouped_query_attention=grouped_query_attention,
        selective_last_layer=selective_last_layer,
        layer_attention=tuple(
            DFlashLayerAttention(layer.self_attn.sliding_window, layer_causal)
            for layer, layer_causal in zip(layers, causal, strict=True)
        ),
    )
    with torch.inference_mode(False), torch.device("meta"):
        model = Qwen3DFlashForCausalLM(cpu_config)
    source = gpu_model.state_dict()
    with torch.inference_mode(False):
        weights = {
            name: source[name].detach().to(device="cpu", copy=True)
            for name in model.state_dict()
        }
        model.load_state_dict(weights, assign=True)
    return model
