# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU building blocks shared by the trainable online draft models.

Each block computes what the vLLM layer it mirrors computes; the parity tests
in ``tests/v1/spec_decode/test_online_draft_parity.py`` compare them against
vLLM's native implementations.
"""

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

LayerKV = tuple[torch.Tensor, torch.Tensor]

# KV-cache quantization scales: runtime attention state, not draft weights.
VLLM_ATTENTION_STATE_SUFFIXES = (
    ".attn._q_scale",
    ".attn._k_scale",
    ".attn._v_scale",
    ".attn._prob_scale",
)


class AttentionBackend(str, Enum):
    """CPU attention implementations for the autoregressive draft models."""

    EAGER = "eager"
    FLASH_ATTENTION = "flash_attention"

    __str__ = str.__str__


@dataclass(slots=True)
class DraftForwardOutput:
    """Final hidden states, the hidden fed to the next draft step, and KV."""

    hidden_states: torch.Tensor
    recurrent_hidden_states: torch.Tensor
    past_key_values: tuple[LayerKV, ...]


class RMSNorm(nn.Module):
    """vLLM RMSNorm; ``unit_offset`` scales by ``1 + weight`` like GemmaRMSNorm."""

    def __init__(self, hidden_size: int, eps: float, *, unit_offset: bool = False):
        super().__init__()
        initial = torch.zeros if unit_offset else torch.ones
        self.weight = nn.Parameter(initial(hidden_size))
        self.variance_epsilon = eps
        self.unit_offset = unit_offset

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            hidden_states = hidden_states + residual
            return self._normalize(hidden_states), hidden_states

        return self._normalize(hidden_states)

    def _normalize(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype

        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states.float() * torch.rsqrt(
            variance + self.variance_epsilon
        )
        weight = self.weight.float()
        if self.unit_offset:
            weight = weight + 1.0
        normalized = normalized * weight

        return normalized.to(dtype=input_dtype)


class GatedMLP(nn.Module):
    """SiLU gate-up MLP with vLLM's fused ``gate_up_proj`` layout."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.intermediate_size = intermediate_size
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(hidden_states).split(
            self.intermediate_size, dim=-1
        )
        return self.down_proj(F.silu(gate) * up)


def rotary_cosine_and_sine(
    positions: torch.Tensor,
    rotary_dim: int,
    rope_theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 NeoX RoPE cosine and sine shaped [tokens, rotary_dim // 2]."""
    frequency_indices = torch.arange(
        0,
        rotary_dim,
        2,
        dtype=torch.float32,
        device=positions.device,
    )
    inverse_frequencies = 1.0 / (rope_theta ** (frequency_indices / rotary_dim))
    frequencies = torch.outer(positions.to(dtype=torch.float32), inverse_frequencies)
    return frequencies.cos(), frequencies.sin()


class RotaryEmbedding(nn.Module):
    """NeoX RoPE on the first ``rotary_dim`` channels of [heads, tokens, head_dim].

    The remaining channels pass through unchanged, as in vLLM partial rotary.
    Drafts copied from vLLM look up the GPU layer's ``cos_sin_cache``, so any
    RoPE scaling matches; drafts built from a config use default RoPE.
    """

    def __init__(self, rotary_dim: int, rope_theta: float) -> None:
        super().__init__()
        self.rotary_dim = rotary_dim
        self.rope_theta = rope_theta
        self.cos_sin_cache: torch.Tensor | None
        self.register_buffer("cos_sin_cache", None, persistent=False)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = positions.to(query.device)
        if self.cos_sin_cache is None:
            cosine, sine = rotary_cosine_and_sine(
                positions, self.rotary_dim, self.rope_theta
            )
        else:
            cosine, sine = self.cos_sin_cache[positions].chunk(2, dim=-1)
        cosine = cosine.to(dtype=query.dtype).unsqueeze(0)
        sine = sine.to(dtype=query.dtype).unsqueeze(0)
        head_dim = query.shape[-1]

        def rotate(tensor: torch.Tensor) -> torch.Tensor:
            rotary, passthrough = tensor.split(
                (self.rotary_dim, head_dim - self.rotary_dim), -1
            )
            first_half, second_half = rotary.chunk(2, dim=-1)
            rotated = torch.cat(
                (
                    first_half * cosine - second_half * sine,
                    second_half * cosine + first_half * sine,
                ),
                dim=-1,
            )
            if self.rotary_dim == head_dim:
                return rotated
            return torch.cat((rotated, passthrough), dim=-1)

        return rotate(query), rotate(key)


def copy_vllm_rotary(model: nn.Module, rotary: nn.Module) -> None:
    """Point every RoPE in ``model`` at one CPU copy of vLLM's cos/sin table."""
    cache = rotary.cos_sin_cache.detach().to(device="cpu", copy=True)
    for module in model.modules():
        if isinstance(module, RotaryEmbedding):
            module.cos_sin_cache = cache


def cached_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    past_key_value: LayerKV | None,
    attention_mask: torch.Tensor | None,
    backend: AttentionBackend,
) -> tuple[torch.Tensor, LayerKV]:
    """Attend [heads, tokens, head_dim] queries to past and new keys.

    Args:
        query: Rotated queries.
        key: Rotated new keys shaped [kv_heads, tokens, head_dim].
        value: New values shaped like ``key``.
        past_key_value: Detached or differentiable earlier keys and values.
        attention_mask: Boolean [query_tokens, key_tokens] mask in which True
            blocks a pair, or None for causal attention after the past.
        backend: Attention implementation.

    Returns:
        Attention output shaped like ``query`` and the concatenated KV.
    """
    num_heads, sequence_length, head_dim = query.shape
    past_length = 0
    if past_key_value is not None:
        past_key, past_value = past_key_value
        past_length = past_key.shape[1]
        key = torch.cat((past_key, key), dim=1)
        value = torch.cat((past_value, value), dim=1)
    present = (key, value)

    total_key_length = past_length + sequence_length
    if attention_mask is None:
        attention_mask = torch.triu(
            torch.ones(
                sequence_length,
                total_key_length,
                dtype=torch.bool,
                device=query.device,
            ),
            diagonal=past_length + 1,
        )

    repeats = num_heads // key.shape[0]
    if repeats != 1:
        key = key.repeat_interleave(repeats, dim=0)
        value = value.repeat_interleave(repeats, dim=0)

    if backend is AttentionBackend.FLASH_ATTENTION:
        # SDPA uses the opposite boolean convention: True entries are
        # visible. A batch dimension is required by the fused kernel.
        try:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                output = F.scaled_dot_product_attention(
                    query.unsqueeze(0),
                    key.unsqueeze(0),
                    value.unsqueeze(0),
                    attn_mask=~attention_mask,
                    dropout_p=0.0,
                )
        except RuntimeError as error:
            raise RuntimeError(
                "flash attention is unavailable for the current device, dtype, or mask"
            ) from error
        return output.squeeze(0), present

    scores = torch.matmul(query, key.transpose(-1, -2)) * (head_dim**-0.5)
    scores = scores.masked_fill(attention_mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores.float(), dim=-1).to(dtype=value.dtype)
    return torch.matmul(weights, value), present
