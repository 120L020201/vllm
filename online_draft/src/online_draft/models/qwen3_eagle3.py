# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class Qwen3Eagle3Config:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_hidden_layers: int
    target_vocab_size: int
    draft_vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    attention_bias: bool = False
    norm_before_residual: bool = False
    norm_before_fc: bool = False
    norm_output: bool = False
    num_aux_hidden_states: int = 3
    fc_norm: bool = False

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be greater than zero")
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be greater than zero")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be greater than zero")
        if self.num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be greater than zero")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be greater than zero")
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be greater than zero")
        if self.target_vocab_size <= 0:
            raise ValueError("target_vocab_size must be greater than zero")
        if self.draft_vocab_size <= 0:
            raise ValueError("draft_vocab_size must be greater than zero")
        if self.draft_vocab_size > self.target_vocab_size:
            raise ValueError("draft_vocab_size must not exceed target_vocab_size")
        if self.num_aux_hidden_states <= 0:
            raise ValueError("num_aux_hidden_states must be greater than zero")
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be greater than zero")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be greater than zero")

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> "Qwen3Eagle3Config":
        raw_eagle_config = values.get("eagle_config")
        if raw_eagle_config is None:
            eagle_config: Mapping[str, Any] = {}
        elif isinstance(raw_eagle_config, Mapping):
            eagle_config = raw_eagle_config
        else:
            raise TypeError("eagle_config must be a mapping")

        hidden_size = int(values["hidden_size"])
        num_attention_heads = int(values["num_attention_heads"])
        if num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be greater than zero")

        raw_head_dim = values.get("head_dim")
        if raw_head_dim is None:
            head_dim = hidden_size // num_attention_heads
        else:
            head_dim = int(raw_head_dim)

        target_vocab_size = int(values["vocab_size"])
        raw_draft_vocab_size = values.get("draft_vocab_size")
        if raw_draft_vocab_size is None:
            draft_vocab_size = target_vocab_size
        else:
            draft_vocab_size = int(raw_draft_vocab_size)

        num_aux_hidden_states = _get_num_aux_hidden_states(
            values,
            eagle_config,
        )

        return cls(
            hidden_size=hidden_size,
            intermediate_size=int(values["intermediate_size"]),
            num_attention_heads=num_attention_heads,
            num_key_value_heads=int(values["num_key_value_heads"]),
            head_dim=head_dim,
            num_hidden_layers=int(values.get("num_hidden_layers", 1)),
            target_vocab_size=target_vocab_size,
            draft_vocab_size=draft_vocab_size,
            rms_norm_eps=float(values.get("rms_norm_eps", 1e-6)),
            rope_theta=float(values.get("rope_theta", 10000.0)),
            attention_bias=bool(values.get("attention_bias", False)),
            norm_before_residual=bool(values.get("norm_before_residual", False)),
            norm_before_fc=bool(
                eagle_config.get(
                    "norm_before_fc",
                    values.get("norm_before_fc", False),
                )
            ),
            norm_output=bool(values.get("norm_output", False)),
            num_aux_hidden_states=num_aux_hidden_states,
            fc_norm=bool(
                eagle_config.get(
                    "fc_norm",
                    values.get("fc_norm", False),
                )
            ),
        )


def _get_num_aux_hidden_states(
    values: Mapping[str, Any],
    eagle_config: Mapping[str, Any],
) -> int:
    raw_count = values.get("num_aux_hidden_states")
    if raw_count is None:
        raw_count = values.get("num_aux_layers")
    if raw_count is not None:
        return int(raw_count)

    aux_layer_ids = values.get("eagle_aux_hidden_state_layer_ids")
    if aux_layer_ids is None:
        aux_layer_ids = eagle_config.get("eagle_aux_hidden_state_layer_ids")
    if aux_layer_ids is None:
        return 3
    if not isinstance(aux_layer_ids, (list, tuple)):
        raise TypeError("eagle_aux_hidden_state_layer_ids must be a list or tuple")
    return len(aux_layer_ids)


class Qwen3Eagle3RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            hidden_states = hidden_states + residual
            return self._normalize(hidden_states), hidden_states

        return self._normalize(hidden_states)

    def _normalize(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype

        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states.float() * torch.rsqrt(
            variance + self.variance_epsilon
        )
        normalized = normalized * self.weight.float()

        return normalized.to(dtype=input_dtype)


def _apply_rotary_embedding(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    rope_theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query.ndim != 3 or key.ndim != 3:
        raise ValueError("query and key must have shape [heads, tokens, dim]")
    if positions.ndim != 1:
        raise ValueError("positions must have shape [tokens]")
    if query.shape[1] != positions.numel():
        raise ValueError("query token count must match positions")
    if key.shape[1] != positions.numel():
        raise ValueError("key token count must match positions")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key head dimensions must match")
    if query.device != key.device:
        raise ValueError("query and key must be on the same device")

    head_dim = query.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError("head dimension must be even")

    frequency_indices = torch.arange(
        0,
        head_dim,
        2,
        dtype=torch.float32,
        device=query.device,
    )
    inverse_frequencies = 1.0 / (rope_theta ** (frequency_indices / head_dim))
    frequencies = torch.outer(
        positions.to(device=query.device, dtype=torch.float32),
        inverse_frequencies,
    )

    cosine = frequencies.cos().to(dtype=query.dtype).unsqueeze(0)
    sine = frequencies.sin().to(dtype=query.dtype).unsqueeze(0)

    rotated_query = _rotate_with_cosine_and_sine(
        query,
        cosine,
        sine,
    )
    rotated_key = _rotate_with_cosine_and_sine(
        key,
        cosine,
        sine,
    )
    return rotated_query, rotated_key


def _rotate_with_cosine_and_sine(
    tensor: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    first_half, second_half = tensor.chunk(2, dim=-1)

    rotated_first = first_half * cosine - second_half * sine
    rotated_second = second_half * cosine + first_half * sine

    return torch.cat(
        (rotated_first, rotated_second),
        dim=-1,
    )


Eagle3LayerKVCache = tuple[torch.Tensor, torch.Tensor]


class Qwen3Eagle3SelfAttention(nn.Module):
    def __init__(
        self,
        config: Qwen3Eagle3Config,
        input_size: int,
    ) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta

        query_size = self.num_attention_heads * self.head_dim
        key_value_size = self.num_key_value_heads * self.head_dim
        qkv_output_size = query_size + 2 * key_value_size

        self.qkv_proj = nn.Linear(
            input_size,
            qkv_output_size,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        past_key_value: Eagle3LayerKVCache | None = None,
    ) -> tuple[torch.Tensor, Eagle3LayerKVCache]:
        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must have shape [tokens, input_size]")
        if positions.ndim != 1:
            raise ValueError("positions must have shape [tokens]")
        if hidden_states.shape[0] != positions.numel():
            raise ValueError("hidden_states token count must match positions")
        if hidden_states.shape[1] != self.qkv_proj.in_features:
            raise ValueError("hidden_states width does not match attention input size")

        sequence_length = hidden_states.shape[0]
        query_size = self.num_attention_heads * self.head_dim
        key_value_size = self.num_key_value_heads * self.head_dim

        qkv = self.qkv_proj(hidden_states)
        query, key, value = qkv.split(
            (query_size, key_value_size, key_value_size),
            dim=-1,
        )

        query = query.view(
            sequence_length,
            self.num_attention_heads,
            self.head_dim,
        ).transpose(0, 1)
        key = key.view(
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(0, 1)
        value = value.view(
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(0, 1)

        query, key = _apply_rotary_embedding(
            query,
            key,
            positions,
            self.rope_theta,
        )

        past_length = 0
        if past_key_value is not None:
            past_key, past_value = past_key_value
            self._validate_past_key_value(
                past_key,
                past_value,
                hidden_states,
            )
            past_length = past_key.shape[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)

        present_key_value = (key, value)

        if self.num_attention_heads != self.num_key_value_heads:
            repeat_count = self.num_attention_heads // self.num_key_value_heads
            attention_key = key.repeat_interleave(
                repeat_count,
                dim=0,
            )
            attention_value = value.repeat_interleave(
                repeat_count,
                dim=0,
            )
        else:
            attention_key = key
            attention_value = value

        attention_scores = torch.matmul(
            query,
            attention_key.transpose(-1, -2),
        )
        attention_scores = attention_scores * (self.head_dim**-0.5)

        total_key_length = past_length + sequence_length
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                total_key_length,
                dtype=torch.bool,
                device=attention_scores.device,
            ),
            diagonal=past_length + 1,
        )
        attention_scores = attention_scores.masked_fill(
            causal_mask,
            torch.finfo(attention_scores.dtype).min,
        )

        attention_weights = torch.softmax(
            attention_scores.float(),
            dim=-1,
        ).to(dtype=attention_value.dtype)

        attention_output = torch.matmul(
            attention_weights,
            attention_value,
        )
        attention_output = attention_output.transpose(
            0,
            1,
        ).reshape(sequence_length, self.hidden_size)

        return self.o_proj(attention_output), present_key_value

    def _validate_past_key_value(
        self,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> None:
        if past_key.ndim != 3 or past_value.ndim != 3:
            raise ValueError(
                "cached key and value must have shape [kv_heads, tokens, head_dim]"
            )
        if past_key.shape != past_value.shape:
            raise ValueError("cached key and value shapes must match")
        if past_key.shape[0] != self.num_key_value_heads:
            raise ValueError("cached key/value head count is invalid")
        if past_key.shape[2] != self.head_dim:
            raise ValueError("cached key/value head dimension is invalid")
        if past_key.device != hidden_states.device:
            raise ValueError("cached key/value and hidden_states devices must match")
        if past_key.dtype != hidden_states.dtype:
            raise ValueError("cached key/value and hidden_states dtypes must match")


class Qwen3Eagle3MLP(nn.Module):
    def __init__(
        self,
        config: Qwen3Eagle3Config,
    ) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_up_proj = nn.Linear(
            self.hidden_size,
            2 * self.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must have shape [tokens, hidden_size]")
        if hidden_states.shape[1] != self.hidden_size:
            raise ValueError("hidden_states width does not match hidden_size")

        gate_and_up = self.gate_up_proj(hidden_states)
        gate, up = gate_and_up.split(
            self.intermediate_size,
            dim=-1,
        )
        activated = F.silu(gate) * up

        return self.down_proj(activated)


class Qwen3Eagle3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Eagle3Config,
        layer_index: int,
    ) -> None:
        super().__init__()

        if layer_index < 0:
            raise ValueError("layer_index must not be negative")

        self.layer_index = layer_index
        self.hidden_size = config.hidden_size
        self.norm_before_residual = config.norm_before_residual

        attention_input_size = (
            2 * config.hidden_size if layer_index == 0 else config.hidden_size
        )

        self.self_attn = Qwen3Eagle3SelfAttention(
            config,
            input_size=attention_input_size,
        )
        self.mlp = Qwen3Eagle3MLP(config)

        self.hidden_norm = Qwen3Eagle3RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.input_layernorm = Qwen3Eagle3RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.post_attention_layernorm = Qwen3Eagle3RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

    def forward(
        self,
        positions: torch.Tensor,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        past_key_value: Eagle3LayerKVCache | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Eagle3LayerKVCache,
    ]:
        self._validate_inputs(
            positions,
            input_embeds,
            hidden_states,
        )

        if self.layer_index == 0:
            normalized_embeds = self.input_layernorm(input_embeds)
            if not isinstance(normalized_embeds, torch.Tensor):
                raise RuntimeError(
                    "input embedding normalization returned invalid output"
                )

            hidden_states, residual = self._normalize_first_layer_hidden_states(
                hidden_states
            )
            hidden_states = torch.cat(
                (normalized_embeds, hidden_states),
                dim=-1,
            )
        else:
            if residual is None:
                raise ValueError("residual is required after the first decoder layer")

            normalized = self.input_layernorm(
                hidden_states,
                residual,
            )
            if not isinstance(normalized, tuple):
                raise RuntimeError("residual normalization returned invalid output")
            hidden_states, residual = normalized

        attention_output, present_key_value = self.self_attn(
            hidden_states,
            positions,
            past_key_value,
        )

        normalized = self.post_attention_layernorm(
            attention_output,
            residual,
        )
        if not isinstance(normalized, tuple):
            raise RuntimeError("post-attention normalization returned invalid output")
        hidden_states, residual = normalized

        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual, present_key_value

    def _normalize_first_layer_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.hidden_norm(hidden_states)
        if not isinstance(normalized, torch.Tensor):
            raise RuntimeError("hidden state normalization returned invalid output")

        residual = normalized if self.norm_before_residual else hidden_states

        return normalized, residual

    def _validate_inputs(
        self,
        positions: torch.Tensor,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> None:
        if positions.ndim != 1:
            raise ValueError("positions must have shape [tokens]")
        if input_embeds.ndim != 2:
            raise ValueError("input_embeds must have shape [tokens, hidden_size]")
        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must have shape [tokens, hidden_size]")
        if input_embeds.shape != hidden_states.shape:
            raise ValueError("input_embeds and hidden_states shapes must match")
        if input_embeds.shape[0] != positions.numel():
            raise ValueError("input token count must match positions")
        if input_embeds.shape[1] != self.hidden_size:
            raise ValueError("input width does not match hidden_size")


Eagle3KVCache = tuple[Eagle3LayerKVCache, ...]


@dataclass(slots=True)
class Eagle3ForwardOutput:
    hidden_states: torch.Tensor
    recurrent_hidden_states: torch.Tensor
    past_key_values: Eagle3KVCache


class Qwen3Eagle3Model(nn.Module):
    def __init__(
        self,
        config: Qwen3Eagle3Config,
    ) -> None:
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_aux_hidden_states = config.num_aux_hidden_states
        self.norm_before_fc = config.norm_before_fc
        self.norm_output = config.norm_output

        self.fc_input_size = config.hidden_size * config.num_aux_hidden_states

        if self.norm_before_fc:
            self.input_norm: Qwen3Eagle3RMSNorm | None = Qwen3Eagle3RMSNorm(
                self.fc_input_size,
                config.rms_norm_eps,
            )
        else:
            self.input_norm = None

        if config.fc_norm:
            self.fc_norm: nn.ModuleList | None = nn.ModuleList(
                [
                    Qwen3Eagle3RMSNorm(
                        config.hidden_size,
                        config.rms_norm_eps,
                    )
                    for _ in range(config.num_aux_hidden_states)
                ]
            )
        else:
            self.fc_norm = None

        self.fc = nn.Linear(
            self.fc_input_size,
            config.hidden_size,
            bias=False,
        )

        self.layers = nn.ModuleList(
            [
                Qwen3Eagle3DecoderLayer(
                    config,
                    layer_index=layer_index,
                )
                for layer_index in range(config.num_hidden_layers)
            ]
        )

        self.norm = Qwen3Eagle3RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

    def combine_hidden_states(
        self,
        auxiliary_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if auxiliary_hidden_states.ndim != 2:
            raise ValueError(
                "auxiliary_hidden_states must have shape [tokens, auxiliary_width]"
            )
        if auxiliary_hidden_states.shape[1] != self.fc_input_size:
            raise ValueError(
                "auxiliary hidden state width does not match "
                "the configured FC input size"
            )

        hidden_states = auxiliary_hidden_states

        if self.input_norm is not None:
            normalized = self.input_norm(hidden_states)
            if not isinstance(normalized, torch.Tensor):
                raise RuntimeError("input normalization returned invalid output")
            hidden_states = normalized

        if self.fc_norm is not None:
            chunks = hidden_states.chunk(
                self.num_aux_hidden_states,
                dim=-1,
            )
            normalized_chunks: list[torch.Tensor] = []

            for norm, chunk in zip(
                self.fc_norm,
                chunks,
                strict=True,
            ):
                normalized = norm(chunk)
                if not isinstance(normalized, torch.Tensor):
                    raise RuntimeError(
                        "auxiliary normalization returned invalid output"
                    )
                normalized_chunks.append(normalized)

            hidden_states = torch.cat(
                normalized_chunks,
                dim=-1,
            )

        return self.fc(hidden_states)

    def forward(
        self,
        positions: torch.Tensor,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_values: Eagle3KVCache | None = None,
    ) -> Eagle3ForwardOutput:
        self._validate_inputs(
            positions,
            input_embeds,
            hidden_states,
            past_key_values,
        )

        residual: torch.Tensor | None = None
        present_key_values: list[Eagle3LayerKVCache] = []

        for layer_index, layer in enumerate(self.layers):
            if past_key_values is None:
                past_key_value = None
            else:
                past_key_value = past_key_values[layer_index]

            hidden_states, residual, present_key_value = layer(
                positions=positions,
                input_embeds=input_embeds,
                hidden_states=hidden_states,
                residual=residual,
                past_key_value=past_key_value,
            )
            present_key_values.append(present_key_value)

        if residual is None:
            raise RuntimeError("model did not produce a residual")

        final_output = self.norm(
            hidden_states,
            residual,
        )
        if not isinstance(final_output, tuple):
            raise RuntimeError("final normalization returned invalid output")

        normalized_hidden_states, hidden_states_before_norm = final_output

        if self.norm_output:
            recurrent_hidden_states = normalized_hidden_states
        else:
            recurrent_hidden_states = hidden_states_before_norm

        return Eagle3ForwardOutput(
            hidden_states=normalized_hidden_states,
            recurrent_hidden_states=recurrent_hidden_states,
            past_key_values=tuple(present_key_values),
        )

    def _validate_inputs(
        self,
        positions: torch.Tensor,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_values: Eagle3KVCache | None,
    ) -> None:
        if positions.ndim != 1:
            raise ValueError("positions must have shape [tokens]")
        if input_embeds.ndim != 2:
            raise ValueError("input_embeds must have shape [tokens, hidden_size]")
        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must have shape [tokens, hidden_size]")
        if input_embeds.shape != hidden_states.shape:
            raise ValueError("input_embeds and hidden_states shapes must match")
        if input_embeds.shape[0] != positions.numel():
            raise ValueError("input token count must match positions")
        if input_embeds.shape[1] != self.hidden_size:
            raise ValueError("input width does not match hidden_size")
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError("past_key_values must contain one entry per layer")


class Qwen3Eagle3ForCausalLM(nn.Module):
    def __init__(
        self,
        config: Qwen3Eagle3Config,
    ) -> None:
        super().__init__()

        self.config = config
        self.model = Qwen3Eagle3Model(config)

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.draft_vocab_size,
            bias=False,
        )

        self.draft_id_to_target_id = nn.Parameter(
            torch.zeros(
                config.draft_vocab_size,
                dtype=torch.long,
            ),
            requires_grad=False,
        )

    def forward(
        self,
        positions: torch.Tensor,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_values: Eagle3KVCache | None = None,
    ) -> Eagle3ForwardOutput:
        return self.model(
            positions=positions,
            input_embeds=input_embeds,
            hidden_states=hidden_states,
            past_key_values=past_key_values,
        )

    def combine_hidden_states(
        self,
        auxiliary_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.combine_hidden_states(auxiliary_hidden_states)

    def compute_draft_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must have shape [tokens, hidden_size]")
        if hidden_states.shape[1] != self.config.hidden_size:
            raise ValueError("hidden_states width does not match hidden_size")

        return self.lm_head(hidden_states)

    def compute_target_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        draft_logits = self.compute_draft_logits(hidden_states)
        target_token_ids = self.get_target_token_ids(device=draft_logits.device)

        target_logits = draft_logits.new_full(
            (
                draft_logits.shape[0],
                self.config.target_vocab_size,
            ),
            float("-inf"),
        )
        target_logits[:, target_token_ids] = draft_logits

        return target_logits

    def get_target_token_ids(
        self,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if device is None:
            device = self.draft_id_to_target_id.device

        draft_token_ids = torch.arange(
            self.config.draft_vocab_size,
            dtype=torch.long,
            device=device,
        )
        offsets = self.draft_id_to_target_id.to(device=device)
        target_token_ids = draft_token_ids + offsets

        if target_token_ids.numel() > 0:
            minimum_id = int(target_token_ids.min().item())
            maximum_id = int(target_token_ids.max().item())

            if minimum_id < 0:
                raise ValueError("draft-to-target mapping contains a negative token ID")
            if maximum_id >= self.config.target_vocab_size:
                raise ValueError("draft-to-target mapping exceeds target vocabulary")

        if target_token_ids.unique().numel() != target_token_ids.numel():
            raise ValueError("draft-to-target mapping contains duplicate token IDs")

        return target_token_ids


def convert_angelslim_eagle3_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert AngelSlim EAGLE3 weights to this model's layout.

    Args:
        state_dict: State dictionary loaded from the AngelSlim checkpoint.

    Returns:
        State dictionary accepted by Qwen3Eagle3ForCausalLM.
    """
    return {
        "draft_id_to_target_id": state_dict["d2t"],
        "model.layers.0.self_attn.qkv_proj.weight": torch.cat(
            (
                state_dict["midlayer.self_attn.q_proj.weight"],
                state_dict["midlayer.self_attn.k_proj.weight"],
                state_dict["midlayer.self_attn.v_proj.weight"],
            ),
            dim=0,
        ),
        "model.layers.0.self_attn.o_proj.weight": state_dict[
            "midlayer.self_attn.o_proj.weight"
        ],
        "model.layers.0.mlp.gate_up_proj.weight": torch.cat(
            (
                state_dict["midlayer.mlp.gate_proj.weight"],
                state_dict["midlayer.mlp.up_proj.weight"],
            ),
            dim=0,
        ),
        "model.layers.0.mlp.down_proj.weight": state_dict[
            "midlayer.mlp.down_proj.weight"
        ],
        "model.layers.0.hidden_norm.weight": state_dict["midlayer.hidden_norm.weight"],
        "model.layers.0.input_layernorm.weight": state_dict[
            "midlayer.input_layernorm.weight"
        ],
        "model.layers.0.post_attention_layernorm.weight": state_dict[
            "midlayer.post_attention_layernorm.weight"
        ],
        "model.norm.weight": state_dict["norm.weight"],
        "model.fc.weight": state_dict["fc.weight"],
        "lm_head.weight": state_dict["lm_head.weight"],
    }


def load_qwen3_eagle3_checkpoint(
    model_directory: str | Path,
    *,
    dtype: torch.dtype = torch.float32,
) -> Qwen3Eagle3ForCausalLM:
    """Load an AngelSlim Qwen3 EAGLE3 checkpoint.

    Args:
        model_directory: Directory containing config.json and model weights.
        dtype: Floating-point dtype used by the loaded model.

    Returns:
        A model with converted checkpoint weights.
    """
    model_path = Path(model_directory)
    config_path = model_path / "config.json"
    checkpoint_path = model_path / "pytorch_model.bin"

    with config_path.open(encoding="utf-8") as config_file:
        raw_config = json.load(config_file)

    if not isinstance(raw_config, Mapping):
        raise TypeError("config.json must contain a JSON object")

    config = Qwen3Eagle3Config.from_dict(raw_config)
    model = Qwen3Eagle3ForCausalLM(config)
    model.to(dtype=dtype)

    source_state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(source_state_dict, Mapping):
        raise TypeError("checkpoint must contain a state dictionary")

    converted_state_dict = convert_angelslim_eagle3_state_dict(source_state_dict)
    model.load_state_dict(converted_state_dict, strict=True)

    return model
