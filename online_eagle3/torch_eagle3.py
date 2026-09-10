# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

Eagle3KVCache = tuple[tuple[torch.Tensor, torch.Tensor], ...]


@dataclass(slots=True)
class TorchEagle3Config:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_hidden_layers: int
    vocab_size: int
    draft_vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    attention_bias: bool = False
    norm_before_residual: bool = False
    norm_before_fc: bool = False
    norm_output: bool = False
    num_aux_hidden_states: int = 3
    fc_norm: bool = False

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TorchEagle3Config:
        eagle_config = values.get("eagle_config") or {}
        aux_layers = values.get("num_aux_hidden_states")
        if aux_layers is None:
            aux_layers = values.get("num_aux_layers")
        if aux_layers is None:
            aux_ids = eagle_config.get("eagle_aux_hidden_state_layer_ids")
            aux_layers = len(aux_ids) if aux_ids else 3

        return cls(
            hidden_size=int(values["hidden_size"]),
            intermediate_size=int(values["intermediate_size"]),
            num_attention_heads=int(values["num_attention_heads"]),
            num_key_value_heads=int(values["num_key_value_heads"]),
            head_dim=int(
                values.get(
                    "head_dim",
                    values["hidden_size"] // values["num_attention_heads"],
                )
            ),
            num_hidden_layers=int(values.get("num_hidden_layers", 1)),
            vocab_size=int(values["vocab_size"]),
            draft_vocab_size=int(values.get("draft_vocab_size", values["vocab_size"])),
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
            num_aux_hidden_states=int(aux_layers),
            fc_norm=bool(values.get("fc_norm", False)),
        )


class TorchRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            x = x + residual
            return self._norm(x), x
        return self._norm(x)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normed = x.float() * torch.rsqrt(variance + self.variance_epsilon)
        return (normed * self.weight.float()).to(dtype)


class TorchEagle3SelfAttention(nn.Module):
    def __init__(self, config: TorchEagle3Config, input_size: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        qkv_size = self.hidden_size + 2 * self.num_kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(input_size, qkv_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_output: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split((q_size, kv_size, kv_size), dim=-1)

        seq_len = hidden_states.shape[0]
        q = q.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(seq_len, self.num_kv_heads, self.head_dim).transpose(0, 1)
        v = v.view(seq_len, self.num_kv_heads, self.head_dim).transpose(0, 1)
        q, k = self._apply_rotary(q, k, positions)

        past_len = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_len = past_k.shape[1]
            k = torch.cat((past_k, k), dim=1)
            v = torch.cat((past_v, v), dim=1)
        if kv_output is not None:
            kv_output.append((k, v))

        if self.num_heads != self.num_kv_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=0)
            v = v.repeat_interleave(repeat, dim=0)

        scores = torch.matmul(q, k.transpose(-1, -2)) * (self.head_dim**-0.5)
        mask = torch.triu(
            torch.ones(
                seq_len, past_len + seq_len, dtype=torch.bool, device=scores.device
            ),
            diagonal=past_len + 1,
        )
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        output = torch.matmul(attn, v)
        output = output.transpose(0, 1).reshape(seq_len, self.hidden_size)
        return self.o_proj(output)

    def _apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, device=q.device).float()
                / self.head_dim
            )
        )
        freqs = torch.outer(positions.to(q.device).float(), inv_freq)
        cos = freqs.cos().to(q.dtype).unsqueeze(0)
        sin = freqs.sin().to(q.dtype).unsqueeze(0)
        return _rotate_half(q, cos, sin), _rotate_half(k, cos, sin)


class TorchEagle3MLP(nn.Module):
    def __init__(self, config: TorchEagle3Config) -> None:
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = nn.Linear(
            config.hidden_size,
            2 * config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(hidden_states).split(
            self.intermediate_size,
            dim=-1,
        )
        return self.down_proj(F.silu(gate) * up)


class TorchEagle3DecoderLayer(nn.Module):
    def __init__(self, config: TorchEagle3Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        input_size = 2 * config.hidden_size if layer_idx == 0 else config.hidden_size
        self.self_attn = TorchEagle3SelfAttention(config, input_size)
        self.mlp = TorchEagle3MLP(config)
        self.hidden_norm = TorchRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.input_layernorm = TorchRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = TorchRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.norm_before_residual = config.norm_before_residual

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_output: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.layer_idx == 0:
            embeds = self.input_layernorm(embeds)
            assert isinstance(embeds, torch.Tensor)
            hidden_states, residual = self._residual_norm(hidden_states)
            hidden_states = torch.cat([embeds, hidden_states], dim=-1)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
            assert isinstance(hidden_states, torch.Tensor)

        hidden_states = self.self_attn(
            positions, hidden_states, past_key_value, kv_output
        )
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states,
            residual,
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def _residual_norm(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.norm_before_residual:
            hidden_states = self.hidden_norm(hidden_states)
            assert isinstance(hidden_states, torch.Tensor)
            return hidden_states, hidden_states
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)
        assert isinstance(hidden_states, torch.Tensor)
        return hidden_states, residual


class TorchEagle3Model(nn.Module):
    def __init__(self, config: TorchEagle3Config) -> None:
        super().__init__()
        self.config = config
        self.use_aux_hidden_state = True
        self.norm_before_fc = config.norm_before_fc
        self.norm_output = config.norm_output
        self.num_aux_hidden_states = config.num_aux_hidden_states
        self.fc_input_size = config.hidden_size * self.num_aux_hidden_states

        if self.norm_before_fc:
            self.input_norm = TorchRMSNorm(self.fc_input_size, config.rms_norm_eps)
        else:
            self.input_norm = None

        if config.fc_norm:
            self.fc_norm = nn.ModuleList(
                [
                    TorchRMSNorm(config.hidden_size, config.rms_norm_eps)
                    for _ in range(self.num_aux_hidden_states)
                ]
            )
        else:
            self.fc_norm = None

        self.fc = nn.Linear(self.fc_input_size, config.hidden_size, bias=False)
        self.layers = nn.ModuleList(
            [
                TorchEagle3DecoderLayer(config, i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = TorchRMSNorm(config.hidden_size, config.rms_norm_eps)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.norm_before_fc:
            assert self.input_norm is not None
            hidden_states = self.input_norm(hidden_states)
            assert isinstance(hidden_states, torch.Tensor)

        if self.fc_norm is not None:
            chunks = hidden_states.chunk(self.num_aux_hidden_states, dim=-1)
            hidden_states = torch.cat(
                [norm(chunk) for norm, chunk in zip(self.fc_norm, chunks)],
                dim=-1,
            )

        return self.fc(hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        input_embeds: torch.Tensor,
        past_key_values: Eagle3KVCache = (),
        kv_output: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del input_ids
        residual = None
        if past_key_values and len(past_key_values) != len(self.layers):
            raise ValueError("KV cache must have one entry per decoder layer")
        for index, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                positions=positions,
                embeds=input_embeds,
                hidden_states=hidden_states,
                residual=residual,
                past_key_value=past_key_values[index] if past_key_values else None,
                kv_output=kv_output,
            )
        hidden_states, hidden_prenorm = self.norm(hidden_states, residual)
        aux_output = hidden_states if self.norm_output else hidden_prenorm
        return hidden_states, aux_output


class TorchEagle3ForCausalLM(nn.Module):
    def __init__(self, config: TorchEagle3Config) -> None:
        super().__init__()
        self.config = config
        self.model = TorchEagle3Model(config)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.draft_vocab_size,
            bias=False,
        )
        self.draft_id_to_target_id = nn.Parameter(
            torch.zeros(config.draft_vocab_size, dtype=torch.long),
            requires_grad=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        past_key_values: Eagle3KVCache = (),
        kv_output: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(
            input_ids,
            positions,
            hidden_states,
            inputs_embeds,
            past_key_values,
            kv_output,
        )

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(hidden_states)

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.compute_draft_logits(hidden_states)
        draft_ids = torch.arange(
            self.config.draft_vocab_size,
            device=logits.device,
            dtype=torch.long,
        )
        target_ids = draft_ids + self.draft_id_to_target_id.to(logits.device)
        full_logits = logits.new_full(
            (logits.shape[0], self.config.vocab_size),
            float("-inf"),
        )
        valid = (target_ids >= 0) & (target_ids < self.config.vocab_size)
        full_logits[:, target_ids[valid]] = logits[:, valid]
        return full_logits


def _rotate_half(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    first, second = torch.chunk(tensor, 2, dim=-1)
    return torch.cat(
        (first * cos - second * sin, second * cos + first * sin),
        dim=-1,
    )
