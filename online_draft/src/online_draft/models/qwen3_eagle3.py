# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from safetensors.torch import load_file

from online_draft.models.layers import (
    VLLM_ATTENTION_STATE_SUFFIXES,
    AttentionBackend,
    DraftForwardOutput,
    GatedMLP,
    LayerKV,
    RMSNorm,
    RotaryEmbedding,
    cached_attention,
    copy_vllm_rotary,
)


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
    attention_backend: AttentionBackend = AttentionBackend.EAGER
    # Leading head channels that RoPE rotates; None rotates all of them.
    rotary_dim: int | None = None
    # vLLM pads the draft head to a multiple of 64 rows; logits drop the padding.
    padded_draft_vocab_size: int | None = None

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
        rope_parameters = values.get("rope_parameters") or {}
        rope_theta = values.get("rope_theta") or rope_parameters.get(
            "rope_theta", 10000.0
        )
        partial_rotary_factor = values.get(
            "partial_rotary_factor"
        ) or rope_parameters.get("partial_rotary_factor", 1.0)
        rotary_dim = int(head_dim * partial_rotary_factor)

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
            rope_theta=float(rope_theta),
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
            attention_backend=_parse_attention_backend(
                values.get(
                    "attention_backend",
                    AttentionBackend.EAGER,
                )
            ),
            rotary_dim=None if rotary_dim == head_dim else rotary_dim,
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


def _parse_attention_backend(
    value: object,
) -> AttentionBackend:
    if isinstance(value, AttentionBackend):
        return value
    if not isinstance(value, str):
        raise TypeError("attention_backend must be a string")

    try:
        return AttentionBackend(value)
    except ValueError as error:
        supported = ", ".join(backend.value for backend in AttentionBackend)
        raise ValueError(f"attention_backend must be one of: {supported}") from error


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
        self.attention_backend = config.attention_backend

        query_size = self.num_attention_heads * self.head_dim
        key_value_size = self.num_key_value_heads * self.head_dim
        qkv_output_size = query_size + 2 * key_value_size

        self.qkv_proj = nn.Linear(
            input_size,
            qkv_output_size,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            query_size,
            self.hidden_size,
            bias=False,
        )
        self.rotary_emb = RotaryEmbedding(
            config.rotary_dim or config.head_dim, config.rope_theta
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        past_key_value: LayerKV | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, LayerKV]:
        sequence_length = hidden_states.shape[0]
        query_size = self.num_attention_heads * self.head_dim
        key_value_size = self.num_key_value_heads * self.head_dim

        query, key, value = self.qkv_proj(hidden_states).split(
            (query_size, key_value_size, key_value_size),
            dim=-1,
        )
        query = query.view(
            sequence_length, self.num_attention_heads, self.head_dim
        ).transpose(0, 1)
        key = key.view(
            sequence_length, self.num_key_value_heads, self.head_dim
        ).transpose(0, 1)
        value = value.view(
            sequence_length, self.num_key_value_heads, self.head_dim
        ).transpose(0, 1)

        query, key = self.rotary_emb(query, key, positions)
        attention_output, present_key_value = cached_attention(
            query,
            key,
            value,
            past_key_value,
            attention_mask,
            self.attention_backend,
        )
        attention_output = attention_output.transpose(0, 1).reshape(
            sequence_length, query_size
        )

        return self.o_proj(attention_output), present_key_value


class Qwen3Eagle3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Eagle3Config,
        layer_index: int,
    ) -> None:
        super().__init__()

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
        self.mlp = GatedMLP(config.hidden_size, config.intermediate_size)

        self.hidden_norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

    def forward(
        self,
        positions: torch.Tensor,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        past_key_value: LayerKV | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        LayerKV,
    ]:
        if self.layer_index == 0:
            normalized_embeds = cast(torch.Tensor, self.input_layernorm(input_embeds))
            hidden_states, residual = self._normalize_first_layer_hidden_states(
                hidden_states
            )
            hidden_states = torch.cat(
                (normalized_embeds, hidden_states),
                dim=-1,
            )
        else:
            hidden_states, residual = cast(
                tuple[torch.Tensor, torch.Tensor],
                self.input_layernorm(hidden_states, residual),
            )

        attention_output, present_key_value = self.self_attn(
            hidden_states,
            positions,
            past_key_value,
            attention_mask,
        )

        hidden_states, residual = cast(
            tuple[torch.Tensor, torch.Tensor],
            self.post_attention_layernorm(attention_output, residual),
        )

        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual, present_key_value

    def _normalize_first_layer_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = cast(torch.Tensor, self.hidden_norm(hidden_states))
        residual = normalized if self.norm_before_residual else hidden_states

        return normalized, residual


Eagle3KVCache = tuple[LayerKV, ...]


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
            self.input_norm: RMSNorm | None = RMSNorm(
                self.fc_input_size,
                config.rms_norm_eps,
            )
        else:
            self.input_norm = None

        if config.fc_norm:
            self.fc_norm: nn.ModuleList | None = nn.ModuleList(
                [
                    RMSNorm(
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

        self.norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

    def combine_hidden_states(
        self,
        auxiliary_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = auxiliary_hidden_states

        if self.input_norm is not None:
            hidden_states = cast(torch.Tensor, self.input_norm(hidden_states))

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
                normalized_chunks.append(norm(chunk))

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
        attention_mask: torch.Tensor | None = None,
    ) -> DraftForwardOutput:
        residual: torch.Tensor | None = None
        present_key_values: list[LayerKV] = []

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
                attention_mask=attention_mask,
            )
            present_key_values.append(present_key_value)

        normalized_hidden_states, hidden_states_before_norm = cast(
            tuple[torch.Tensor, torch.Tensor], self.norm(hidden_states, residual)
        )

        if self.norm_output:
            recurrent_hidden_states = normalized_hidden_states
        else:
            recurrent_hidden_states = hidden_states_before_norm

        return DraftForwardOutput(
            hidden_states=normalized_hidden_states,
            recurrent_hidden_states=recurrent_hidden_states,
            past_key_values=tuple(present_key_values),
        )


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
            config.padded_draft_vocab_size or config.draft_vocab_size,
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
        attention_mask: torch.Tensor | None = None,
    ) -> DraftForwardOutput:
        return self.model(
            positions=positions,
            input_embeds=input_embeds,
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
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
        return self.lm_head(hidden_states)[..., : self.config.draft_vocab_size]

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


def load_qwen3_eagle3_from_vllm(
    gpu_model: nn.Module,
    *,
    attention_backend: AttentionBackend | str | None = None,
) -> Qwen3Eagle3ForCausalLM:
    """Copy the loaded vLLM EAGLE3 draft into a trainable CPU model.

    Structure, norm, and RoPE settings are read from the GPU modules rather
    than config.json, so the CPU model computes what the GPU draft computes.
    Settings this CPU model does not implement are rejected.

    Args:
        gpu_model: Loaded ``Eagle3LlamaForCausalLM`` draft.
        attention_backend: CPU attention implementation; defaults to eager.

    Returns:
        A CPU model holding copies of the draft-owned GPU weights.

    Raises:
        ValueError: If the draft uses a setting the CPU model does not support,
            or its weights do not match the CPU parameter layout.
    """
    config = gpu_model.config
    draft = gpu_model.model
    layers = draft.layers
    attention = layers[0].self_attn
    rotary = attention.rotary_emb

    unsupported = {
        "aux hidden states disabled": not draft.use_aux_hidden_state,
        "parallel drafting": getattr(gpu_model, "use_parallel_drafting", False),
        "logit scale": gpu_model.logits_processor.scale != 1.0,
        "non-NeoX rotary": not rotary.is_neox_style,
        "attention scaling": attention.scaling != attention.head_dim**-0.5,
        "missing draft-to-target map": gpu_model.draft_id_to_target_id is None,
    }
    if any(unsupported.values()):
        names = ", ".join(name for name, found in unsupported.items() if found)
        raise ValueError(f"online EAGLE3 CPU model does not support: {names}")

    cpu_config = Qwen3Eagle3Config(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_attention_heads=attention.total_num_heads,
        num_key_value_heads=attention.total_num_kv_heads,
        head_dim=attention.head_dim,
        num_hidden_layers=len(layers),
        target_vocab_size=config.vocab_size,
        draft_vocab_size=config.draft_vocab_size,
        rms_norm_eps=layers[0].input_layernorm.variance_epsilon,
        rope_theta=float(rotary.base),
        attention_bias=attention.qkv_proj.bias is not None,
        norm_before_residual=bool(getattr(config, "norm_before_residual", False)),
        norm_before_fc=draft.norm_before_fc,
        norm_output=draft.norm_output,
        num_aux_hidden_states=draft.num_aux_hidden_states,
        fc_norm=draft.fc_norm is not None,
        attention_backend=(
            AttentionBackend.EAGER
            if attention_backend is None
            else _parse_attention_backend(attention_backend)
        ),
        rotary_dim=(
            None if rotary.rotary_dim == attention.head_dim else rotary.rotary_dim
        ),
        padded_draft_vocab_size=(
            None
            if gpu_model.lm_head.weight.shape[0] == config.draft_vocab_size
            else gpu_model.lm_head.weight.shape[0]
        ),
    )
    with torch.inference_mode(False), torch.device("meta"):
        model = Qwen3Eagle3ForCausalLM(cpu_config)

    source = gpu_model.state_dict()
    # Embeddings are shared with the target; KV-cache quantization scales are
    # runtime attention state. Neither is a trainable draft weight.
    draft_owned = {
        name
        for name in source
        if name != "model.embed_tokens.weight"
        and not name.endswith(VLLM_ATTENTION_STATE_SUFFIXES)
    }
    if draft_owned != set(model.state_dict()):
        missing = sorted(set(model.state_dict()) - draft_owned)
        extra = sorted(draft_owned - set(model.state_dict()))
        raise ValueError(
            f"EAGLE3 draft weights do not match the CPU layout: "
            f"missing={missing}, unsupported={extra}"
        )

    with torch.inference_mode(False):
        weights = {
            name: source[name].detach().to(device="cpu", copy=True)
            for name in model.state_dict()
        }
        model.load_state_dict(weights, assign=True)
        copy_vllm_rotary(model, rotary)
    return model


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
    attention_backend: AttentionBackend | str | None = None,
) -> Qwen3Eagle3ForCausalLM:
    """Load an AngelSlim Qwen3 EAGLE3 checkpoint from disk.

    Used by the methods harness. Online training copies the loaded vLLM
    draft with load_qwen3_eagle3_from_vllm instead.

    Args:
        model_directory: Directory containing config.json and model weights.
        dtype: Floating-point dtype used by the loaded model.
        attention_backend: Optional runtime override for the attention
            implementation stored in config.json.

    Returns:
        A model with converted checkpoint weights.
    """
    model_path = Path(model_directory)
    config_path = model_path / "config.json"
    safetensors_path = model_path / "model.safetensors"

    with config_path.open(encoding="utf-8") as config_file:
        raw_config = json.load(config_file)

    if not isinstance(raw_config, Mapping):
        raise TypeError("config.json must contain a JSON object")

    config = Qwen3Eagle3Config.from_dict(raw_config)
    if attention_backend is not None:
        config = replace(
            config,
            attention_backend=_parse_attention_backend(attention_backend),
        )

    model = Qwen3Eagle3ForCausalLM(config)
    model.to(dtype=dtype)

    if safetensors_path.is_file():
        source_state_dict = load_file(safetensors_path, device="cpu")
    else:
        source_state_dict = torch.load(
            model_path / "pytorch_model.bin",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    if not isinstance(source_state_dict, Mapping):
        raise TypeError("checkpoint must contain a state dictionary")

    converted_state_dict = convert_angelslim_eagle3_state_dict(source_state_dict)
    model.load_state_dict(converted_state_dict, strict=True)

    return model
