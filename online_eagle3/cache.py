# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from .torch_eagle3 import Eagle3KVCache, TorchEagle3ForCausalLM


def forward_with_cache(
    model: TorchEagle3ForCausalLM,
    positions: torch.Tensor,
    embeds: torch.Tensor,
    hidden: torch.Tensor,
    cache: Eagle3KVCache,
) -> tuple[torch.Tensor, torch.Tensor, Eagle3KVCache]:
    output: list[tuple[torch.Tensor, torch.Tensor]] = []
    normalized, recurrent = model(
        input_ids=torch.zeros_like(positions),
        positions=positions,
        hidden_states=hidden,
        inputs_embeds=embeds,
        past_key_values=cache,
        kv_output=output,
    )
    return normalized, recurrent, tuple(output)


def cache_prefix(cache: Eagle3KVCache, length: int) -> Eagle3KVCache:
    return tuple((k[:, :length], v[:, :length]) for k, v in cache)


def append_confirmed(
    model: TorchEagle3ForCausalLM,
    cache: Eagle3KVCache,
    positions: torch.Tensor,
    embeds: torch.Tensor,
    auxiliary: torch.Tensor,
) -> Eagle3KVCache:
    length = cache[0][0].shape[1] if cache else 0
    keep = positions >= length
    positions, embeds, auxiliary = positions[keep], embeds[keep], auxiliary[keep]
    if positions.numel() == 0:
        return cache
    expected = torch.arange(length, length + positions.numel())
    if not torch.equal(positions, expected):
        raise ValueError("CPU history must append contiguous positions without gaps")
    _, _, result = forward_with_cache(
        model, positions, embeds, model.combine_hidden_states(auxiliary), cache
    )
    return tuple((k.detach(), v.detach()) for k, v in result)
