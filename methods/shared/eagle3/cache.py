# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, cast

import torch

from methods.shared.eagle3.batch import validate_eagle3_rounds
from methods.shared.eagle3.draft_model import AutoregressiveDraftModel
from methods.shared.qwen3_eagle3 import Eagle3KVCache

if TYPE_CHECKING:
    from methods.shared.eagle3.window import (
        Eagle3TrainingWindow,
    )

PersistentEagle3KVCache = Eagle3KVCache | None


def persistent_cache_length(
    cache: PersistentEagle3KVCache,
) -> int:
    """Return the number of positions in a persistent cache."""
    return 0 if cache is None else cache[0][0].shape[1]


def append_confirmed_window_to_persistent_cache(
    model: AutoregressiveDraftModel,
    window: "Eagle3TrainingWindow",
    persistent_cache: PersistentEagle3KVCache,
) -> Eagle3KVCache:
    """Append a window's confirmed path under updated model weights.

    Existing persistent KV remains fixed. When history is empty, the
    first round must contain the full prompt. All new prompt and confirmed
    positions are submitted in one no-grad model forward.

    Args:
        model: CPU EAGLE3 model after the optimizer update.
        window: Consecutive verification rounds in the completed update.
        persistent_cache: Detached history through the first anchor, or
            None for the first window.

    Returns:
        Detached persistent history through the window's final confirmed
        token.

    Raises:
        ValueError: If history, batches, or positions are inconsistent.
    """
    validate_eagle3_rounds(model, window.rounds)
    return _append_confirmed_window(model, window, persistent_cache)


def _append_confirmed_window(
    model: AutoregressiveDraftModel,
    window: "Eagle3TrainingWindow",
    persistent_cache: PersistentEagle3KVCache,
) -> Eagle3KVCache:
    """Append the window's confirmed rows, and the prompt to an empty history."""
    first_round = window.rounds[0]

    position_parts: list[torch.Tensor] = []
    input_embed_parts: list[torch.Tensor] = []
    auxiliary_hidden_state_parts: list[torch.Tensor] = []

    if persistent_cache is None:
        position_parts.append(first_round.prefix_positions)
        input_embed_parts.append(first_round.prefix_input_embeds)
        auxiliary_hidden_state_parts.append(first_round.prefix_aux_hidden_states)

    for round_batch in window.rounds:
        position_parts.append(round_batch.confirmed_positions)
        input_embed_parts.append(round_batch.confirmed_input_embeds)
        auxiliary_hidden_state_parts.append(round_batch.confirmed_aux_hidden_states)

    # Every round has confirmed rows, so the append always yields a cache.
    updated_cache = _append_fixed_inputs(
        model=model,
        cache=persistent_cache,
        positions=torch.cat(position_parts, dim=0),
        input_embeds=torch.cat(input_embed_parts, dim=0),
        auxiliary_hidden_states=torch.cat(
            auxiliary_hidden_state_parts,
            dim=0,
        ),
    )
    return cast(Eagle3KVCache, updated_cache)


def _append_fixed_inputs(
    *,
    model: AutoregressiveDraftModel,
    cache: PersistentEagle3KVCache,
    positions: torch.Tensor,
    input_embeds: torch.Tensor,
    auxiliary_hidden_states: torch.Tensor,
) -> PersistentEagle3KVCache:
    current_length = persistent_cache_length(cache)

    keep = positions >= current_length
    new_positions = positions[keep]

    if new_positions.numel() == 0:
        return cache

    new_input_embeds = input_embeds[keep]
    new_auxiliary_hidden_states = auxiliary_hidden_states[keep]

    # Disable an outer inference_mode context because inference tensors
    # cannot safely condition the later differentiable canvas. no_grad
    # gives ordinary detached tensors that autograd may read as constants.
    with torch.inference_mode(False), torch.no_grad():
        new_hidden_states = model.combine_hidden_states(new_auxiliary_hidden_states)
        output = model(
            positions=new_positions,
            input_embeds=new_input_embeds,
            hidden_states=new_hidden_states,
            past_key_values=cache,
        )

    return _detach_cache(output.past_key_values)


def _detach_cache(
    cache: Eagle3KVCache,
) -> Eagle3KVCache:
    return tuple(
        (
            key.detach(),
            value.detach(),
        )
        for key, value in cache
    )
