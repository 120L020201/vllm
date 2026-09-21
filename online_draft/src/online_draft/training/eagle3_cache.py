# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch
from online_draft.models.qwen3_eagle3 import (
    Eagle3KVCache,
    Qwen3Eagle3ForCausalLM,
)
from online_draft.training.eagle3_batch import (
    Eagle3DistillationBatch,
)

if TYPE_CHECKING:
    from online_draft.training.eagle3_window import (
        Eagle3TrainingWindow,
    )

PersistentEagle3KVCache = Eagle3KVCache | None


def persistent_cache_length(
    cache: PersistentEagle3KVCache,
) -> int:
    """Return the number of positions in a detached persistent cache.

    Args:
        cache: Persistent cache, or None when history is empty.

    Returns:
        The shared sequence length of every cache layer.

    Raises:
        ValueError: If the cache is malformed, differentiable, or an
            inference tensor.
    """
    if cache is None:
        return 0

    if not cache:
        raise ValueError("persistent cache must contain at least one layer")

    cache_length: int | None = None

    for key, value in cache:
        if key.ndim != 3 or value.ndim != 3:
            raise ValueError("persistent key and value must be three-dimensional")

        if key.shape != value.shape:
            raise ValueError("persistent key and value shapes must match")

        if (
            key.requires_grad
            or value.requires_grad
            or key.grad_fn is not None
            or value.grad_fn is not None
        ):
            raise ValueError("persistent KV cache must be detached")

        if torch.is_inference(key) or torch.is_inference(value):
            raise ValueError("persistent KV cache must not use inference tensors")

        layer_length = key.shape[1]

        if cache_length is None:
            cache_length = layer_length
        elif layer_length != cache_length:
            raise ValueError("all persistent cache layers must have one length")

    if cache_length is None:
        raise RuntimeError("persistent cache length could not be determined")

    return cache_length


def prepare_training_prefix(
    model: Qwen3Eagle3ForCausalLM,
    batch: Eagle3DistillationBatch,
    persistent_cache: PersistentEagle3KVCache,
) -> PersistentEagle3KVCache:
    """Prepare the detached prefix used by a training canvas.

    On the first round, positions before the anchor are forwarded under
    no_grad to construct a temporary prefix. On later rounds, the prefix
    is a detached view of persistent history ending immediately before
    the anchor.

    The anchor itself is never present in the returned prefix because it
    must be recomputed by the differentiable canvas forward.

    Args:
        model: CPU EAGLE3 model before the current optimizer update.
        batch: Current verification-round tensors.
        persistent_cache: Detached request history.

    Returns:
        A detached cache containing positions before the anchor.

    Raises:
        ValueError: If history does not align with the current anchor.
    """
    _validate_batch(model, batch)

    history_length = persistent_cache_length(persistent_cache)
    anchor_position = int(batch.proposal_positions[0].item())

    if history_length == 0:
        first_position = int(batch.prefill_positions[0].item())
        if first_position != 0:
            raise ValueError("first observation must contain the full prompt")

        # The initial prompt prefix is conditioning context only. It is
        # created without gradients and discarded after the canvas step.
        prefix_cache = _append_fixed_inputs(
            model=model,
            cache=None,
            positions=batch.prefill_positions[:-1],
            input_embeds=batch.prefill_input_embeds[:-1],
            auxiliary_hidden_states=(batch.prefill_aux_hidden_states[:-1]),
        )

        if persistent_cache_length(prefix_cache) != anchor_position:
            raise RuntimeError("initial prefix cache does not end before the anchor")

        return prefix_cache

    expected_history_length = anchor_position + 1
    if history_length != expected_history_length:
        raise ValueError("persistent history must end at the proposal anchor")

    # Persistent history already contains the anchor. Exclude that last
    # entry so the canvas can recompute it with gradients exactly once.
    return _slice_cache_prefix(
        persistent_cache,
        prefix_length=anchor_position,
    )


def append_confirmed_to_persistent_cache(
    model: Qwen3Eagle3ForCausalLM,
    batch: Eagle3DistillationBatch,
    persistent_cache: PersistentEagle3KVCache,
) -> Eagle3KVCache:
    """Append the confirmed path with updated model weights.

    This function must be called after the optimizer update. It never
    reuses temporary canvas KV. Missing prefill positions are appended on
    the first round, followed by every confirmed token, including the
    final correction or bonus token.

    All appended KV entries are built under no_grad and detached before
    becoming persistent request history.

    Args:
        model: CPU EAGLE3 model after the current optimizer update.
        batch: Current verification-round tensors.
        persistent_cache: Detached history from earlier rounds.

    Returns:
        Detached persistent history through the final confirmed token.

    Raises:
        ValueError: If the existing history or appended positions are
            inconsistent.
    """
    _validate_batch(model, batch)

    history_length = persistent_cache_length(persistent_cache)
    anchor_position = int(batch.proposal_positions[0].item())

    if history_length not in (
        0,
        anchor_position + 1,
    ):
        raise ValueError("persistent history must be empty or end at the anchor")

    # On the first round this builds the complete prompt with updated
    # weights. On later rounds every prefill position is already cached,
    # so no model forward is performed for this call.
    updated_cache = _append_fixed_inputs(
        model=model,
        cache=persistent_cache,
        positions=batch.prefill_positions,
        input_embeds=batch.prefill_input_embeds,
        auxiliary_hidden_states=(batch.prefill_aux_hidden_states),
    )

    # Every true-path token is rebuilt with updated weights. This includes
    # accepted draft tokens and the final correction or bonus token.
    updated_cache = _append_fixed_inputs(
        model=model,
        cache=updated_cache,
        positions=batch.confirmed_positions,
        input_embeds=batch.confirmed_input_embeds,
        auxiliary_hidden_states=(batch.confirmed_aux_hidden_states),
    )

    expected_length = int(batch.confirmed_positions[-1].item()) + 1
    actual_length = persistent_cache_length(updated_cache)

    if actual_length != expected_length:
        raise RuntimeError(
            "persistent history does not end at the final confirmed token"
        )

    if updated_cache is None:
        raise RuntimeError("confirmed append did not create a cache")

    return updated_cache


def append_confirmed_window_to_persistent_cache(
    model: Qwen3Eagle3ForCausalLM,
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
    for round_batch in window.rounds:
        _validate_batch(model, round_batch)

    first_round = window.rounds[0]
    history_length = persistent_cache_length(persistent_cache)
    first_anchor_position = window.start_anchor_position

    if history_length not in (
        0,
        first_anchor_position + 1,
    ):
        raise ValueError("persistent history must be empty or end at the first anchor")

    position_parts: list[torch.Tensor] = []
    input_embed_parts: list[torch.Tensor] = []
    auxiliary_hidden_state_parts: list[torch.Tensor] = []

    if history_length == 0:
        first_position = int(first_round.prefill_positions[0].item())
        if first_position != 0:
            raise ValueError("first observation must contain the full prompt")

        position_parts.append(first_round.prefill_positions)
        input_embed_parts.append(first_round.prefill_input_embeds)
        auxiliary_hidden_state_parts.append(first_round.prefill_aux_hidden_states)

    for round_batch in window.rounds:
        position_parts.append(round_batch.confirmed_positions)
        input_embed_parts.append(round_batch.confirmed_input_embeds)
        auxiliary_hidden_state_parts.append(round_batch.confirmed_aux_hidden_states)

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

    expected_length = window.end_confirmed_position + 1
    if persistent_cache_length(updated_cache) != expected_length:
        raise RuntimeError(
            "persistent history does not end at the window's final confirmed token"
        )

    if updated_cache is None:
        raise RuntimeError("confirmed window append did not create a cache")

    return updated_cache


def prepare_training_window_spine(
    model: Qwen3Eagle3ForCausalLM,
    window: "Eagle3TrainingWindow",
    persistent_cache: PersistentEagle3KVCache,
) -> Eagle3KVCache:
    """Build detached confirmed history for packed training branches.

    The current pre-update weights build every confirmed position in the
    window. The resulting cache is temporary: both window objectives use
    branch masks that expose only positions before each branch anchor, and
    the cache is discarded after the differentiable forward.

    Args:
        model: CPU EAGLE3 model before the window optimizer update.
        window: Consecutive verification rounds in the pending update.
        persistent_cache: Detached history through the first anchor, or
            None for the first window.

    Returns:
        Detached confirmed history through the window's final token.

    Raises:
        ValueError: If history, batches, or positions are inconsistent.
    """
    return append_confirmed_window_to_persistent_cache(
        model,
        window,
        persistent_cache,
    )


def _append_fixed_inputs(
    *,
    model: Qwen3Eagle3ForCausalLM,
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

    expected_positions = torch.arange(
        current_length,
        current_length + new_positions.numel(),
        dtype=torch.long,
        device=new_positions.device,
    )
    if not torch.equal(
        new_positions,
        expected_positions,
    ):
        raise ValueError("persistent KV must append contiguous positions")

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

    result = _detach_cache(output.past_key_values)
    expected_length = current_length + new_positions.numel()

    if persistent_cache_length(result) != expected_length:
        raise RuntimeError("model returned an invalid persistent cache length")

    return result


def _slice_cache_prefix(
    cache: PersistentEagle3KVCache,
    *,
    prefix_length: int,
) -> PersistentEagle3KVCache:
    cache_length = persistent_cache_length(cache)

    if not 0 <= prefix_length <= cache_length:
        raise ValueError("cache prefix length is outside persistent history")

    if prefix_length == 0:
        return None

    if cache is None:
        raise RuntimeError("nonempty prefix requires a persistent cache")

    return tuple(
        (
            key[:, :prefix_length].detach(),
            value[:, :prefix_length].detach(),
        )
        for key, value in cache
    )


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


def _validate_batch(
    model: Qwen3Eagle3ForCausalLM,
    batch: Eagle3DistillationBatch,
) -> None:
    batch.validate(
        hidden_size=model.config.hidden_size,
        num_aux_hidden_states=(model.config.num_aux_hidden_states),
        draft_vocab_size=model.config.draft_vocab_size,
        feature_dtype=batch.prefill_input_embeds.dtype,
    )
