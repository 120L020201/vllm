# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""TTS-owned EAGLE3 training window."""

from typing import cast

import torch

from methods.tts_common.loss import forward_kl_loss, window_forward_kl_loss
from methods.tts_common.training.eagle3.cache import (
    PersistentEagle3KVCache,
    _append_confirmed_window,
    persistent_cache_length,
)
from methods.tts_common.training.eagle3.draft_model import AutoregressiveDraftModel
from methods.tts_common.training.eagle3.inputs import (
    prepare_eagle3_confirmed_path_inputs,
    prepare_eagle3_proposal_window_inputs,
)
from methods.tts_common.training.eagle3.window import (
    Eagle3TrainingWindow,
    Eagle3WindowMode,
    Eagle3WindowResult,
)
from methods.tts_common.training.qwen3_eagle3 import Eagle3KVCache
from methods.tts_common.training.runtime import profile_range
from methods.tts_common.training.trainer import DraftTrainer

# EAGLE-3 weights its training-time-test step i by 0.8**i.
EAGLE3_POSITION_DECAY_RHO = 0.8


def _unrolled_proposal_logits(
    model: AutoregressiveDraftModel,
    window: Eagle3TrainingWindow,
    training_prefix: Eagle3KVCache,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Redraft each proposal row from the previous row's trainable output.

    Each round restarts from its anchor and sees only confirmed spine
    positions before the anchor plus its own earlier rows, as in the
    packed proposal-canvas mask. Unlike the packed forward, row k's
    hidden input is row k-1's differentiable output instead of the
    captured inference value, so gradients cross the draft recurrence.
    """
    spine_length = persistent_cache_length(training_prefix)
    logits: list[torch.Tensor] = []
    for round_batch in window.rounds:
        anchor_position = int(round_batch.proposal_positions[0].item())
        input_embeds = torch.cat(
            (
                round_batch.prefix_input_embeds[-1:],
                round_batch.draft_token_input_embeds,
            ),
            dim=0,
        )
        hidden_states = model.combine_hidden_states(
            round_batch.prefix_aux_hidden_states[-1:]
        )
        cache = training_prefix
        for depth in range(round_batch.draft_length):
            attention_mask = torch.ones(
                (1, spine_length + depth + 1),
                dtype=torch.bool,
                device=hidden_states.device,
            )
            attention_mask[:, :anchor_position] = False
            attention_mask[:, spine_length:] = False
            output = model(
                positions=round_batch.proposal_positions[depth : depth + 1],
                input_embeds=input_embeds[depth : depth + 1],
                hidden_states=hidden_states,
                past_key_values=cache,
                attention_mask=attention_mask,
            )
            cache = output.past_key_values
            hidden_states = output.recurrent_hidden_states
            logits.append(model.compute_draft_logits(output.hidden_states))
    teacher_probabilities = torch.cat(
        tuple(round_batch.teacher_probabilities for round_batch in window.rounds),
        dim=0,
    )
    return torch.cat(logits, dim=0), teacher_probabilities


def _proposal_depths(
    window: Eagle3TrainingWindow,
    mode: Eagle3WindowMode,
    *,
    device: torch.device,
) -> torch.Tensor:
    counts = tuple(
        (
            round_batch.draft_length
            if mode is Eagle3WindowMode.PROPOSAL_CANVAS
            or round_batch.rejection_position is None
            else round_batch.confirmed_length
        )
        for round_batch in window.rounds
    )
    return torch.cat(
        tuple(torch.arange(count, dtype=torch.long, device=device) for count in counts)
    )


def train_tts_window(
    trainer: DraftTrainer,
    window: Eagle3TrainingWindow,
    persistent_cache: PersistentEagle3KVCache,
    mode: Eagle3WindowMode,
    profile_timeline: bool = False,
    *,
    loss_mode: str = "uniform",
    position_decay_rho: float | None = None,
    recurrent_gradients: bool = False,
) -> tuple[Eagle3KVCache, Eagle3WindowResult]:
    """Train one TTS window independently of the online-draft trainer config.

    Args:
        position_decay_rho: Weight row k by rho**k, overriding loss_mode.
        recurrent_gradients: Unroll the proposal so gradients reach earlier
            rows through each row's recurrent hidden input.
    """
    model = cast(AutoregressiveDraftModel, trainer.model)
    confirmed_length = sum(
        round_batch.confirmed_length for round_batch in window.rounds
    )
    if recurrent_gradients and mode is not Eagle3WindowMode.PROPOSAL_CANVAS:
        raise ValueError("recurrent gradients require the proposal canvas")

    with torch.inference_mode(False), torch.enable_grad():
        with profile_range("tts.prepare_spine", profile_timeline):
            training_prefix = _append_confirmed_window(
                model,
                window,
                persistent_cache,
            )
            spine_length = persistent_cache_length(training_prefix)

        if recurrent_gradients:
            with profile_range("tts.forward", profile_timeline):
                logits, teacher_probabilities = _unrolled_proposal_logits(
                    model, window, training_prefix
                )
            input_length = logits.shape[0]
        else:
            with profile_range("tts.prepare_inputs", profile_timeline):
                if mode is Eagle3WindowMode.CONFIRMED_PATH:
                    training_inputs = prepare_eagle3_confirmed_path_inputs(
                        model,
                        window,
                        spine_length=spine_length,
                    )
                else:
                    training_inputs = prepare_eagle3_proposal_window_inputs(
                        model,
                        window,
                        spine_length=spine_length,
                    )

            with profile_range("tts.forward", profile_timeline):
                output = model(
                    positions=training_inputs.positions,
                    input_embeds=training_inputs.input_embeds,
                    hidden_states=training_inputs.hidden_states,
                    past_key_values=training_prefix,
                    attention_mask=training_inputs.attention_mask,
                )
                logits = model.compute_draft_logits(output.hidden_states)
            teacher_probabilities = training_inputs.teacher_probabilities
            input_length = training_inputs.input_length
            del output
            del training_inputs

        with profile_range("tts.loss", profile_timeline):
            proposal_depths = _proposal_depths(window, mode, device=logits.device)
            if position_decay_rho is None:
                loss = forward_kl_loss(
                    logits,
                    teacher_probabilities,
                    mode=loss_mode,
                    proposal_depths=proposal_depths,
                )
            else:
                loss = window_forward_kl_loss(
                    logits,
                    teacher_probabilities,
                    proposal_depths,
                    position_decay_rho,
                )

        target_count = teacher_probabilities.shape[0]
        loss_value = trainer.backward_and_step(
            loss,
            profile_timeline=profile_timeline,
        )

        del loss
        del logits
        del teacher_probabilities
        del training_prefix

    with profile_range("tts.rebuild_cache", profile_timeline):
        updated_cache = _append_confirmed_window(
            model,
            window,
            persistent_cache,
        )
        updated_cache_length = persistent_cache_length(updated_cache)

    return updated_cache, Eagle3WindowResult(
        loss=loss_value,
        model_version=trainer.version,
        mode=mode,
        round_count=window.round_count,
        input_length=input_length,
        target_count=target_count,
        confirmed_length=confirmed_length,
        persistent_cache_length=updated_cache_length,
        mean_acceptance_length=window.mean_acceptance_length,
    )
