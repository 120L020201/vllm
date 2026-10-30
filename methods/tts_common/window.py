# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""TTS-owned EAGLE3 training window."""

from typing import cast

import torch
from online_draft.models.qwen3_eagle3 import Eagle3KVCache
from online_draft.runtime import profile_range
from online_draft.training.eagle3.cache import (
    PersistentEagle3KVCache,
    _append_confirmed_window,
    persistent_cache_length,
)
from online_draft.training.eagle3.draft_model import AutoregressiveDraftModel
from online_draft.training.eagle3.inputs import (
    prepare_eagle3_confirmed_path_inputs,
    prepare_eagle3_proposal_window_inputs,
)
from online_draft.training.eagle3.window import (
    Eagle3TrainingWindow,
    Eagle3WindowMode,
    Eagle3WindowResult,
)
from online_draft.training.trainer import DraftTrainer

from methods.tts_common.loss import forward_kl_loss


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
) -> tuple[Eagle3KVCache, Eagle3WindowResult]:
    """Train one TTS window independently of the online-draft trainer config."""
    model = cast(AutoregressiveDraftModel, trainer.model)
    confirmed_length = sum(
        round_batch.confirmed_length for round_batch in window.rounds
    )

    with torch.inference_mode(False), torch.enable_grad():
        with profile_range("tts.prepare_spine", profile_timeline):
            training_prefix = _append_confirmed_window(
                model,
                window,
                persistent_cache,
            )
            spine_length = persistent_cache_length(training_prefix)

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

        with profile_range("tts.loss", profile_timeline):
            loss = forward_kl_loss(
                logits,
                training_inputs.teacher_probabilities,
                mode=loss_mode,
                proposal_depths=_proposal_depths(
                    window,
                    mode,
                    device=logits.device,
                ),
            )

        input_length = training_inputs.input_length
        target_count = training_inputs.target_count
        loss_value = trainer.backward_and_step(
            loss,
            profile_timeline=profile_timeline,
        )

        del loss
        del logits
        del output
        del training_inputs
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
