# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from enum import Enum
from typing import cast

import torch

from methods.tts_common.training.eagle3.batch import Eagle3DistillationBatch
from methods.tts_common.training.eagle3.cache import (
    PersistentEagle3KVCache,
    _append_confirmed_window,
    persistent_cache_length,
)
from methods.tts_common.training.eagle3.distill import (
    window_forward_kl_loss,
)
from methods.tts_common.training.eagle3.draft_model import AutoregressiveDraftModel
from methods.tts_common.training.eagle3.inputs import (
    prepare_eagle3_confirmed_path_inputs,
    prepare_eagle3_proposal_window_inputs,
)
from methods.tts_common.training.qwen3_eagle3 import Eagle3KVCache
from methods.tts_common.training.runtime import profile_range
from methods.tts_common.training.trainer import DraftTrainer


class Eagle3WindowMode(str, Enum):
    """Supported EAGLE3 window training objectives."""

    CONFIRMED_PATH = "confirmed_path"
    PROPOSAL_CANVAS = "proposal_canvas"

    __str__ = str.__str__


@dataclass(frozen=True, slots=True)
class Eagle3TrainingWindow:
    """Consecutive verification rounds updated by one optimizer step.

    The runtime decides when enough rounds have been collected; each round
    anchors at the previous round's final confirmed position.
    """

    rounds: tuple[Eagle3DistillationBatch, ...]

    @property
    def round_count(self) -> int:
        """Return the number of verification rounds."""
        return len(self.rounds)

    @property
    def start_anchor_position(self) -> int:
        """Return the anchor position of the first round."""
        return int(self.rounds[0].proposal_positions[0].item())

    @property
    def end_confirmed_position(self) -> int:
        """Return the final confirmed position of the window."""
        return int(self.rounds[-1].confirmed_positions[-1].item())

    @property
    def proposal_target_count(self) -> int:
        """Return the number of proposal-canvas loss positions."""
        return sum(round_batch.draft_length for round_batch in self.rounds)

    @property
    def confirmed_path_target_count(self) -> int:
        """Return the confirmed-path loss count, excluding bonus targets."""
        return sum(
            (
                round_batch.draft_length
                if round_batch.rejection_position is None
                else round_batch.confirmed_length
            )
            for round_batch in self.rounds
        )

    @property
    def mean_acceptance_length(self) -> float:
        """Return one plus the mean number of accepted draft tokens."""
        accepted_tokens = sum(
            (
                round_batch.draft_length
                if round_batch.rejection_position is None
                else round_batch.rejection_position
            )
            for round_batch in self.rounds
        )
        return 1 + accepted_tokens / self.round_count


@dataclass(frozen=True, slots=True)
class Eagle3WindowResult:
    """Metrics produced by one window-level optimizer update."""

    loss: float
    model_version: int
    mode: Eagle3WindowMode
    round_count: int
    input_length: int
    target_count: int
    confirmed_length: int
    persistent_cache_length: int
    mean_acceptance_length: float


def train_eagle3_rounds(
    trainer: DraftTrainer,
    rounds: tuple[Eagle3DistillationBatch, ...],
    persistent_cache: PersistentEagle3KVCache,
    profile_timeline: bool = False,
    *,
    mode: Eagle3WindowMode = Eagle3WindowMode.CONFIRMED_PATH,
) -> tuple[Eagle3KVCache, Eagle3WindowResult]:
    """Train the rounds selected by the shared asynchronous pipeline."""
    return train_eagle3_window(
        trainer, Eagle3TrainingWindow(rounds), persistent_cache, mode, profile_timeline
    )


def train_eagle3_window(
    trainer: DraftTrainer,
    window: Eagle3TrainingWindow,
    persistent_cache: PersistentEagle3KVCache,
    mode: Eagle3WindowMode,
    profile_timeline: bool = False,
) -> tuple[Eagle3KVCache, Eagle3WindowResult]:
    """Train one EAGLE3 window with one optimizer update.

    Both modes perform one differentiable packed forward and one optimizer
    step behind an explicit per-round isolation mask. Confirmed-path keeps
    only query rows with confirmed targets; proposal-canvas keeps every
    draft row. Temporary differentiable KV and the pre-update spine are
    discarded before updated confirmed KV is built.

    Args:
        trainer: Owner of the CPU model, optimizer, and weight version.
        window: Consecutive verification rounds in this update.
        persistent_cache: Detached history through the first anchor, or
            None for the first window.
        mode: Window input and loss organization strategy.
        profile_timeline: Whether to annotate the training stages.

    Returns:
        Updated persistent KV and metrics for the completed update.

    Raises:
        ValueError: If the loss or gradients are invalid (see DraftTrainer).
    """
    model = cast(AutoregressiveDraftModel, trainer.model)

    confirmed_length = sum(
        round_batch.confirmed_length for round_batch in window.rounds
    )

    with torch.inference_mode(False), torch.enable_grad():
        with profile_range("online_draft.prepare_spine", profile_timeline):
            # Pre-update weights; discarded after the forward. Branch masks
            # expose only positions before each anchor.
            training_prefix = _append_confirmed_window(
                model,
                window,
                persistent_cache,
            )
            spine_length = persistent_cache_length(training_prefix)

        with profile_range("online_draft.prepare_inputs", profile_timeline):
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

        with profile_range("online_draft.forward", profile_timeline):
            output = model(
                positions=training_inputs.positions,
                input_embeds=training_inputs.input_embeds,
                hidden_states=training_inputs.hidden_states,
                past_key_values=training_prefix,
                attention_mask=training_inputs.attention_mask,
            )
            hidden_states_for_loss = output.hidden_states
            logits = model.compute_draft_logits(hidden_states_for_loss)

        with profile_range("online_draft.loss", profile_timeline):
            loss = window_forward_kl_loss(
                student_logits=logits,
                teacher_probabilities=training_inputs.teacher_probabilities,
                rejection_target_mask=training_inputs.rejection_target_mask,
            )

        input_length = training_inputs.input_length
        target_count = training_inputs.target_count
        loss_value = trainer.backward_and_step(
            loss,
            profile_timeline=profile_timeline,
        )

        del loss
        del logits
        del hidden_states_for_loss
        del output
        del training_inputs
        del training_prefix

    with profile_range("online_draft.rebuild_cache", profile_timeline):
        updated_cache = _append_confirmed_window(
            model,
            window,
            persistent_cache,
        )
        updated_cache_length = persistent_cache_length(updated_cache)

    result = Eagle3WindowResult(
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

    return updated_cache, result
