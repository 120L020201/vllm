# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch
from online_draft.models.qwen3_eagle3 import (
    Eagle3KVCache,
    Qwen3Eagle3ForCausalLM,
)
from online_draft.training.eagle3_batch import (
    Eagle3DistillationBatch,
)
from online_draft.training.eagle3_cache import (
    PersistentEagle3KVCache,
    append_confirmed_to_persistent_cache,
    persistent_cache_length,
    prepare_training_prefix,
)
from online_draft.training.eagle3_distill import (
    forward_kl_loss,
)
from online_draft.training.eagle3_inputs import (
    prepare_eagle3_training_inputs,
)
from online_draft.training.trainer import DraftTrainer


@dataclass(frozen=True, slots=True)
class Eagle3StepResult:
    """Metrics produced after one complete EAGLE3 training step."""

    loss: float
    model_version: int
    draft_length: int
    confirmed_length: int
    persistent_cache_length: int


def train_eagle3_round(
    trainer: DraftTrainer,
    batch: Eagle3DistillationBatch,
    persistent_cache: PersistentEagle3KVCache,
) -> tuple[Eagle3KVCache, Eagle3StepResult]:
    """Train one parallel EAGLE3 canvas and append confirmed KV.


    Temporary canvas KV is discarded after backward. After the optimizer
    update, the confirmed path is forwarded under no_grad and appended to
    persistent request history.

    This function does not retain the batch or temporary forward tensors.

    Args:
        trainer: Owner of the CPU model, optimizer, and weight version.
        batch: Prepared CPU tensors for one verification round.
        persistent_cache: Detached request history before this round.

    Returns:
        Updated persistent KV and metrics for the completed step.

    Raises:
        TypeError: If the trainer does not own a Qwen3 EAGLE3 model.
        ValueError: If the batch, history, loss, or gradients are invalid.
    """
    model = trainer.model
    if not isinstance(
        model,
        Qwen3Eagle3ForCausalLM,
    ):
        raise TypeError("EAGLE3 training requires Qwen3Eagle3ForCausalLM")

    draft_length = batch.draft_length
    confirmed_length = batch.confirmed_length

    # The GPU runner may call the CPU trainer while an outer
    # torch.inference_mode context is active. Disable it explicitly so
    # the canvas can construct a normal autograd graph.
    with torch.inference_mode(False), torch.enable_grad():
        # This prefix never receives gradients. On the first round it is
        # built from prompt positions before the anchor. On later rounds
        # it is a temporary view of persistent history without the anchor.
        training_prefix = prepare_training_prefix(
            model,
            batch,
            persistent_cache,
        )

        # Row zero recomputes the anchor through the trainable FC layer.
        # Later rows use fixed recurrent hidden states captured on GPU.
        training_inputs = prepare_eagle3_training_inputs(
            model,
            batch,
        )

        # This is the only differentiable draft-model forward in the
        # round. Causal attention handles dependencies between every
        # parallel canvas position.
        output = model(
            positions=training_inputs.positions,
            input_embeds=training_inputs.input_embeds,
            hidden_states=training_inputs.hidden_states,
            past_key_values=training_prefix,
        )
        logits = model.compute_draft_logits(output.hidden_states)

        loss = forward_kl_loss(
            student_logits=logits,
            teacher_probabilities=(training_inputs.teacher_probabilities),
            rejection_position=(training_inputs.rejection_position),
        )

        loss_value = trainer.backward_and_step(loss)

        # output.past_key_values contains differentiable temporary canvas
        # KV. It must never replace persistent request history.
        del loss
        del logits
        del output
        del training_inputs
        del training_prefix

    # This happens only after optimizer.step(). It recomputes the true
    # confirmed path with updated weights and returns detached KV.
    updated_cache = append_confirmed_to_persistent_cache(
        model,
        batch,
        persistent_cache,
    )
    updated_cache_length = persistent_cache_length(updated_cache)

    result = Eagle3StepResult(
        loss=loss_value,
        model_version=trainer.version,
        draft_length=draft_length,
        confirmed_length=confirmed_length,
        persistent_cache_length=updated_cache_length,
    )

    return updated_cache, result
