# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from collections.abc import Sequence

import torch

from .cache import append_confirmed, cache_prefix, forward_with_cache
from .data import DistillationBatch, TrainObservation
from .loss import distillation_loss
from .qwen3_trainer import Qwen3Eagle3CpuTrainer
from .torch_eagle3 import TorchEagle3ForCausalLM

logger = logging.getLogger(__name__)


def qwen3_eagle3_distillation_step(
    trainer: Qwen3Eagle3CpuTrainer,
    observations: Sequence[TrainObservation],
) -> None:
    """S=1: batch fixed-hidden inputs, update, then append with new weights."""
    if len(observations) != 1:
        raise ValueError("Append-only CPU KV training requires S=1")
    with torch.inference_mode(False), torch.enable_grad():
        _step(trainer, observations[0])


def _step(trainer: Qwen3Eagle3CpuTrainer, observation: TrainObservation) -> None:
    model = trainer.model
    if not isinstance(model, TorchEagle3ForCausalLM):
        raise TypeError("KV distillation requires TorchEagle3ForCausalLM")
    if trainer.request_id not in (None, observation.request_id):
        raise ValueError("Reset CPU history before switching requests")
    if trainer.last_step_id is not None and observation.step_id <= trainer.last_step_id:
        raise ValueError("Duplicate or out-of-order observation")
    dtype = next(p.dtype for p in model.parameters() if p.is_floating_point())
    batch = DistillationBatch.from_payload(observation.payload, dtype)
    batch.validate(model.config.hidden_size, model.config.num_aux_hidden_states)
    positions = batch.prefill_positions
    embeds = batch.prefill_input_embeds
    auxiliary = batch.prefill_aux_hidden_states
    proposal_embeds = batch.proposal_input_embeds
    proposal_hidden = batch.proposal_hidden_states
    proposal_positions = batch.proposal_positions
    teacher = batch.teacher_probs
    confirmed_positions = batch.confirmed_positions
    confirmed_embeds = batch.confirmed_input_embeds
    confirmed_auxiliary = batch.confirmed_aux_hidden_states
    depth = proposal_positions.numel()
    if trainer.cache_length:
        if trainer.cache_length != int(positions[-1]) + 1:
            raise ValueError("CPU history does not end at the proposal anchor")
        # Recompute the boundary query with gradients, without replacing its
        # persistent KV. Earlier history remains a detached conditioning input.
        cache = cache_prefix(trainer.kv_cache, int(positions[-1]))
        replay_positions = positions[-1:]
        replay_embeds = embeds[-1:]
        replay_auxiliary = auxiliary[-1:]
    else:
        if int(positions[0]) != 0:
            raise ValueError("First observation must include the full prompt prefill")
        cache = ()
        replay_positions, replay_embeds, replay_auxiliary = positions, embeds, auxiliary

    trainer.zero_grad()
    with torch.profiler.record_function("online_eagle3.cpu_forward"):
        # Fixed GPU recurrent inputs allow one causal forward for the canvas.
        # Keep feature-fusion and within-canvas attention gradients on CPU.
        output, recurrent, cache = forward_with_cache(
            model,
            torch.cat((replay_positions, proposal_positions[1:])),
            torch.cat((replay_embeds, proposal_embeds[1:])),
            torch.cat(
                (model.combine_hidden_states(replay_auxiliary), proposal_hidden[1:])
            ),
            cache,
        )
        logits = model.compute_draft_logits(output[-depth:])
        loss = distillation_loss(logits, teacher)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite distillation loss")
    with torch.profiler.record_function("online_eagle3.cpu_backward"):
        loss.backward()
    if trainer.config.check_gradients:
        with torch.profiler.record_function("online_eagle3.cpu_check_gradients"):
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in model.parameters()
            ):
                raise ValueError("Nonfinite distillation gradient")
    trainer.step()
    trainer.last_loss = loss.detach().item()
    trainer.zero_grad()
    del cache, output, recurrent, logits, loss

    with torch.no_grad(), torch.profiler.record_function("online_eagle3.cpu_append_kv"):
        # Only newly confirmed target-feature inputs enter persistent history.
        # Both bootstrap and append run AFTER optimizer.step().
        history = append_confirmed(
            model, trainer.kv_cache, positions, embeds, auxiliary
        )
        history = append_confirmed(
            model, history, confirmed_positions, confirmed_embeds, confirmed_auxiliary
        )
    trainer.kv_cache = history
    trainer.request_id = observation.request_id
    trainer.last_step_id = observation.step_id
    logger.info(
        "S=1 CPU distillation: request=%s step=%d version=%d loss=%.6f "
        "canvas=%d cpu_kv_length=%d replay=fixed_hidden_parallel",
        observation.request_id,
        observation.step_id,
        trainer.version,
        trainer.last_loss,
        depth,
        trainer.cache_length,
    )
