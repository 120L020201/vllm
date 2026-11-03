# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Gemma4 MTP window training over the frozen target's K/V.

The draft never writes K/V, so the persistent history is the captured target
K/V appended as is: no draft forward builds or rebuilds it. Every draft step
of a round runs at the round's anchor position, as on the GPU.
"""

from dataclasses import dataclass
from typing import cast

import torch
from online_draft.models.gemma4_mtp import Gemma4MTPForCausalLM, TargetKV
from online_draft.runtime import profile_range
from online_draft.training.eagle3.batch import Eagle3DistillationBatch
from online_draft.training.eagle3.distill import window_forward_kl_loss
from online_draft.training.trainer import DraftTrainer


@dataclass(frozen=True)
class Gemma4MTPBatch:
    """One EAGLE3-layout round plus the target K/V it adds to history.

    ``kv`` covers the prompt and confirmed rows in the first round, then only
    the confirmed rows, one tensor per shared target layer.
    """

    round: Eagle3DistillationBatch
    kv: TargetKV


@dataclass(frozen=True)
class Gemma4MTPWindowResult:
    model_version: int
    loss: float
    round_count: int
    target_count: int
    persistent_cache_length: int


def _query_rows(batch: Eagle3DistillationBatch):
    """Anchor row plus draft rows whose targets are confirmed."""
    count = (
        batch.draft_length
        if batch.rejection_position is None
        else batch.confirmed_length
    )
    embeds = torch.cat((batch.prefix_input_embeds[-1:], batch.draft_token_input_embeds))
    hidden = torch.cat(
        (batch.prefix_aux_hidden_states[-1:], batch.draft_input_hidden_states)
    )
    rejected = torch.zeros(count, dtype=torch.bool)
    if batch.rejection_position is not None:
        rejected[batch.rejection_position] = True
    positions = batch.prefix_positions[-1:].expand(count)
    return (
        positions,
        embeds[:count],
        hidden[:count],
        batch.teacher_probabilities[:count],
        rejected,
    )


def train_gemma4_mtp_window(
    trainer: DraftTrainer,
    rounds: tuple[Gemma4MTPBatch, ...],
    cache: TargetKV | None,
    profile_timeline: bool = False,
    *,
    max_train_rounds: int = 0,
) -> tuple[TargetKV, Gemma4MTPWindowResult]:
    """Append the window's target K/V, then train the last rounds once."""
    model = cast(Gemma4MTPForCausalLM, trainer.model)
    parts = [batch.kv for batch in rounds]
    if cache is not None:
        parts.insert(0, cache)
    history = tuple(torch.cat(layer) for layer in zip(*parts, strict=True))

    selected = rounds[-max_train_rounds:] if max_train_rounds else rounds
    positions, embeds, hidden, teacher, rejected = (
        torch.cat(column)
        for column in zip(*(_query_rows(batch.round) for batch in selected))
    )
    with torch.inference_mode(False), torch.enable_grad():
        with profile_range("online_draft.forward", profile_timeline):
            draft_hidden, _ = model(positions, embeds, hidden, history)
            logits = model.lm_head(draft_hidden)
        with profile_range("online_draft.loss", profile_timeline):
            loss = window_forward_kl_loss(logits, teacher, rejected)
        loss_value = trainer.backward_and_step(loss, profile_timeline=profile_timeline)
        del loss, logits, draft_hidden

    return history, Gemma4MTPWindowResult(
        model_version=trainer.version,
        loss=loss_value,
        round_count=len(selected),
        target_count=teacher.shape[0],
        persistent_cache_length=history[0].shape[0],
    )
