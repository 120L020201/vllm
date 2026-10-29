# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import cast

import torch
from online_draft.models.qwen3_dflash import ContextKV, Qwen3DFlashForCausalLM
from online_draft.runtime import profile_range
from online_draft.training.dflash.batch import DFlashBatch
from online_draft.training.dflash.distill import dflash_kl_loss, position_weights
from online_draft.training.trainer import DraftTrainer


@dataclass(frozen=True)
class DFlashCache:
    layers: ContextKV
    # Exclusive ends and CPU projection weight versions.
    segments: tuple[tuple[int, int], ...]

    @property
    def length(self) -> int:
        return self.layers[0][0].shape[1]


@dataclass(frozen=True)
class DFlashWindowResult:
    model_version: int
    loss: float
    round_count: int
    target_count: int
    persistent_cache_length: int


def append_context(prefix: ContextKV | None, suffix: ContextKV) -> ContextKV:
    if prefix is None:
        return suffix
    return tuple(
        (torch.cat((pk, sk), dim=1), torch.cat((pv, sv), dim=1))
        for (pk, pv), (sk, sv) in zip(prefix, suffix, strict=True)
    )


def _first_query_round(rounds: tuple[DFlashBatch, ...], max_train_rounds: int) -> int:
    """Rounds before this index only contribute context, not query blocks."""
    return max(0, len(rounds) - max_train_rounds) if max_train_rounds else 0


def forward_dflash_window(
    model: Qwen3DFlashForCausalLM,
    rounds: tuple[DFlashBatch, ...],
    cache: DFlashCache | None,
    max_train_rounds: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return aligned logits and teacher probabilities for selected full blocks."""
    old_length = 0 if cache is None else cache.length
    context = None if cache is None else cache.layers
    new_features = []
    teachers, selected_rows = [], []
    anchors, block_ids = [], []
    context_length = old_length
    query_length = 0
    first_query = _first_query_round(rounds, max_train_rounds)
    for index, batch in enumerate(rounds):
        features = batch.prompt_fc if index == 0 else rounds[index - 1].confirmed_fc
        if features is not None:
            context_length += features.shape[0]
            new_features.append(features)
        if index < first_query:
            continue
        query_count = batch.input_ids.numel()
        anchors.extend([batch.anchor] * query_count)
        block_ids.extend([index] * query_count)
        count = batch.target_count
        selected_rows.append(torch.arange(query_length + 1, query_length + count + 1))
        query_length += query_count
        teachers.append(batch.teacher_probabilities)

    if new_features:
        context = append_context(
            context,
            model.project_context(
                torch.cat(new_features), torch.arange(old_length, context_length)
            ),
        )
    # The first round carries the prompt, so context is never empty here.
    context = cast(ContextKV, context)
    blocks = torch.tensor(block_ids)
    attention_mask = torch.cat(
        (
            torch.arange(context_length)[None, :] < torch.tensor(anchors)[:, None],
            blocks[:, None] == blocks[None, :],
        ),
        dim=1,
    )
    rows = torch.cat(selected_rows)
    hidden = model(
        torch.cat([batch.input_ids for batch in rounds[first_query:]]),
        torch.cat([batch.positions for batch in rounds[first_query:]]),
        context,
        attention_mask,
        block_size=rounds[0].input_ids.numel(),
        output_rows=rows if model.config.selective_last_layer else None,
    )
    logits = model.lm_head(
        hidden if model.config.selective_last_layer else hidden[rows]
    )
    return logits, torch.cat(teachers)


def train_dflash_window(
    trainer: DraftTrainer,
    rounds: tuple[DFlashBatch, ...],
    cache: DFlashCache | None,
    profile_timeline: bool = False,
    *,
    max_train_rounds: int = 0,
) -> tuple[DFlashCache, DFlashWindowResult]:
    """Train recent complete blocks; append all confirmed target context KV."""
    model = cast(Qwen3DFlashForCausalLM, trainer.model)
    old_length = 0 if cache is None else cache.length
    selected = rounds[_first_query_round(rounds, max_train_rounds) :]
    with torch.inference_mode(False), torch.enable_grad():
        with profile_range("online_draft.forward", profile_timeline):
            logits, teacher = forward_dflash_window(
                model, rounds, cache, max_train_rounds
            )
        with profile_range("online_draft.loss", profile_timeline):
            loss = dflash_kl_loss(logits, teacher, position_weights(selected))
        loss_value = trainer.backward_and_step(loss, profile_timeline=profile_timeline)
        del loss, logits, teacher

    new_features = [batch.confirmed_fc for batch in rounds]
    if rounds[0].prompt_fc is not None:
        new_features.insert(0, rounds[0].prompt_fc)
    features = torch.cat(new_features)
    positions = torch.arange(old_length, old_length + features.shape[0])
    with (
        torch.inference_mode(False),
        torch.no_grad(),
        profile_range("online_draft.rebuild_cache", profile_timeline),
    ):
        layers = append_context(
            None if cache is None else cache.layers,
            model.project_context(features, positions),
        )
    end = old_length + features.shape[0]
    segments = (() if cache is None else cache.segments) + ((end, trainer.version),)
    updated = DFlashCache(layers, segments)
    return updated, DFlashWindowResult(
        model_version=trainer.version,
        loss=loss_value,
        round_count=len(selected),
        target_count=sum(batch.target_count for batch in selected),
        persistent_cache_length=end,
    )
