# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Training helpers shared by online EAGLE3 policy adapters."""

from collections.abc import Iterable

from online_draft.models.qwen3_eagle3 import Eagle3KVCache
from online_draft.training.eagle3_batch import Eagle3DistillationBatch
from online_draft.training.eagle3_cache import PersistentEagle3KVCache
from online_draft.training.eagle3_window import (
    Eagle3TrainingWindow,
    Eagle3WindowMode,
    train_eagle3_window,
)
from online_draft.training.trainer import DraftTrainer


def train_batches(
    trainer: DraftTrainer,
    batches: Iterable[Eagle3DistillationBatch],
    *,
    persistent_cache: PersistentEagle3KVCache = None,
) -> tuple[Eagle3KVCache | None, tuple[float, ...]]:
    """Train consecutive batches and return their final cache and losses."""
    cache = persistent_cache
    losses: list[float] = []
    for batch in batches:
        cache, result = train_eagle3_window(
            trainer=trainer,
            window=Eagle3TrainingWindow(rounds=(batch,)),
            persistent_cache=cache,
            mode=Eagle3WindowMode.CONFIRMED_PATH,
        )
        losses.append(result.loss)
    return cache, tuple(losses)
