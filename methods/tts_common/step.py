# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Training helpers shared by online EAGLE3 policy adapters."""

from collections.abc import Iterable

from methods.tts_common.batch import Eagle3DistillationBatch
from methods.tts_common.training.eagle3.cache import PersistentEagle3KVCache
from methods.tts_common.training.eagle3.window import (
    Eagle3TrainingWindow,
    Eagle3WindowMode,
)
from methods.tts_common.training.qwen3_eagle3 import Eagle3KVCache
from methods.tts_common.training.trainer import DraftTrainer
from methods.tts_common.window import train_tts_window


def train_batches(
    trainer: DraftTrainer,
    batches: Iterable[Eagle3DistillationBatch],
    *,
    persistent_cache: PersistentEagle3KVCache = None,
    mode: Eagle3WindowMode = Eagle3WindowMode.CONFIRMED_PATH,
) -> tuple[Eagle3KVCache | None, tuple[float, ...]]:
    """Train each batch with one update and return its final cache and losses."""
    cache = persistent_cache
    losses: list[float] = []
    for batch in batches:
        cache, result = train_tts_window(
            trainer=trainer,
            window=Eagle3TrainingWindow(rounds=(batch,)),
            persistent_cache=cache,
            mode=mode,
        )
        losses.append(result.loss)
    return cache, tuple(losses)
