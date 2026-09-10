# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch
import torch.nn as nn

from .config import Qwen3Eagle3TrainerConfig
from .torch_eagle3 import Eagle3KVCache
from .weights import (
    TrainableWeightSnapshot,
    export_trainable_state_dict,
    freeze_parameters,
    get_trainable_named_parameters,
    load_trainable_state_dict,
)


class Qwen3Eagle3CpuTrainer:
    """Minimal CPU-side weight manager for Qwen3 EAGLE3 draft updates."""

    def __init__(
        self,
        model: nn.Module,
        config: Qwen3Eagle3TrainerConfig | None = None,
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.AdamW,
    ) -> None:
        self.model = model
        self.config = config or Qwen3Eagle3TrainerConfig()
        self.frozen_names = freeze_parameters(self.model, self.config.frozen_prefixes)
        trainable_params = [
            parameter
            for _, parameter in get_trainable_named_parameters(
                self.model, self.config.frozen_prefixes
            )
        ]
        self.optimizer = optimizer_cls(
            trainable_params,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self._version = 0
        self.kv_cache: Eagle3KVCache = ()
        self.request_id: str | None = None
        self.last_step_id: int | None = None
        self.last_loss: float | None = None

    @property
    def cache_length(self) -> int:
        return self.kv_cache[0][0].shape[1] if self.kv_cache else 0

    def clear_request_state(self) -> None:
        self.kv_cache = ()
        self.request_id = None
        self.last_step_id = None
        self.last_loss = None

    @property
    def version(self) -> int:
        return self._version

    def snapshot(self) -> TrainableWeightSnapshot:
        with torch.profiler.record_function("online_eagle3.cpu_snapshot"):
            return TrainableWeightSnapshot(
                version=self._version,
                state_dict=export_trainable_state_dict(
                    self.model, self.config.frozen_prefixes
                ),
            )

    def load_snapshot(
        self,
        snapshot: TrainableWeightSnapshot,
        *,
        strict: bool = True,
    ) -> None:
        load_trainable_state_dict(
            self.model,
            snapshot.state_dict,
            self.config.frozen_prefixes,
            strict=strict,
        )
        self._version = snapshot.version

    def restore_snapshot(
        self,
        snapshot: TrainableWeightSnapshot,
        *,
        strict: bool = True,
    ) -> None:
        self.load_snapshot(snapshot, strict=strict)
        self.clear_optimizer_state()
        self.clear_request_state()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def clear_optimizer_state(self) -> None:
        self.optimizer.state.clear()
        self.zero_grad()

    def step(self) -> None:
        self.optimizer.step()
        self._version += 1

    def trainable_parameter_names(self) -> list[str]:
        return [
            name
            for name, _ in get_trainable_named_parameters(
                self.model, self.config.frozen_prefixes
            )
        ]
