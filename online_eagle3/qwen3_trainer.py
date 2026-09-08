# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .weights import (
    QWEN3_EAGLE3_FROZEN_PREFIXES,
    TrainableWeightSnapshot,
    export_trainable_state_dict,
    freeze_parameters,
    get_trainable_named_parameters,
    load_trainable_state_dict,
)


@dataclass(slots=True)
class Qwen3Eagle3TrainerConfig:
    frozen_prefixes: tuple[str, ...] = QWEN3_EAGLE3_FROZEN_PREFIXES
    lr: float = 1e-5
    weight_decay: float = 0.0


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
        with torch.profiler.record_function("online_eagle3.cpu_load_snapshot"):
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

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def clear_optimizer_state(self) -> None:
        with torch.profiler.record_function("online_eagle3.cpu_clear_optimizer_state"):
            self.optimizer.state.clear()
            self.zero_grad()

    def step(self) -> None:
        with torch.profiler.record_function("online_eagle3.cpu_trainer_step"):
            self.optimizer.step()
            self._version += 1

    def trainable_parameter_names(self) -> list[str]:
        return [
            name
            for name, _ in get_trainable_named_parameters(
                self.model, self.config.frozen_prefixes
            )
        ]
