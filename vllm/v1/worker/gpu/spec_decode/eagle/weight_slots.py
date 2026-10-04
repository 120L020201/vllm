# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(slots=True)
class Eagle3WeightSlots:
    """Own two GPU copies of draft-only parameters."""

    model: nn.Module
    slots: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]

    @classmethod
    def from_models(
        cls,
        target_model: nn.Module,
        draft_model: nn.Module,
    ) -> "Eagle3WeightSlots":
        target_parameter_ids = {
            id(parameter) for parameter in target_model.parameters()
        }

        slot_a = {
            name: parameter.detach()
            for name, parameter in draft_model.named_parameters()
            if id(parameter) not in target_parameter_ids
        }

        slot_b = {name: torch.empty_like(tensor) for name, tensor in slot_a.items()}

        for name, tensor in slot_a.items():
            slot_b[name].copy_(tensor)

        return cls(
            model=draft_model,
            slots=(slot_a, slot_b),
        )

    @property
    def owned_names(self) -> tuple[str, ...]:
        return tuple(self.slots[0])

    def bind(self, weights: Mapping[str, torch.Tensor]) -> None:
        parameters = dict(self.model.named_parameters())

        for name, tensor in weights.items():
            parameters[name].data = tensor
