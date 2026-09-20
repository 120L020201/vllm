# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """Settings for one online draft trainer."""

    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    fused: bool = False
    frozen_parameter_prefixes: tuple[str, ...] = ()
    check_gradients: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if any(not prefix for prefix in self.frozen_parameter_prefixes):
            raise ValueError("frozen parameter prefixes must not be empty")


class DraftTrainer:
    """Own a trainable model, optimizer, gradients, and weight version."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or TrainerConfig()

        self.frozen_parameter_names = self._freeze_configured_parameters()
        self._trainable_named_parameters = tuple(
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        )

        if not self._trainable_named_parameters:
            raise ValueError("trainer requires at least one trainable parameter")

        self.optimizer = torch.optim.AdamW(
            [parameter for _, parameter in self._trainable_named_parameters],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            fused=self.config.fused,
        )

        self._version = 0
        self.last_loss: float | None = None
        self.model.train()

    @property
    def version(self) -> int:
        return self._version

    @property
    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self._trainable_named_parameters)

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def clear_optimizer_state(self) -> None:
        self.optimizer.state.clear()
        self.zero_grad()

    def backward_and_step(
        self,
        loss: torch.Tensor,
    ) -> float:
        """Backpropagate a finite scalar loss and update the model.

        Args:
            loss: Scalar loss connected to the trainer's model.

        Returns:
            The detached FP32 loss value.

        Raises:
            ValueError: If the loss is invalid or disconnected, or if a
                checked gradient is nonfinite.
        """
        self.zero_grad()

        if loss.ndim != 0:
            raise ValueError("loss must be a scalar")
        if not loss.requires_grad:
            raise ValueError("loss must require gradients")
        if not torch.isfinite(loss.detach()):
            raise ValueError("loss must be finite")

        try:
            loss.backward()

            if not any(
                parameter.grad is not None
                for _, parameter in self._trainable_named_parameters
            ):
                raise ValueError("loss is not connected to any trainable parameter")

            if self.config.check_gradients:
                nonfinite_names = self._nonfinite_gradient_names()
                if nonfinite_names:
                    raise ValueError(
                        "nonfinite gradients: " + ", ".join(nonfinite_names)
                    )

            self.optimizer.step()
        except Exception:
            self.zero_grad()
            raise

        loss_value = float(loss.detach().float().item())
        self._version += 1
        self.last_loss = loss_value
        self.zero_grad()

        return loss_value

    def _freeze_configured_parameters(
        self,
    ) -> tuple[str, ...]:
        frozen_names: list[str] = []

        for name, parameter in self.model.named_parameters():
            if any(
                _matches_prefix(name, prefix)
                for prefix in self.config.frozen_parameter_prefixes
            ):
                parameter.requires_grad_(False)
                frozen_names.append(name)

        return tuple(frozen_names)

    def _nonfinite_gradient_names(self) -> list[str]:
        return [
            name
            for name, parameter in self._trainable_named_parameters
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        ]


def _matches_prefix(
    parameter_name: str,
    prefix: str,
) -> bool:
    return parameter_name == prefix or parameter_name.startswith(f"{prefix}.")
