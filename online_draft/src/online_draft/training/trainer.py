# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from online_draft.runtime import profile_range
from online_draft.training.master_adamw import (
    MasterWeightAdamW,
    supports_master_weight_adamw,
)
from online_draft.training.persistent_grads import install_persistent_weight_grads


@torch.compile(fullgraph=True)
def _gradients_are_finite(gradients: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([torch.isfinite(gradient).all() for gradient in gradients]).all()


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """Settings for one online draft trainer."""

    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    fused: bool = True
    frozen_parameter_prefixes: tuple[str, ...] = ()
    check_gradients: bool = False
    master_weights: bool = False
    lr_reference_rows: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if any(not prefix for prefix in self.frozen_parameter_prefixes):
            raise ValueError("frozen parameter prefixes must not be empty")
        if self.lr_reference_rows is not None and (
            not math.isfinite(self.lr_reference_rows) or self.lr_reference_rows <= 0
        ):
            raise ValueError("lr_reference_rows must be finite and positive")


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

        self._initial_weights = tuple(
            (name, parameter.detach().clone())
            for name, parameter in self._trainable_named_parameters
        )

        install_persistent_weight_grads(self.model, self.trainable_parameter_names)

        # The native optimizer keeps its FP32 masters in its own state; without
        # it, the optimizer updates FP32 copies of the parameters.
        native_master_weights = self.config.master_weights and (
            supports_master_weight_adamw(
                parameter for _, parameter in self._trainable_named_parameters
            )
        )
        self._copies_master_weights = (
            self.config.master_weights and not native_master_weights
        )
        self._optimizer_parameters = [
            parameter.detach().float().clone().requires_grad_()
            if self._copies_master_weights
            else parameter
            for _, parameter in self._trainable_named_parameters
        ]
        self.optimizer: torch.optim.Optimizer
        if native_master_weights:
            self.optimizer = MasterWeightAdamW(
                self._optimizer_parameters,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self._optimizer_parameters,
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

    @property
    def initial_weights(self) -> tuple[tuple[str, torch.Tensor], ...]:
        return self._initial_weights

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        if self._copies_master_weights:
            self.model.zero_grad(set_to_none=True)

    def clear_optimizer_state(self) -> None:
        self.optimizer.state.clear()
        self.zero_grad()

    @torch.no_grad()
    def reset(self) -> None:
        for (_, parameter), (_, initial_weight) in zip(
            self._trainable_named_parameters,
            self._initial_weights,
            strict=True,
        ):
            parameter.copy_(initial_weight)
        if self._copies_master_weights:
            for master, (_, initial) in zip(
                self._optimizer_parameters, self._initial_weights, strict=True
            ):
                master.copy_(initial)

        self.clear_optimizer_state()
        self._version = 0
        self.last_loss = None

    def backward_and_step(
        self,
        loss: torch.Tensor,
        *,
        supervised_rows: int | None = None,
        profile_timeline: bool = False,
    ) -> float:
        """Backpropagate a finite scalar loss and update the model.

        Args:
            loss: Scalar loss connected to the trainer's model.
            supervised_rows: Rows behind the loss. With ``lr_reference_rows``
                set, the step uses ``lr * min(1, rows / lr_reference_rows)``.
            profile_timeline: Whether to annotate backward and optimizer.

        Returns:
            The detached FP32 loss value.

        Raises:
            ValueError: If the loss is nonfinite or disconnected, or if a
                checked gradient is nonfinite.
        """
        self.zero_grad()

        if not loss.requires_grad:
            raise ValueError("loss must require gradients")
        if not torch.isfinite(loss.detach()):
            raise ValueError("loss must be finite")

        try:
            with profile_range("online_draft.backward", profile_timeline):
                loss.backward()

            if not any(
                parameter.grad is not None
                for _, parameter in self._trainable_named_parameters
            ):
                raise ValueError("loss is not connected to any trainable parameter")

            if self.config.check_gradients:
                gradients = [
                    parameter.grad
                    for _, parameter in self._trainable_named_parameters
                    if parameter.grad is not None
                ]
                if gradients and not _gradients_are_finite(gradients).item():
                    raise ValueError("nonfinite gradients")

            with profile_range("online_draft.optimizer", profile_timeline):
                lr_scale = self._lr_scale(supervised_rows)
                for group in self.optimizer.param_groups:
                    group["lr"] = self.config.learning_rate * lr_scale
                if self._copies_master_weights:
                    for master, (_, parameter) in zip(
                        self._optimizer_parameters,
                        self._trainable_named_parameters,
                        strict=True,
                    ):
                        master.grad = (
                            None if parameter.grad is None else parameter.grad.float()
                        )
                self.optimizer.step()
                if self._copies_master_weights:
                    with torch.no_grad():
                        for master, (_, parameter) in zip(
                            self._optimizer_parameters,
                            self._trainable_named_parameters,
                            strict=True,
                        ):
                            parameter.copy_(master)
        except Exception:
            self.zero_grad()
            raise

        loss_value = float(loss.detach().float().item())
        self._version += 1
        self.last_loss = loss_value
        self.zero_grad()

        return loss_value

    def _lr_scale(self, supervised_rows: int | None) -> float:
        reference_rows = self.config.lr_reference_rows
        if reference_rows is None or supervised_rows is None:
            return 1.0
        return min(1.0, supervised_rows / reference_rows)

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


def _matches_prefix(
    parameter_name: str,
    prefix: str,
) -> bool:
    return parameter_name == prefix or parameter_name.startswith(f"{prefix}.")
