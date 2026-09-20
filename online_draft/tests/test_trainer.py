# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
import torch.nn as nn
from online_draft.training.trainer import (
    DraftTrainer,
    TrainerConfig,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trainable = nn.Linear(3, 2)
        self.frozen = nn.Linear(3, 2)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return self.trainable(inputs) + self.frozen(inputs)


def test_successful_step_updates_only_trainable_parameters() -> None:
    torch.manual_seed(0)

    model = TinyModel()
    trainer = DraftTrainer(
        model,
        TrainerConfig(
            learning_rate=0.1,
            frozen_parameter_prefixes=("frozen",),
        ),
    )

    old_trainable_weight = model.trainable.weight.detach().clone()
    old_frozen_weight = model.frozen.weight.detach().clone()
    old_frozen_bias = model.frozen.bias.detach().clone()

    inputs = torch.randn(4, 3)
    loss = model(inputs).float().square().mean()
    loss_value = trainer.backward_and_step(loss)

    assert trainer.version == 1
    assert trainer.last_loss == loss_value
    assert trainer.frozen_parameter_names == (
        "frozen.weight",
        "frozen.bias",
    )
    assert trainer.trainable_parameter_names == (
        "trainable.weight",
        "trainable.bias",
    )

    assert not torch.equal(
        model.trainable.weight,
        old_trainable_weight,
    )
    assert torch.equal(
        model.frozen.weight,
        old_frozen_weight,
    )
    assert torch.equal(
        model.frozen.bias,
        old_frozen_bias,
    )

    assert all(parameter.grad is None for parameter in model.parameters())


def test_bfloat16_optimizer_state() -> None:
    torch.manual_seed(1)

    model = nn.Linear(
        3,
        2,
        bias=False,
    ).to(dtype=torch.bfloat16)
    trainer = DraftTrainer(
        model,
        TrainerConfig(learning_rate=1e-2),
    )

    inputs = torch.randn(
        4,
        3,
        dtype=torch.bfloat16,
    )
    loss = model(inputs).float().square().mean()

    trainer.backward_and_step(loss)

    state = trainer.optimizer.state[model.weight]

    assert model.weight.dtype == torch.bfloat16
    assert state["exp_avg"].dtype == torch.bfloat16
    assert state["exp_avg_sq"].dtype == torch.bfloat16
    assert state["step"].dtype == torch.float32


def test_fused_bfloat16_optimizer_state() -> None:
    torch.manual_seed(2)

    model = nn.Linear(
        3,
        2,
        bias=False,
    ).to(dtype=torch.bfloat16)
    trainer = DraftTrainer(
        model,
        TrainerConfig(
            learning_rate=1e-2,
            fused=True,
        ),
    )

    inputs = torch.randn(
        4,
        3,
        dtype=torch.bfloat16,
    )
    loss = model(inputs).float().square().mean()

    trainer.backward_and_step(loss)

    state = trainer.optimizer.state[model.weight]

    assert trainer.optimizer.defaults["fused"] is True
    assert trainer.version == 1
    assert model.weight.dtype == torch.bfloat16
    assert state["exp_avg"].dtype == torch.bfloat16
    assert state["exp_avg_sq"].dtype == torch.bfloat16
    assert state["step"].dtype == torch.float32


def test_nonfinite_loss_does_not_update_model() -> None:
    torch.manual_seed(3)

    model = nn.Linear(3, 2)
    trainer = DraftTrainer(
        model,
        TrainerConfig(learning_rate=0.1),
    )
    old_weight = model.weight.detach().clone()

    inputs = torch.randn(4, 3)
    loss = model(inputs).sum() * torch.tensor(float("nan"))

    with pytest.raises(
        ValueError,
        match="loss must be finite",
    ):
        trainer.backward_and_step(loss)

    assert trainer.version == 0
    assert trainer.last_loss is None
    assert torch.equal(model.weight, old_weight)
    assert not trainer.optimizer.state
    assert model.weight.grad is None


def test_nonfinite_gradient_does_not_update_model() -> None:
    torch.manual_seed(4)

    model = nn.Linear(3, 2)
    trainer = DraftTrainer(
        model,
        TrainerConfig(learning_rate=0.1),
    )
    old_weight = model.weight.detach().clone()

    hook = model.weight.register_hook(
        lambda gradient: torch.full_like(
            gradient,
            float("nan"),
        )
    )

    inputs = torch.randn(4, 3)
    loss = model(inputs).float().square().mean()

    with pytest.raises(
        ValueError,
        match="nonfinite gradients",
    ):
        trainer.backward_and_step(loss)

    hook.remove()

    assert trainer.version == 0
    assert trainer.last_loss is None
    assert torch.equal(model.weight, old_weight)
    assert not trainer.optimizer.state
    assert model.weight.grad is None


def test_unrelated_loss_does_not_advance_version() -> None:
    model = nn.Linear(2, 2)
    trainer = DraftTrainer(model)
    loss = torch.tensor(1.0, requires_grad=True)

    with pytest.raises(
        ValueError,
        match="not connected",
    ):
        trainer.backward_and_step(loss)

    assert trainer.version == 0
    assert trainer.last_loss is None
    assert not trainer.optimizer.state
    assert all(parameter.grad is None for parameter in model.parameters())
