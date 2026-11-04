# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy

import online_draft.training.trainer as trainer_module
import pytest
import torch
import torch.nn as nn
from online_draft.training.master_adamw import MasterWeightAdamW
from online_draft.training.persistent_grads import (
    PersistentGradLinear,
    install_persistent_weight_grads,
)
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


def test_master_weights_accumulate_small_updates_and_reset_between_requests():
    """Updates below one BF16 ULP must accumulate in the FP32 master."""
    model = nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        model.weight.fill_(1)
    trainer = DraftTrainer(
        model,
        TrainerConfig(
            learning_rate=1e-4, fused=True, master_weights=True, check_gradients=False
        ),
    )
    for _ in range(30):
        trainer.backward_and_step(model.weight.float().sum())
    assert model.weight.item() < 1
    assert model.weight.grad is None
    state = next(iter(trainer.optimizer.state.values()))
    assert state["exp_avg"].dtype == torch.float32
    assert state["exp_avg_sq"].dtype == torch.float32
    trainer.reset()
    assert model.weight.item() == 1
    assert not trainer.optimizer.state
    trainer.backward_and_step(model.weight.float().sum())
    assert model.weight.item() == 1


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
        TrainerConfig(learning_rate=1e-2, fused=False),
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
        TrainerConfig(learning_rate=0.1, check_gradients=True),
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


def test_reset_restores_initial_training_state() -> None:
    torch.manual_seed(5)

    model = TinyModel()
    trainer = DraftTrainer(
        model,
        TrainerConfig(
            learning_rate=0.1,
            frozen_parameter_prefixes=("frozen",),
        ),
    )
    initial_parameters = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }

    inputs = torch.randn(4, 3)
    loss = model(inputs).float().square().mean()
    trainer.backward_and_step(loss)

    assert trainer.version == 1
    assert trainer.last_loss is not None
    assert trainer.optimizer.state

    trainer.reset()

    assert trainer.version == 0
    assert trainer.last_loss is None
    assert not trainer.optimizer.state
    assert all(parameter.grad is None for parameter in model.parameters())
    assert tuple(name for name, _ in trainer.initial_weights) == (
        "trainable.weight",
        "trainable.bias",
    )

    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, initial_parameters[name])


@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
def test_master_adamw_tracks_fp32_adamw(weight_decay: float) -> None:
    """FP32 masters follow FP32 AdamW on the same BF16 gradients."""
    torch.manual_seed(6)
    reference = nn.Linear(64, 33)
    model = nn.Linear(64, 33).to(dtype=torch.bfloat16)
    reference.load_state_dict(
        {name: value.float() for name, value in model.state_dict().items()}
    )
    reference_optimizer = torch.optim.AdamW(
        reference.parameters(), lr=1e-2, weight_decay=weight_decay
    )
    optimizer = MasterWeightAdamW(
        model.parameters(), lr=1e-2, weight_decay=weight_decay
    )

    for _ in range(3):
        for parameter in model.parameters():
            parameter.grad = torch.randn_like(parameter)
        for expected, parameter in zip(reference.parameters(), model.parameters()):
            expected.grad = parameter.grad.float()
        reference_optimizer.step()
        optimizer.step()

    for expected, parameter in zip(reference.parameters(), model.parameters()):
        master = optimizer.state[parameter]["master"]
        torch.testing.assert_close(master, expected, rtol=1e-5, atol=1e-6)
        assert torch.equal(parameter.detach(), master.to(torch.bfloat16))
        assert optimizer.state[parameter]["step"] == 3


def test_master_adamw_rejects_non_bf16_parameters() -> None:
    with pytest.raises(ValueError, match="contiguous CPU BF16"):
        MasterWeightAdamW(nn.Linear(2, 2).parameters(), lr=1e-3)


def test_master_weights_fall_back_to_fp32_copies_without_native_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both master-weight implementations take the same FP32 AdamW steps."""
    torch.manual_seed(10)
    model = nn.Linear(64, 33, dtype=torch.bfloat16)
    fallback_model = copy.deepcopy(model)
    config = TrainerConfig(
        learning_rate=1e-3, master_weights=True, check_gradients=False
    )
    trainer = DraftTrainer(model, config)
    monkeypatch.setattr(
        trainer_module, "supports_master_weight_adamw", lambda parameters: False
    )
    fallback = DraftTrainer(fallback_model, config)
    assert isinstance(trainer.optimizer, MasterWeightAdamW)
    assert type(fallback.optimizer) is torch.optim.AdamW

    for _ in range(3):
        inputs = torch.randn(8, 64, dtype=torch.bfloat16)
        trainer.backward_and_step(model(inputs).float().square().mean())
        fallback.backward_and_step(fallback_model(inputs).float().square().mean())

    for actual, expected in zip(model.parameters(), fallback_model.parameters()):
        torch.testing.assert_close(actual, expected)


def test_persistent_weight_grads_match_autograd_and_reuse_storage() -> None:
    """Buffers allocate on first backward, so installing in inference mode works."""
    torch.manual_seed(8)
    reference = nn.Sequential(
        nn.Linear(4, 4, bias=False),
        nn.ReLU(),
        nn.Linear(4, 2, bias=False),
    )
    model = copy.deepcopy(reference)
    with torch.inference_mode():
        patched = install_persistent_weight_grads(
            model, (name for name, _ in model.named_parameters())
        )
    assert patched == ("0.weight", "2.weight")

    storage = None
    for _ in range(2):
        inputs = torch.randn(5, 4, requires_grad=True)
        reference_inputs = inputs.detach().clone().requires_grad_(True)
        model(inputs).square().sum().backward()
        reference(reference_inputs).square().sum().backward()
        torch.testing.assert_close(inputs.grad, reference_inputs.grad)
        for (name, actual), expected in zip(
            model.named_parameters(), reference.parameters()
        ):
            torch.testing.assert_close(actual.grad, expected.grad, msg=name)
        pointers = model[2].weight.grad.data_ptr()
        assert storage in (None, pointers)
        storage = pointers
        for module in (model, reference):
            module.zero_grad(set_to_none=True)


def test_trainer_step_matches_plain_adamw_with_persistent_grads() -> None:
    torch.manual_seed(9)
    reference_model = nn.Sequential(
        nn.Linear(3, 4, bias=False), nn.ReLU(), nn.Linear(4, 2, bias=False)
    )
    model = copy.deepcopy(reference_model)
    reference_optimizer = torch.optim.AdamW(
        reference_model.parameters(), lr=1e-2, weight_decay=0.0
    )
    trainer = DraftTrainer(
        model, TrainerConfig(check_gradients=False, learning_rate=1e-2)
    )
    assert isinstance(model[0], PersistentGradLinear)

    for _ in range(2):
        inputs = torch.randn(4, 3)
        trainer.backward_and_step(model(inputs).square().mean())
        reference_optimizer.zero_grad(set_to_none=True)
        reference_model(inputs).square().mean().backward()
        reference_optimizer.step()
    for actual, expected in zip(model.parameters(), reference_model.parameters()):
        torch.testing.assert_close(actual, expected)
        assert actual.grad is None


@pytest.mark.parametrize(
    ("reference_rows", "rows", "expected_scale"),
    [(None, 2, 1.0), (4.0, 2, 0.5), (4.0, 8, 1.0), (4.0, None, 1.0)],
)
def test_lr_reference_rows_scale_small_windows(
    reference_rows: float | None, rows: int | None, expected_scale: float
) -> None:
    """Adam's first step moves each weight by lr, so the move shows the scale."""
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)
    trainer = DraftTrainer(
        model,
        TrainerConfig(
            learning_rate=1e-2,
            check_gradients=False,
            lr_reference_rows=reference_rows,
        ),
    )

    trainer.backward_and_step(model.weight.sum(), supervised_rows=rows)

    torch.testing.assert_close(
        model.weight.detach(), torch.full((1, 1), 1 - 1e-2 * expected_scale)
    )
