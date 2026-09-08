from __future__ import annotations

import torch
import torch.nn as nn

from online_eagle3.qwen3_trainer import Qwen3Eagle3CpuTrainer
from online_eagle3.weights import (
    export_trainable_state_dict,
    freeze_parameters,
    load_trainable_state_dict,
)


class _ToyEagle3Module(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(8, 4)
        self.model.fc = nn.Linear(4, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.draft_id_to_target_id = nn.Parameter(
            torch.zeros(8, dtype=torch.long), requires_grad=False
        )
        self.mask_hidden = nn.Parameter(torch.ones(1, 4), requires_grad=False)


def test_freeze_and_export_trainable_state() -> None:
    model = _ToyEagle3Module()

    frozen_names = freeze_parameters(model)
    assert set(frozen_names) == {
        "model.embed_tokens.weight",
        "lm_head.weight",
        "draft_id_to_target_id",
        "mask_hidden",
    }

    trainable_state = export_trainable_state_dict(model)
    assert set(trainable_state) == {"model.fc.weight", "model.fc.bias"}

    for name, parameter in model.named_parameters():
        if name in frozen_names:
            assert not parameter.requires_grad
        else:
            assert parameter.requires_grad


def test_load_trainable_state_only_restores_trainable_weights() -> None:
    model = _ToyEagle3Module()
    freeze_parameters(model)
    snapshot = export_trainable_state_dict(model)
    fc_weight = model.model.fc.weight
    fc_bias = model.model.fc.bias
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.ones_like(parameter))

    load_trainable_state_dict(model, snapshot)

    assert model.model.fc.weight is fc_weight
    assert model.model.fc.bias is fc_bias
    assert torch.allclose(model.model.fc.weight, snapshot["model.fc.weight"])
    assert torch.allclose(model.model.fc.bias, snapshot["model.fc.bias"])
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            expected = frozen_before[name] + torch.ones_like(parameter)
            if parameter.is_floating_point():
                assert torch.allclose(parameter, expected)
            else:
                assert torch.equal(parameter, expected)


def test_cpu_trainer_tracks_version_and_trainable_names() -> None:
    model = _ToyEagle3Module()
    trainer = Qwen3Eagle3CpuTrainer(model)

    assert trainer.version == 0
    assert trainer.trainable_parameter_names() == ["model.fc.weight", "model.fc.bias"]

    snapshot = trainer.snapshot()
    assert snapshot.version == 0
    assert set(snapshot.state_dict) == {"model.fc.weight", "model.fc.bias"}

    trainer.step()
    assert trainer.version == 1
