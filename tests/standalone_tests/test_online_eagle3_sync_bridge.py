# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch
import torch.nn as nn

from online_eagle3.async_bridge import TrainObservation
from online_eagle3.qwen3_trainer import Qwen3Eagle3CpuTrainer
from online_eagle3.sync_bridge import (
    Qwen3Eagle3LazySyncBridge,
    Qwen3Eagle3SyncBridge,
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


def _train_step(
    trainer: Qwen3Eagle3CpuTrainer,
    observations: tuple[TrainObservation, ...],
) -> None:
    assert observations
    trainer.zero_grad()
    loss = trainer.model.model.fc.weight.sum()
    loss.backward()
    trainer.step()


def test_sync_bridge_applies_updates_and_resets() -> None:
    model = _ToyEagle3Module()
    trainer = Qwen3Eagle3CpuTrainer(model)
    bridge = Qwen3Eagle3SyncBridge(trainer, _train_step)

    baseline = trainer.snapshot()

    bridge.observe_step("req-1", 0)
    updated = bridge.maybe_apply_pending_weights()

    assert updated is not None
    assert updated.version == 1
    assert not torch.equal(
        updated.state_dict["model.fc.weight"],
        baseline.state_dict["model.fc.weight"],
    )

    bridge.reset_request("req-1")
    reset_snapshot = bridge.maybe_apply_pending_weights()

    assert reset_snapshot is not None
    assert reset_snapshot.version == baseline.version
    assert torch.equal(
        reset_snapshot.state_dict["model.fc.weight"],
        baseline.state_dict["model.fc.weight"],
    )
    assert torch.equal(
        reset_snapshot.state_dict["model.fc.bias"],
        baseline.state_dict["model.fc.bias"],
    )


def test_sync_bridge_waits_for_update_interval() -> None:
    model = _ToyEagle3Module()
    trainer = Qwen3Eagle3CpuTrainer(model)
    bridge = Qwen3Eagle3SyncBridge(trainer, _train_step, update_interval=2)

    bridge.observe_step("req-1", 0)
    assert bridge.maybe_apply_pending_weights() is None

    bridge.observe_step("req-1", 1)
    updated = bridge.maybe_apply_pending_weights()

    assert updated is not None
    assert updated.version == 1


def test_sync_bridge_reset_drops_pending_observations() -> None:
    model = _ToyEagle3Module()
    trainer = Qwen3Eagle3CpuTrainer(model)
    bridge = Qwen3Eagle3SyncBridge(trainer, _train_step, update_interval=2)

    baseline = trainer.snapshot()

    bridge.observe_step("req-1", 0)
    bridge.reset_request("req-1")
    reset_snapshot = bridge.maybe_apply_pending_weights()

    assert reset_snapshot is not None
    assert reset_snapshot.version == baseline.version
    assert torch.equal(
        reset_snapshot.state_dict["model.fc.weight"],
        baseline.state_dict["model.fc.weight"],
    )

    bridge.observe_step("req-1", 1)
    assert bridge.maybe_apply_pending_weights() is None


def test_lazy_sync_bridge_defers_loading_until_observation() -> None:
    load_count = 0

    def _make_bridge() -> Qwen3Eagle3SyncBridge:
        nonlocal load_count
        load_count += 1
        return Qwen3Eagle3SyncBridge(
            Qwen3Eagle3CpuTrainer(_ToyEagle3Module()),
            _train_step,
        )

    bridge = Qwen3Eagle3LazySyncBridge(_make_bridge)

    assert not bridge.is_loaded
    assert bridge.maybe_apply_pending_weights() is None
    bridge.reset_request("req-1")
    assert load_count == 0

    bridge.observe_step("req-1", 0)

    assert bridge.is_loaded
    assert load_count == 1
    assert bridge.maybe_apply_pending_weights() is not None
