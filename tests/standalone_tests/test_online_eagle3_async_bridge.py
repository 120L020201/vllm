from __future__ import annotations

import time

import torch
import torch.nn as nn

from online_eagle3.async_bridge import Qwen3Eagle3AsyncBridge, TrainObservation
from online_eagle3.qwen3_trainer import Qwen3Eagle3CpuTrainer


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


def _wait_for_snapshot(bridge: Qwen3Eagle3AsyncBridge, timeout_s: float = 5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        snapshot = bridge.maybe_apply_pending_weights()
        if snapshot is not None:
            return snapshot
        time.sleep(0.01)
    raise TimeoutError("timed out waiting for background trainer")


def test_async_bridge_applies_updates_and_resets() -> None:
    model = _ToyEagle3Module()
    trainer = Qwen3Eagle3CpuTrainer(model)
    bridge = Qwen3Eagle3AsyncBridge(trainer, _train_step)

    baseline = trainer.snapshot()

    bridge.observe_step("req-1", 0)
    updated = _wait_for_snapshot(bridge)

    assert updated.version == 1
    assert not torch.equal(
        updated.state_dict["model.fc.weight"],
        baseline.state_dict["model.fc.weight"],
    )

    bridge.reset_request("req-1")
    reset_snapshot = _wait_for_snapshot(bridge)

    assert reset_snapshot.version == baseline.version
    assert torch.equal(
        reset_snapshot.state_dict["model.fc.weight"],
        baseline.state_dict["model.fc.weight"],
    )
    assert torch.equal(
        reset_snapshot.state_dict["model.fc.bias"],
        baseline.state_dict["model.fc.bias"],
    )

    bridge.shutdown()


def test_async_bridge_waits_for_update_interval() -> None:
    model = _ToyEagle3Module()
    trainer = Qwen3Eagle3CpuTrainer(model)
    bridge = Qwen3Eagle3AsyncBridge(trainer, _train_step, update_interval=2)

    bridge.observe_step("req-1", 0)
    time.sleep(0.05)

    assert bridge.maybe_apply_pending_weights() is None

    bridge.observe_step("req-1", 1)
    updated = _wait_for_snapshot(bridge)

    assert updated.version == 1

    bridge.shutdown()


def test_async_bridge_reset_drops_pending_observations() -> None:
    model = _ToyEagle3Module()
    trainer = Qwen3Eagle3CpuTrainer(model)
    bridge = Qwen3Eagle3AsyncBridge(trainer, _train_step, update_interval=2)

    baseline = trainer.snapshot()

    bridge.observe_step("req-1", 0)
    bridge.reset_request("req-1")
    reset_snapshot = _wait_for_snapshot(bridge)

    assert reset_snapshot.version == baseline.version
    assert torch.equal(
        reset_snapshot.state_dict["model.fc.weight"],
        baseline.state_dict["model.fc.weight"],
    )

    bridge.observe_step("req-1", 1)
    time.sleep(0.05)

    assert bridge.maybe_apply_pending_weights() is None

    bridge.shutdown()
