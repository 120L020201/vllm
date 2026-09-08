# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import threading
import time

import torch
import torch.nn as nn

from online_eagle3.async_bridge import (
    Qwen3Eagle3AsyncBridge,
    Qwen3Eagle3LazyBridge,
    TrainObservation,
)
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


def test_async_bridge_keeps_only_latest_snapshot() -> None:
    model = _ToyEagle3Module()
    trainer = Qwen3Eagle3CpuTrainer(model)
    bridge = Qwen3Eagle3AsyncBridge(trainer, _train_step)

    for step_id in range(3):
        bridge.observe_step("req-1", step_id)

    deadline = time.monotonic() + 5.0
    snapshot = None
    while time.monotonic() < deadline:
        snapshot = bridge.maybe_apply_pending_weights()
        if snapshot is not None and snapshot.version == 3:
            break
        time.sleep(0.01)

    assert snapshot is not None
    assert snapshot.version == 3
    assert bridge.maybe_apply_pending_weights() is None

    bridge.shutdown()


def test_async_bridge_reset_skips_queued_observations() -> None:
    model = _ToyEagle3Module()
    trainer = Qwen3Eagle3CpuTrainer(model)
    first_step_started = threading.Event()
    release_first_step = threading.Event()
    trained_step_ids: list[int] = []

    def blocking_train_step(
        trainer: Qwen3Eagle3CpuTrainer,
        observations: tuple[TrainObservation, ...],
    ) -> None:
        trained_step_ids.extend(observation.step_id for observation in observations)
        first_step_started.set()
        if observations[0].step_id == 0:
            assert release_first_step.wait(timeout=5.0)
        _train_step(trainer, observations)

    bridge = Qwen3Eagle3AsyncBridge(trainer, blocking_train_step)

    bridge.observe_step("req-1", 0)
    assert first_step_started.wait(timeout=5.0)
    for step_id in range(1, 4):
        bridge.observe_step("req-1", step_id)
    bridge.reset_request("req-1")
    release_first_step.set()

    reset_snapshot = _wait_for_snapshot(bridge)

    assert reset_snapshot.version == 0
    assert trained_step_ids == [0]

    bridge.shutdown()


def test_async_bridge_shutdown_is_idempotent() -> None:
    model = _ToyEagle3Module()
    trainer = Qwen3Eagle3CpuTrainer(model)
    bridge = Qwen3Eagle3AsyncBridge(trainer, _train_step)

    bridge.shutdown()
    bridge.shutdown()
    bridge.observe_step("req-1", 0)
    bridge.reset_request("req-1")

    assert not bridge._thread.is_alive()
    assert bridge.maybe_apply_pending_weights() is None


def test_lazy_bridge_defers_loading_until_observation() -> None:
    created_count = 0

    def create_bridge() -> Qwen3Eagle3AsyncBridge:
        nonlocal created_count
        created_count += 1
        return Qwen3Eagle3AsyncBridge(
            Qwen3Eagle3CpuTrainer(_ToyEagle3Module()),
            _train_step,
        )

    bridge = Qwen3Eagle3LazyBridge(create_bridge)

    assert not bridge.is_loaded
    assert bridge.maybe_apply_pending_weights() is None
    bridge.reset_request("req-1")

    assert not bridge.is_loaded
    assert created_count == 0

    bridge.observe_step("req-1", 0)
    snapshot = _wait_for_snapshot(bridge)

    assert bridge.is_loaded
    assert created_count == 1
    assert snapshot.version == 1

    bridge.shutdown()
