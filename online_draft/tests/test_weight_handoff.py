# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field

import torch
from online_draft.runtime.draft_weight_installer import DraftWeightInstaller
from online_draft.runtime.weight_handoff import Eagle3WeightHandoff
from online_draft.runtime.weight_snapshot import DraftWeightSnapshot


@dataclass
class _SnapshotSource:
    snapshots: list[DraftWeightSnapshot] = field(default_factory=list)

    def poll_snapshot(self) -> DraftWeightSnapshot | None:
        if not self.snapshots:
            return None
        return self.snapshots.pop(0)


class _DelayedInstaller:
    def __init__(self, results: list[int | None]) -> None:
        self.results = results
        self.staged: list[DraftWeightSnapshot] = []
        self.calls: list[str] = []

    def stage(self, snapshot: DraftWeightSnapshot) -> None:
        self.calls.append("stage")
        self.staged.append(snapshot)

    def commit_if_ready(self) -> int | None:
        self.calls.append("commit")
        return self.results.pop(0)


def _snapshot(version: int) -> DraftWeightSnapshot:
    return DraftWeightSnapshot.from_named_tensors(
        [("weight", torch.tensor([version], dtype=torch.float32))],
        version=version,
        pin_memory=False,
    )


def test_advance_returns_only_successful_commit() -> None:
    source = _SnapshotSource([_snapshot(1)])
    installer = _DelayedInstaller([None, 1])
    handoff = Eagle3WeightHandoff(
        snapshot_source=source,
        installer=installer,
    )

    assert handoff.advance() is None
    assert handoff.advance() == 1
    assert installer.calls == ["stage", "commit", "commit"]


def test_real_installer_switches_version() -> None:
    active = {"weight": torch.tensor([0.0])}
    staging = {"weight": torch.tensor([0.0])}

    installer = DraftWeightInstaller(
        (active, staging),
        mutable_names=("weight",),
    )
    source = _SnapshotSource([_snapshot(1)])
    handoff = Eagle3WeightHandoff(
        snapshot_source=source,
        installer=installer,
    )

    assert handoff.advance() == 1
    assert installer.active_version == 1
    assert torch.equal(
        installer.active_weights["weight"],
        torch.tensor([1.0]),
    )
