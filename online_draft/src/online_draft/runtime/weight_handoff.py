# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Protocol

from online_draft.runtime.weight_snapshot import DraftWeightSnapshot


class _SnapshotSource(Protocol):
    def poll_snapshot(self) -> DraftWeightSnapshot | None: ...


class _WeightInstaller(Protocol):
    def stage(self, snapshot: DraftWeightSnapshot) -> None: ...

    def commit_if_ready(self) -> int | None: ...


class Eagle3WeightHandoff:
    """Move CPU snapshots to GPU slots and commit completed copies."""

    def __init__(
        self,
        *,
        snapshot_source: _SnapshotSource,
        installer: _WeightInstaller,
    ) -> None:
        self._snapshot_source = snapshot_source
        self._installer = installer

    def advance(self) -> int | None:
        """Stage a ready snapshot and commit it when its copy is complete."""
        snapshot = self._snapshot_source.poll_snapshot()
        if snapshot is not None:
            self._installer.stage(snapshot)

        return self._installer.commit_if_ready()
