# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Iterable, Mapping
from threading import Lock

import torch

from online_draft.runtime.weight_snapshot import DraftWeightSnapshot


class DraftWeightInstaller:
    """Stage draft weights asynchronously and switch one completed slot."""

    def __init__(
        self,
        slots: tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]],
        *,
        mutable_names: Iterable[str],
        initial_version: int = 0,
    ) -> None:
        if len(slots) != 2:
            raise ValueError("slots must contain two mappings")

        self._slots = slots
        self._names = tuple(slots[0])
        self._mutable_names = tuple(mutable_names)
        if not self._names:
            raise ValueError("slots must not be empty")
        if not self._mutable_names:
            raise ValueError("mutable_names must not be empty")
        if tuple(slots[1]) != self._names:
            raise ValueError("slot names do not match")
        if len(set(self._mutable_names)) != len(self._mutable_names):
            raise ValueError("mutable_names must not contain duplicates")
        if not set(self._mutable_names).issubset(self._names):
            raise ValueError("mutable_names must be present in both slots")

        self._device = slots[0][self._names[0]].device
        for name in self._names:
            first = slots[0][name]
            second = slots[1][name]
            if first.device != self._device or second.device != self._device:
                raise ValueError("all slot tensors must use one device")
            if first.shape != second.shape or first.dtype != second.dtype:
                raise ValueError(f"slot tensor metadata does not match: {name}")

        self._active_slot = 0
        self._active_version = initial_version
        self._staged: tuple[DraftWeightSnapshot, torch.cuda.Event | None] | None = None
        self._lock = Lock()
        self._h2d_stream = (
            torch.cuda.Stream(device=self._device)
            if self._device.type == "cuda"
            else None
        )

    @property
    def active_version(self) -> int:
        return self._active_version

    @property
    def staged_version(self) -> int | None:
        staged = self._staged
        return None if staged is None else staged[0].version

    @property
    def active_weights(self) -> Mapping[str, torch.Tensor]:
        return self._slots[self._active_slot]

    def stage(self, snapshot: DraftWeightSnapshot) -> None:
        """Copy a CPU snapshot into the inactive device slot."""
        with self._lock:
            self._validate_snapshot(snapshot)
            if self._staged is not None:
                raise RuntimeError("a draft weight update is already staged")

            inactive_slot = 1 - self._active_slot
            destination = self._slots[inactive_slot]
            if self._h2d_stream is None:
                for name, source in snapshot.tensors:
                    destination[name].copy_(source)
                event = None
            else:
                with torch.cuda.stream(self._h2d_stream):
                    for name, source in snapshot.tensors:
                        destination[name].copy_(source, non_blocking=True)
                    event = torch.cuda.Event()
                    event.record(self._h2d_stream)

            self._staged = (snapshot, event)

    def commit_if_ready(self) -> int | None:
        """Switch to the staged slot without waiting for H2D."""
        with self._lock:
            if self._staged is None:
                return None

            snapshot, event = self._staged
            if event is not None and not event.query():
                return None

            self._active_slot = 1 - self._active_slot
            self._active_version = snapshot.version
            self._staged = None
            return self._active_version

    def _validate_snapshot(self, snapshot: DraftWeightSnapshot) -> None:
        if snapshot.version <= self._active_version:
            raise ValueError("snapshot version must be newer than active version")

        if tuple(name for name, _ in snapshot.tensors) != self._mutable_names:
            raise ValueError("snapshot tensor names do not match the installer")

        for name, tensor in snapshot.tensors:
            if tensor.device.type != "cpu":
                raise ValueError(f"snapshot tensor must be on CPU: {name}")
            if tensor.shape != self._slots[0][name].shape:
                raise ValueError(f"snapshot shape does not match: {name}")
            if tensor.dtype != self._slots[0][name].dtype:
                raise ValueError(f"snapshot dtype does not match: {name}")
