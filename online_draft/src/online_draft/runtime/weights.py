# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from threading import Lock

import torch

from online_draft.runtime import profile_range


@dataclass(frozen=True, slots=True)
class DraftWeightSnapshot:
    """Detached CPU weights owned by one install attempt."""

    version: int
    tensors: tuple[tuple[str, torch.Tensor], ...]

    @classmethod
    def from_named_tensors(
        cls,
        named_tensors: Iterable[tuple[str, torch.Tensor]],
        *,
        version: int,
        pin_memory: bool = True,
    ) -> DraftWeightSnapshot:
        """Copy named tensors into CPU-owned storage."""
        use_pinned_memory = pin_memory and torch.cuda.is_available()
        tensors = []
        for name, tensor in named_tensors:
            cpu_tensor = torch.empty_like(
                tensor,
                device="cpu",
                pin_memory=use_pinned_memory,
            )
            cpu_tensor.copy_(tensor.detach())
            tensors.append((name, cpu_tensor))

        return cls(version=version, tensors=tuple(tensors))


class DraftWeightInstaller:
    """Stage draft weights asynchronously and switch one completed slot."""

    def __init__(
        self,
        slots: tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]],
        *,
        initial_version: int = 0,
        profile_timeline: bool = False,
        prepare_slot: Callable[[int, Mapping[str, torch.Tensor]], None] | None = None,
    ) -> None:
        self._slots = slots
        self._device = next(iter(slots[0].values())).device
        self._active_slot = 0
        self._initial_version = initial_version
        self._active_version = initial_version
        self._profile_timeline = profile_timeline
        self._prepare_slot = prepare_slot
        self._last_use: list[torch.cuda.Event | None] = [None, None]
        self._staged: tuple[DraftWeightSnapshot, torch.cuda.Event | None] | None = None
        self._lock = Lock()
        self._h2d_stream = (
            torch.cuda.Stream(device=self._device)
            if self._device.type == "cuda"
            else None
        )

    @property
    def active_slot(self) -> int:
        return self._active_slot

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

    def advance(self, snapshot: DraftWeightSnapshot | None) -> int | None:
        """Stage an available snapshot and commit a completed copy without waiting."""
        if snapshot is not None:
            self.stage(snapshot)
        return self.commit_if_ready()

    def stage(self, snapshot: DraftWeightSnapshot) -> None:
        """Copy a CPU snapshot into the inactive device slot."""
        with self._lock:
            if self._staged is not None:
                raise RuntimeError("a draft weight update is already staged")

            inactive_slot = 1 - self._active_slot
            destination = self._slots[inactive_slot]
            profile_name = f"online_draft.h2d_stage[version={snapshot.version}]"
            with profile_range(profile_name, self._profile_timeline):
                if self._h2d_stream is None:
                    for name, source in snapshot.tensors:
                        destination[name].copy_(source)
                    if self._prepare_slot is not None:
                        self._prepare_slot(inactive_slot, destination)
                    event = None
                else:
                    with torch.cuda.stream(self._h2d_stream):
                        last_use = self._last_use[inactive_slot]
                        if last_use is not None:
                            self._h2d_stream.wait_event(last_use)
                        for name, source in snapshot.tensors:
                            destination[name].copy_(source, non_blocking=True)
                        if self._prepare_slot is not None:
                            self._prepare_slot(inactive_slot, destination)
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

            profile_name = f"online_draft.h2d_commit[version={snapshot.version}]"
            with profile_range(profile_name, self._profile_timeline):
                if self._h2d_stream is not None:
                    last_use = torch.cuda.Event()
                    last_use.record(torch.cuda.current_stream(self._device))
                    self._last_use[self._active_slot] = last_use
                self._active_slot = 1 - self._active_slot
                self._active_version = snapshot.version
                self._staged = None
            return self._active_version

    def reset(
        self,
        initial_weights: Iterable[tuple[str, torch.Tensor]],
    ) -> None:
        """Restore initial weights and discard the staged update."""
        weights = tuple(initial_weights)

        with self._lock:
            if self._h2d_stream is not None:
                self._h2d_stream.synchronize()

            destination = self._slots[self._active_slot]
            for name, source in weights:
                destination[name].copy_(source)
            if self._prepare_slot is not None:
                self._prepare_slot(self._active_slot, destination)

            self._staged = None
            self._active_version = self._initial_version
