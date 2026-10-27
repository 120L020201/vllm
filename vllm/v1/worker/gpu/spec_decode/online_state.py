# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Iterable, Mapping
from typing import Protocol

import torch
from online_draft.runtime.contracts import CaptureAdapter
from online_draft.runtime.weights import DraftWeightInstaller, DraftWeightSnapshot

from vllm.logger import init_logger

logger = init_logger(__name__)


class OnlineWeightSlots(Protocol):
    """Two GPU parameter slots that a draft model can bind."""

    @property
    def slots(
        self,
    ) -> tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]: ...

    @property
    def owned_names(self) -> tuple[str, ...]: ...

    def bind(self, weights: Mapping[str, torch.Tensor]) -> None: ...


class DraftOnlineState:
    """Own draft weight installation and request lifecycle state."""

    def __init__(
        self,
        slots: OnlineWeightSlots,
        *,
        prepare_slot: Callable[[int, Mapping[str, torch.Tensor]], None] | None = None,
        profile_timeline: bool,
        log_updates: bool,
    ) -> None:
        self.slots = slots
        self.installer = DraftWeightInstaller(
            slots=slots.slots,
            prepare_slot=prepare_slot,
            profile_timeline=profile_timeline,
        )
        self.adapter: CaptureAdapter
        self.poll_snapshot: Callable[[], DraftWeightSnapshot | None]
        self.log_updates = log_updates
        self.finished_request: dict | None = None
        self._last_installed_round = 0

    def advance(self) -> int | None:
        version = self.installer.advance(self.poll_snapshot())
        if version is None:
            return None
        self.slots.bind(self.installer.active_weights)
        return version

    def close_epoch(self, request_id: str) -> None:
        self.adapter.close_epoch(request_id)
        if not self.log_updates:
            return

        completed_rounds = self.adapter.completed_rounds
        rounds_since_install = completed_rounds - self._last_installed_round
        self._last_installed_round = completed_rounds
        logger.info(
            "Online draft installed weights: request_id=%s version=%d rounds=%d",
            request_id,
            self.installer.active_version,
            rounds_since_install,
        )

    def finish(
        self,
        request_id: str,
        initial_weights: Iterable[tuple[str, torch.Tensor]],
    ) -> None:
        self.finished_request = {
            "request_id": request_id,
            "gpu_version": self.installer.active_version,
            "completed_rounds": self.adapter.completed_rounds,
        }
        self.adapter.finish_request(request_id)
        self.installer.reset(initial_weights)
        self.slots.bind(self.installer.active_weights)
        self._last_installed_round = 0
