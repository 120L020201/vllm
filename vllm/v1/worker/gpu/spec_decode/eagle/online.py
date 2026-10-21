# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Iterable

import torch
import torch.nn as nn
from online_draft.runtime.capture.eagle3 import Eagle3CaptureAdapter
from online_draft.runtime.weights import DraftWeightInstaller, DraftWeightSnapshot

from vllm.logger import init_logger
from vllm.v1.worker.gpu.spec_decode.weight_slots import DraftWeightSlots

logger = init_logger(__name__)


class Eagle3OnlineState:
    """Own EAGLE3 weight installation and request lifecycle state."""

    def __init__(
        self,
        target_model: nn.Module,
        draft_model: nn.Module,
        mutable_names: Iterable[str],
        *,
        profile_timeline: bool,
        log_updates: bool,
    ) -> None:
        self.slots = DraftWeightSlots.from_models(
            target_model, draft_model, mutable_names
        )
        self.installer = DraftWeightInstaller(
            slots=self.slots.slots,
            mutable_names=self.slots.owned_names,
            profile_timeline=profile_timeline,
        )
        self.adapter: Eagle3CaptureAdapter
        self.poll_snapshot: Callable[[], DraftWeightSnapshot | None]
        self.log_updates = log_updates
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
        self.adapter.finish_request(request_id)
        self.installer.reset(initial_weights)
        self.slots.bind(self.installer.active_weights)
        self._last_installed_round = 0
