# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.v1.worker.gpu.input_batch import InputBatch

if TYPE_CHECKING:
    from online_draft.runtime.contracts import CaptureAdapter
    from online_draft.runtime.weights import DraftWeightInstaller, DraftWeightSnapshot

    from vllm.v1.worker.gpu.spec_decode.online_state import DraftOnlineState


def completes_prefill(input_batch: InputBatch) -> bool:
    """Whether the single scheduled request finishes its prompt this step."""
    return bool(
        input_batch.num_computed_prefill_tokens_np[0]
        + input_batch.num_scheduled_tokens[0]
        >= input_batch.prefill_len_np[0]
    )


class OnlineDraftSpeculator:
    """Online draft training hooks shared by EAGLE3 and DFlash speculators.

    Subclasses decide which tensors to capture; this mixin owns the capture,
    install, and epoch-close order and the request lifecycle.
    """

    model: nn.Module
    online_draft_training_enabled: bool
    # Online drafts switch between two weight slots, each with its graphs.
    num_weight_slots: int
    online_state: "DraftOnlineState | None"
    capture_adapter: "CaptureAdapter | None"
    _online_target_model: nn.Module | None

    def _init_online_draft(self) -> None:
        self.online_draft_training_enabled = envs.VLLM_ONLINE_DRAFT_TRAIN
        self.num_weight_slots = 2 if self.online_draft_training_enabled else 1
        self.online_state = None
        self.capture_adapter = None
        self._online_target_model = None

    def _bind_weight_slot(self, slot: int) -> None:
        """Point the draft parameters at one weight slot, e.g. for capture."""
        if self.online_state is not None:
            slots = self.online_state.slots
            slots.bind(slots.slots[slot])

    def _active_weight_slot(self) -> int:
        if self.online_state is None:
            return 0
        return self.online_state.installer.active_slot

    def _keep_online_target(self, target_model: nn.Module) -> None:
        if self.online_draft_training_enabled:
            self._online_target_model = target_model

    def _create_online_state(
        self,
        mutable_names: Iterable[str],
        *,
        profile_timeline: bool,
        log_updates: bool,
    ) -> "DraftOnlineState":
        from vllm.v1.worker.gpu.spec_decode.online_state import DraftOnlineState
        from vllm.v1.worker.gpu.spec_decode.weight_slots import DraftWeightSlots

        assert self._online_target_model is not None
        return DraftOnlineState(
            DraftWeightSlots.from_models(
                self._online_target_model, self.model, mutable_names
            ),
            profile_timeline=profile_timeline,
            log_updates=log_updates,
        )

    def configure_online_training(
        self,
        mutable_names: Iterable[str],
        *,
        profile_timeline: bool = False,
        log_updates: bool = False,
    ) -> None:
        self.online_state = self._create_online_state(
            mutable_names,
            profile_timeline=profile_timeline,
            log_updates=log_updates,
        )

    @property
    def draft_weight_installer(self) -> "DraftWeightInstaller | None":
        return None if self.online_state is None else self.online_state.installer

    def set_capture_adapter(self, adapter: "CaptureAdapter") -> None:
        assert self.online_state is not None
        self.capture_adapter = adapter
        self.online_state.adapter = adapter

    def set_snapshot_source(
        self,
        poll_snapshot: Callable[[], "DraftWeightSnapshot | None"],
    ) -> None:
        assert self.online_state is not None
        self.online_state.poll_snapshot = poll_snapshot

    def on_target_logits_ready(self, logits: torch.Tensor) -> None:
        if self.capture_adapter is not None:
            self.capture_adapter.capture_teacher_logits(logits)

    def finish_request(
        self,
        request_id: str,
        initial_weights: Iterable[tuple[str, torch.Tensor]],
    ) -> None:
        assert self.online_state is not None
        self.online_state.finish(request_id, initial_weights)

    def _capture_target_and_install(
        self,
        input_batch: InputBatch,
        **target: Any,
    ) -> None:
        """Submit this step's target rows, then install any finished upload.

        The packet enters the FIFO before its epoch-close marker, and the
        caller runs weight-dependent draft compute only after this returns.
        """
        assert self.capture_adapter is not None
        assert self.online_state is not None
        assert input_batch.num_reqs == 1

        request_id = input_batch.req_ids[0]
        self.capture_adapter.capture_target(
            request_id=request_id,
            final_prefill=completes_prefill(input_batch),
            **target,
        )
        if self.online_state.advance() is not None:
            self.online_state.close_epoch(request_id)
