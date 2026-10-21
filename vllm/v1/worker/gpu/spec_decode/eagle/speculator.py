# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model

if TYPE_CHECKING:
    from online_draft.runtime.capture.eagle3 import Eagle3CaptureAdapter
    from online_draft.runtime.weights import DraftWeightInstaller, DraftWeightSnapshot

    from vllm.v1.worker.gpu.spec_decode.eagle.online import Eagle3OnlineState


class EagleSpeculator(AutoRegressiveSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self.online_draft_training_enabled = envs.VLLM_ONLINE_DRAFT_TRAIN
        self.explicit_input_embeds = self.online_draft_training_enabled
        self._online_target_model: nn.Module | None = None
        self.online_state: Eagle3OnlineState | None = None
        self.capture_adapter: Eagle3CaptureAdapter | None = None

    def set_capture_adapter(self, adapter: "Eagle3CaptureAdapter") -> None:
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

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        eagle_model = load_eagle_model(target_model, self.vllm_config)

        if self.online_draft_training_enabled:
            self._online_target_model = target_model

        return eagle_model

    def configure_online_training(
        self,
        mutable_names: Iterable[str],
        *,
        profile_timeline: bool = False,
        log_updates: bool = False,
    ) -> None:
        from vllm.v1.worker.gpu.spec_decode.eagle.online import Eagle3OnlineState

        assert self._online_target_model is not None
        self.online_state = Eagle3OnlineState(
            target_model=self._online_target_model,
            draft_model=self.model,
            mutable_names=mutable_names,
            profile_timeline=profile_timeline,
            log_updates=log_updates,
        )

    @property
    def draft_weight_installer(self) -> "DraftWeightInstaller | None":
        return None if self.online_state is None else self.online_state.installer

    def finish_request(
        self,
        request_id: str,
        initial_weights: Iterable[tuple[str, torch.Tensor]],
    ) -> None:
        assert self.online_state is not None
        self.online_state.finish(request_id, initial_weights)

    def on_prefill_input_embeds_ready(
        self,
        num_reqs: int,
        input_batch: InputBatch,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        capture_enabled: bool,
    ) -> None:
        if capture_enabled and self.capture_adapter is not None:
            assert num_reqs == 1
            assert aux_hidden_states is not None
            assert self.last_prefill_input_embeds is not None
            assert self.last_prefill_positions is not None
            assert self.last_prefill_length is not None

            self.capture_adapter.capture_prefill(
                request_id=input_batch.req_ids[0],
                verify_row_indices=input_batch.logits_indices,
                target_aux_hidden_states=aux_hidden_states,
                num_sampled=num_sampled[0],
                prefill_input_embeds=self.last_prefill_input_embeds,
                prefill_positions=self.last_prefill_positions,
                prefill_length=num_sampled[0],
            )

        if self.online_state is None:
            return

        if self.online_state.advance() is not None:
            self.online_state.close_epoch(input_batch.req_ids[0])

    def on_proposal_ready(
        self,
        input_batch: InputBatch,
        aux_hidden_states: list[torch.Tensor] | None,
        capture_enabled: bool,
    ) -> None:
        if not capture_enabled or self.capture_adapter is None:
            return

        assert input_batch.num_reqs == 1
        assert aux_hidden_states is not None

        self.capture_adapter.capture_proposal(
            request_id=input_batch.req_ids[0],
            verify_row_indices=input_batch.logits_indices,
            target_aux_hidden_states=aux_hidden_states,
            proposal=self.get_proposal_snapshot(),
        )
