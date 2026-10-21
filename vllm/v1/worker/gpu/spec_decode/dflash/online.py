# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Iterable, Mapping

import torch
from online_draft.runtime.capture.dflash import DFlashCaptureAdapter
from online_draft.runtime.weights import DraftWeightInstaller, DraftWeightSnapshot

from vllm.logger import init_logger
from vllm.v1.worker.gpu.spec_decode.eagle.weight_slots import DraftWeightSlots

logger = init_logger(__name__)


class DFlashWeightSlots:
    """Keep context-projection copies in the same versioned slot as parameters."""

    def __init__(self, target_model, draft_model, mutable_names: Iterable[str]):
        self.weights = DraftWeightSlots.from_models(
            target_model, draft_model, mutable_names
        )
        self.model = draft_model.model
        self.parameters = dict(draft_model.named_parameters())
        self.slots = self.weights.slots
        self.owned_names = self.weights.owned_names
        self.derived = tuple(
            {
                "_fused_kv_weight": self.model._fused_kv_weight.clone(),
                "_k_norm_weights": self.model._k_norm_weights.clone(),
            }
            for _ in range(2)
        )

    @torch.no_grad()
    def prepare(self, slot: int, weights: Mapping[str, torch.Tensor]) -> None:
        kv_weights, norms = [], []
        for index, layer in enumerate(self.model.layers):
            prefix = f"model.layers.{index}.self_attn"
            qkv_name = f"{prefix}.qkv_proj.weight"
            norm_name = f"{prefix}.k_norm.weight"
            qkv = weights.get(qkv_name, self.parameters[qkv_name])
            kv_weights.append(qkv[layer.self_attn.q_size :])
            norms.append(weights.get(norm_name, self.parameters[norm_name]))
        torch.cat(kv_weights, dim=0, out=self.derived[slot]["_fused_kv_weight"])
        torch.stack(norms, dim=0, out=self.derived[slot]["_k_norm_weights"])

    def bind(self, weights: Mapping[str, torch.Tensor]) -> None:
        self.weights.bind(weights)
        slot = 0 if weights is self.slots[0] else 1
        self.model._fused_kv_weight = self.derived[slot]["_fused_kv_weight"]
        self.model._k_norm_weights = self.derived[slot]["_k_norm_weights"]
        self.model._hidden_norm_weight = self.model.hidden_norm.weight.data


class DFlashOnlineState:
    def __init__(
        self, target_model, draft_model, mutable_names, *, profile_timeline, log_updates
    ):
        self.slots = DFlashWeightSlots(target_model, draft_model, mutable_names)
        self.installer = DraftWeightInstaller(
            self.slots.slots,
            mutable_names=self.slots.owned_names,
            prepare_slot=self.slots.prepare,
            profile_timeline=profile_timeline,
        )
        self.adapter: DFlashCaptureAdapter
        self.poll_snapshot: Callable[[], DraftWeightSnapshot | None]
        self.log_updates = log_updates
        self._last_installed_round = 0
        self.finished_request: dict | None = None

    def advance(self, request_id: str) -> int | None:
        version = self.installer.advance(self.poll_snapshot())
        if version is None:
            return None
        self.slots.bind(self.installer.active_weights)
        return version

    def close_epoch(self, request_id: str) -> None:
        self.adapter.close_epoch(request_id)
        if self.log_updates:
            completed = self.adapter.completed_rounds
            logger.info(
                "Online DFlash installed version=%d completed_rounds=%d "
                "window_rounds=%d",
                self.installer.active_version,
                completed,
                completed - self._last_installed_round,
            )
            self._last_installed_round = completed

    def finish(self, request_id: str, initial_weights) -> None:
        self.finished_request = {
            "request_id": request_id,
            "gpu_version": self.installer.active_version,
            "completed_rounds": self.adapter.completed_rounds,
        }
        self.adapter.finish_request(request_id)
        self.installer.reset(initial_weights)
        self.slots.bind(self.installer.active_weights)
        self._last_installed_round = 0
