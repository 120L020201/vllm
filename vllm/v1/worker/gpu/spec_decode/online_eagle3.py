# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from online_eagle3.data import WeightUpdateBridge
from online_eagle3.loss import project_teacher_distribution
from online_eagle3.weights import load_trainable_state_dict
from vllm.logger import init_logger
from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger(__name__)
_INTERNAL_REQUEST_PREFIXES = ("_dummy_req_", "_warmup_")


class OnlineEagle3Adapter:
    """Translate GPU proposal/verify callbacks into standalone CPU observations."""

    def __init__(self, model, bridge: WeightUpdateBridge, depth: int) -> None:
        self.model = model
        self.bridge = bridge
        self.depth = depth
        self._step_id = 0
        self._trace: dict[str, list[torch.Tensor]] | None = None
        self._auxiliary: torch.Tensor | None = None
        self._pending: dict[str, torch.Tensor] | None = None
        self._request_id: str | None = None
        self._logged_first_weight_apply = False

    def start_proposal(
        self,
        input_batch: InputBatch,
        auxiliary: torch.Tensor | None,
        *,
        skip: bool = False,
    ) -> None:
        self._trace = None
        self._auxiliary = None
        if skip or any(
            req.startswith(_INTERNAL_REQUEST_PREFIXES) for req in input_batch.req_ids
        ):
            return
        if input_batch.num_reqs != 1:
            raise ValueError("Online EAGLE3 requires exactly one request")
        if self._pending is not None:
            raise RuntimeError("Verify the pending proposal before proposing again")
        self._request_id = input_batch.req_ids[0]
        self._trace = {
            key: []
            for key in (
                "proposal_input_ids",
                "proposal_input_embeds",
                "proposal_positions",
                "proposal_hidden_states",
            )
        }
        self._auxiliary = auxiliary

    def record_step(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        if self._trace is None:
            return
        values = {
            "proposal_input_ids": input_ids[indices],
            "proposal_input_embeds": self.model.embed_input_ids(input_ids[indices]),
            "proposal_positions": positions[indices],
            "proposal_hidden_states": hidden_states[indices],
        }
        for name, value in values.items():
            self._trace[name].append(value.detach().clone())

    def record_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        last_token_indices: torch.Tensor,
    ) -> None:
        if self._trace is None:
            return
        end = int(last_token_indices[0].item()) + 1
        if self._auxiliary is None:
            raise ValueError("Online distillation requires target auxiliary features")
        values = {
            "prefill_input_embeds": self.model.embed_input_ids(input_ids[:end]),
            "prefill_positions": positions[:end],
            "prefill_aux_hidden_states": self._auxiliary[:end],
        }
        for name, value in values.items():
            self._trace[name] = [value.detach().clone()]
        self.record_step(input_ids, positions, hidden_states, last_token_indices[:1])
        self._auxiliary = None

    def finish_proposal(self) -> None:
        if self._trace is None:
            return
        self._pending = {
            name: torch.cat(values) for name, values in self._trace.items()
        }
        self._trace = None
        if self._pending["proposal_positions"].numel() != self.depth:
            raise ValueError("Incomplete online EAGLE3 proposal capture")

    def capture_teacher_logits(self, logits: torch.Tensor) -> None:
        if self._pending is None:
            return
        if logits.shape[0] != self.depth + 1:
            raise ValueError(
                "Online S=1 training requires a complete verification round"
            )
        probs, coverage = project_teacher_distribution(
            logits[: self.depth], self.model.draft_id_to_target_id
        )
        self._pending["teacher_probs"] = probs
        self._pending["teacher_coverage"] = coverage

    def observe_verify(
        self,
        input_batch: InputBatch,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        auxiliary: list[torch.Tensor] | None,
    ) -> None:
        if self._pending is None:
            return
        if input_batch.req_ids != [self._request_id]:
            raise ValueError("Verification request does not match the pending proposal")
        payload = self._pending
        if "teacher_probs" not in payload:
            raise ValueError("Missing teacher distributions for online distillation")
        count = int(num_sampled[0].item())
        if not 1 <= count <= self.depth + 1:
            raise ValueError("Invalid confirmed path length")
        if not auxiliary:
            raise ValueError("Online distillation requires target auxiliary features")
        shifted_ids = torch.cat(
            (
                input_batch.input_ids[1:count],
                sampled_token_ids[0, count - 1 : count],
            )
        )
        payload["confirmed_input_embeds"] = (
            self.model.embed_input_ids(shifted_ids).detach().clone()
        )
        payload["confirmed_positions"] = input_batch.positions[:count].detach().clone()
        payload["confirmed_aux_hidden_states"] = torch.cat(
            [value[:count] for value in auxiliary], dim=-1
        ).detach()
        self._pending = None
        self.bridge.observe_step(input_batch.req_ids[0], self._step_id, payload)
        self._step_id += 1

    def apply_weights(self) -> None:
        snapshot = self.bridge.maybe_apply_pending_weights()
        if snapshot is None:
            return
        with torch.profiler.record_function("online_eagle3.gpu_apply_weights"):
            load_trainable_state_dict(self.model, snapshot.state_dict)
        if not self._logged_first_weight_apply:
            logger.info(
                "Applied first online EAGLE3 GPU draft weight snapshot: version=%d",
                snapshot.version,
            )
            self._logged_first_weight_apply = True

    def reset_request(self, request_id: str) -> None:
        if request_id.startswith(_INTERNAL_REQUEST_PREFIXES):
            return
        if self._request_id is not None and request_id != self._request_id:
            return
        self._trace = self._pending = None
        self._auxiliary = None
        self._request_id = None
        self.bridge.reset_request(request_id)
        self.apply_weights()

    def shutdown(self) -> None:
        self._trace = self._pending = None
        self._auxiliary = None
        self.bridge.shutdown()
