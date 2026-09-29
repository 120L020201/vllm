# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Translate vLLM EAGLE3 callbacks into standalone training batches."""

import logging

import torch
from online_draft.training.eagle3_batch import Eagle3DistillationBatch
from online_draft.training.eagle3_distill import project_teacher_distribution

from methods.utils.bridge import PolicyBridge, WeightSnapshot

logger = logging.getLogger(__name__)
_INTERNAL_REQUEST_PREFIXES = ("_dummy_req_", "_warmup_")


def _cpu_clone(tensor: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode(False):
        return tensor.detach().to(device="cpu").clone()


class OnlineEagle3Adapter:
    """Capture one-request EAGLE3 proposals and publish CPU updates."""

    def __init__(self, model, bridge: PolicyBridge, depth: int) -> None:
        self.model = model
        self.bridge = bridge
        self.depth = depth
        self._step_id = 0
        self._trace: dict[str, list[torch.Tensor]] | None = None
        self._pending: dict[str, torch.Tensor] | None = None
        self._request_id: str | None = None
        self._sampled_token_ids: torch.Tensor | None = None
        self._num_sampled: torch.Tensor | None = None
        self._logged_first_weight_apply = False

    def start_proposal(
        self,
        input_batch,
        auxiliary: list[torch.Tensor] | None,
        *,
        skip: bool = False,
    ) -> None:
        self._trace = None
        if skip or any(
            request_id.startswith(_INTERNAL_REQUEST_PREFIXES)
            for request_id in input_batch.req_ids
        ):
            return
        if input_batch.num_reqs != 1:
            raise ValueError("online methods require exactly one request")
        if self._pending is not None:
            raise RuntimeError("verify the pending proposal before proposing again")
        if not auxiliary:
            raise ValueError("online methods require target auxiliary hidden states")
        self._request_id = input_batch.req_ids[0]
        self._trace = {
            "prefill_input_embeds": [],
            "prefill_positions": [],
            "prefill_aux_hidden_states": [
                torch.cat(auxiliary, dim=-1).detach().clone()
            ],
            "proposal_positions": [],
            "draft_token_input_embeds": [],
            "draft_recurrent_hidden_states": [],
        }

    def record_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        last_token_indices: torch.Tensor,
    ) -> None:
        if self._trace is None:
            return
        end = int(last_token_indices[0].item()) + 1
        self._trace["prefill_input_embeds"] = [
            self.model.embed_input_ids(input_ids[:end]).detach().clone()
        ]
        self._trace["prefill_positions"] = [positions[:end].detach().clone()]
        self._trace["prefill_aux_hidden_states"] = [
            self._trace["prefill_aux_hidden_states"][0][:end]
        ]
        self._trace["proposal_positions"].append(
            positions[last_token_indices[:1]].detach().clone()
        )

    def record_draft_step(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        if self._trace is None:
            return
        self._trace["proposal_positions"].append(positions[indices].detach().clone())
        self._trace["draft_token_input_embeds"].append(
            self.model.embed_input_ids(input_ids[indices]).detach().clone()
        )
        self._trace["draft_recurrent_hidden_states"].append(
            hidden_states[indices].detach().clone()
        )

    def finish_proposal(self) -> None:
        if self._trace is None:
            return
        pending: dict[str, torch.Tensor] = {}
        for name, values in self._trace.items():
            if values:
                pending[name] = _cpu_clone(torch.cat(values))
        self._trace = None
        if pending["proposal_positions"].numel() != self.depth:
            raise ValueError("incomplete online EAGLE3 proposal capture")
        hidden_size = pending["prefill_input_embeds"].shape[-1]
        pending.setdefault(
            "draft_token_input_embeds",
            torch.empty(0, hidden_size, dtype=pending["prefill_input_embeds"].dtype),
        )
        pending.setdefault(
            "draft_recurrent_hidden_states",
            torch.empty(0, hidden_size, dtype=pending["prefill_input_embeds"].dtype),
        )
        self._pending = pending

    def capture_verification(
        self,
        logits: torch.Tensor,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
    ) -> None:
        if self._pending is None:
            return
        if logits.shape[0] != self.depth + 1:
            raise ValueError("online methods require a complete verification round")
        target_token_ids = self.bridge.trainer.model.get_target_token_ids(
            device=logits.device
        )
        probabilities, _ = project_teacher_distribution(
            logits[: self.depth], target_token_ids
        )
        self._pending["teacher_probabilities"] = _cpu_clone(probabilities)
        self._sampled_token_ids = _cpu_clone(sampled_token_ids)
        self._num_sampled = _cpu_clone(num_sampled)

    def observe_pending_verification(
        self,
        input_batch,
        auxiliary: list[torch.Tensor] | None,
    ) -> None:
        if self._pending is None:
            return
        if input_batch.req_ids != [self._request_id]:
            raise ValueError("verification request does not match pending proposal")
        if self._sampled_token_ids is None or self._num_sampled is None:
            raise ValueError("missing verification result")
        if not auxiliary:
            raise ValueError("online methods require target auxiliary hidden states")
        count = int(self._num_sampled[0].item())
        if not 1 <= count <= self.depth + 1:
            raise ValueError("invalid confirmed path length")
        shifted_ids = torch.cat(
            (
                input_batch.input_ids[1:count],
                self._sampled_token_ids.to(input_batch.input_ids.device)[
                    0, count - 1 : count
                ],
            )
        )
        batch = Eagle3DistillationBatch(
            prefill_positions=self._pending["prefill_positions"],
            prefill_input_embeds=self._pending["prefill_input_embeds"],
            prefill_aux_hidden_states=self._pending["prefill_aux_hidden_states"],
            proposal_positions=self._pending["proposal_positions"],
            draft_token_input_embeds=self._pending["draft_token_input_embeds"],
            draft_recurrent_hidden_states=(
                self._pending["draft_recurrent_hidden_states"]
            ),
            teacher_probabilities=self._pending["teacher_probabilities"],
            confirmed_positions=_cpu_clone(input_batch.positions[:count]),
            confirmed_input_embeds=_cpu_clone(self.model.embed_input_ids(shifted_ids)),
            confirmed_aux_hidden_states=_cpu_clone(
                torch.cat([value[:count] for value in auxiliary], dim=-1)
            ),
            rejection_position=None if count == self.depth + 1 else count - 1,
        )
        assert self._request_id is not None
        self.bridge.observe_step(self._request_id, self._step_id, batch)
        self._step_id += 1
        self._pending = None
        self._sampled_token_ids = None
        self._num_sampled = None

    def apply_weights(self) -> None:
        snapshot = self.bridge.maybe_apply_pending_weights()
        if snapshot is None:
            return
        self._apply_snapshot(snapshot)
        if not self._logged_first_weight_apply:
            logger.info(
                "applied first online EAGLE3 GPU snapshot: version=%d",
                snapshot.version,
            )
            self._logged_first_weight_apply = True

    def _apply_snapshot(self, snapshot: WeightSnapshot) -> None:
        state = self.model.state_dict()
        missing = sorted(set(snapshot.state_dict) - set(state))
        if missing:
            raise KeyError(f"GPU draft is missing trainable weights: {missing}")
        with torch.no_grad():
            for name, source in snapshot.state_dict.items():
                target = state[name]
                if target.shape != source.shape:
                    raise ValueError(f"shape mismatch for GPU draft weight {name}")
                target.copy_(source.to(device=target.device, dtype=target.dtype))

    def reset_request(self, request_id: str) -> None:
        if request_id.startswith(_INTERNAL_REQUEST_PREFIXES):
            return
        if self._request_id is not None and request_id != self._request_id:
            return
        self._trace = None
        self._pending = None
        self._sampled_token_ids = None
        self._num_sampled = None
        self._request_id = None
        self._step_id = 0
        self.bridge.reset_request(request_id)
        self.apply_weights()

    def shutdown(self) -> None:
        self._trace = None
        self._pending = None
        self.bridge.shutdown()
