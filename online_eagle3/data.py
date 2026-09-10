# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Protocol

import torch

from .weights import TrainableWeightSnapshot

if TYPE_CHECKING:
    from .qwen3_trainer import Qwen3Eagle3CpuTrainer


@dataclass(slots=True)
class TrainObservation:
    request_id: str
    step_id: int
    payload: dict[str, torch.Tensor] = field(default_factory=dict)


Qwen3Eagle3StepFn = Callable[
    ["Qwen3Eagle3CpuTrainer", Sequence[TrainObservation]], None
]


class WeightUpdateBridge(Protocol):
    def observe_step(
        self,
        request_id: str,
        step_id: int,
        payload: Mapping[str, torch.Tensor] | None = None,
    ) -> None: ...

    def maybe_apply_pending_weights(self) -> TrainableWeightSnapshot | None: ...

    def reset_request(self, request_id: str) -> None: ...

    def shutdown(self) -> None: ...


@dataclass(slots=True)
class DistillationBatch:
    """One canvas: prefill [P,H], proposal [K,H], confirmed [C,H].

    Auxiliary features have width 3H; teacher probabilities have draft-vocab
    width. Positions are absolute. Only the confirmed path enters persistent KV.
    """

    prefill_positions: torch.Tensor
    prefill_input_embeds: torch.Tensor
    prefill_aux_hidden_states: torch.Tensor
    proposal_positions: torch.Tensor
    proposal_input_embeds: torch.Tensor
    proposal_hidden_states: torch.Tensor
    teacher_probs: torch.Tensor
    confirmed_positions: torch.Tensor
    confirmed_input_embeds: torch.Tensor
    confirmed_aux_hidden_states: torch.Tensor

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, torch.Tensor], dtype: torch.dtype
    ) -> "DistillationBatch":
        # Clone outside inference_mode so autograd can save these inputs.
        values = {}
        with (
            torch.inference_mode(False),
            torch.profiler.record_function("online_eagle3.cpu_prepare_batch"),
        ):
            for item in fields(cls):
                value = payload[item.name].detach().to(device="cpu").clone()
                target_dtype = (
                    torch.long
                    if item.name.endswith("positions")
                    else torch.float32
                    if item.name == "teacher_probs"
                    else dtype
                )
                values[item.name] = value.to(target_dtype)
        return cls(**values)

    def validate(self, hidden_size: int, num_aux: int) -> None:
        for prefix in ("prefill", "proposal", "confirmed"):
            positions = getattr(self, f"{prefix}_positions")
            embeds = getattr(self, f"{prefix}_input_embeds")
            if positions.ndim != 1 or embeds.shape != (positions.numel(), hidden_size):
                raise ValueError(
                    f"{prefix} positions and embeddings have invalid shapes"
                )
            if prefix != "proposal":
                auxiliary = getattr(self, f"{prefix}_aux_hidden_states")
                if auxiliary.shape != (positions.numel(), hidden_size * num_aux):
                    raise ValueError(f"{prefix} auxiliary features have invalid shapes")
        if self.teacher_probs.ndim != 2:
            raise ValueError("Teacher probabilities must have shape [K, V]")
        positions = self.prefill_positions
        proposal_positions = self.proposal_positions
        proposal_embeds = self.proposal_input_embeds
        proposal_hidden = self.proposal_hidden_states
        teacher = self.teacher_probs
        confirmed_positions = self.confirmed_positions
        depth = proposal_positions.numel()
        if depth == 0 or teacher.shape[0] != depth or proposal_embeds.shape[0] != depth:
            raise ValueError("Teacher and proposal depths do not match")
        if proposal_hidden.shape != proposal_embeds.shape:
            raise ValueError("Proposal hidden states must match the embedding shape")
        if positions.numel() == 0 or not torch.equal(
            positions, torch.arange(int(positions[0]), int(positions[-1]) + 1)
        ):
            raise ValueError("Prefill positions must be nonempty and contiguous")
        if not torch.equal(
            proposal_positions,
            torch.arange(int(positions[-1]), int(positions[-1]) + depth),
        ):
            raise ValueError("Proposal positions do not align with prefill")
        if (
            not torch.equal(
                confirmed_positions,
                torch.arange(
                    int(positions[-1]) + 1,
                    int(positions[-1]) + 1 + confirmed_positions.numel(),
                ),
            )
            or not 1 <= confirmed_positions.numel() <= depth + 1
        ):
            raise ValueError(
                "Confirmed positions do not align with the verified canvas"
            )
