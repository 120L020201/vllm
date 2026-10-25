# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, fields, replace
from typing import Any, ClassVar, Generic, Protocol, TypeVar

import torch
from typing_extensions import Self

from online_draft.runtime import profile_range
from online_draft.runtime.contracts import PreparedRound
from online_draft.runtime.cpu_worker import CPUWorkerConfig
from online_draft.runtime.transport import CaptureQueue, PinnedStaging


class CaptureBuffers(PinnedStaging):
    """Pinned staging with the fields every target packet carries."""

    def __init__(
        self,
        *,
        max_tokens: int,
        teacher_shape: tuple[int, int],
        teacher_dtype: torch.dtype,
        layouts: Mapping[str, tuple[tuple[int, ...], torch.dtype]],
    ) -> None:
        super().__init__(
            {
                "positions": ((max_tokens,), torch.int64),
                "metadata": ((2,), torch.int64),
                "teacher_logits": (teacher_shape, teacher_dtype),
                **layouts,
            }
        )


class CaptureProposal(Protocol):
    """GPU proposal verified by the next target step."""

    source_weight_version: int

    def to_cpu(
        self,
        buffers: PinnedStaging,
        *,
        non_blocking: bool = False,
    ) -> Self: ...


ProposalT = TypeVar("ProposalT", bound=CaptureProposal)
RowsT = TypeVar("RowsT")
BatchT = TypeVar("BatchT")


@dataclass(slots=True, kw_only=True)
class TargetPacket(Generic[ProposalT]):
    """One target step's rows, with the pending proposal they verify.

    Metadata holds the confirmed leading row count followed by num_sampled.
    Prompt chunks carry no proposal. Subclass tensor fields are copied into
    the staging entry of the same name.
    """

    request_id: str
    step_id: int
    proposal: ProposalT | None
    teacher_logits: torch.Tensor | None
    positions: torch.Tensor
    metadata: torch.Tensor
    buffers: PinnedStaging
    ready: torch.cuda.Event | None = None

    _shared_fields: ClassVar[frozenset[str]] = frozenset(
        ("request_id", "step_id", "proposal", "teacher_logits", "buffers", "ready")
    )

    @property
    def device(self) -> torch.device:
        return self.positions.device

    def to_cpu(self, *, non_blocking: bool = False) -> Self:
        def copy(name: str, tensor: torch.Tensor) -> torch.Tensor:
            return self.buffers.copy(name, tensor, non_blocking)

        tensors = {
            field.name: copy(field.name, getattr(self, field.name))
            for field in fields(self)
            if field.name not in self._shared_fields
        }
        return replace(
            self,
            proposal=(
                None
                if self.proposal is None
                else self.proposal.to_cpu(self.buffers, non_blocking=non_blocking)
            ),
            teacher_logits=(
                None
                if self.teacher_logits is None
                else copy("teacher_logits", self.teacher_logits)
            ),
            ready=None,
            **tensors,
        )


@dataclass(slots=True)
class ConfirmedRows(Generic[RowsT]):
    """Owned CPU copies of one packet's confirmed rows."""

    positions: torch.Tensor
    features: RowsT


PacketT = TypeVar("PacketT", bound=TargetPacket)


def teacher_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Softmax teacher logits in FP32 and renormalize the rows in FP64."""
    probabilities = logits.float().softmax(-1)
    probabilities /= probabilities.sum(-1, keepdim=True, dtype=torch.float64)
    return probabilities


class TargetHistoryBuilder(Generic[PacketT, RowsT, BatchT]):
    """Build CPU rounds from target packets on the capture worker.

    Owns the request's contiguous history: prompt chunks accumulate until
    the first packet that verifies a proposal. Subclasses copy their rows
    and assemble the model-specific batch.
    """

    def __init__(
        self,
        *,
        on_round: Callable[[PreparedRound[BatchT]], object],
        on_epoch_close: Callable[[str], object],
        on_finish: Callable[[str], object],
    ) -> None:
        self._on_round = on_round
        self._on_epoch_close = on_epoch_close
        self._on_finish = on_finish

        self._request_id: str | None = None
        self._history_end = 0
        self._prompt: list[ConfirmedRows[RowsT]] = []

    def consume(self, packet: PacketT) -> None:
        self._set_request(packet.request_id)

        count, sampled = packet.metadata.tolist()
        positions = packet.positions[:count]
        expected_positions = torch.arange(
            self._history_end,
            self._history_end + count,
            dtype=torch.long,
        )
        if not torch.equal(positions, expected_positions):
            raise ValueError(
                "online draft capture requires contiguous target positions "
                "(disable prefix caching)"
            )
        self._history_end += count

        # The next transfer overwrites staging; keep owned copies only.
        confirmed = ConfirmedRows(positions.clone(), self._own_rows(packet, count))

        proposal = packet.proposal
        if proposal is None:
            self._prompt.append(confirmed)
            return

        if count != sampled:
            raise ValueError("confirmed rows must equal num_sampled")
        if packet.teacher_logits is None:
            raise RuntimeError("teacher logits are missing")

        prompt, self._prompt = self._prompt, []
        batch = self._build_batch(proposal, packet.teacher_logits, confirmed, prompt)
        self._on_round(
            PreparedRound(
                request_id=packet.request_id,
                step_id=packet.step_id,
                source_weight_version=proposal.source_weight_version,
                batch=batch,
            )
        )

    def close_epoch(self, request_id: str) -> None:
        if self._request_id != request_id:
            raise RuntimeError("closed epoch does not match active request")

        self._on_epoch_close(request_id)

    def finish_request(self, request_id: str) -> None:
        if self._request_id is not None and self._request_id != request_id:
            raise RuntimeError("finished request does not match active request")

        self._request_id = None
        self._history_end = 0
        self._prompt.clear()
        self._reset_history()

        self._on_finish(request_id)

    def _set_request(self, request_id: str) -> None:
        if self._request_id is None:
            self._request_id = request_id
        elif self._request_id != request_id:
            raise RuntimeError("capture runtime received multiple active requests")

    def _own_rows(self, packet: PacketT, count: int) -> RowsT:
        """Clone the first count model-specific rows out of staging."""
        raise NotImplementedError

    def _build_batch(
        self,
        proposal: Any,
        teacher_logits: torch.Tensor,
        confirmed: ConfirmedRows[RowsT],
        prompt: list[ConfirmedRows[RowsT]],
    ) -> BatchT:
        """Assemble one round; prompt is empty after the first round."""
        raise NotImplementedError

    def _reset_history(self) -> None:
        """Drop model-specific history at request finish."""


class CaptureAdapterBase(Generic[ProposalT]):
    """Turn inference callbacks into ordered target packets.

    Runs on the inference thread. A proposal is saved only after the prompt
    completes, the next target step's teacher logits are kept only for a
    saved proposal, and both travel in that step's packet.
    """

    _packet_type: ClassVar[type[TargetPacket]]

    def __init__(
        self,
        *,
        buffers: PinnedStaging,
        builder: TargetHistoryBuilder,
        source_weight_version: Callable[[], int],
        target_token_ids: torch.Tensor | None = None,
        on_queue_size: Callable[[int], None] | None = None,
        profile_timeline: bool = False,
        cpu_config: CPUWorkerConfig | None = None,
    ) -> None:
        self._buffers = buffers
        self._source_weight_version = source_weight_version
        self._target_token_ids = (
            None if target_token_ids is None else target_token_ids.detach()
        )
        self._profile_timeline = profile_timeline

        self._request_id: str | None = None
        self._step_id = 0
        self._capture_proposal = False
        self._proposal: ProposalT | None = None
        self._teacher_logits: torch.Tensor | None = None

        self._batch_builder = builder
        self._queue = CaptureQueue(
            consume=builder.consume,
            on_finish=builder.finish_request,
            on_close_epoch=builder.close_epoch,
            on_queue_size=on_queue_size,
            profile_timeline=profile_timeline,
            cpu_config=cpu_config,
        )

    @property
    def completed_rounds(self) -> int:
        return self._step_id

    def capture_teacher_logits(self, logits: torch.Tensor) -> None:
        """Keep teacher logits before sampling modifies them."""
        if self._proposal is None:
            return

        with self._profile("online_draft.capture_teacher"):
            if self._target_token_ids is not None:
                logits = logits.index_select(-1, self._target_token_ids)
            self._teacher_logits = logits.detach().clone()

    def finish_request(self, request_id: str) -> None:
        if self._request_id is not None and request_id != self._request_id:
            raise RuntimeError("request does not match active capture")

        self._queue.finish_request(request_id)

        self._request_id = None
        self._capture_proposal = False
        self._proposal = None
        self._teacher_logits = None
        self._step_id = 0

    def close_epoch(self, request_id: str) -> None:
        if request_id != self._request_id:
            raise RuntimeError("request does not match active capture")

        self._queue.close_epoch(request_id)

    def close(self) -> None:
        self._queue.close()

    def raise_if_failed(self) -> None:
        self._queue.raise_if_failed()

    def _profile(self, name: str) -> AbstractContextManager:
        return profile_range(name, self._profile_timeline)

    def _submit_target(
        self,
        *,
        request_id: str,
        final_prefill: bool,
        positions: torch.Tensor,
        confirmed_length: torch.Tensor,
        num_sampled: torch.Tensor,
        **features: torch.Tensor,
    ) -> None:
        """Queue one packet; features must already be owned GPU copies."""
        if self._request_id is None:
            self._request_id = request_id
        elif request_id != self._request_id:
            raise RuntimeError("another request is already being captured")

        if self._proposal is not None and self._teacher_logits is None:
            raise RuntimeError("teacher logits are missing")

        packet = self._packet_type(
            request_id=request_id,
            step_id=self._step_id,
            proposal=self._proposal,
            teacher_logits=self._teacher_logits,
            positions=positions.detach().clone(),
            metadata=torch.stack(
                (
                    confirmed_length.detach().reshape(()).long(),
                    num_sampled.detach().reshape(()).long(),
                )
            ),
            buffers=self._buffers,
            **features,
        )
        if packet.device.type == "cuda":
            packet.ready = torch.cuda.Event()
            packet.ready.record(torch.cuda.current_stream(packet.device))

        self._queue.submit(packet)

        if self._proposal is not None:
            self._step_id += 1
        self._proposal = None
        self._teacher_logits = None
        self._capture_proposal = final_prefill

    def _save_proposal(self, build: Callable[[int], ProposalT]) -> None:
        """Save a proposal for the next target step once the prompt is done."""
        if not self._capture_proposal:
            return

        with self._profile("online_draft.capture_proposal"):
            self._proposal = build(self._source_weight_version())
