# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from dataclasses import dataclass, replace

import torch

from online_draft.runtime import profile_range
from online_draft.runtime.contracts import PreparedRound
from online_draft.runtime.cpu_worker import CPUWorkerConfig
from online_draft.runtime.transport import CaptureQueue
from online_draft.training.dflash.window import DFlashBatch


class DFlashCaptureBuffers:
    """One pinned staging area, exclusively reused by the serial capture worker."""

    def __init__(
        self,
        max_tokens,
        num_query_tokens,
        feature_size,
        vocab_size,
        *,
        dtype,
        teacher_dtype,
    ):
        layouts = {
            "proposal_metadata": ((2, num_query_tokens), torch.int64),
            "target_fc": ((max_tokens, feature_size), dtype),
            "target_metadata": ((max_tokens + 2,), torch.int64),
            "teacher_logits": ((num_query_tokens, vocab_size), teacher_dtype),
        }
        with torch.inference_mode(False):
            self.tensors = {
                name: torch.empty(
                    shape,
                    dtype=dtype,
                    device="cpu",
                    pin_memory=torch.cuda.is_available(),
                )
                for name, (shape, dtype) in layouts.items()
            }

    def copy(self, name, tensor, non_blocking):
        destination = self.tensors[name].view(-1)[: tensor.numel()].view(tensor.shape)
        destination.copy_(tensor.detach(), non_blocking=non_blocking)
        return destination


@dataclass(frozen=True)
class DFlashProposal:
    """Snapshot rows contain input token IDs and absolute query positions."""

    metadata: torch.Tensor
    source_weight_version: int

    def to_cpu(self, buffers, *, non_blocking: bool = False) -> "DFlashProposal":
        return replace(
            self,
            metadata=buffers.copy("proposal_metadata", self.metadata, non_blocking),
        )


@dataclass(frozen=True)
class DFlashCapturePacket:
    """Target metadata contains positions followed by rejected/sample counts."""

    request_id: str
    step_id: int
    proposal: DFlashProposal | None
    target_fc: torch.Tensor
    target_metadata: torch.Tensor
    teacher_logits: torch.Tensor | None
    ready: torch.cuda.Event | None
    buffers: DFlashCaptureBuffers

    @property
    def device(self) -> torch.device:
        return self.target_fc.device

    def to_cpu(self, *, non_blocking: bool = False) -> "DFlashCapturePacket":
        def copy(name, tensor):
            return self.buffers.copy(name, tensor, non_blocking)

        return replace(
            self,
            proposal=(
                None
                if self.proposal is None
                else self.proposal.to_cpu(self.buffers, non_blocking=non_blocking)
            ),
            target_fc=copy("target_fc", self.target_fc),
            target_metadata=copy("target_metadata", self.target_metadata),
            teacher_logits=(
                None
                if self.teacher_logits is None
                else copy("teacher_logits", self.teacher_logits)
            ),
            ready=None,
        )


class DFlashCaptureAdapter:
    """Capture FC outputs and prepare aligned rounds on the D2H worker."""

    def __init__(
        self,
        *,
        buffers: DFlashCaptureBuffers,
        on_round: Callable,
        on_epoch_close: Callable,
        on_finish: Callable,
        on_queue_size: Callable | None = None,
        profile_timeline: bool = False,
        cpu_config: CPUWorkerConfig | None = None,
    ):
        self._on_round = on_round
        self._buffers = buffers
        self._on_finish = on_finish
        self._profile_timeline = profile_timeline
        self._proposal: DFlashProposal | None = None
        self._teacher: torch.Tensor | None = None
        self._step_id = 0
        self._capture_proposal = False
        # Only the capture worker owns the CPU history below.
        self._prompt: list[torch.Tensor] = []
        self._history_end = 0
        self._queue = CaptureQueue(
            transfer_chunk_size=1,
            consume_chunk=self._consume,
            on_close_epoch=on_epoch_close,
            on_finish=self._finish_cpu,
            on_queue_size=on_queue_size,
            profile_timeline=profile_timeline,
            cpu_config=cpu_config,
        )

    @property
    def completed_rounds(self) -> int:
        return self._step_id

    def capture_teacher_logits(self, logits: torch.Tensor) -> None:
        if self._proposal is not None:
            with profile_range("online_draft.capture_teacher", self._profile_timeline):
                self._teacher = logits.detach().clone()

    def capture_target(
        self,
        *,
        request_id: str,
        fc_output: torch.Tensor,
        positions: torch.Tensor,
        num_rejected: torch.Tensor,
        num_sampled: torch.Tensor,
        final_prefill: bool,
    ) -> None:
        with profile_range("online_draft.capture_target", self._profile_timeline):
            fc_output = fc_output.detach().clone()
            metadata = torch.cat(
                (positions, num_rejected.reshape(1), num_sampled.reshape(1))
            )
            ready = None
            if fc_output.is_cuda:
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(fc_output.device))
            self._queue.submit(
                DFlashCapturePacket(
                    request_id,
                    self._step_id,
                    self._proposal,
                    fc_output,
                    metadata,
                    self._teacher,
                    ready,
                    self._buffers,
                )
            )
        if self._proposal is not None:
            self._step_id += 1
        self._proposal = None
        self._teacher = None
        self._capture_proposal = final_prefill

    def capture_proposal(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        version: int,
    ) -> None:
        if self._capture_proposal:
            with profile_range("online_draft.capture_proposal", self._profile_timeline):
                self._proposal = DFlashProposal(
                    torch.stack((input_ids, positions)), version
                )

    def _consume(self, packets: tuple[DFlashCapturePacket, ...]) -> None:
        for packet in packets:
            target_positions = packet.target_metadata[:-2]
            rejected, sampled = packet.target_metadata[-2:].tolist()
            count = target_positions.numel() - rejected
            positions = target_positions[:count]
            assert torch.equal(
                positions, torch.arange(self._history_end, self._history_end + count)
            ), "DFlash requires continuous target context (disable prefix caching)"
            self._history_end += count
            confirmed = packet.target_fc[:count].clone()
            if packet.proposal is None:
                self._prompt.append(confirmed)
                continue
            proposal = packet.proposal
            input_ids, query_positions = proposal.metadata.unbind()
            assert int(positions[0]) == int(query_positions[0])
            assert packet.teacher_logits is not None
            k = input_ids.numel() - 1
            targets = min(sampled, k)
            assert targets > 0
            probabilities = packet.teacher_logits[:targets].float().softmax(-1)
            probabilities /= probabilities.sum(-1, keepdim=True, dtype=torch.float64)
            batch = DFlashBatch(
                input_ids=input_ids.clone(),
                positions=query_positions.clone(),
                prompt_fc=torch.cat(self._prompt) if self._prompt else None,
                confirmed_fc=confirmed,
                teacher_probabilities=probabilities,
                rejection_position=sampled - 1 if sampled <= k else None,
            )
            self._prompt.clear()
            self._on_round(
                PreparedRound(
                    packet.request_id,
                    packet.step_id,
                    proposal.source_weight_version,
                    batch,
                )
            )

    def close_epoch(self, request_id: str) -> None:
        self._queue.close_epoch(request_id)

    def _finish_cpu(self, request_id: str) -> None:
        self._prompt.clear()
        self._history_end = 0
        self._on_finish(request_id)

    def finish_request(self, request_id: str) -> None:
        self._queue.finish_request(request_id)
        self._proposal = None
        self._teacher = None
        self._step_id = 0
        self._capture_proposal = False

    def raise_if_failed(self) -> None:
        self._queue.raise_if_failed()

    def close(self) -> None:
        self._queue.close()
