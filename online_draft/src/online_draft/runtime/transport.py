# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from queue import Full, Queue
from threading import Lock, Thread

import torch

MAX_PENDING_CAPTURE = 200


class CaptureQueueFullError(RuntimeError):
    """The bounded capture queue is full."""


@dataclass(slots=True)
class Eagle3PromptContext:
    """Prompt data transferred once for one request."""

    prefill_input_embeds: torch.Tensor
    prefill_positions: torch.Tensor
    prefill_length: torch.Tensor
    prefill_aux_hidden_states: tuple[torch.Tensor, ...]
    prefill_aux_row_indices: torch.Tensor

    def to_cpu(self) -> Eagle3PromptContext:
        def copy(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.detach().to(device="cpu", copy=True)

        return replace(
            self,
            prefill_input_embeds=copy(self.prefill_input_embeds),
            prefill_positions=copy(self.prefill_positions),
            prefill_length=copy(self.prefill_length),
            prefill_aux_hidden_states=tuple(
                copy(tensor) for tensor in self.prefill_aux_hidden_states
            ),
            prefill_aux_row_indices=copy(self.prefill_aux_row_indices),
        )


@dataclass(slots=True)
class Eagle3CapturePacket:
    """One GPU-resident EAGLE3 verification round."""

    request_id: str
    step_id: int
    source_weight_version: int

    prompt_context: Eagle3PromptContext | None

    proposal_token_input_embeds: torch.Tensor
    proposal_recurrent_hidden_states: torch.Tensor

    teacher_logits: torch.Tensor
    verify_row_indices: torch.Tensor
    target_aux_hidden_states: tuple[torch.Tensor, ...]
    num_sampled: torch.Tensor

    confirmed_prefill_input_embeds: torch.Tensor
    confirmed_prefill_positions: torch.Tensor
    confirmed_prefill_length: torch.Tensor

    ready: torch.cuda.Event | None = None

    def to_cpu(self) -> Eagle3CapturePacket:
        def copy(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.detach().to(device="cpu", copy=True)

        prompt_context = (
            None if self.prompt_context is None else self.prompt_context.to_cpu()
        )

        return replace(
            self,
            prompt_context=prompt_context,
            proposal_token_input_embeds=copy(self.proposal_token_input_embeds),
            proposal_recurrent_hidden_states=copy(
                self.proposal_recurrent_hidden_states
            ),
            teacher_logits=copy(self.teacher_logits),
            verify_row_indices=copy(self.verify_row_indices),
            target_aux_hidden_states=tuple(
                copy(tensor) for tensor in self.target_aux_hidden_states
            ),
            num_sampled=copy(self.num_sampled),
            confirmed_prefill_input_embeds=copy(self.confirmed_prefill_input_embeds),
            confirmed_prefill_positions=copy(self.confirmed_prefill_positions),
            confirmed_prefill_length=copy(self.confirmed_prefill_length),
            ready=None,
        )


@dataclass(frozen=True, slots=True)
class _FinishRequest:
    request_id: str


@dataclass(frozen=True, slots=True)
class _ReleasePending:
    request_id: str


class Eagle3CaptureQueue:
    """Transfer bounded GPU packet chunks to the CPU worker."""

    def __init__(
        self,
        *,
        transfer_chunk_size: int,
        consume_chunk: Callable[
            [tuple[Eagle3CapturePacket, ...]],
            None,
        ],
        on_finish: Callable[[str], None],
        on_release: Callable[[str], None],
    ) -> None:
        if transfer_chunk_size <= 0:
            raise ValueError("transfer_chunk_size must be positive")

        self._transfer_chunk_size = transfer_chunk_size
        self._consume_chunk = consume_chunk
        self._on_finish = on_finish
        self._on_release = on_release
        self._queue: Queue[object] = Queue(MAX_PENDING_CAPTURE)
        self._stop = object()

        self._error: BaseException | None = None
        self._error_lock = Lock()

        self._worker = Thread(
            target=self._run,
            name="eagle3-capture",
            daemon=True,
        )
        self._worker.start()

    def submit(self, packet: Eagle3CapturePacket) -> None:
        self.raise_if_failed()
        try:
            self._queue.put_nowait(packet)
        except Full as error:
            queue_error = CaptureQueueFullError(
                f"capture queue is full ({MAX_PENDING_CAPTURE} packets)"
            )
            self._set_error(queue_error)
            raise queue_error from error

    def finish_request(self, request_id: str) -> None:
        try:
            self._queue.put_nowait(_FinishRequest(request_id))
        except Full as error:
            queue_error = CaptureQueueFullError(
                f"capture queue is full ({MAX_PENDING_CAPTURE} packets)"
            )
            self._set_error(queue_error)
            raise queue_error from error

    def release_pending(self, request_id: str) -> None:
        self.raise_if_failed()
        try:
            self._queue.put_nowait(_ReleasePending(request_id))
        except Full as error:
            queue_error = CaptureQueueFullError(
                f"capture queue is full ({MAX_PENDING_CAPTURE} packets)"
            )
            self._set_error(queue_error)
            raise queue_error from error

    def close(self) -> None:
        self._queue.put(self._stop)
        self._worker.join()
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        with self._error_lock:
            error = self._error

        if error is not None:
            raise error

    def _run(self) -> None:
        chunk: list[Eagle3CapturePacket] = []
        request_id: str | None = None

        def consume_chunk() -> None:
            if not chunk:
                return

            cpu_chunk = tuple(packet.to_cpu() for packet in chunk)
            self._consume_chunk(cpu_chunk)
            chunk.clear()

        while True:
            item = self._queue.get()

            try:
                if item is self._stop:
                    return

                if isinstance(item, _FinishRequest):
                    chunk.clear()
                    request_id = None
                    self._on_finish(item.request_id)
                    continue

                if isinstance(item, _ReleasePending):
                    consume_chunk()
                    self._on_release(item.request_id)
                    continue

                packet = item
                assert isinstance(packet, Eagle3CapturePacket)

                if request_id is None:
                    request_id = packet.request_id
                elif packet.request_id != request_id:
                    raise RuntimeError(
                        "capture queue received multiple active requests"
                    )

                if packet.ready is not None:
                    packet.ready.synchronize()

                chunk.append(packet)

                if len(chunk) == self._transfer_chunk_size:
                    consume_chunk()
            except Exception as error:
                chunk.clear()
                self._set_error(error)
            finally:
                self._queue.task_done()

    def _set_error(self, error: BaseException) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = error
