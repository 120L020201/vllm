# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from queue import Queue
from threading import Event, Lock, Thread

import torch


def _copy_to_cpu(
    tensor: torch.Tensor,
    *,
    non_blocking: bool,
) -> torch.Tensor:
    cpu_tensor = torch.empty_like(
        tensor,
        device="cpu",
        pin_memory=non_blocking,
    )
    cpu_tensor.copy_(
        tensor.detach(),
        non_blocking=non_blocking,
    )
    return cpu_tensor


@dataclass(slots=True)
class Eagle3PromptContext:
    """Prompt data transferred once for one request."""

    prefill_input_embeds: torch.Tensor
    prefill_positions: torch.Tensor
    prefill_length: torch.Tensor
    prefill_aux_hidden_states: tuple[torch.Tensor, ...]
    prefill_aux_row_indices: torch.Tensor

    def to_cpu(
        self,
        *,
        non_blocking: bool = False,
    ) -> Eagle3PromptContext:
        return replace(
            self,
            prefill_input_embeds=_copy_to_cpu(
                self.prefill_input_embeds,
                non_blocking=non_blocking,
            ),
            prefill_positions=_copy_to_cpu(
                self.prefill_positions,
                non_blocking=non_blocking,
            ),
            prefill_length=_copy_to_cpu(
                self.prefill_length,
                non_blocking=non_blocking,
            ),
            prefill_aux_hidden_states=tuple(
                _copy_to_cpu(
                    tensor,
                    non_blocking=non_blocking,
                )
                for tensor in self.prefill_aux_hidden_states
            ),
            prefill_aux_row_indices=_copy_to_cpu(
                self.prefill_aux_row_indices,
                non_blocking=non_blocking,
            ),
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

    def to_cpu(
        self,
        *,
        non_blocking: bool = False,
    ) -> Eagle3CapturePacket:
        prompt_context = (
            None
            if self.prompt_context is None
            else self.prompt_context.to_cpu(
                non_blocking=non_blocking,
            )
        )

        return replace(
            self,
            prompt_context=prompt_context,
            proposal_token_input_embeds=_copy_to_cpu(
                self.proposal_token_input_embeds,
                non_blocking=non_blocking,
            ),
            proposal_recurrent_hidden_states=_copy_to_cpu(
                self.proposal_recurrent_hidden_states,
                non_blocking=non_blocking,
            ),
            teacher_logits=_copy_to_cpu(
                self.teacher_logits,
                non_blocking=non_blocking,
            ),
            verify_row_indices=_copy_to_cpu(
                self.verify_row_indices,
                non_blocking=non_blocking,
            ),
            target_aux_hidden_states=tuple(
                _copy_to_cpu(
                    tensor,
                    non_blocking=non_blocking,
                )
                for tensor in self.target_aux_hidden_states
            ),
            num_sampled=_copy_to_cpu(
                self.num_sampled,
                non_blocking=non_blocking,
            ),
            confirmed_prefill_input_embeds=_copy_to_cpu(
                self.confirmed_prefill_input_embeds,
                non_blocking=non_blocking,
            ),
            confirmed_prefill_positions=_copy_to_cpu(
                self.confirmed_prefill_positions,
                non_blocking=non_blocking,
            ),
            confirmed_prefill_length=_copy_to_cpu(
                self.confirmed_prefill_length,
                non_blocking=non_blocking,
            ),
            ready=None,
        )


@dataclass(frozen=True, slots=True)
class _FinishRequest:
    request_id: str
    done: Event


@dataclass(frozen=True, slots=True)
class _CloseEpoch:
    request_id: str


class Eagle3CaptureQueue:
    """Transfer GPU packet chunks to the CPU worker in FIFO order."""

    def __init__(
        self,
        *,
        transfer_chunk_size: int,
        consume_chunk: Callable[
            [tuple[Eagle3CapturePacket, ...]],
            None,
        ],
        on_finish: Callable[[str], None],
        on_close_epoch: Callable[[str], None],
        on_queue_size: Callable[[int], None] | None = None,
    ) -> None:
        if transfer_chunk_size <= 0:
            raise ValueError("transfer_chunk_size must be positive")

        self._transfer_chunk_size = transfer_chunk_size
        self._consume_chunk = consume_chunk
        self._on_finish = on_finish
        self._on_close_epoch = on_close_epoch
        self._on_queue_size = on_queue_size
        self._queue: Queue[object] = Queue()
        self._stop = object()

        self._error: BaseException | None = None
        self._error_lock = Lock()

        self._worker = Thread(
            target=self._run,
            name="eagle3-capture",
            daemon=True,
        )
        self._worker.start()

    def submit(self, packet: Eagle3CapturePacket) -> int:
        self.raise_if_failed()
        self._queue.put_nowait(packet)
        queue_size = self._queue.qsize()

        if self._on_queue_size is not None:
            self._on_queue_size(queue_size)

        return queue_size

    def close_epoch(self, request_id: str) -> None:
        self.raise_if_failed()
        self._queue.put_nowait(_CloseEpoch(request_id))

    def finish_request(self, request_id: str) -> None:
        self.raise_if_failed()

        item = _FinishRequest(
            request_id=request_id,
            done=Event(),
        )
        self._queue.put_nowait(item)

        item.done.wait()
        self.raise_if_failed()

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
        active_request_id: str | None = None
        d2h_stream: torch.cuda.Stream | None = None

        def consume_chunk() -> None:
            nonlocal d2h_stream

            if not chunk:
                return

            if chunk[0].ready is None:
                cpu_chunk = tuple(packet.to_cpu() for packet in chunk)
            else:
                if d2h_stream is None:
                    d2h_stream = torch.cuda.Stream(
                        device=chunk[0].proposal_token_input_embeds.device,
                    )

                cpu_packets: list[Eagle3CapturePacket] = []
                with torch.cuda.stream(d2h_stream):
                    for packet in chunk:
                        assert packet.ready is not None
                        d2h_stream.wait_event(packet.ready)
                        cpu_packets.append(
                            packet.to_cpu(non_blocking=True),
                        )

                    copy_done = torch.cuda.Event()
                    copy_done.record(d2h_stream)

                copy_done.synchronize()
                cpu_chunk = tuple(cpu_packets)

            self._consume_chunk(cpu_chunk)
            chunk.clear()

        while True:
            item = self._queue.get()

            try:
                if item is self._stop:
                    return

                if isinstance(item, _FinishRequest):
                    if (
                        active_request_id is not None
                        and active_request_id != item.request_id
                    ):
                        raise RuntimeError(
                            "finished request does not match active request"
                        )

                    chunk.clear()
                    self._on_finish(item.request_id)
                    active_request_id = None
                    continue

                if isinstance(item, _CloseEpoch):
                    if active_request_id != item.request_id:
                        raise RuntimeError("closed epoch does not match active request")

                    consume_chunk()
                    self._on_close_epoch(item.request_id)
                    continue

                packet = item
                assert isinstance(packet, Eagle3CapturePacket)

                if active_request_id is None:
                    active_request_id = packet.request_id
                elif packet.request_id != active_request_id:
                    raise RuntimeError(
                        "capture queue received multiple active requests"
                    )

                chunk.append(packet)

                if len(chunk) == self._transfer_chunk_size:
                    consume_chunk()
            except Exception as error:
                chunk.clear()
                self._set_error(error)
                return
            finally:
                if isinstance(item, _FinishRequest):
                    item.done.set()
                self._queue.task_done()

    def _set_error(self, error: BaseException) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = error
