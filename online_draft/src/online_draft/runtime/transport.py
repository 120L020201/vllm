# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from queue import Queue
from threading import Event, Lock, Thread
from typing import cast

import torch

from online_draft.runtime import profile_range
from online_draft.runtime.contracts import CapturePacket
from online_draft.runtime.cpu_worker import CPUWorkerConfig


class PinnedStaging:
    """Pinned D2H destinations reused by the serial capture worker.

    Each copy overwrites the previous one for the same name, so the capture
    queue must consume one packet per transfer, and consumers must clone
    whatever they keep beyond that packet.
    """

    def __init__(
        self,
        layouts: Mapping[str, tuple[tuple[int, ...], torch.dtype]],
    ) -> None:
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

    def copy(
        self,
        name: str,
        tensor: torch.Tensor,
        non_blocking: bool,
    ) -> torch.Tensor:
        storage = self.tensors[name]
        if tensor.dtype != storage.dtype:
            raise ValueError(
                f"{name} uses {tensor.dtype}, but staging stores {storage.dtype}"
            )
        if tensor.numel() > storage.numel():
            raise ValueError(
                f"{name} has {tensor.numel()} elements, "
                f"but staging holds {storage.numel()}"
            )

        destination = storage.view(-1)[: tensor.numel()].view(tensor.shape)
        destination.copy_(tensor.detach(), non_blocking=non_blocking)
        return destination


@dataclass(frozen=True, slots=True)
class _FinishRequest:
    request_id: str
    done: Event


@dataclass(frozen=True, slots=True)
class _CloseEpoch:
    request_id: str


class CaptureQueue:
    """Transfer GPU packets to the CPU worker one at a time in FIFO order.

    One packet per transfer lets packets reuse a single pinned staging area.
    """

    def __init__(
        self,
        *,
        consume: Callable[[CapturePacket], None],
        on_finish: Callable[[str], None],
        on_close_epoch: Callable[[str], None],
        on_queue_size: Callable[[int], None] | None = None,
        profile_timeline: bool = False,
        cpu_config: CPUWorkerConfig | None = None,
    ) -> None:
        self._consume = consume
        self._on_finish = on_finish
        self._on_close_epoch = on_close_epoch
        self._on_queue_size = on_queue_size
        self._profile_timeline = profile_timeline
        self._cpu_config = cpu_config
        self._queue: Queue[object] = Queue()
        self._stop = object()

        self._error: BaseException | None = None
        self._error_lock = Lock()

        self._worker = Thread(
            target=self._run,
            name="online-draft-capture",
            daemon=True,
        )
        self._worker.start()

    def submit(self, packet: CapturePacket) -> int:
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
        active_request_id: str | None = None
        d2h_stream: torch.cuda.Stream | None = None

        def consume(packet: CapturePacket) -> None:
            nonlocal d2h_stream

            with profile_range("online_draft.d2h", self._profile_timeline):
                if packet.ready is None:
                    cpu_packet = packet.to_cpu()
                else:
                    if d2h_stream is None:
                        d2h_stream = torch.cuda.Stream(device=packet.device)

                    with torch.cuda.stream(d2h_stream):
                        d2h_stream.wait_event(packet.ready)
                        cpu_packet = packet.to_cpu(non_blocking=True)
                        # Sleep rather than spin: a spinning core lowers the
                        # turbo clock of the inference thread.
                        copy_done = torch.cuda.Event(blocking=True)
                        copy_done.record(d2h_stream)

                    copy_done.synchronize()

            with profile_range("online_draft.prepare", self._profile_timeline):
                self._consume(cpu_packet)

        while True:
            item = self._queue.get()

            try:
                if item is self._stop:
                    return

                if self._cpu_config is not None:
                    config, self._cpu_config = self._cpu_config, None
                    config.apply()

                if isinstance(item, _FinishRequest):
                    if (
                        active_request_id is not None
                        and active_request_id != item.request_id
                    ):
                        raise RuntimeError(
                            "finished request does not match active request"
                        )

                    self._on_finish(item.request_id)
                    active_request_id = None
                    continue

                if self._error is not None:
                    continue

                if isinstance(item, _CloseEpoch):
                    if active_request_id != item.request_id:
                        raise RuntimeError("closed epoch does not match active request")

                    self._on_close_epoch(item.request_id)
                    continue

                packet = cast(CapturePacket, item)

                if active_request_id is None:
                    active_request_id = packet.request_id
                elif packet.request_id != active_request_id:
                    raise RuntimeError(
                        "capture queue received multiple active requests"
                    )

                consume(packet)
            except Exception as error:
                self._set_error(error)
            finally:
                if isinstance(item, _FinishRequest):
                    item.done.set()
                self._queue.task_done()

    def _set_error(self, error: BaseException) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = error
