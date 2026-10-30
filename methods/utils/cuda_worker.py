# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run GPU work on a dedicated CUDA stream without blocking its caller."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import torch

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class CudaStreamResult(Generic[T]):
    value: T
    completed: torch.cuda.Event


@dataclass(frozen=True, slots=True)
class _CudaWorkItem:
    function: Callable[[], Any]
    inputs: Any
    ready: torch.cuda.Event | None
    future: concurrent.futures.Future[CudaStreamResult[Any]]


def _record_tensors_on_stream(value: Any, stream: torch.cuda.Stream) -> None:
    if isinstance(value, torch.Tensor):
        if value.device.type == "cuda":
            value.record_stream(stream)
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            _record_tensors_on_stream(getattr(value, field.name), stream)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _record_tensors_on_stream(key, stream)
            _record_tensors_on_stream(item, stream)
        return
    if isinstance(value, (tuple, list, set)):
        for item in value:
            _record_tensors_on_stream(item, stream)


class CudaStreamWorker:
    """Serialize submitted jobs on one side stream and report completion events."""

    def __init__(self, device: torch.device, *, name: str) -> None:
        if device.type != "cuda":
            raise ValueError("CUDA stream workers require a CUDA device")
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self._inbox: queue.Queue[_CudaWorkItem | None] = queue.Queue()
        self._lock = threading.Lock()
        self._error: BaseException | None = None
        self._closed = False

        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(device))
        self.stream.wait_event(ready)

        self._thread = threading.Thread(
            target=self._work,
            daemon=True,
            name=name,
        )
        self._thread.start()

    def submit(
        self,
        function: Callable[[], T],
        *,
        inputs: Any = None,
        ready: torch.cuda.Event | None = None,
    ) -> concurrent.futures.Future[CudaStreamResult[T]]:
        with self._lock:
            if self._closed:
                raise RuntimeError("CUDA stream worker is closed")
            if self._error is not None:
                raise RuntimeError("CUDA stream worker failed") from self._error
        future: concurrent.futures.Future[CudaStreamResult[T]] = (
            concurrent.futures.Future()
        )
        self._inbox.put(_CudaWorkItem(function, inputs, ready, future))
        return future

    def _work(self) -> None:
        with torch.accelerator.device_index(self.device.index):
            while True:
                item = self._inbox.get()
                if item is None:
                    return
                try:
                    with self._lock:
                        error = self._error
                    if error is not None:
                        raise RuntimeError("CUDA stream worker failed") from error
                    with torch.cuda.stream(self.stream):
                        if item.ready is not None:
                            self.stream.wait_event(item.ready)
                        _record_tensors_on_stream(item.inputs, self.stream)
                        value = item.function()
                        completed = torch.cuda.Event()
                        completed.record(self.stream)
                    item.future.set_result(CudaStreamResult(value, completed))
                except BaseException as error:
                    with self._lock:
                        if self._error is None:
                            self._error = error
                    item.future.set_exception(error)

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._inbox.put(None)
        if threading.current_thread() is not self._thread:
            self._thread.join()
        with torch.accelerator.device_index(self.device.index):
            self.stream.synchronize()
