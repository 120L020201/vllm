# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

import torch
from typing_extensions import Self

if TYPE_CHECKING:
    from online_draft.training.trainer import DraftTrainer

BatchT = TypeVar("BatchT")


@dataclass(frozen=True, slots=True)
class PreparedRound(Generic[BatchT]):
    """One CPU batch with request and weight metadata."""

    request_id: str
    step_id: int
    source_weight_version: int
    batch: BatchT


class WindowResult(Protocol):
    """Metrics the pipeline reads after one window update."""

    @property
    def model_version(self) -> int: ...

    @property
    def round_count(self) -> int: ...

    @property
    def target_count(self) -> int: ...


class TrainRounds(Protocol):
    """Train one window and return its updated persistent cache.

    The cache is method-specific and None before the first window.
    """

    def __call__(
        self,
        trainer: DraftTrainer,
        rounds: tuple[Any, ...],
        cache: Any,
        profile_timeline: bool,
        /,
    ) -> tuple[Any, WindowResult]: ...


class CapturePacket(Protocol):
    """GPU payload explicitly copied by the capture worker."""

    request_id: str
    ready: torch.cuda.Event | None

    @property
    def device(self) -> torch.device: ...

    def to_cpu(self, *, non_blocking: bool = False) -> Self: ...


class CaptureAdapter(Protocol):
    """Inference-thread capture interface shared by online draft methods.

    capture_target and capture_proposal take model-specific tensors.
    """

    @property
    def capture_target(self) -> Callable[..., None]: ...

    @property
    def capture_proposal(self) -> Callable[..., None]: ...

    @property
    def completed_rounds(self) -> int: ...

    def capture_teacher_logits(self, logits: torch.Tensor) -> None: ...

    def close_epoch(self, request_id: str) -> None: ...

    def finish_request(self, request_id: str) -> None: ...

    def raise_if_failed(self) -> None: ...

    def close(self) -> None: ...
