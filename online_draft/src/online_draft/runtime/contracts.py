# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, Self, TypeVar

import torch

BatchT = TypeVar("BatchT")


@dataclass(frozen=True, slots=True)
class PreparedRound(Generic[BatchT]):
    """One CPU batch with request and weight metadata."""

    request_id: str
    step_id: int
    source_weight_version: int
    batch: BatchT


class CapturePacket(Protocol):
    """GPU payload explicitly copied by the capture worker."""

    request_id: str
    ready: torch.cuda.Event | None

    @property
    def device(self) -> torch.device: ...

    def to_cpu(self, *, non_blocking: bool = False) -> Self: ...
