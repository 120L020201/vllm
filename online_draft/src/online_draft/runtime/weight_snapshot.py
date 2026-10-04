# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class DraftWeightSnapshot:
    """Detached CPU weights owned by one install attempt."""

    version: int
    tensors: tuple[tuple[str, torch.Tensor], ...]

    @classmethod
    def from_named_tensors(
        cls,
        named_tensors: Iterable[tuple[str, torch.Tensor]],
        *,
        version: int,
        pin_memory: bool = True,
    ) -> DraftWeightSnapshot:
        """Copy named tensors into CPU-owned storage."""
        use_pinned_memory = pin_memory and torch.cuda.is_available()
        tensors = []
        for name, tensor in named_tensors:
            cpu_tensor = torch.empty_like(
                tensor,
                device="cpu",
                pin_memory=use_pinned_memory,
            )
            cpu_tensor.copy_(tensor.detach())
            tensors.append((name, cpu_tensor))

        return cls(version=version, tensors=tuple(tensors))
