# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import dataclass

import torch

from .weights import QWEN3_EAGLE3_FROZEN_PREFIXES


@dataclass(frozen=True, slots=True)
class Qwen3Eagle3TrainerConfig:
    """CPU training settings, independent of the inference engine."""

    frozen_prefixes: tuple[str, ...] = QWEN3_EAGLE3_FROZEN_PREFIXES
    lr: float = 1e-5
    weight_decay: float = 0.0
    dtype: torch.dtype = torch.float32
    torch_threads: int | None = None
    update_interval: int = 1
    check_gradients: bool = True

    def __post_init__(self) -> None:
        if self.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("CPU dtype must be float32 or bfloat16")
        if self.update_interval != 1:
            raise ValueError("Append-only CPU KV training requires S=1")
        if self.torch_threads is not None and self.torch_threads < 1:
            raise ValueError("torch_threads must be >= 1")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
