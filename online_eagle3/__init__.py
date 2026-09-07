# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .qwen3_trainer import Qwen3Eagle3CpuTrainer
from .weights import (
    QWEN3_EAGLE3_FROZEN_PREFIXES,
    TrainableWeightSnapshot,
    export_trainable_state_dict,
    freeze_parameters,
    get_trainable_named_parameters,
    load_trainable_state_dict,
)

__all__ = [
    "QWEN3_EAGLE3_FROZEN_PREFIXES",
    "Qwen3Eagle3CpuTrainer",
    "TrainableWeightSnapshot",
    "export_trainable_state_dict",
    "freeze_parameters",
    "get_trainable_named_parameters",
    "load_trainable_state_dict",
]
