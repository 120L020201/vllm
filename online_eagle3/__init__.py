# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .qwen3_trainer import Qwen3Eagle3CpuTrainer
from .async_bridge import Qwen3Eagle3AsyncBridge, ResetRequest, TrainObservation
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
    "Qwen3Eagle3AsyncBridge",
    "Qwen3Eagle3CpuTrainer",
    "ResetRequest",
    "TrainableWeightSnapshot",
    "TrainObservation",
    "export_trainable_state_dict",
    "freeze_parameters",
    "get_trainable_named_parameters",
    "load_trainable_state_dict",
]
