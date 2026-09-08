# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .qwen3_trainer import Qwen3Eagle3CpuTrainer
from .async_bridge import (
    Qwen3Eagle3AsyncBridge,
    Qwen3Eagle3StepFn,
    ResetRequest,
    TrainObservation,
)
from .observations import (
    IGNORE_LABEL,
    DraftTrainLabels,
    DraftVerifyLabels,
    build_target_to_draft_map,
    build_verified_target_labels,
    map_target_to_draft_labels,
)
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
    "Qwen3Eagle3StepFn",
    "Qwen3Eagle3CpuTrainer",
    "ResetRequest",
    "IGNORE_LABEL",
    "DraftTrainLabels",
    "DraftVerifyLabels",
    "TrainableWeightSnapshot",
    "TrainObservation",
    "build_target_to_draft_map",
    "build_verified_target_labels",
    "export_trainable_state_dict",
    "freeze_parameters",
    "get_trainable_named_parameters",
    "load_trainable_state_dict",
    "map_target_to_draft_labels",
]
