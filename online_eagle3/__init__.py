# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .async_bridge import (
    Qwen3Eagle3AsyncBridge,
    ResetRequest,
)
from .ce_step import qwen3_eagle3_ce_step
from .checkpoint import convert_eagle3_checkpoint_state, load_torch_eagle3_model
from .config import Qwen3Eagle3TrainerConfig
from .data import DistillationBatch, Qwen3Eagle3StepFn, TrainObservation
from .distillation import qwen3_eagle3_distillation_step
from .factory import create_cpu_bridge
from .observations import (
    IGNORE_LABEL,
    DraftTrainLabels,
    DraftVerifyLabels,
    build_target_to_draft_map,
    build_verified_target_labels,
    map_target_to_draft_labels,
)
from .qwen3_trainer import Qwen3Eagle3CpuTrainer
from .sync_bridge import Qwen3Eagle3LazySyncBridge, Qwen3Eagle3SyncBridge
from .torch_eagle3 import (
    TorchEagle3Config,
    TorchEagle3ForCausalLM,
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
    "Qwen3Eagle3LazySyncBridge",
    "Qwen3Eagle3SyncBridge",
    "Qwen3Eagle3StepFn",
    "Qwen3Eagle3CpuTrainer",
    "ResetRequest",
    "qwen3_eagle3_ce_step",
    "IGNORE_LABEL",
    "DraftTrainLabels",
    "DraftVerifyLabels",
    "TrainableWeightSnapshot",
    "TrainObservation",
    "TorchEagle3Config",
    "TorchEagle3ForCausalLM",
    "build_target_to_draft_map",
    "build_verified_target_labels",
    "convert_eagle3_checkpoint_state",
    "export_trainable_state_dict",
    "freeze_parameters",
    "get_trainable_named_parameters",
    "load_torch_eagle3_model",
    "load_trainable_state_dict",
    "map_target_to_draft_labels",
    "Qwen3Eagle3TrainerConfig",
    "DistillationBatch",
    "qwen3_eagle3_distillation_step",
    "create_cpu_bridge",
]
