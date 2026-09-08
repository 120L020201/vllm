# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from typing import Any

import torch

from .async_bridge import Qwen3Eagle3AsyncBridge, Qwen3Eagle3LazyBridge
from .ce_step import qwen3_eagle3_ce_step
from .qwen3_trainer import Qwen3Eagle3CpuTrainer, Qwen3Eagle3TrainerConfig
from .torch_eagle3 import load_torch_eagle3_model

_TRUE_VALUES = {"1", "true", "yes", "on"}
_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def maybe_create_qwen3_eagle3_bridge(
    vllm_config: Any,
) -> Qwen3Eagle3LazyBridge | None:
    """Create the experimental CPU update bridge when explicitly enabled."""
    if os.environ.get("VLLM_ONLINE_EAGLE3", "0").lower() not in _TRUE_VALUES:
        return None

    speculative_config = getattr(vllm_config, "speculative_config", None)
    if speculative_config is None or speculative_config.method != "eagle3":
        raise ValueError("VLLM_ONLINE_EAGLE3 requires EAGLE3 speculative decoding")

    _verify_single_rank(vllm_config)

    draft_model_config = speculative_config.draft_model_config
    draft_model_path = (
        os.environ.get("VLLM_ONLINE_EAGLE3_DRAFT_MODEL") or draft_model_config.model
    )
    if not draft_model_path:
        raise ValueError("Unable to resolve EAGLE3 draft model path")

    trainer_config = Qwen3Eagle3TrainerConfig(
        lr=float(os.environ.get("VLLM_ONLINE_EAGLE3_LR", "1e-5")),
        weight_decay=float(os.environ.get("VLLM_ONLINE_EAGLE3_WEIGHT_DECAY", "0.0")),
    )
    dtype = _get_dtype()
    update_interval = int(os.environ.get("VLLM_ONLINE_EAGLE3_UPDATE_INTERVAL", "1"))

    def create_bridge() -> Qwen3Eagle3AsyncBridge:
        _maybe_set_torch_threads()
        with torch.inference_mode(False):
            trainer = Qwen3Eagle3CpuTrainer(
                load_torch_eagle3_model(draft_model_path, dtype=dtype),
                trainer_config,
            )
        return Qwen3Eagle3AsyncBridge(
            trainer,
            qwen3_eagle3_ce_step,
            update_interval=update_interval,
        )

    return Qwen3Eagle3LazyBridge(create_bridge)


def _verify_single_rank(vllm_config: Any) -> None:
    parallel_config = getattr(vllm_config, "parallel_config", None)
    tensor_parallel_size = getattr(parallel_config, "tensor_parallel_size", 1)
    pipeline_parallel_size = getattr(parallel_config, "pipeline_parallel_size", 1)
    data_parallel_size = getattr(parallel_config, "data_parallel_size", 1)
    if (
        tensor_parallel_size != 1
        or pipeline_parallel_size != 1
        or data_parallel_size != 1
    ):
        raise ValueError("VLLM_ONLINE_EAGLE3 currently supports only TP=PP=DP=1")


def _maybe_set_torch_threads() -> None:
    value = os.environ.get("VLLM_ONLINE_EAGLE3_TORCH_THREADS")
    if value is None:
        return
    torch.set_num_threads(max(1, int(value)))


def _get_dtype() -> torch.dtype:
    value = os.environ.get("VLLM_ONLINE_EAGLE3_DTYPE", "float32").lower()
    try:
        return _DTYPES[value]
    except KeyError as exc:
        raise ValueError(
            f"VLLM_ONLINE_EAGLE3_DTYPE must be one of {sorted(_DTYPES)}, got {value!r}"
        ) from exc
