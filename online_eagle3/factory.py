# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger

from .ce_step import qwen3_eagle3_ce_step
from .qwen3_trainer import Qwen3Eagle3CpuTrainer, Qwen3Eagle3TrainerConfig
from .sync_bridge import Qwen3Eagle3LazySyncBridge, Qwen3Eagle3SyncBridge
from .torch_eagle3 import load_torch_eagle3_model

logger = init_logger("vllm.online_eagle3.factory")

_TRUE_VALUES = {"1", "true", "yes", "on"}
_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def maybe_create_qwen3_eagle3_sync_bridge(
    vllm_config: VllmConfig,
) -> Qwen3Eagle3LazySyncBridge | None:
    if not _env_enabled("VLLM_ONLINE_EAGLE3"):
        return None

    speculative_config = vllm_config.speculative_config
    if speculative_config is None or speculative_config.method != "eagle3":
        raise ValueError("VLLM_ONLINE_EAGLE3 requires --speculative-config eagle3")

    _require_single_request(vllm_config)
    _require_single_parallel_worker(vllm_config)

    draft_model_path = os.environ.get("VLLM_ONLINE_EAGLE3_DRAFT_MODEL")
    if draft_model_path is None:
        draft_model_path = getattr(
            speculative_config.draft_model_config,
            "model",
            None,
        )
    if not draft_model_path:
        raise ValueError(
            "VLLM_ONLINE_EAGLE3 requires VLLM_ONLINE_EAGLE3_DRAFT_MODEL "
            "or a draft model path in the speculative config"
        )

    update_interval = _get_int_env("VLLM_ONLINE_EAGLE3_UPDATE_INTERVAL", 1)
    lr = _get_float_env("VLLM_ONLINE_EAGLE3_LR", 1e-5)
    weight_decay = _get_float_env("VLLM_ONLINE_EAGLE3_WEIGHT_DECAY", 0.0)
    dtype = _get_dtype_env("VLLM_ONLINE_EAGLE3_DTYPE", torch.float32)
    torch_threads = _get_optional_int_env("VLLM_ONLINE_EAGLE3_TORCH_THREADS")

    logger.info(
        "Enabled synchronous online EAGLE3 updates: draft_model=%s, "
        "update_interval=%d, lr=%s, weight_decay=%s, dtype=%s, "
        "torch_threads=%s",
        draft_model_path,
        update_interval,
        lr,
        weight_decay,
        dtype,
        torch_threads,
    )

    def bridge_factory() -> Qwen3Eagle3SyncBridge:
        if torch_threads is not None:
            torch.set_num_threads(torch_threads)
        logger.info("Loading online EAGLE3 CPU draft from %s", draft_model_path)
        with torch.inference_mode(False), torch.enable_grad():
            cpu_model = load_torch_eagle3_model(draft_model_path, dtype=dtype)
            cpu_model.train()
            trainer = Qwen3Eagle3CpuTrainer(
                cpu_model,
                Qwen3Eagle3TrainerConfig(
                    lr=lr,
                    weight_decay=weight_decay,
                ),
            )
        return Qwen3Eagle3SyncBridge(
            trainer,
            qwen3_eagle3_ce_step,
            update_interval=update_interval,
        )

    return Qwen3Eagle3LazySyncBridge(bridge_factory)


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


def _get_int_env(name: str, default: int) -> int:
    value = _get_optional_int_env(name)
    if value is None:
        return default
    return value


def _get_optional_int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    value = int(raw)
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _get_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _get_dtype_env(name: str, default: torch.dtype) -> torch.dtype:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    dtype = _DTYPES.get(raw.strip().lower())
    if dtype is None:
        raise ValueError(f"{name} must be one of {sorted(_DTYPES)}")
    return dtype


def _require_single_request(vllm_config: VllmConfig) -> None:
    _require_config_value(
        vllm_config.scheduler_config,
        "max_num_seqs",
        1,
        "VLLM_ONLINE_EAGLE3 currently supports only --max-num-seqs 1",
    )


def _require_single_parallel_worker(vllm_config: VllmConfig) -> None:
    parallel_config = vllm_config.parallel_config
    _require_config_value(
        parallel_config,
        "tensor_parallel_size",
        1,
        "VLLM_ONLINE_EAGLE3 currently supports only tensor_parallel_size=1",
    )
    _require_config_value(
        parallel_config,
        "pipeline_parallel_size",
        1,
        "VLLM_ONLINE_EAGLE3 currently supports only pipeline_parallel_size=1",
    )
    _require_config_value(
        parallel_config,
        "data_parallel_size",
        1,
        "VLLM_ONLINE_EAGLE3 currently supports only data_parallel_size=1",
    )


def _require_config_value(
    config: Any,
    name: str,
    expected: int,
    message: str,
) -> None:
    actual = getattr(config, name, expected)
    if actual != expected:
        raise ValueError(f"{message}; got {name}={actual}")
