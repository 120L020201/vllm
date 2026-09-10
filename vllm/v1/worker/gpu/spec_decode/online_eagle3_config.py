# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import os
from typing import Any

import torch

from online_eagle3.config import Qwen3Eagle3TrainerConfig
from online_eagle3.factory import create_cpu_bridge
from online_eagle3.sync_bridge import Qwen3Eagle3LazySyncBridge
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

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
    speculative = vllm_config.speculative_config
    if speculative is None or speculative.method != "eagle3":
        raise ValueError("VLLM_ONLINE_EAGLE3 requires --speculative-config eagle3")
    _require_single_request(vllm_config)
    _require_single_parallel_worker(vllm_config)
    if getattr(speculative, "parallel_drafting", False):
        raise ValueError("Online S=1 training requires autoregressive drafting")
    if getattr(vllm_config.scheduler_config, "enable_chunked_prefill", False):
        raise ValueError("Online S=1 training requires --no-enable-chunked-prefill")
    if getattr(vllm_config.cache_config, "enable_prefix_caching", False):
        raise ValueError("Online S=1 training requires --no-enable-prefix-caching")
    model_path = os.environ.get("VLLM_ONLINE_EAGLE3_DRAFT_MODEL") or getattr(
        speculative.draft_model_config, "model", None
    )
    if not model_path:
        raise ValueError("Online EAGLE3 requires a draft model checkpoint")
    config = Qwen3Eagle3TrainerConfig(
        dtype=_get_dtype_env("VLLM_ONLINE_EAGLE3_DTYPE", torch.float32),
        lr=_get_float_env("VLLM_ONLINE_EAGLE3_LR", 1e-5),
        weight_decay=_get_float_env("VLLM_ONLINE_EAGLE3_WEIGHT_DECAY", 0.0),
        torch_threads=_get_optional_int_env("VLLM_ONLINE_EAGLE3_TORCH_THREADS"),
        update_interval=_get_int_env("VLLM_ONLINE_EAGLE3_UPDATE_INTERVAL", 1),
        check_gradients=_get_bool_env("VLLM_ONLINE_EAGLE3_CHECK_GRADIENTS", True),
    )
    # Route standalone training logs through the host's configured handler.
    core_logger = logging.getLogger("online_eagle3")
    host_logger = logging.getLogger("vllm")
    if not core_logger.handlers:
        core_logger.handlers = host_logger.handlers[:]
        core_logger.setLevel(host_logger.getEffectiveLevel())
        core_logger.propagate = False
    logger.info(
        "Enabled synchronous online EAGLE3 updates: draft_model=%s, config=%s",
        model_path,
        config,
    )
    profiler = getattr(vllm_config, "profiler_config", None)
    return create_cpu_bridge(
        model_path, config, trace_dir=getattr(profiler, "torch_profiler_dir", None)
    )


def _get_bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    if raw.strip().lower() not in _TRUE_VALUES | {"0", "false", "no", "off"}:
        raise ValueError(f"{name} must be a boolean")
    return raw.strip().lower() in _TRUE_VALUES


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
