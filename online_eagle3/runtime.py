# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging
import os
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .config import Qwen3Eagle3TrainerConfig

logger = logging.getLogger(__name__)


def get_cpu_runtime(model: torch.nn.Module) -> dict[str, Any]:
    cpu_name = platform.processor() or platform.machine() or "unknown"
    flags: list[str] = []
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            key, sep, value = line.partition(":")
            if sep and key.strip() == "model name":
                cpu_name = value.strip()
            if sep and key.strip() == "flags":
                flags = value.split()
                break
    except OSError:
        pass
    dtype = next(p.dtype for p in model.parameters() if p.is_floating_point())
    return {
        "cpu_model": cpu_name,
        "cpu_flags": flags,
        "torch_version": str(torch.__version__),
        "torch_build_config": torch.__config__.show(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cpu_draft_dtype": "FP32" if dtype == torch.float32 else "BF16",
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "thread_env": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_PROC_BIND",
                "OMP_PLACES",
            )
        },
    }


def write_cpu_runtime(
    trace_dir: str | Path | None,
    runtime: dict[str, Any],
    config: Qwen3Eagle3TrainerConfig,
    model_path: str,
) -> None:
    if not trace_dir or "://" in str(trace_dir):
        return
    path = Path(trace_dir) / "online_eagle3_cpu_runtime.json"
    settings = asdict(config)
    settings["dtype"] = str(config.dtype)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {**runtime, "training_config": settings, "draft_model": model_path},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError:
        logger.warning("Failed to write online EAGLE3 CPU runtime to %s", path)
