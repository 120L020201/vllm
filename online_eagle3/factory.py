# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from pathlib import Path

import torch

from .checkpoint import load_torch_eagle3_model
from .config import Qwen3Eagle3TrainerConfig
from .data import Qwen3Eagle3StepFn
from .distillation import qwen3_eagle3_distillation_step
from .qwen3_trainer import Qwen3Eagle3CpuTrainer
from .runtime import get_cpu_runtime, write_cpu_runtime
from .sync_bridge import Qwen3Eagle3LazySyncBridge, Qwen3Eagle3SyncBridge

logger = logging.getLogger(__name__)


def create_cpu_bridge(
    model_path: str,
    config: Qwen3Eagle3TrainerConfig | None = None,
    *,
    step_fn: Qwen3Eagle3StepFn = qwen3_eagle3_distillation_step,
    trace_dir: str | Path | None = None,
) -> Qwen3Eagle3LazySyncBridge:
    """Build request-local training without importing an inference engine."""
    config = config or Qwen3Eagle3TrainerConfig()

    def load() -> Qwen3Eagle3SyncBridge:
        if config.torch_threads is not None:
            torch.set_num_threads(config.torch_threads)
        logger.info("Loading online EAGLE3 CPU draft from %s", model_path)
        with (
            torch.inference_mode(False),
            torch.enable_grad(),
            torch.profiler.record_function("online_eagle3.cpu_load_model"),
        ):
            model = load_torch_eagle3_model(model_path, dtype=config.dtype)
            model.train()
            runtime = get_cpu_runtime(model)
            logger.info(
                "Online EAGLE3 CPU runtime: cpu_model=%s, torch_num_threads=%d, "
                "cpu_draft_dtype=%s",
                runtime["cpu_model"],
                runtime["torch_num_threads"],
                runtime["cpu_draft_dtype"],
            )
            write_cpu_runtime(trace_dir, runtime, config, model_path)
            trainer = Qwen3Eagle3CpuTrainer(model, config)
        return Qwen3Eagle3SyncBridge(
            trainer, step_fn, update_interval=config.update_interval, fail_on_error=True
        )

    return Qwen3Eagle3LazySyncBridge(load)
