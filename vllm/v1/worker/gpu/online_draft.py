# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import torch
from online_draft.models import (
    load_qwen3_eagle3_checkpoint,
    validate_qwen3_eagle3_weights,
)
from online_draft.runtime.async_pipeline import AsyncDraftPipeline
from online_draft.runtime.capture.eagle3 import Eagle3CaptureAdapter
from online_draft.runtime.cpu_worker import CPUWorkerConfig
from online_draft.training.eagle3.window import train_eagle3_rounds
from online_draft.training.trainer import DraftTrainer, TrainerConfig

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator

if TYPE_CHECKING:
    from online_draft.runtime.capture.dflash import DFlashCaptureAdapter

    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

logger = init_logger(__name__)


@dataclass
class OnlineDraftRuntime:
    """Own online training components attached to one GPU worker."""

    trainer: DraftTrainer
    pipeline: AsyncDraftPipeline
    capture_adapter: Eagle3CaptureAdapter | DFlashCaptureAdapter
    speculator: EagleSpeculator | DFlashSpeculator
    inference_cpu: CPUWorkerConfig

    def enable_capture(self) -> None:
        self.inference_cpu.apply()
        self.speculator.set_capture_adapter(self.capture_adapter)
        logger.debug("Online draft capture enabled after warmup")

    def raise_if_failed(self) -> None:
        self.pipeline.raise_if_failed()
        self.capture_adapter.raise_if_failed()

    def finish_requests(self, request_ids: set[str]) -> None:
        if self.speculator.capture_adapter is None:
            return
        for request_id in request_ids:
            self.speculator.finish_request(request_id, self.trainer.initial_weights)

    def close(self) -> None:
        self.capture_adapter.close()
        self.pipeline.close()


def create_online_draft(
    vllm_config: VllmConfig,
    model_runner: GPUModelRunner,
    device: torch.device,
) -> OnlineDraftRuntime:
    """Build the CPU training pipeline and connect it to the loaded speculator."""
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None

    draft_model_config = speculative_config.draft_model_config
    assert draft_model_config is not None

    gpu_draft_model = model_runner.get_draft_model()
    assert gpu_draft_model is not None

    additional_config = cast(dict[str, Any], vllm_config.additional_config)
    online_config = additional_config.get("online_draft", {})
    cpu_threads = online_config.get("cpu_threads", {})
    cpu_affinity = online_config.get("cpu_affinity", {})
    cpu_configs = {
        role: CPUWorkerConfig(
            threads=cpu_threads.get(role),
            cpus=tuple(cpu_affinity.get(role, ())),
        )
        for role in ("inference", "capture", "training")
    }
    profile_online_draft = envs.VLLM_ONLINE_DRAFT_PROFILE
    log_queue_sizes = online_config.get("log_queue_sizes", False)
    speculator = model_runner.speculator
    assert speculator is not None

    is_dflash = speculative_config.method == "dflash"
    if is_dflash:
        from online_draft.models.qwen3_dflash import load_qwen3_dflash_from_vllm
        from online_draft.runtime.capture.dflash import (
            DFlashCaptureAdapter,
            DFlashCaptureBuffers,
        )
        from online_draft.training.dflash.window import train_dflash_window

        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
            DFlashSpeculator,
        )

        if (
            vllm_config.parallel_config.tensor_parallel_size != 1
            or vllm_config.parallel_config.pipeline_parallel_size != 1
            or vllm_config.scheduler_config.max_num_seqs != 1
            or vllm_config.cache_config.enable_prefix_caching
            or speculator._speculator_name != "DFlash"
            or vllm_config.model_config.hf_config.model_type != "qwen3"
        ):
            raise ValueError(
                "online DFlash requires one GPU/request and prefix caching disabled"
            )
        cpu_model = load_qwen3_dflash_from_vllm(
            gpu_draft_model,
            grouped_query_attention=online_config.get("grouped_query_attention", False),
            selective_last_layer=online_config.get("selective_last_layer", False),
        )
    else:
        cpu_model = load_qwen3_eagle3_checkpoint(
            draft_model_config.model,
            dtype=vllm_config.model_config.dtype,
            attention_backend=online_config.get("attention_backend"),
        )
        validate_qwen3_eagle3_weights(cpu_model, gpu_draft_model.state_dict())

    trainer = DraftTrainer(
        cpu_model,
        TrainerConfig(
            learning_rate=online_config.get("learning_rate", 1e-5),
            fused=online_config.get("fused_optimizer", is_dflash),
            frozen_parameter_prefixes=tuple(
                online_config.get("frozen_parameter_prefixes", ())
            ),
            check_gradients=online_config.get("check_gradients", True),
            master_weights=online_config.get("master_weights", False),
        ),
    )

    assert isinstance(speculator, DFlashSpeculator if is_dflash else EagleSpeculator)

    speculator.configure_online_training(
        trainer.trainable_parameter_names,
        profile_timeline=profile_online_draft,
        log_updates=online_config.get("log_updates", False),
    )

    installer = speculator.draft_weight_installer
    assert installer is not None

    queue_logger = lambda name: (
        lambda size: logger.info("Online draft %s queue size: %d", name, size)
    )
    train_queue_log = queue_logger("train") if log_queue_sizes else None
    capture_queue_log = queue_logger("capture") if log_queue_sizes else None
    train_rounds: Callable
    if is_dflash:
        max_train_rounds = online_config.get("max_train_rounds", 16)
        loss_decay_gamma = online_config.get("loss_decay_gamma", 7.0)
        if max_train_rounds < 0 or not 0 < loss_decay_gamma < float("inf"):
            raise ValueError(
                "DFlash requires max_train_rounds >= 0 and finite loss_decay_gamma > 0"
            )
        train_rounds = partial(
            train_dflash_window,
            max_train_rounds=max_train_rounds,
            loss_decay_gamma=loss_decay_gamma,
        )
    else:
        train_rounds = train_eagle3_rounds

    pipeline = AsyncDraftPipeline(
        trainer=trainer,
        train_rounds=train_rounds,
        on_queue_size=train_queue_log,
        profile_timeline=profile_online_draft,
        cpu_config=cpu_configs["training"],
    )
    if is_dflash:
        adapter = DFlashCaptureAdapter(
            buffers=DFlashCaptureBuffers(
                vllm_config.scheduler_config.max_num_batched_tokens,
                speculative_config.num_speculative_tokens + 1,
                cpu_model.config.hidden_size,
                cpu_model.config.vocab_size,
                dtype=vllm_config.model_config.dtype,
                teacher_dtype=vllm_config.model_config.head_dtype
                or vllm_config.model_config.dtype,
            ),
            on_round=pipeline.submit_round,
            on_epoch_close=pipeline.close_epoch,
            on_finish=pipeline.finish_request,
            on_queue_size=capture_queue_log,
            profile_timeline=profile_online_draft,
            cpu_config=cpu_configs["capture"],
        )
    else:
        adapter = Eagle3CaptureAdapter(
            target_token_ids=cpu_model.get_target_token_ids().to(device),
            source_weight_version=lambda: installer.active_version,
            hidden_size=cpu_model.config.hidden_size,
            num_aux_hidden_states=cpu_model.config.num_aux_hidden_states,
            draft_vocab_size=cpu_model.config.draft_vocab_size,
            feature_dtype=vllm_config.model_config.dtype,
            transfer_chunk_size=1,
            on_round=pipeline.submit_round,
            on_epoch_close=pipeline.close_epoch,
            on_finish=pipeline.finish_request,
            on_queue_size=capture_queue_log,
            profile_timeline=profile_online_draft,
            cpu_config=cpu_configs["capture"],
        )
    speculator.set_snapshot_source(pipeline.poll_snapshot)

    return OnlineDraftRuntime(
        trainer=trainer,
        pipeline=pipeline,
        capture_adapter=adapter,
        speculator=speculator,
        inference_cpu=cpu_configs["inference"],
    )
