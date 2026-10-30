# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch
from online_draft.models import Qwen3Eagle3ForCausalLM, load_qwen3_eagle3_from_vllm
from online_draft.runtime.async_pipeline import AsyncDraftPipeline
from online_draft.runtime.capture.eagle3 import (
    Eagle3CaptureAdapter,
    Eagle3CaptureBuffers,
)
from online_draft.runtime.contracts import TrainRounds
from online_draft.runtime.cpu_worker import CPUWorkerConfig
from online_draft.training.eagle3.draft_model import AutoregressiveDraftConfig
from online_draft.training.eagle3.window import train_eagle3_rounds
from online_draft.training.trainer import DraftTrainer, TrainerConfig
from torch import nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator
from vllm.v1.worker.gpu.spec_decode.online import OnlineDraftSpeculator

if TYPE_CHECKING:
    from online_draft.runtime.capture.dflash import DFlashCaptureAdapter

    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(__name__)


@dataclass
class OnlineDraftRuntime:
    """Own online training components attached to one GPU worker."""

    trainer: DraftTrainer
    pipeline: AsyncDraftPipeline
    capture_adapter: Eagle3CaptureAdapter | DFlashCaptureAdapter
    speculator: OnlineDraftSpeculator
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


def _autoregressive_capture_adapter(
    vllm_config: VllmConfig,
    draft_config: AutoregressiveDraftConfig,
    target_token_ids: torch.Tensor | None,
    **capture: Any,
) -> Eagle3CaptureAdapter:
    """Capture for drafts trained by the EAGLE3-style window (EAGLE3, MTP)."""
    assert vllm_config.speculative_config is not None
    model_config = vllm_config.model_config
    return Eagle3CaptureAdapter(
        buffers=Eagle3CaptureBuffers(
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.speculative_config.num_speculative_tokens,
            draft_config.hidden_size,
            draft_config.num_aux_hidden_states,
            draft_config.draft_vocab_size,
            dtype=model_config.dtype,
            teacher_dtype=model_config.head_dtype or model_config.dtype,
        ),
        target_token_ids=target_token_ids,
        **capture,
    )


@dataclass(frozen=True, kw_only=True)
class _Eagle3Recipe:
    """EAGLE3 captures before its trainable FC; fields are its settings."""

    name: ClassVar[str] = "EAGLE3"
    target_model_types: ClassVar[frozenset[str]] = frozenset({"qwen3", "qwen3_5"})

    attention_backend: str | None = None

    @staticmethod
    def supports(vllm_config: VllmConfig, speculator: Any) -> bool:
        assert vllm_config.speculative_config is not None
        return vllm_config.speculative_config.method == "eagle3"

    def load_model(self, gpu_draft_model: nn.Module) -> nn.Module:
        return load_qwen3_eagle3_from_vllm(
            gpu_draft_model, attention_backend=self.attention_backend
        )

    def check(self, trainer: DraftTrainer, speculator: Any) -> None:
        assert isinstance(speculator, EagleSpeculator)

    def train_rounds(self) -> TrainRounds:
        return train_eagle3_rounds

    def create_adapter(
        self,
        vllm_config: VllmConfig,
        cpu_model: nn.Module,
        device: torch.device,
        **capture: Any,
    ) -> Eagle3CaptureAdapter:
        assert isinstance(cpu_model, Qwen3Eagle3ForCausalLM)
        return _autoregressive_capture_adapter(
            vllm_config,
            cpu_model.config,
            cpu_model.get_target_token_ids().to(device),
            **capture,
        )


@dataclass(frozen=True, kw_only=True)
class _MTPRecipe:
    """Qwen3/Qwen3.5 MTP reuses the EAGLE3 capture and training path.

    The draft fuses the target's final hidden state inside its layer, so it
    captures one feature row and distills over the full shared vocabulary.
    """

    name: ClassVar[str] = "MTP"
    target_model_types: ClassVar[frozenset[str]] = frozenset({"qwen3", "qwen3_5"})

    attention_backend: str | None = None

    @staticmethod
    def supports(vllm_config: VllmConfig, speculator: Any) -> bool:
        assert vllm_config.speculative_config is not None
        return vllm_config.speculative_config.method == "mtp"

    def load_model(self, gpu_draft_model: nn.Module) -> nn.Module:
        model_type = gpu_draft_model.config.model_type
        if model_type == "qwen3_mtp":
            from online_draft.models.qwen3_mtp import load_qwen3_mtp_from_vllm

            loader = load_qwen3_mtp_from_vllm
        elif model_type in ("qwen3_5_mtp", "qwen3_5_text"):
            from online_draft.models.qwen3_5_mtp import load_qwen3_5_mtp_from_vllm

            loader = load_qwen3_5_mtp_from_vllm
        else:
            raise ValueError(f"online MTP cannot load draft type {model_type!r}")
        return loader(gpu_draft_model, attention_backend=self.attention_backend)

    def check(self, trainer: DraftTrainer, speculator: Any) -> None:
        from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

        assert isinstance(speculator, MTPSpeculator)
        assert "lm_head.weight" not in trainer.trainable_parameter_names, (
            "online MTP shares its head with the target"
        )

    def train_rounds(self) -> TrainRounds:
        return train_eagle3_rounds

    def create_adapter(
        self,
        vllm_config: VllmConfig,
        cpu_model: nn.Module,
        device: torch.device,
        **capture: Any,
    ) -> Eagle3CaptureAdapter:
        from online_draft.models.qwen3_5_mtp import Qwen3_5MTPForCausalLM
        from online_draft.models.qwen3_mtp import Qwen3MTPForCausalLM

        assert isinstance(cpu_model, (Qwen3MTPForCausalLM, Qwen3_5MTPForCausalLM))
        return _autoregressive_capture_adapter(
            vllm_config, cpu_model.config, None, **capture
        )


@dataclass(frozen=True, kw_only=True)
class _DFlashRecipe:
    """DFlash captures its frozen FC output; fields are its settings."""

    name: ClassVar[str] = "DFlash"
    target_model_types: ClassVar[frozenset[str]] = frozenset({"qwen3", "qwen3_5"})

    grouped_query_attention: bool = False
    selective_last_layer: bool = False
    max_train_rounds: int = 16
    loss_gamma: float | None = None
    loss_normalization: str = "mean"
    loss_rejection_weight: float = 1.0
    loss_objective: str = "kl"
    loss_overlap_weight: float = 1.0
    loss_temperature: float = 1.0
    loss_top_k: int = 0
    loss_top_p: float = 1.0
    loss_reference_weight: float = 0.0
    loss_reference_decay_tokens: int = 0

    def __post_init__(self) -> None:
        if self.max_train_rounds < 0:
            raise ValueError("DFlash requires max_train_rounds >= 0")
        if self.loss_gamma is not None and (
            not math.isfinite(self.loss_gamma) or self.loss_gamma <= 0
        ):
            raise ValueError("DFlash loss gamma must be finite and positive")
        if self.loss_normalization not in ("mean", "weights", "block"):
            raise ValueError(
                "DFlash loss normalization must be mean, weights, or block"
            )
        if (
            not math.isfinite(self.loss_rejection_weight)
            or self.loss_rejection_weight <= 0
        ):
            raise ValueError("DFlash rejection weight must be finite and positive")
        if self.loss_objective not in ("kl", "overlap", "ce", "kl_overlap"):
            raise ValueError("DFlash loss objective is invalid")
        if not math.isfinite(self.loss_overlap_weight) or self.loss_overlap_weight <= 0:
            raise ValueError("DFlash overlap weight must be finite and positive")
        if not math.isfinite(self.loss_temperature) or self.loss_temperature <= 0:
            raise ValueError("DFlash loss temperature must be finite and positive")
        if self.loss_top_k < 0:
            raise ValueError("DFlash loss top-k must be nonnegative")
        if not 0 < self.loss_top_p <= 1:
            raise ValueError("DFlash loss top-p must be in (0, 1]")
        if (
            not math.isfinite(self.loss_reference_weight)
            or self.loss_reference_weight < 0
        ):
            raise ValueError("DFlash reference loss weight must be nonnegative")
        if self.loss_reference_decay_tokens < 0:
            raise ValueError("DFlash reference decay tokens must be nonnegative")

    @staticmethod
    def supports(vllm_config: VllmConfig, speculator: Any) -> bool:
        return speculator._speculator_name == "DFlash"

    def load_model(self, gpu_draft_model: nn.Module) -> nn.Module:
        from online_draft.models.qwen3_dflash import load_qwen3_dflash_from_vllm

        return load_qwen3_dflash_from_vllm(
            gpu_draft_model,
            grouped_query_attention=self.grouped_query_attention,
            selective_last_layer=self.selective_last_layer,
        )

    def check(self, trainer: DraftTrainer, speculator: Any) -> None:
        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
            DFlashSpeculator,
        )

        assert isinstance(speculator, DFlashSpeculator)
        assert not any(
            name.startswith("model.fc.") for name in trainer.trainable_parameter_names
        ), "online DFlash captures FC output and installs weights after FC"

    def train_rounds(self) -> TrainRounds:
        from online_draft.training.dflash.window import train_dflash_window

        return partial(
            train_dflash_window,
            max_train_rounds=self.max_train_rounds,
            loss_gamma=self.loss_gamma,
            loss_normalization=self.loss_normalization,
            loss_rejection_weight=self.loss_rejection_weight,
            loss_objective=self.loss_objective,
            loss_overlap_weight=self.loss_overlap_weight,
            loss_temperature=self.loss_temperature,
            loss_top_k=self.loss_top_k,
            loss_top_p=self.loss_top_p,
            loss_reference_weight=self.loss_reference_weight,
            loss_reference_decay_tokens=self.loss_reference_decay_tokens,
        )

    def create_adapter(
        self,
        vllm_config: VllmConfig,
        cpu_model: nn.Module,
        device: torch.device,
        **capture: Any,
    ) -> DFlashCaptureAdapter:
        from online_draft.models.qwen3_dflash import Qwen3DFlashForCausalLM
        from online_draft.runtime.capture.dflash import (
            DFlashCaptureAdapter,
            DFlashCaptureBuffers,
        )

        assert isinstance(cpu_model, Qwen3DFlashForCausalLM)
        assert vllm_config.speculative_config is not None
        model_config = vllm_config.model_config
        return DFlashCaptureAdapter(
            buffers=DFlashCaptureBuffers(
                vllm_config.scheduler_config.max_num_batched_tokens,
                vllm_config.speculative_config.num_speculative_tokens + 1,
                cpu_model.config.hidden_size,
                cpu_model.config.vocab_size,
                dtype=model_config.dtype,
                teacher_dtype=model_config.head_dtype or model_config.dtype,
            ),
            **capture,
        )


_Recipe = _Eagle3Recipe | _DFlashRecipe | _MTPRecipe

# Setting name -> TrainerConfig field.
_TRAINER_SETTINGS = {
    "learning_rate": "learning_rate",
    "fused_optimizer": "fused",
    "check_gradients": "check_gradients",
    "master_weights": "master_weights",
    "persistent_grads": "persistent_grads",
    "lr_reference_rows": "lr_reference_rows",
    "frozen_parameter_prefixes": "frozen_parameter_prefixes",
    "loss_mode": "loss_mode",
}
_RUNTIME_SETTINGS = {
    "cpu_threads",
    "cpu_affinity",
    "log_queue_sizes",
    "log_updates",
    "continuous_training",
    "train_schedule",
}
_CPU_ROLES = ("inference", "capture", "training")


@dataclass(frozen=True)
class _OnlineDraftSettings:
    """Parsed ``additional_config["online_draft"]``."""

    recipe: _Recipe
    trainer: TrainerConfig
    cpu: dict[str, CPUWorkerConfig]
    log_queue_sizes: bool
    log_updates: bool
    continuous_training: bool
    train_schedule: tuple[int, ...] | None


def _recipe_type(vllm_config: VllmConfig) -> type[_Recipe]:
    assert vllm_config.speculative_config is not None
    method = vllm_config.speculative_config.method
    if method == "dflash":
        return _DFlashRecipe
    if method == "mtp":
        return _MTPRecipe
    return _Eagle3Recipe


def _parse_online_draft_settings(
    values: Mapping[str, Any], recipe_type: type[_Recipe]
) -> _OnlineDraftSettings:
    """Reject settings the selected method does not read."""
    recipe_settings = {field.name for field in fields(recipe_type)}
    unknown = set(values) - _TRAINER_SETTINGS.keys() - _RUNTIME_SETTINGS
    unknown -= recipe_settings
    if unknown:
        raise ValueError(
            f"unknown online {recipe_type.name} settings: {', '.join(sorted(unknown))}"
        )

    cpu_threads = values.get("cpu_threads", {})
    cpu_affinity = values.get("cpu_affinity", {})
    unknown_roles = {*cpu_threads, *cpu_affinity} - set(_CPU_ROLES)
    if unknown_roles:
        raise ValueError(
            f"unknown online draft CPU roles: {', '.join(sorted(unknown_roles))}"
        )

    trainer = {
        field: values[name]
        for name, field in _TRAINER_SETTINGS.items()
        if name in values
    }
    if "frozen_parameter_prefixes" in trainer:
        trainer["frozen_parameter_prefixes"] = tuple(
            trainer["frozen_parameter_prefixes"]
        )
    train_schedule = tuple(values.get("train_schedule", ()))
    if train_schedule and (
        any(value < 1 for value in train_schedule)
        or any(left >= right for left, right in zip(train_schedule, train_schedule[1:]))
    ):
        raise ValueError("train_schedule must be strictly increasing rounds")

    return _OnlineDraftSettings(
        recipe=recipe_type(
            **{name: values[name] for name in recipe_settings & set(values)}
        ),
        trainer=TrainerConfig(**trainer),
        cpu={
            role: CPUWorkerConfig(
                threads=cpu_threads.get(role), cpus=tuple(cpu_affinity.get(role, ()))
            )
            for role in _CPU_ROLES
        },
        log_queue_sizes=values.get("log_queue_sizes", False),
        log_updates=values.get("log_updates", False),
        continuous_training=values.get("continuous_training", False),
        train_schedule=train_schedule or None,
    )


def _validate_online_draft_config(vllm_config: VllmConfig, speculator: Any) -> None:
    """Reject configs the online capture and training path does not support."""
    recipe_type = _recipe_type(vllm_config)
    parallel_config = vllm_config.parallel_config

    requirements = {
        f"method={recipe_type.name.lower()}": recipe_type.supports(
            vllm_config, speculator
        ),
        "tensor_parallel_size=1": parallel_config.tensor_parallel_size == 1,
        "pipeline_parallel_size=1": parallel_config.pipeline_parallel_size == 1,
        "max_num_seqs=1": vllm_config.scheduler_config.max_num_seqs == 1,
        "enable_prefix_caching=False": (
            not vllm_config.cache_config.enable_prefix_caching
        ),
        f"target model_type in {sorted(recipe_type.target_model_types)}": (
            vllm_config.model_config.hf_config.model_type
            in recipe_type.target_model_types
        ),
        "text-only draft inputs": not speculator.supports_mm_inputs,
    }

    unmet = [requirement for requirement, ok in requirements.items() if not ok]
    if unmet:
        raise ValueError(f"online {recipe_type.name} requires {', '.join(unmet)}")


def create_online_draft(
    vllm_config: VllmConfig,
    model_runner: GPUModelRunner,
    device: torch.device,
) -> OnlineDraftRuntime:
    """Build the CPU training pipeline and connect it to the loaded speculator."""
    gpu_draft_model = model_runner.get_draft_model()
    assert gpu_draft_model is not None
    speculator = model_runner.speculator
    assert isinstance(speculator, OnlineDraftSpeculator)

    _validate_online_draft_config(vllm_config, speculator)
    additional_config = cast(dict[str, Any], vllm_config.additional_config)
    settings = _parse_online_draft_settings(
        additional_config.get("online_draft", {}), _recipe_type(vllm_config)
    )
    recipe = settings.recipe
    profile_online_draft = envs.VLLM_ONLINE_DRAFT_PROFILE

    cpu_model = recipe.load_model(gpu_draft_model)
    trainer = DraftTrainer(cpu_model, settings.trainer)
    recipe.check(trainer, speculator)

    speculator.configure_online_training(
        trainer.trainable_parameter_names,
        profile_timeline=profile_online_draft,
        log_updates=settings.log_updates,
    )
    installer = speculator.draft_weight_installer
    assert installer is not None

    def queue_logger(name: str) -> Callable[[int], None] | None:
        if not settings.log_queue_sizes:
            return None
        return lambda size: logger.info("Online draft %s queue size: %d", name, size)

    pipeline = AsyncDraftPipeline(
        trainer=trainer,
        train_rounds=recipe.train_rounds(),
        on_queue_size=queue_logger("train"),
        profile_timeline=profile_online_draft,
        cpu_config=settings.cpu["training"],
        continuous_training=settings.continuous_training,
        train_schedule=settings.train_schedule,
    )
    adapter = recipe.create_adapter(
        vllm_config,
        cpu_model,
        device,
        source_weight_version=lambda: installer.active_version,
        on_round=pipeline.submit_round,
        on_epoch_close=pipeline.close_epoch,
        on_finish=pipeline.finish_request,
        on_queue_size=queue_logger("capture"),
        profile_timeline=profile_online_draft,
        cpu_config=settings.cpu["capture"],
    )
    speculator.set_snapshot_source(pipeline.poll_snapshot)

    return OnlineDraftRuntime(
        trainer=trainer,
        pipeline=pipeline,
        capture_adapter=adapter,
        speculator=speculator,
        inference_cpu=settings.cpu["inference"],
    )
