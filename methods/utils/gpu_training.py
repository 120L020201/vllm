# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Synchronous CUDA training policies for online EAGLE3 methods."""

import copy
import json
import logging
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import online_draft.training.eagle3_inputs as eagle3_inputs
import torch
from online_draft.models.qwen3_eagle3 import (
    Eagle3KVCache,
    Qwen3Eagle3Config,
    Qwen3Eagle3ForCausalLM,
    convert_angelslim_eagle3_state_dict,
)
from online_draft.training.eagle3_batch import Eagle3DistillationBatch
from online_draft.training.eagle3_cache import (
    append_confirmed_window_to_persistent_cache,
)
from online_draft.training.eagle3_window import Eagle3TrainingWindow
from online_draft.training.trainer import DraftTrainer, TrainerConfig
from safetensors.torch import load_file as load_safetensors_file

from methods.random_sampling.gating import BernoulliGate
from methods.tts_common.step import train_batches
from methods.utils.stats import WorkerStats

logger = logging.getLogger(__name__)

_ORIGINAL_VALIDATE_TENSOR_PROPERTIES = (
    Eagle3DistillationBatch._validate_tensor_properties
)
_ORIGINAL_VALIDATE_POSITIONS = Eagle3DistillationBatch._validate_positions
_ORIGINAL_VALIDATE_TRAINING_BATCH = eagle3_inputs._validate_eagle3_training_batch


@dataclass(frozen=True, slots=True)
class GpuWeightSnapshot:
    version: int
    state_dict: dict[str, torch.Tensor]


def _validate_cuda_tensor_properties(self: Eagle3DistillationBatch) -> None:
    names = (
        "prefill_positions",
        "proposal_positions",
        "confirmed_positions",
        "prefill_input_embeds",
        "prefill_aux_hidden_states",
        "draft_token_input_embeds",
        "draft_recurrent_hidden_states",
        "confirmed_input_embeds",
        "confirmed_aux_hidden_states",
        "teacher_probabilities",
    )
    devices = set()
    for name in names:
        tensor = getattr(self, name)
        devices.add(tensor.device)
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} must remain on CUDA")
        if torch.is_inference(tensor):
            raise ValueError(f"{name} must not be an inference tensor")
        if tensor.requires_grad:
            raise ValueError(f"{name} must be detached")
    if len(devices) != 1:
        raise ValueError("all training tensors must use one CUDA device")


def _validate_cuda_positions(self: Eagle3DistillationBatch) -> None:
    device = self.prefill_positions.device
    prefill_start = int(self.prefill_positions[0].item())
    if prefill_start < 0:
        raise ValueError("prefill positions must not be negative")
    expected_prefill = torch.arange(
        prefill_start,
        prefill_start + self.prefill_positions.numel(),
        dtype=torch.long,
        device=device,
    )
    if not torch.equal(self.prefill_positions, expected_prefill):
        raise ValueError("prefill positions must be contiguous")

    anchor_position = int(self.prefill_positions[-1].item())
    expected_proposal = torch.arange(
        anchor_position,
        anchor_position + self.draft_length,
        dtype=torch.long,
        device=device,
    )
    if not torch.equal(self.proposal_positions, expected_proposal):
        raise ValueError("proposal positions must start at the prefill anchor")

    expected_confirmed = torch.arange(
        anchor_position + 1,
        anchor_position + 1 + self.confirmed_length,
        dtype=torch.long,
        device=device,
    )
    if not torch.equal(self.confirmed_positions, expected_confirmed):
        raise ValueError("confirmed positions must follow the prefill anchor")


def _validate_cuda_training_batch(
    model: Qwen3Eagle3ForCausalLM,
    batch: Eagle3DistillationBatch,
) -> None:
    feature_dtype = batch.prefill_input_embeds.dtype
    batch.validate(
        hidden_size=model.config.hidden_size,
        num_aux_hidden_states=model.config.num_aux_hidden_states,
        draft_vocab_size=model.config.draft_vocab_size,
        feature_dtype=feature_dtype,
    )
    floating_parameters = [
        parameter for parameter in model.parameters() if parameter.is_floating_point()
    ]
    devices = {parameter.device for parameter in floating_parameters}
    dtypes = {parameter.dtype for parameter in floating_parameters}
    if len(devices) != 1 or next(iter(devices)).type != "cuda":
        raise ValueError("EAGLE3 training model must use one CUDA device")
    if devices != {batch.prefill_input_embeds.device}:
        raise ValueError("training model and batch must use the same CUDA device")
    if dtypes != {feature_dtype}:
        raise ValueError("batch feature dtype must match the model parameter dtype")


def install_cuda_training_validation() -> None:
    """Make the reusable EAGLE3 objective validate CUDA-resident batches."""

    Eagle3DistillationBatch._validate_tensor_properties = (
        _validate_cuda_tensor_properties
    )
    Eagle3DistillationBatch._validate_positions = _validate_cuda_positions
    eagle3_inputs._validate_eagle3_training_batch = _validate_cuda_training_batch


def uninstall_cuda_training_validation() -> None:
    """Restore the reusable objective's default CPU validation."""

    Eagle3DistillationBatch._validate_tensor_properties = (
        _ORIGINAL_VALIDATE_TENSOR_PROPERTIES
    )
    Eagle3DistillationBatch._validate_positions = _ORIGINAL_VALIDATE_POSITIONS
    eagle3_inputs._validate_eagle3_training_batch = _ORIGINAL_VALIDATE_TRAINING_BATCH


def load_gpu_training_model(
    model_directory: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Qwen3Eagle3ForCausalLM:
    """Load the trainable EAGLE3 copy directly onto the serving CUDA device."""

    if device.type != "cuda":
        raise ValueError("online draft training requires a CUDA device")
    model_path = Path(model_directory)
    with (model_path / "config.json").open(encoding="utf-8") as config_file:
        raw_config = json.load(config_file)
    if not isinstance(raw_config, Mapping):
        raise TypeError("config.json must contain a JSON object")

    with torch.inference_mode(False):
        model = Qwen3Eagle3ForCausalLM(Qwen3Eagle3Config.from_dict(raw_config))
        model.to(device=device, dtype=dtype)
        safetensors_path = model_path / "model.safetensors"
        if safetensors_path.is_file():
            source = load_safetensors_file(safetensors_path, device=str(device))
        else:
            source = torch.load(
                model_path / "pytorch_model.bin",
                map_location=device,
                weights_only=True,
            )
        if not isinstance(source, Mapping):
            raise TypeError("checkpoint must contain a state dictionary")
        model.load_state_dict(
            convert_angelslim_eagle3_state_dict(source),
            strict=True,
        )
    return model


def export_gpu_state(
    trainer: DraftTrainer,
    *,
    clone: bool = True,
) -> dict[str, torch.Tensor]:
    state = trainer.model.state_dict()
    result = {
        name: state[name].detach().clone() if clone else state[name].detach()
        for name in trainer.trainable_parameter_names
    }
    if any(tensor.device.type != "cuda" for tensor in result.values()):
        raise RuntimeError("training snapshots must remain on CUDA")
    return result


def _initial_gpu_state(trainer: DraftTrainer) -> dict[str, torch.Tensor]:
    result = {name: tensor.detach() for name, tensor in trainer.initial_weights}
    if set(result) != set(trainer.trainable_parameter_names):
        raise KeyError("initial weights do not match the trainer")
    if any(tensor.device.type != "cuda" for tensor in result.values()):
        raise RuntimeError("initial training weights must remain on CUDA")
    return result


def _load_gpu_state(
    trainer: DraftTrainer,
    state_dict: dict[str, torch.Tensor],
) -> None:
    current = trainer.model.state_dict()
    expected = set(trainer.trainable_parameter_names)
    if set(state_dict) != expected:
        raise KeyError("trainable snapshot does not match the trainer")
    with torch.no_grad():
        for name, source in state_dict.items():
            target = current[name]
            if source.device != target.device:
                raise ValueError("training weights must not leave CUDA")
            target.copy_(source)
    trainer.clear_optimizer_state()
    trainer.last_loss = None


class GpuChunkEnsemble:
    """Three persistent OSpec learners resident on the serving GPU."""

    def __init__(
        self,
        trainer: DraftTrainer,
        learning_rates: tuple[float, ...],
        epsilon: float,
        merged_state: dict[str, torch.Tensor] | None = None,
    ) -> None:
        if len(learning_rates) != 3 or any(rate <= 0 for rate in learning_rates):
            raise ValueError("ospec requires three positive learning rates")
        if not math.isfinite(epsilon) or epsilon < 0:
            raise ValueError("epsilon must be finite and nonnegative")

        config = trainer.config
        self.learners = [trainer]
        trainer.optimizer.param_groups[0]["lr"] = learning_rates[0]
        for learning_rate in learning_rates[1:]:
            self.learners.append(
                DraftTrainer(
                    copy.deepcopy(trainer.model),
                    TrainerConfig(
                        learning_rate=learning_rate,
                        weight_decay=config.weight_decay,
                        fused=config.fused,
                        frozen_parameter_prefixes=config.frozen_parameter_prefixes,
                        check_gradients=config.check_gradients,
                    ),
                )
            )
        device = next(trainer.model.parameters()).device
        self.epsilon = epsilon
        self.cumulative_losses = torch.zeros(
            3,
            dtype=torch.float64,
            device=device,
        )
        self._merged_state = merged_state or export_gpu_state(trainer)
        expected = set(trainer.trainable_parameter_names)
        if set(self._merged_state) != expected:
            raise KeyError("OSpec merge destination does not match the trainer")
        for name, tensor in self._merged_state.items():
            reference = trainer.model.state_dict()[name]
            if tensor.device != reference.device or tensor.shape != reference.shape:
                raise ValueError("OSpec merge destination must share CUDA and shape")
        for learner in self.learners:
            learner._initial_weights = ()
        self.version = 0
        self.last_update_steps = 0

    def update(
        self,
        requests: tuple[tuple[Eagle3DistillationBatch, ...], ...],
    ) -> GpuWeightSnapshot | None:
        if not requests:
            return None
        self.last_update_steps = len(self.learners) * sum(
            len(batches) for batches in requests
        )
        for index, learner in enumerate(self.learners):
            for batches in requests:
                _, losses = train_batches(learner, batches)
                self.cumulative_losses[index] += sum(losses)

        weights = torch.softmax(-self.epsilon * self.cumulative_losses, dim=0)
        factors = weights.tolist()
        states = [export_gpu_state(learner, clone=False) for learner in self.learners]
        with torch.no_grad():
            for name, merged in self._merged_state.items():
                merged.copy_(states[0][name]).mul_(factors[0])
                for factor, state in zip(factors[1:], states[1:], strict=True):
                    merged.add_(state[name], alpha=factor)
        self.version += 1
        return GpuWeightSnapshot(
            version=self.version,
            state_dict=self._merged_state,
        )


class GpuPolicyBridge:
    """Run TTS/OSpec updates synchronously without a CPU training worker."""

    def __init__(
        self,
        trainer: DraftTrainer,
        *,
        method: str,
        merge_destination: dict[str, torch.Tensor] | None = None,
    ) -> None:
        if method not in {"tts", "random_sampling", "ospec"}:
            raise ValueError(f"unsupported method: {method}")
        if next(trainer.model.parameters()).device.type != "cuda":
            raise ValueError("online draft trainer must be on CUDA")

        self.method = method
        self.stride = int(os.environ.get("OSD_UPDATE_STRIDE", "1"))
        self.chunk_size = int(os.environ.get("OSD_CHUNK_SIZE", "5"))
        if self.stride < 1 or self.chunk_size < 1:
            raise ValueError("stride and chunk size must be positive")

        self.gate = None
        if method == "random_sampling":
            self.gate = BernoulliGate(
                float(os.environ.get("OSD_PROBABILITY", "0.5")),
                int(os.environ.get("OSD_SEED", "0")),
                self.stride,
            )

        self.ensemble = None
        if method == "ospec":
            rates = tuple(
                float(value)
                for value in os.environ.get(
                    "OSD_ENSEMBLE_LRS",
                    "1e-5,2e-5,3e-5",
                ).split(",")
            )
            self.ensemble = GpuChunkEnsemble(
                trainer,
                rates,
                float(os.environ.get("OSD_EPSILON", "0.1")),
                merge_destination,
            )
        self.trainer = trainer
        self._baseline = (
            None
            if self.ensemble is not None
            else GpuWeightSnapshot(
                version=trainer.version,
                state_dict=_initial_gpu_state(trainer),
            )
        )
        self._latest: GpuWeightSnapshot | None = None
        self._request_id: str | None = None
        self._next_step_id = 0
        self._rounds = 0
        self._persistent_cache: Eagle3KVCache | None = None
        self._current_request: list[Eagle3DistillationBatch] = []
        self._chunk_requests: list[tuple[Eagle3DistillationBatch, ...]] = []
        self.stats = WorkerStats()
        self.log_updates = os.environ.get("OSD_LOG_UPDATES", "0") == "1"

    def observe_step(
        self,
        request_id: str,
        step_id: int,
        batch: Eagle3DistillationBatch,
    ) -> None:
        self.stats.begin_request()
        self.stats.enqueue()
        updates = 0
        try:
            updates = self._observe(request_id, step_id, batch)
        finally:
            self.stats.complete(updates)
        if updates and self.log_updates:
            logger.info(
                "%s GPU draft update: request=%s step=%d version=%d",
                self.method,
                request_id,
                step_id,
                self.trainer.version,
            )

    def _observe(
        self,
        request_id: str,
        step_id: int,
        batch: Eagle3DistillationBatch,
    ) -> int:
        if self._request_id is None:
            self._request_id = request_id
        if request_id != self._request_id:
            raise ValueError("methods support one active request per worker")
        if step_id != self._next_step_id:
            raise ValueError(f"expected step {self._next_step_id}, got {step_id}")
        self._next_step_id += 1
        self._rounds += 1

        if self.ensemble is not None:
            self._current_request.append(batch)
            return 0

        should_train = self._rounds % self.stride == 0
        if self.gate is not None:
            should_train = self.gate.select(request_id)
        if should_train:
            self._persistent_cache, _ = train_batches(
                self.trainer,
                (batch,),
                persistent_cache=self._persistent_cache,
            )
            self._latest = GpuWeightSnapshot(
                version=self.trainer.version,
                state_dict=export_gpu_state(self.trainer, clone=False),
            )
            return 1

        self._persistent_cache = append_confirmed_window_to_persistent_cache(
            self.trainer.model,
            Eagle3TrainingWindow(rounds=(batch,)),
            self._persistent_cache,
        )
        return 0

    def maybe_apply_pending_weights(self) -> GpuWeightSnapshot | None:
        result = self._latest
        self._latest = None
        return result

    def reset_request(self, request_id: str) -> None:
        if self._request_id is not None and request_id != self._request_id:
            return
        if self.gate is not None:
            self.gate.reset(request_id)
        if self.ensemble is not None:
            if self._current_request:
                self._chunk_requests.append(tuple(self._current_request))
            self._current_request.clear()
            if len(self._chunk_requests) >= self.chunk_size:
                snapshot = self.ensemble.update(tuple(self._chunk_requests))
                if snapshot is not None:
                    self._latest = snapshot
                    self.stats.add_updates(self.ensemble.last_update_steps)
                    if self.log_updates:
                        logger.info(
                            "ospec GPU draft merge: version=%d updates=%d",
                            snapshot.version,
                            self.ensemble.last_update_steps,
                        )
                self._chunk_requests.clear()
        elif os.environ.get("OSD_KEEP_WEIGHTS", "0") != "1":
            assert self._baseline is not None
            _load_gpu_state(self.trainer, self._baseline.state_dict)
            self._latest = self._baseline

        self._request_id = None
        self._next_step_id = 0
        self._rounds = 0
        self._persistent_cache = None
        self.stats.end_request()

    def shutdown(self) -> None:
        self._current_request.clear()
        self._chunk_requests.clear()


def create_gpu_policy_bridge(
    model_path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    learning_rate: float,
    method: str,
    inference_model=None,
) -> GpuPolicyBridge:
    install_cuda_training_validation()
    model = load_gpu_training_model(
        model_path,
        device=device,
        dtype=dtype,
    )
    trainer = DraftTrainer(
        model,
        TrainerConfig(
            learning_rate=learning_rate,
            fused=os.environ.get("OSD_FUSED_OPTIMIZER", "1") == "1",
        ),
    )
    merge_destination = None
    if method == "ospec" and inference_model is not None:
        inference_state = inference_model.state_dict()
        merge_destination = {
            name: inference_state[name].detach()
            for name in trainer.trainable_parameter_names
        }
    return GpuPolicyBridge(
        trainer,
        method=method,
        merge_destination=merge_destination,
    )
