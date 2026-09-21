# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from enum import Enum

from online_draft.models.qwen3_eagle3 import Eagle3KVCache
from online_draft.training.eagle3_batch import Eagle3DistillationBatch
from online_draft.training.eagle3_cache import (
    PersistentEagle3KVCache,
    persistent_cache_length,
)
from online_draft.training.eagle3_window import (
    Eagle3TrainingWindow,
    Eagle3WindowMode,
    Eagle3WindowResult,
)


class Eagle3TrainStatus(str, Enum):
    """Completion states returned by an EAGLE3 executor."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Eagle3Observation:
    """One prepared GPU verification round delivered to the runtime."""

    request_id: str
    request_generation: int
    step_id: int
    source_weight_version: int
    batch: Eagle3DistillationBatch

    def __post_init__(self) -> None:
        _validate_request_scope(
            request_id=self.request_id,
            request_generation=self.request_generation,
        )
        _validate_nonnegative_int(self.step_id, "step_id")
        _validate_nonnegative_int(
            self.source_weight_version,
            "source_weight_version",
        )

        if not isinstance(self.batch, Eagle3DistillationBatch):
            raise TypeError("batch must be an Eagle3DistillationBatch")

    @property
    def anchor_position(self) -> int:
        """Return the proposal anchor position."""
        return int(self.batch.proposal_positions[0].item())

    @property
    def end_confirmed_position(self) -> int:
        """Return the final confirmed position."""
        return int(self.batch.confirmed_positions[-1].item())


@dataclass(frozen=True, slots=True)
class Eagle3TrainTask:
    """One ordered observation window submitted to an executor."""

    task_id: int
    observations: tuple[Eagle3Observation, ...]
    mode: Eagle3WindowMode
    persistent_cache: PersistentEagle3KVCache = None
    window: Eagle3TrainingWindow = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_nonnegative_int(self.task_id, "task_id")

        if not isinstance(self.observations, tuple):
            raise TypeError("observations must be a tuple")
        if not self.observations:
            raise ValueError("task must contain at least one observation")
        if not isinstance(self.mode, Eagle3WindowMode):
            raise TypeError("mode must be an Eagle3WindowMode")

        first = self.observations[0]
        for index, observation in enumerate(self.observations):
            if not isinstance(observation, Eagle3Observation):
                raise TypeError("every task entry must be an Eagle3Observation")
            if observation.request_id != first.request_id:
                raise ValueError("task observations must share one request_id")
            if observation.request_generation != first.request_generation:
                raise ValueError("task observations must share one request_generation")
            if observation.step_id != first.step_id + index:
                raise ValueError("task observation step_ids must be consecutive")

        window = Eagle3TrainingWindow(
            rounds=tuple(observation.batch for observation in self.observations)
        )
        object.__setattr__(self, "window", window)

        cache_length = persistent_cache_length(self.persistent_cache)
        if cache_length == 0:
            first_position = int(window.rounds[0].prefill_positions[0].item())
            if first_position != 0:
                raise ValueError("first task must contain the full prompt")
            anchor_only_observations = self.observations[1:]
        elif cache_length != window.start_anchor_position + 1:
            raise ValueError("persistent cache must end at the first task anchor")
        else:
            anchor_only_observations = self.observations

        for observation in anchor_only_observations:
            if observation.batch.prefill_positions.numel() != 1:
                raise ValueError(
                    "observations without prompt bootstrap must contain "
                    "exactly one prefill anchor"
                )

    @property
    def is_initial(self) -> bool:
        """Return whether this task bootstraps the prompt history."""
        return self.persistent_cache is None

    @property
    def prompt_prefix_length(self) -> int:
        """Return the initial prompt rows preceding the first anchor."""
        if not self.is_initial:
            return 0

        return self.window.rounds[0].prefill_positions.numel() - 1

    @property
    def request_id(self) -> str:
        """Return the request identifier shared by the window."""
        return self.observations[0].request_id

    @property
    def request_generation(self) -> int:
        """Return the request generation shared by the window."""
        return self.observations[0].request_generation

    @property
    def start_step_id(self) -> int:
        """Return the first consumed GPU step."""
        return self.observations[0].step_id

    @property
    def end_step_id(self) -> int:
        """Return the last consumed GPU step."""
        return self.observations[-1].step_id

    @property
    def source_weight_versions(self) -> tuple[int, ...]:
        """Return the GPU proposal version used by every observation."""
        return tuple(
            observation.source_weight_version for observation in self.observations
        )


@dataclass(frozen=True, slots=True)
class Eagle3TrainResult:
    """Success or failure returned after executing one training task."""

    task_id: int
    request_id: str
    request_generation: int
    start_step_id: int
    end_step_id: int
    source_weight_versions: tuple[int, ...]
    mode: Eagle3WindowMode
    status: Eagle3TrainStatus
    cpu_start_version: int
    cpu_end_version: int
    training_result: Eagle3WindowResult | None = None
    updated_persistent_cache: Eagle3KVCache | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        _validate_nonnegative_int(self.task_id, "task_id")
        _validate_request_scope(
            request_id=self.request_id,
            request_generation=self.request_generation,
        )
        _validate_nonnegative_int(self.start_step_id, "start_step_id")
        _validate_nonnegative_int(self.end_step_id, "end_step_id")
        _validate_nonnegative_int(self.cpu_start_version, "cpu_start_version")
        _validate_nonnegative_int(self.cpu_end_version, "cpu_end_version")

        if self.end_step_id < self.start_step_id:
            raise ValueError("end_step_id must not precede start_step_id")
        if not isinstance(self.source_weight_versions, tuple):
            raise TypeError("source_weight_versions must be a tuple")
        if len(self.source_weight_versions) != self.round_count:
            raise ValueError("source_weight_versions must match the consumed steps")
        for version in self.source_weight_versions:
            _validate_nonnegative_int(version, "source weight version")
        if not isinstance(self.mode, Eagle3WindowMode):
            raise TypeError("mode must be an Eagle3WindowMode")
        if not isinstance(self.status, Eagle3TrainStatus):
            raise TypeError("status must be an Eagle3TrainStatus")
        if self.cpu_end_version < self.cpu_start_version:
            raise ValueError("cpu_end_version must not precede cpu_start_version")

        if self.status is Eagle3TrainStatus.SUCCEEDED:
            self._validate_success()
        else:
            self._validate_failure()

    @property
    def round_count(self) -> int:
        """Return the number of consumed GPU steps."""
        return self.end_step_id - self.start_step_id + 1

    @classmethod
    def succeeded(
        cls,
        *,
        task: Eagle3TrainTask,
        cpu_start_version: int,
        updated_persistent_cache: Eagle3KVCache,
        training_result: Eagle3WindowResult,
    ) -> "Eagle3TrainResult":
        """Build a successful task result from executor outputs."""
        return cls(
            task_id=task.task_id,
            request_id=task.request_id,
            request_generation=task.request_generation,
            start_step_id=task.start_step_id,
            end_step_id=task.end_step_id,
            source_weight_versions=task.source_weight_versions,
            mode=task.mode,
            status=Eagle3TrainStatus.SUCCEEDED,
            cpu_start_version=cpu_start_version,
            cpu_end_version=training_result.model_version,
            training_result=training_result,
            updated_persistent_cache=updated_persistent_cache,
        )

    @classmethod
    def failed(
        cls,
        *,
        task: Eagle3TrainTask,
        cpu_start_version: int,
        cpu_end_version: int,
        error: str,
    ) -> "Eagle3TrainResult":
        """Build a failed task result without publishing a cache."""
        return cls(
            task_id=task.task_id,
            request_id=task.request_id,
            request_generation=task.request_generation,
            start_step_id=task.start_step_id,
            end_step_id=task.end_step_id,
            source_weight_versions=task.source_weight_versions,
            mode=task.mode,
            status=Eagle3TrainStatus.FAILED,
            cpu_start_version=cpu_start_version,
            cpu_end_version=cpu_end_version,
            error=error,
        )

    def _validate_success(self) -> None:
        if self.training_result is None:
            raise ValueError("successful result requires training_result")
        if self.updated_persistent_cache is None:
            raise ValueError("successful result requires updated persistent cache")
        if self.error is not None:
            raise ValueError("successful result must not contain an error")
        if self.cpu_end_version != self.cpu_start_version + 1:
            raise ValueError("successful task must perform exactly one CPU update")
        if self.training_result.model_version != self.cpu_end_version:
            raise ValueError("training result must match cpu_end_version")
        if self.training_result.mode is not self.mode:
            raise ValueError("training result mode must match the task mode")
        if self.training_result.round_count != self.round_count:
            raise ValueError("training result round count must match the task")

        cache_length = persistent_cache_length(self.updated_persistent_cache)
        if cache_length != self.training_result.persistent_cache_length:
            raise ValueError("updated cache length must match the training result")

    def _validate_failure(self) -> None:
        if self.training_result is not None:
            raise ValueError("failed result must not contain training_result")
        if self.updated_persistent_cache is not None:
            raise ValueError("failed result must not publish a persistent cache")
        if not isinstance(self.error, str) or not self.error:
            raise ValueError("failed result requires a nonempty error")
        if self.cpu_end_version > self.cpu_start_version + 1:
            raise ValueError("failed task advanced the CPU version more than once")


def _validate_request_scope(
    *,
    request_id: str,
    request_generation: int,
) -> None:
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request_id must be a nonempty string")
    _validate_nonnegative_int(request_generation, "request_generation")


def _validate_nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
