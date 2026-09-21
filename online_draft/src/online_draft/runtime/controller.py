# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from online_draft.runtime.contracts import (
    Eagle3Observation,
    Eagle3TrainResult,
    Eagle3TrainStatus,
    Eagle3TrainTask,
)
from online_draft.runtime.executor import Eagle3Executor
from online_draft.runtime.state import Eagle3RequestState
from online_draft.training.eagle3_window import Eagle3WindowMode


@dataclass(frozen=True, slots=True)
class Eagle3ControllerConfig:
    """Configuration for one synchronous EAGLE3 controller."""

    window_size: int = 1
    mode: Eagle3WindowMode = Eagle3WindowMode.CONFIRMED_PATH

    def __post_init__(self) -> None:
        if (
            isinstance(self.window_size, bool)
            or not isinstance(self.window_size, int)
            or self.window_size <= 0
        ):
            raise ValueError("window_size must be a positive integer")
        if not isinstance(self.mode, Eagle3WindowMode):
            raise TypeError("mode must be an Eagle3WindowMode")


class Eagle3SyncController:
    """Accumulate one request's observations and execute fixed windows."""

    def __init__(
        self,
        executor: Eagle3Executor,
        config: Eagle3ControllerConfig | None = None,
    ) -> None:
        if not isinstance(executor, Eagle3Executor):
            raise TypeError("executor must implement Eagle3Executor")
        if config is not None and not isinstance(config, Eagle3ControllerConfig):
            raise TypeError("config must be an Eagle3ControllerConfig")

        self.executor = executor
        self.config = config or Eagle3ControllerConfig()
        self._request_state: Eagle3RequestState | None = None
        self._pending_observations: list[Eagle3Observation] = []
        self._failure: Eagle3TrainResult | None = None
        self._next_task_id = 0
        self._latest_generations: dict[str, int] = {}

    @property
    def request_state(self) -> Eagle3RequestState | None:
        """Return the active request state, if any."""
        return self._request_state

    @property
    def pending_count(self) -> int:
        """Return the number of observations waiting for a full window."""
        return len(self._pending_observations)

    @property
    def failure(self) -> Eagle3TrainResult | None:
        """Return the failure that stopped the active request, if any."""
        return self._failure

    def start_request(
        self,
        *,
        request_id: str,
        request_generation: int,
    ) -> None:
        """Start one request generation.

        Raises:
            RuntimeError: If another request is still active.
            ValueError: If the generation does not advance a reused ID.
        """
        if self._request_state is not None:
            raise RuntimeError("a request is already active")

        state = Eagle3RequestState(
            request_id=request_id,
            request_generation=request_generation,
        )
        latest_generation = self._latest_generations.get(request_id)
        if latest_generation is not None and request_generation <= latest_generation:
            raise ValueError("reused request_id must advance request_generation")

        self._request_state = state
        self._latest_generations[request_id] = request_generation
        self._pending_observations.clear()
        self._failure = None

    def observe(
        self,
        observation: Eagle3Observation,
    ) -> Eagle3TrainResult | None:
        """Accept one round and synchronously execute a complete window."""
        if not isinstance(observation, Eagle3Observation):
            raise TypeError("observation must be an Eagle3Observation")

        state = self._require_active_state()
        if self._failure is not None:
            raise RuntimeError("failed request must be finished before reuse")

        self._validate_next_observation(state, observation)
        candidate = (*self._pending_observations, observation)

        if len(candidate) < self.config.window_size:
            self._pending_observations.append(observation)
            return None

        task = Eagle3TrainTask(
            task_id=self._next_task_id,
            observations=candidate,
            mode=self.config.mode,
            persistent_cache=state.persistent_cache,
        )
        state.validate_window_scope(
            request_id=task.request_id,
            request_generation=task.request_generation,
            start_step_id=task.start_step_id,
            end_step_id=task.end_step_id,
        )
        self._next_task_id += 1
        self._pending_observations.clear()

        cpu_start_version = self.executor.model_version
        try:
            result = self.executor.execute(task)
        except Exception as error:
            result = Eagle3TrainResult.failed(
                task=task,
                cpu_start_version=cpu_start_version,
                cpu_end_version=self.executor.model_version,
                error=f"{type(error).__name__}: {error}",
            )

        if not self._result_matches_task(result, task):
            result = Eagle3TrainResult.failed(
                task=task,
                cpu_start_version=result.cpu_start_version,
                cpu_end_version=result.cpu_end_version,
                error="executor returned a result for a different task",
            )

        if result.status is Eagle3TrainStatus.FAILED:
            self._failure = result
            return result

        if result.updated_persistent_cache is None:
            raise RuntimeError("successful result did not publish a cache")

        try:
            state.commit_window(
                request_id=result.request_id,
                request_generation=result.request_generation,
                start_step_id=result.start_step_id,
                end_step_id=result.end_step_id,
                persistent_cache=result.updated_persistent_cache,
            )
        except Exception as error:
            result = Eagle3TrainResult.failed(
                task=task,
                cpu_start_version=result.cpu_start_version,
                cpu_end_version=result.cpu_end_version,
                error=f"{type(error).__name__}: {error}",
            )
            self._failure = result

        return result

    def finish_request(
        self,
        *,
        request_id: str,
        request_generation: int,
    ) -> int:
        """Finish the active request and discard an incomplete tail window.

        Returns:
            Number of pending observations discarded during cleanup.
        """
        state = self._require_active_state()
        state.validate_request_scope(
            request_id=request_id,
            request_generation=request_generation,
        )

        discarded_count = len(self._pending_observations)
        state.reset()
        self._pending_observations.clear()
        self._failure = None
        self._request_state = None
        return discarded_count

    def _require_active_state(self) -> Eagle3RequestState:
        if self._request_state is None:
            raise RuntimeError("no request is active")
        return self._request_state

    def _validate_next_observation(
        self,
        state: Eagle3RequestState,
        observation: Eagle3Observation,
    ) -> None:
        state.validate_request_scope(
            request_id=observation.request_id,
            request_generation=observation.request_generation,
        )

        expected_step_id = state.next_step_id + len(self._pending_observations)
        if observation.step_id != expected_step_id:
            raise ValueError(
                f"observation step_id must equal next expected step {expected_step_id}"
            )

        if self._pending_observations:
            previous_end = self._pending_observations[-1].end_confirmed_position
            if observation.anchor_position != previous_end:
                raise ValueError(
                    "observation anchor must equal the previous confirmed end"
                )
            if observation.batch.prefill_positions.numel() != 1:
                raise ValueError("noninitial observation must contain one anchor")
        elif state.cache_length == 0:
            if int(observation.batch.prefill_positions[0].item()) != 0:
                raise ValueError("first observation must contain the full prompt")
        else:
            if observation.anchor_position != state.cache_length - 1:
                raise ValueError("observation anchor must match persistent history")
            if observation.batch.prefill_positions.numel() != 1:
                raise ValueError("noninitial observation must contain one anchor")

    @staticmethod
    def _result_matches_task(
        result: Eagle3TrainResult,
        task: Eagle3TrainTask,
    ) -> bool:
        return (
            result.task_id == task.task_id
            and result.request_id == task.request_id
            and result.request_generation == task.request_generation
            and result.start_step_id == task.start_step_id
            and result.end_step_id == task.end_step_id
            and result.source_weight_versions == task.source_weight_versions
            and result.mode is task.mode
        )
