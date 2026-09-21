# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Protocol, runtime_checkable

from online_draft.runtime.contracts import (
    Eagle3TrainResult,
    Eagle3TrainTask,
)
from online_draft.training.eagle3_window import train_eagle3_window
from online_draft.training.trainer import DraftTrainer


@runtime_checkable
class Eagle3Executor(Protocol):
    """Executor interface shared by synchronous and future workers."""

    @property
    def model_version(self) -> int:
        """Return the current CPU model version."""
        ...

    def execute(self, task: Eagle3TrainTask) -> Eagle3TrainResult:
        """Execute one training task."""
        ...


class DirectEagle3Executor:
    """Execute EAGLE3 training tasks synchronously on the caller thread."""

    def __init__(self, trainer: DraftTrainer) -> None:
        if not isinstance(trainer, DraftTrainer):
            raise TypeError("trainer must be a DraftTrainer")

        self.trainer = trainer

    @property
    def model_version(self) -> int:
        """Return the current CPU model version."""
        return self.trainer.version

    def execute(self, task: Eagle3TrainTask) -> Eagle3TrainResult:
        """Execute one task and return a success or failure result.

        Args:
            task: Validated window and persistent history to train.

        Returns:
            A result that publishes updated KV only after full success.

        Raises:
            TypeError: If task is not an Eagle3TrainTask.
        """
        if not isinstance(task, Eagle3TrainTask):
            raise TypeError("task must be an Eagle3TrainTask")

        cpu_start_version = self.trainer.version

        try:
            updated_cache, training_result = train_eagle3_window(
                trainer=self.trainer,
                window=task.window,
                persistent_cache=task.persistent_cache,
                mode=task.mode,
            )
        except Exception as error:
            return Eagle3TrainResult.failed(
                task=task,
                cpu_start_version=cpu_start_version,
                cpu_end_version=self.trainer.version,
                error=f"{type(error).__name__}: {error}",
            )

        return Eagle3TrainResult.succeeded(
            task=task,
            cpu_start_version=cpu_start_version,
            updated_persistent_cache=updated_cache,
            training_result=training_result,
        )
