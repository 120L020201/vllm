# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Chunk-wise ensemble of three CPU EAGLE3 learners."""

import copy
import math
from dataclasses import dataclass

import torch
from online_draft.training.eagle3_batch import Eagle3DistillationBatch
from online_draft.training.trainer import DraftTrainer, TrainerConfig

from methods.tts_common.step import train_batches


@dataclass(frozen=True, slots=True)
class WeightSnapshot:
    version: int
    state_dict: dict[str, torch.Tensor]


def export_trainable_state(trainer: DraftTrainer) -> dict[str, torch.Tensor]:
    state = trainer.model.state_dict()
    return {
        name: state[name].detach().cpu().clone()
        for name in trainer.trainable_parameter_names
    }


class ChunkEnsemble:
    def __init__(
        self,
        trainer: DraftTrainer,
        learning_rates: tuple[float, ...],
        epsilon: float,
    ) -> None:
        if len(learning_rates) != 3 or any(lr <= 0 for lr in learning_rates):
            raise ValueError("ospec requires three positive learning rates")
        if not math.isfinite(epsilon) or epsilon < 0:
            raise ValueError("epsilon must be finite and nonnegative")
        self.learners = [
            DraftTrainer(
                copy.deepcopy(trainer.model),
                TrainerConfig(
                    learning_rate=learning_rate,
                    weight_decay=trainer.config.weight_decay,
                    frozen_parameter_prefixes=(
                        trainer.config.frozen_parameter_prefixes
                    ),
                    check_gradients=trainer.config.check_gradients,
                ),
            )
            for learning_rate in learning_rates
        ]
        self.epsilon = epsilon
        self.cumulative_losses = torch.zeros(3, dtype=torch.float64)
        self.version = 0
        self.last_update_steps = 0

    def update(
        self,
        requests: tuple[tuple[Eagle3DistillationBatch, ...], ...],
    ) -> WeightSnapshot | None:
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
        states = [export_trainable_state(learner) for learner in self.learners]
        merged = {
            name: sum(
                float(weight) * state[name].float()
                for weight, state in zip(weights, states, strict=True)
            ).to(dtype=states[0][name].dtype)
            for name in states[0]
        }
        self.version += 1
        return WeightSnapshot(version=self.version, state_dict=merged)
