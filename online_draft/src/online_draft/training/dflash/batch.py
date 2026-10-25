# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DFlashBatch:
    """One verified DFlash query block.

    Block row 0 holds the anchor token and row j predicts the token j
    positions later. The first ``target_count`` predictions are supervised:
    accepted drafts plus the correction, or every draft when all are
    accepted. ``confirmed_fc`` holds the frozen FC output of the verified
    rows starting at the anchor; it becomes context for the next round.
    """

    input_ids: torch.Tensor
    positions: torch.Tensor
    # Whole-prompt FC output for the first round, else None.
    prompt_fc: torch.Tensor | None
    confirmed_fc: torch.Tensor
    teacher_probabilities: torch.Tensor
    rejection_position: int | None

    @property
    def anchor(self) -> int:
        return int(self.positions[0])

    @property
    def target_count(self) -> int:
        return self.teacher_probabilities.shape[0]

    def validate(
        self,
        *,
        hidden_size: int,
        vocab_size: int,
        feature_dtype: torch.dtype,
    ) -> None:
        """Validate device, dtypes, shapes, positions, and supervision."""
        tensors = {
            "input_ids": self.input_ids,
            "positions": self.positions,
            "confirmed_fc": self.confirmed_fc,
            "teacher_probabilities": self.teacher_probabilities,
        }
        if self.prompt_fc is not None:
            tensors["prompt_fc"] = self.prompt_fc
        for name, tensor in tensors.items():
            if tensor.device.type != "cpu":
                raise ValueError(f"{name} must already be on the CPU")

        block_size = self.input_ids.numel()
        if self.input_ids.dtype != torch.long or self.input_ids.ndim != 1:
            raise ValueError("input_ids must be a one-dimensional torch.long tensor")
        if block_size < 2:
            raise ValueError("query block must contain an anchor and a draft")
        expected_positions = torch.arange(self.anchor, self.anchor + block_size)
        if self.positions.dtype != torch.long or not torch.equal(
            self.positions, expected_positions
        ):
            raise ValueError("query positions must be contiguous from the anchor")

        feature_shapes = {"confirmed_fc": self.confirmed_fc}
        if self.prompt_fc is not None:
            feature_shapes["prompt_fc"] = self.prompt_fc
            if self.prompt_fc.shape[0] != self.anchor:
                raise ValueError(
                    "prompt_fc must cover every position before the anchor"
                )
        for name, features in feature_shapes.items():
            if features.ndim != 2 or features.shape[1] != hidden_size:
                raise ValueError(f"{name} must have shape [tokens, {hidden_size}]")
            if features.dtype != feature_dtype:
                raise ValueError(f"{name} must use {feature_dtype}")

        confirmed_length = self.confirmed_fc.shape[0]
        if not 1 <= confirmed_length <= block_size:
            raise ValueError("confirmed length must be between 1 and the block size")
        expected_targets = min(confirmed_length, block_size - 1)
        if self.teacher_probabilities.shape != (expected_targets, vocab_size):
            raise ValueError(
                f"teacher_probabilities must have shape ({expected_targets}, "
                f"{vocab_size})"
            )
        if self.teacher_probabilities.dtype != torch.float32:
            raise ValueError("teacher_probabilities must use torch.float32")

        expected_rejection = (
            None if confirmed_length == block_size else confirmed_length - 1
        )
        if self.rejection_position != expected_rejection:
            raise ValueError("rejection_position does not match the confirmed length")
