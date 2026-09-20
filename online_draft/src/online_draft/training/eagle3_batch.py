# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch

_POSITION_FIELD_NAMES = (
    "prefill_positions",
    "proposal_positions",
    "confirmed_positions",
)

_FEATURE_FIELD_NAMES = (
    "prefill_input_embeds",
    "prefill_aux_hidden_states",
    "draft_token_input_embeds",
    "draft_recurrent_hidden_states",
    "confirmed_input_embeds",
    "confirmed_aux_hidden_states",
)


@dataclass(slots=True)
class Eagle3DistillationBatch:
    """Prepared CPU tensors for one EAGLE3 verification round.

    The first proposal hidden input is reconstructed from the final
    prefill auxiliary hidden state through the trainable FC layer.
    Later proposal hidden inputs are fixed recurrent hidden states
    captured from GPU draft generation.
    """

    prefill_positions: torch.Tensor
    prefill_input_embeds: torch.Tensor
    prefill_aux_hidden_states: torch.Tensor

    proposal_positions: torch.Tensor
    draft_token_input_embeds: torch.Tensor
    draft_recurrent_hidden_states: torch.Tensor
    teacher_probabilities: torch.Tensor

    confirmed_positions: torch.Tensor
    confirmed_input_embeds: torch.Tensor
    confirmed_aux_hidden_states: torch.Tensor

    rejection_position: int | None

    @property
    def draft_length(self) -> int:
        return self.proposal_positions.numel()

    @property
    def confirmed_length(self) -> int:
        return self.confirmed_positions.numel()

    def validate(
        self,
        *,
        hidden_size: int,
        num_aux_hidden_states: int,
        draft_vocab_size: int,
        feature_dtype: torch.dtype,
    ) -> None:
        """Validate shapes, positions, dtypes, and rejection boundaries."""
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_aux_hidden_states <= 0:
            raise ValueError("num_aux_hidden_states must be positive")
        if draft_vocab_size <= 0:
            raise ValueError("draft_vocab_size must be positive")
        if feature_dtype not in (
            torch.float32,
            torch.bfloat16,
        ):
            raise ValueError("feature_dtype must be float32 or bfloat16")

        self._validate_tensor_properties()
        self._validate_dtypes(feature_dtype)
        self._validate_shapes(
            hidden_size=hidden_size,
            num_aux_hidden_states=num_aux_hidden_states,
            draft_vocab_size=draft_vocab_size,
        )
        self._validate_positions()
        self._validate_rejection_position()

    def _validate_tensor_properties(self) -> None:
        tensor_names = (
            *_POSITION_FIELD_NAMES,
            *_FEATURE_FIELD_NAMES,
            "teacher_probabilities",
        )

        for name in tensor_names:
            tensor = getattr(self, name)

            if tensor.device.type != "cpu":
                raise ValueError(f"{name} must already be on the CPU")
            if torch.is_inference(tensor):
                raise ValueError(f"{name} must not be an inference tensor")
            if tensor.requires_grad:
                raise ValueError(f"{name} must be detached")

    def _validate_dtypes(
        self,
        feature_dtype: torch.dtype,
    ) -> None:
        for name in _POSITION_FIELD_NAMES:
            tensor = getattr(self, name)

            if tensor.dtype != torch.long:
                raise ValueError(f"{name} must use torch.long")

        for name in _FEATURE_FIELD_NAMES:
            tensor = getattr(self, name)

            if tensor.dtype != feature_dtype:
                raise ValueError(f"{name} must use {feature_dtype}")

        if self.teacher_probabilities.dtype != torch.float32:
            raise ValueError("teacher_probabilities must use torch.float32")

    def _validate_shapes(
        self,
        *,
        hidden_size: int,
        num_aux_hidden_states: int,
        draft_vocab_size: int,
    ) -> None:
        for name in _POSITION_FIELD_NAMES:
            positions = getattr(self, name)

            if positions.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")

        prefill_length = self.prefill_positions.numel()
        draft_length = self.draft_length
        confirmed_length = self.confirmed_length
        auxiliary_size = hidden_size * num_aux_hidden_states

        if prefill_length == 0:
            raise ValueError("prefill must not be empty")
        if draft_length == 0:
            raise ValueError("proposal must not be empty")
        if not 1 <= confirmed_length <= draft_length + 1:
            raise ValueError("confirmed length must be between 1 and draft_length + 1")

        expected_shapes = {
            "prefill_input_embeds": (
                prefill_length,
                hidden_size,
            ),
            "prefill_aux_hidden_states": (
                prefill_length,
                auxiliary_size,
            ),
            "draft_token_input_embeds": (
                draft_length - 1,
                hidden_size,
            ),
            "draft_recurrent_hidden_states": (
                draft_length - 1,
                hidden_size,
            ),
            "teacher_probabilities": (
                draft_length,
                draft_vocab_size,
            ),
            "confirmed_input_embeds": (
                confirmed_length,
                hidden_size,
            ),
            "confirmed_aux_hidden_states": (
                confirmed_length,
                auxiliary_size,
            ),
        }

        for name, expected_shape in expected_shapes.items():
            tensor = getattr(self, name)

            if tensor.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}")

    def _validate_positions(self) -> None:
        prefill_start = int(self.prefill_positions[0].item())
        if prefill_start < 0:
            raise ValueError("prefill positions must not be negative")

        expected_prefill = torch.arange(
            prefill_start,
            prefill_start + self.prefill_positions.numel(),
            dtype=torch.long,
        )
        if not torch.equal(
            self.prefill_positions,
            expected_prefill,
        ):
            raise ValueError("prefill positions must be contiguous")

        anchor_position = int(self.prefill_positions[-1].item())
        expected_proposal = torch.arange(
            anchor_position,
            anchor_position + self.draft_length,
            dtype=torch.long,
        )
        if not torch.equal(
            self.proposal_positions,
            expected_proposal,
        ):
            raise ValueError("proposal positions must start at the prefill anchor")

        expected_confirmed = torch.arange(
            anchor_position + 1,
            anchor_position + 1 + self.confirmed_length,
            dtype=torch.long,
        )
        if not torch.equal(
            self.confirmed_positions,
            expected_confirmed,
        ):
            raise ValueError("confirmed positions must follow the prefill anchor")

    def _validate_rejection_position(self) -> None:
        rejection_position = self.rejection_position

        if rejection_position is None:
            expected_confirmed_length = self.draft_length + 1
        else:
            if isinstance(rejection_position, bool) or not isinstance(
                rejection_position, int
            ):
                raise TypeError("rejection_position must be an integer or None")
            if not 0 <= rejection_position < self.draft_length:
                raise ValueError("rejection_position must be within the draft sequence")

            expected_confirmed_length = rejection_position + 1

        if self.confirmed_length != expected_confirmed_length:
            raise ValueError("confirmed length does not match rejection_position")
