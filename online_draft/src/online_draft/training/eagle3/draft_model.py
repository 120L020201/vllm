# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""What EAGLE3-style window training needs from a CPU draft model.

EAGLE3 and MTP drafts both prefill on target rows and then decode from their
own hidden state, so the same batch, cache, and window code trains any model
with this interface.
"""

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import torch
from online_draft.models.layers import DraftForwardOutput, LayerKV
from torch import nn


class AutoregressiveDraftConfig(Protocol):
    @property
    def hidden_size(self) -> int: ...

    @property
    def num_aux_hidden_states(self) -> int:
        """Target feature rows concatenated per position (1 for MTP)."""
        ...

    @property
    def draft_vocab_size(self) -> int: ...


@runtime_checkable
class AutoregressiveDraftModel(Protocol):
    @property
    def config(self) -> AutoregressiveDraftConfig: ...

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: ...

    def combine_hidden_states(
        self, auxiliary_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Map captured target features to the draft's hidden input."""
        ...

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

    def __call__(
        self,
        *,
        positions: torch.Tensor,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_values: tuple[LayerKV, ...] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> DraftForwardOutput: ...
