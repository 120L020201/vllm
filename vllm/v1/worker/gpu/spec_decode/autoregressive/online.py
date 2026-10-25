# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.online import OnlineDraftSpeculator

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
        DraftProposalSnapshot,
    )


class AutoRegressiveOnlineDraft(OnlineDraftSpeculator):
    """Online capture for drafts that prefill on target rows, then decode from
    their own hidden state (EAGLE3, MTP).

    Subclasses only choose which target features the draft consumes; the
    capture boundary sits before the draft's first weight-dependent op.
    """

    explicit_input_embeds: bool
    last_prefill_input_embeds: torch.Tensor | None
    last_prefill_positions: torch.Tensor | None
    last_prefill_length: torch.Tensor | None
    # Provided by AutoRegressiveSpeculator, which follows this mixin in the MRO.
    get_proposal_snapshot: Callable[[], "DraftProposalSnapshot"]

    def _init_autoregressive_online_draft(self) -> None:
        self._init_online_draft()
        self.explicit_input_embeds = self.online_draft_training_enabled

    def _online_target_features(
        self,
        aux_hidden_states: list[torch.Tensor] | None,
        target_hidden_states: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        """Target rows the draft combines with the next token's embedding."""
        raise NotImplementedError

    def on_prefill_input_embeds_ready(
        self,
        num_reqs: int,
        input_batch: InputBatch,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        capture_enabled: bool,
        target_hidden_states: torch.Tensor | None = None,
    ) -> None:
        if not capture_enabled or self.capture_adapter is None:
            return

        assert self.last_prefill_input_embeds is not None
        assert self.last_prefill_positions is not None
        assert self.last_prefill_length is not None

        # Draft prefill row i pairs target row i with the next token.
        num_tokens = input_batch.num_tokens
        features = self._online_target_features(aux_hidden_states, target_hidden_states)
        self._capture_target_and_install(
            input_batch,
            input_embeds=self.last_prefill_input_embeds[:num_tokens],
            positions=self.last_prefill_positions[:num_tokens],
            aux_hidden_states=[feature[:num_tokens] for feature in features],
            confirmed_length=self.last_prefill_length,
            num_sampled=num_sampled[0],
        )

    def on_proposal_ready(
        self,
        input_batch: InputBatch,
        aux_hidden_states: list[torch.Tensor] | None,
        capture_enabled: bool,
    ) -> None:
        if not capture_enabled or self.capture_adapter is None:
            return

        proposal = self.get_proposal_snapshot()
        self.capture_adapter.capture_proposal(
            token_input_embeds=proposal.draft_token_input_embeds,
            input_hidden_states=proposal.draft_input_hidden_states,
        )
