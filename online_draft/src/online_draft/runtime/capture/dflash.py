# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import torch

from online_draft.runtime.capture.base import (
    CaptureAdapterBase,
    CaptureBuffers,
    ConfirmedRows,
    TargetHistoryBuilder,
    TargetPacket,
    teacher_probabilities,
)
from online_draft.runtime.contracts import PreparedRound
from online_draft.runtime.cpu_worker import CPUWorkerConfig
from online_draft.runtime.transport import PinnedStaging
from online_draft.training.dflash.batch import DFlashBatch


class DFlashCaptureBuffers(CaptureBuffers):
    """DFlash packet layout in the shared pinned staging."""

    def __init__(
        self,
        max_tokens: int,
        num_query_tokens: int,
        feature_size: int,
        vocab_size: int,
        *,
        dtype: torch.dtype,
        teacher_dtype: torch.dtype,
    ) -> None:
        super().__init__(
            max_tokens=max_tokens,
            teacher_shape=(num_query_tokens, vocab_size),
            teacher_dtype=teacher_dtype,
            layouts={
                "proposal_metadata": ((2, num_query_tokens), torch.int64),
                "target_fc": ((max_tokens, feature_size), dtype),
            },
        )


@dataclass(frozen=True, slots=True)
class DFlashProposal:
    """Snapshot rows contain input token IDs and absolute query positions."""

    metadata: torch.Tensor
    source_weight_version: int

    def to_cpu(
        self,
        buffers: PinnedStaging,
        *,
        non_blocking: bool = False,
    ) -> DFlashProposal:
        return replace(
            self,
            metadata=buffers.copy("proposal_metadata", self.metadata, non_blocking),
        )


@dataclass(slots=True, kw_only=True)
class DFlashCapturePacket(TargetPacket[DFlashProposal]):
    """Target rows: the frozen FC output for each scheduled token."""

    target_fc: torch.Tensor


class _DFlashBatchBuilder(
    TargetHistoryBuilder[DFlashCapturePacket, torch.Tensor, DFlashBatch]
):
    """Assemble one complete DFlash query block per round."""

    def _own_rows(self, packet: DFlashCapturePacket, count: int) -> torch.Tensor:
        return packet.target_fc[:count].clone()

    def _build_batch(
        self,
        proposal: DFlashProposal,
        teacher_logits: torch.Tensor,
        confirmed: ConfirmedRows[torch.Tensor],
        prompt: list[ConfirmedRows[torch.Tensor]],
    ) -> DFlashBatch:
        input_ids, query_positions = proposal.metadata.unbind()
        if int(confirmed.positions[0]) != int(query_positions[0]):
            raise ValueError("DFlash query block must start at the verified anchor")

        # One block predicts k tokens after its anchor.
        k = input_ids.numel() - 1
        confirmed_length = confirmed.positions.numel()
        targets = min(confirmed_length, k)

        return DFlashBatch(
            input_ids=input_ids.clone(),
            positions=query_positions.clone(),
            prompt_fc=torch.cat([rows.features for rows in prompt]) if prompt else None,
            confirmed_fc=confirmed.features,
            teacher_probabilities=teacher_probabilities(teacher_logits[:targets]),
            rejection_position=(
                confirmed_length - 1 if confirmed_length <= k else None
            ),
        )


class DFlashCaptureAdapter(CaptureAdapterBase[DFlashProposal]):
    """Capture DFlash FC outputs and query blocks."""

    _packet_type = DFlashCapturePacket

    def __init__(
        self,
        *,
        buffers: DFlashCaptureBuffers,
        source_weight_version: Callable[[], int],
        on_round: Callable[[PreparedRound[DFlashBatch]], object],
        on_epoch_close: Callable[[str], object],
        on_finish: Callable[[str], object],
        on_queue_size: Callable[[int], None] | None = None,
        profile_timeline: bool = False,
        cpu_config: CPUWorkerConfig | None = None,
    ) -> None:
        super().__init__(
            buffers=buffers,
            builder=_DFlashBatchBuilder(
                on_round=on_round,
                on_epoch_close=on_epoch_close,
                on_finish=on_finish,
            ),
            source_weight_version=source_weight_version,
            on_queue_size=on_queue_size,
            profile_timeline=profile_timeline,
            cpu_config=cpu_config,
        )

    def capture_target(
        self,
        *,
        request_id: str,
        final_prefill: bool,
        fc_output: torch.Tensor,
        positions: torch.Tensor,
        num_rejected: torch.Tensor,
        num_sampled: torch.Tensor,
    ) -> None:
        with self._profile("online_draft.capture_target"):
            self._submit_target(
                request_id=request_id,
                final_prefill=final_prefill,
                positions=positions,
                confirmed_length=positions.shape[0] - num_rejected,
                num_sampled=num_sampled,
                target_fc=fc_output.detach().clone(),
            )

    def capture_proposal(
        self,
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        self._save_proposal(
            lambda version: DFlashProposal(torch.stack((input_ids, positions)), version)
        )
