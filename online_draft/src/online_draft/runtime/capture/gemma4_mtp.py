# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Gemma4 MTP capture: the EAGLE3 packet plus the shared target K/V rows."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from online_draft.runtime.capture.base import ConfirmedRows
from online_draft.runtime.capture.eagle3 import (
    Eagle3CaptureAdapter,
    Eagle3CaptureBuffers,
    Eagle3CapturePacket,
    Eagle3Proposal,
    _Eagle3BatchBuilder,
)
from online_draft.runtime.transport import PinnedStaging
from online_draft.training.gemma4_mtp.window import Gemma4MTPBatch

# Staging and packet fields for the target's last sliding and full layer.
KV_FIELDS = ("sliding_kv", "full_kv")


class Gemma4MTPCaptureBuffers(Eagle3CaptureBuffers):
    def __init__(
        self,
        max_tokens: int,
        num_speculative_tokens: int,
        hidden_size: int,
        vocab_size: int,
        kv_shapes: tuple[tuple[int, int], ...],
        *,
        dtype: torch.dtype,
        teacher_dtype: torch.dtype,
    ) -> None:
        super().__init__(
            max_tokens,
            num_speculative_tokens,
            hidden_size,
            1,
            vocab_size,
            dtype=dtype,
            teacher_dtype=teacher_dtype,
        )
        layouts = {
            name: ((max_tokens, *shape), dtype)
            for name, shape in zip(KV_FIELDS, kv_shapes, strict=True)
        }
        self.tensors |= PinnedStaging(layouts).tensors


@dataclass(slots=True, kw_only=True)
class Gemma4MTPCapturePacket(Eagle3CapturePacket):
    """Rows also carry target K|V: [tokens, kv_heads, 2 * head_dim]."""

    sliding_kv: torch.Tensor
    full_kv: torch.Tensor


class _Gemma4MTPBatchBuilder(_Eagle3BatchBuilder):
    """Build the EAGLE3 round, then attach the K/V rows it adds to history."""

    def _own_rows(self, packet, count):
        return (
            *super()._own_rows(packet, count),
            packet.sliding_kv[:count].clone(),
            packet.full_kv[:count].clone(),
        )

    def _build_batch(
        self,
        proposal: Eagle3Proposal,
        teacher_logits: torch.Tensor,
        confirmed: ConfirmedRows,
        prompt: list[ConfirmedRows],
    ) -> Gemma4MTPBatch:  # type: ignore[override]
        def eagle3_rows(rows: ConfirmedRows) -> ConfirmedRows:
            return ConfirmedRows(rows.positions, rows.features[:2])

        batch = super()._build_batch(
            proposal,
            teacher_logits,
            eagle3_rows(confirmed),
            [eagle3_rows(rows) for rows in prompt],
        )
        history = [rows.features[2:] for rows in (*prompt, confirmed)]
        return Gemma4MTPBatch(
            batch, tuple(torch.cat(layer) for layer in zip(*history, strict=True))
        )


class Gemma4MTPCaptureAdapter(Eagle3CaptureAdapter):
    """Pass ``sliding_kv`` and ``full_kv`` to ``capture_target``."""

    _packet_type = Gemma4MTPCapturePacket
    _builder_type = _Gemma4MTPBatchBuilder
