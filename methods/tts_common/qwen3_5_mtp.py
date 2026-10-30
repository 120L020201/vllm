# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""TTS adapter for the Qwen3.5 MTP training model."""

from types import MethodType

import torch
import torch.nn.functional as F
from online_draft.models.qwen3_5_mtp import (
    Qwen3_5MTPForCausalLM,
    load_qwen3_5_mtp_from_vllm,
)
from torch import nn


def _compute_draft_logits(
    self: Qwen3_5MTPForCausalLM,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    return F.linear(hidden_states, self.lm_head.weight)


def _get_target_token_ids(
    self: Qwen3_5MTPForCausalLM,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    if device is None:
        device = self.lm_head.weight.device
    return torch.arange(self.config.vocab_size, dtype=torch.long, device=device)


def load_qwen3_5_mtp_for_tts(
    gpu_model: nn.Module,
    *,
    attention_backend: str | None = None,
    lm_head: nn.Module | None = None,
) -> Qwen3_5MTPForCausalLM:
    """Load Qwen3.5 MTP with the TTS identity vocabulary and frozen head."""
    model = load_qwen3_5_mtp_from_vllm(
        gpu_model,
        attention_backend=attention_backend,
    )
    if lm_head is not None:
        del model.lm_head
    model.compute_draft_logits = MethodType(_compute_draft_logits, model)
    model.get_target_token_ids = MethodType(_get_target_token_ids, model)
    return model
