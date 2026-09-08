# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .async_bridge import TrainObservation
from .observations import (
    DraftTrainLabels,
    build_target_to_draft_map,
    build_verified_target_labels,
    map_target_to_draft_labels,
)
from .qwen3_trainer import Qwen3Eagle3CpuTrainer


def qwen3_eagle3_ce_step(
    trainer: Qwen3Eagle3CpuTrainer,
    observations: Sequence[TrainObservation],
) -> None:
    """Run one supervised CE update over completed draft-verify iterations."""
    losses: list[torch.Tensor] = []
    trainer.zero_grad()

    for observation in observations:
        loss = _loss_for_observation(trainer, observation)
        if loss is not None:
            losses.append(loss)

    if not losses:
        trainer.zero_grad()
        return

    torch.stack(losses).mean().backward()
    trainer.step()


def _loss_for_observation(
    trainer: Qwen3Eagle3CpuTrainer,
    observation: TrainObservation,
) -> torch.Tensor | None:
    payload = observation.payload
    required_keys = (
        "proposal_input_ids",
        "proposal_input_embeds",
        "proposal_positions",
        "proposal_hidden_states",
        "sampled_token_ids",
        "num_sampled",
        "proposal_num_speculative_tokens",
    )
    if any(key not in payload for key in required_keys):
        return None

    num_speculative_tokens = int(
        payload["proposal_num_speculative_tokens"].reshape(-1)[0].item()
    )
    verify_labels = build_verified_target_labels(
        _cpu_long(payload["sampled_token_ids"]),
        _cpu_long(payload["num_sampled"]),
        num_speculative_tokens,
    )
    target_to_draft = _get_target_to_draft_map(trainer.model)

    model_device, model_dtype = _model_device_and_dtype(trainer.model)
    input_ids = _cpu_long(payload["proposal_input_ids"][:num_speculative_tokens])
    positions = _cpu_long(payload["proposal_positions"][:num_speculative_tokens])
    hidden_states = _cpu_float(
        payload["proposal_hidden_states"][:num_speculative_tokens],
        device=model_device,
        dtype=model_dtype,
    )
    aux_hidden_states = payload.get("proposal_aux_hidden_states")
    if aux_hidden_states is not None and hasattr(
        trainer.model, "combine_hidden_states"
    ):
        aux_hidden_states = _cpu_float(
            aux_hidden_states,
            device=model_device,
            dtype=model_dtype,
        )
        combined_hidden_states = trainer.model.combine_hidden_states(aux_hidden_states)
        hidden_states = hidden_states.clone()
        hidden_states[: combined_hidden_states.shape[0]] = combined_hidden_states
    input_embeds = _cpu_float(
        payload["proposal_input_embeds"][:num_speculative_tokens],
        device=model_device,
        dtype=model_dtype,
    )

    output = trainer.model(
        input_ids=input_ids.to(model_device),
        positions=positions.to(model_device),
        hidden_states=hidden_states,
        inputs_embeds=input_embeds,
    )
    hidden_output = output[0] if isinstance(output, tuple) else output

    if target_to_draft is not None:
        train_labels = map_target_to_draft_labels(
            verify_labels.target_token_ids,
            target_to_draft,
            verify_labels.loss_mask,
        )
        logits = _compute_draft_logits(trainer.model, hidden_output)
    else:
        train_labels = DraftTrainLabels(
            draft_token_ids=verify_labels.target_token_ids,
            loss_mask=verify_labels.loss_mask,
        )
        logits = trainer.model.compute_logits(hidden_output)

    active_mask = train_labels.loss_mask.to(logits.device)
    if not active_mask.any():
        return None

    return F.cross_entropy(
        logits[active_mask],
        train_labels.draft_token_ids.to(logits.device)[active_mask],
    )


def _get_target_to_draft_map(model: torch.nn.Module) -> torch.Tensor | None:
    d2t = getattr(model, "draft_id_to_target_id", None)
    config = getattr(model, "config", None)
    target_vocab_size = getattr(config, "vocab_size", None)
    if d2t is None or target_vocab_size is None:
        return None
    if isinstance(d2t, torch.nn.Parameter):
        d2t = d2t.data
    return build_target_to_draft_map(d2t.detach().cpu(), int(target_vocab_size))


def _compute_draft_logits(
    model: torch.nn.Module,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    compute_draft_logits = getattr(model, "compute_draft_logits", None)
    if compute_draft_logits is not None:
        return compute_draft_logits(hidden_states)

    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise AttributeError("model must define compute_draft_logits or lm_head")
    logits = lm_head(hidden_states)
    return logits[0] if isinstance(logits, tuple) else logits


def _model_device_and_dtype(model: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.device, parameter.dtype
    return torch.device("cpu"), torch.float32


def _cpu_long(tensor: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode(False):
        return tensor.detach().cpu().clone().to(torch.long)


def _cpu_float(
    tensor: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    with torch.inference_mode(False):
        return tensor.detach().cpu().clone().to(device=device, dtype=dtype)
