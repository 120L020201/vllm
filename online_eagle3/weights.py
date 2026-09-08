# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn

QWEN3_EAGLE3_FROZEN_PREFIXES: tuple[str, ...] = (
    "model.embed_tokens",
    "lm_head",
    "draft_id_to_target_id",
    "mask_hidden",
)


@dataclass(slots=True)
class TrainableWeightSnapshot:
    version: int
    state_dict: dict[str, torch.Tensor]


def _matches_prefix(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(f"{prefix}.")


def is_frozen_name(name: str, frozen_prefixes: Sequence[str]) -> bool:
    return any(_matches_prefix(name, prefix) for prefix in frozen_prefixes)


def freeze_parameters(
    module: nn.Module,
    frozen_prefixes: Sequence[str] = QWEN3_EAGLE3_FROZEN_PREFIXES,
) -> list[str]:
    frozen_names: list[str] = []
    for name, parameter in module.named_parameters():
        if is_frozen_name(name, frozen_prefixes):
            parameter.requires_grad_(False)
            frozen_names.append(name)
        else:
            parameter.requires_grad_(True)
    return frozen_names


def get_trainable_named_parameters(
    module: nn.Module,
    frozen_prefixes: Sequence[str] = QWEN3_EAGLE3_FROZEN_PREFIXES,
):
    for name, parameter in module.named_parameters():
        if not is_frozen_name(name, frozen_prefixes):
            yield name, parameter


def export_trainable_state_dict(
    module: nn.Module,
    frozen_prefixes: Sequence[str] = QWEN3_EAGLE3_FROZEN_PREFIXES,
) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    for name, parameter in get_trainable_named_parameters(module, frozen_prefixes):
        state_dict[name] = parameter.detach().cpu().clone()
    return state_dict


def load_trainable_state_dict(
    module: nn.Module,
    trainable_state: Mapping[str, torch.Tensor],
    frozen_prefixes: Sequence[str] = QWEN3_EAGLE3_FROZEN_PREFIXES,
    *,
    strict: bool = True,
) -> None:
    current_parameters = dict(module.named_parameters())
    expected_keys = set(
        name for name in current_parameters if not is_frozen_name(name, frozen_prefixes)
    )
    incoming_keys = set(trainable_state)

    missing_keys = sorted(expected_keys - incoming_keys)
    unexpected_keys = sorted(incoming_keys - expected_keys)
    if strict and (missing_keys or unexpected_keys):
        raise KeyError(
            "Trainable state mismatch: "
            f"missing={missing_keys}, unexpected={unexpected_keys}"
        )

    with torch.no_grad():
        for name, source in trainable_state.items():
            if name not in expected_keys:
                continue
            target = current_parameters[name]
            if target.shape != source.shape:
                raise ValueError(
                    f"Shape mismatch for {name}: "
                    f"expected={tuple(target.shape)}, got={tuple(source.shape)}"
                )
            target.copy_(
                source.to(device=target.device, dtype=target.dtype),
                non_blocking=True,
            )
