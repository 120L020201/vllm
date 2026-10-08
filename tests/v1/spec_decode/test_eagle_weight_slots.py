# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn

from vllm.v1.worker.gpu.spec_decode.eagle.weight_slots import (
    Eagle3WeightSlots,
)


def test_only_selected_draft_owned_parameters_receive_slots() -> None:
    target = nn.Module()
    target.embed_tokens = nn.Embedding(4, 3)

    draft = nn.Module()
    draft.embed_tokens = target.embed_tokens
    draft.projection = nn.Linear(3, 2)
    draft.lm_head = nn.Linear(3, 4, bias=False)

    mutable_names = (
        "projection.weight",
        "projection.bias",
    )
    slots = Eagle3WeightSlots.from_models(
        target,
        draft,
        mutable_names,
    )

    assert slots.owned_names == mutable_names

    for name in slots.owned_names:
        assert torch.equal(slots.slots[0][name], slots.slots[1][name])


def test_bind_switches_draft_owned_parameters() -> None:
    target = nn.Module()
    target.embed_tokens = nn.Embedding(4, 3)

    draft = nn.Module()
    draft.embed_tokens = target.embed_tokens
    draft.projection = nn.Linear(3, 2)

    slots = Eagle3WeightSlots.from_models(
        target,
        draft,
        (
            "projection.weight",
            "projection.bias",
        ),
    )

    new_weight = torch.full_like(
        slots.slots[1]["projection.weight"],
        3.0,
    )
    slots.slots[1]["projection.weight"].copy_(new_weight)

    slots.bind(slots.slots[1])

    assert torch.equal(draft.projection.weight, new_weight)
    assert draft.embed_tokens is target.embed_tokens
