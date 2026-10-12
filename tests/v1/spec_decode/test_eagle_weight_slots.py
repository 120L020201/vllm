# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
from online_draft.runtime.draft_weight_installer import DraftWeightInstaller
from online_draft.runtime.weight_snapshot import DraftWeightSnapshot

from vllm.v1.worker.gpu.spec_decode.eagle.speculator import (
    EagleSpeculator,
    logger,
)
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


def test_install_committed_weights_binds_slot_before_closing_epoch() -> None:
    target = nn.Module()
    draft = nn.Module()
    draft.projection = nn.Linear(2, 2)
    mutable_names = tuple(dict(draft.named_parameters()))
    slots = Eagle3WeightSlots.from_models(target, draft, mutable_names)
    installer = DraftWeightInstaller(
        slots.slots,
        mutable_names=mutable_names,
    )
    snapshot = DraftWeightSnapshot.from_named_tensors(
        [
            (name, torch.full_like(tensor, 3.0))
            for name, tensor in slots.slots[0].items()
        ],
        version=1,
        pin_memory=False,
    )
    installer.stage(snapshot)
    assert installer.commit_if_ready() == 1

    closed: list[str] = []

    def close_epoch(request_id: str) -> None:
        for name, parameter in draft.named_parameters():
            assert parameter.data_ptr() == installer.active_weights[name].data_ptr()
        closed.append(request_id)

    speculator = object.__new__(EagleSpeculator)
    speculator.capture_adapter = SimpleNamespace(close_epoch=close_epoch)
    speculator.draft_weight_installer = installer
    speculator.draft_weight_slots = slots

    speculator.install_committed_weights("request-0")

    assert closed == ["request-0"]


def test_finish_request_restores_initial_active_weights() -> None:
    target = nn.Module()
    draft = nn.Module()
    draft.projection = nn.Linear(2, 2)
    mutable_names = tuple(dict(draft.named_parameters()))
    slots = Eagle3WeightSlots.from_models(target, draft, mutable_names)
    initial_weights = tuple(
        (name, tensor.clone()) for name, tensor in slots.slots[0].items()
    )
    installer = DraftWeightInstaller(
        slots.slots,
        mutable_names=mutable_names,
    )
    snapshot = DraftWeightSnapshot.from_named_tensors(
        [
            (name, torch.full_like(tensor, 3.0))
            for name, tensor in slots.slots[0].items()
        ],
        version=1,
        pin_memory=False,
    )
    installer.stage(snapshot)
    assert installer.commit_if_ready() == 1
    slots.bind(installer.active_weights)

    finished: list[str] = []

    def finish_request(request_id: str) -> None:
        assert installer.active_version == 1
        finished.append(request_id)

    speculator = object.__new__(EagleSpeculator)
    speculator.capture_adapter = SimpleNamespace(finish_request=finish_request)
    speculator.draft_weight_installer = installer
    speculator.draft_weight_slots = slots
    speculator._last_installed_round = 7

    speculator.finish_request("request-0", initial_weights)

    assert finished == ["request-0"]
    assert speculator._last_installed_round == 0
    assert installer.active_version == 0
    assert installer.staged_version is None
    parameters = dict(draft.named_parameters())
    for name, expected in initial_weights:
        parameter = parameters[name]
        assert torch.equal(parameter, expected)
        assert parameter.data_ptr() == installer.active_weights[name].data_ptr()


def test_weight_install_logs_rounds_since_previous_install() -> None:
    installed: list[str] = []
    speculator = object.__new__(EagleSpeculator)
    speculator.capture_adapter = SimpleNamespace(completed_rounds=5)
    speculator.weight_handoff = SimpleNamespace(advance=lambda: 2)
    speculator._profile_online_draft = True
    speculator._last_installed_round = 2
    speculator.install_committed_weights = installed.append

    with patch.object(logger, "info") as log_info:
        speculator.on_prefill_input_embeds_ready(
            num_reqs=1,
            input_batch=SimpleNamespace(req_ids=["request-0"]),
            aux_hidden_states=None,
            num_sampled=torch.tensor([1]),
            capture_enabled=False,
        )

    assert installed == ["request-0"]
    assert speculator._last_installed_round == 5
    log_info.assert_called_once_with(
        "Online draft installed weights: request_id=%s version=%d rounds=%d",
        "request-0",
        2,
        3,
    )


def test_weight_install_skips_profile_observation_when_disabled() -> None:
    installed: list[str] = []
    speculator = object.__new__(EagleSpeculator)
    speculator.capture_adapter = SimpleNamespace()
    speculator.weight_handoff = SimpleNamespace(advance=lambda: 2)
    speculator._profile_online_draft = False
    speculator.install_committed_weights = installed.append

    with patch.object(logger, "info") as log_info:
        speculator.on_prefill_input_embeds_ready(
            num_reqs=1,
            input_batch=SimpleNamespace(req_ids=["request-0"]),
            aux_hidden_states=None,
            num_sampled=torch.tensor([1]),
            capture_enabled=False,
        )

    assert installed == ["request-0"]
    log_info.assert_not_called()
