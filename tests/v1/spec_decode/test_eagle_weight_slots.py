# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from online_draft.runtime.draft_weight_installer import DraftWeightInstaller
from online_draft.runtime.weight_snapshot import DraftWeightSnapshot

from vllm.v1.worker.gpu.spec_decode.eagle.speculator import (
    EagleSpeculator,
    logger,
)
from vllm.v1.worker.gpu.spec_decode.eagle.weight_slots import (
    Eagle3WeightSlots,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graphs")
@torch.no_grad()
def test_dflash_graphs_read_each_updated_slot_and_reset_weights(monkeypatch):
    """Captured pointers must follow slot reuse, including a reset on either slot."""
    from online_draft.models.qwen3_dflash import (
        Qwen3DFlashConfig,
        Qwen3DFlashForCausalLM,
    )

    from vllm.v1.worker.gpu.spec_decode.dflash.online import DFlashWeightSlots

    def cuda_attention(query, context_key, context_value, key, value, mask):
        return F.scaled_dot_product_attention(
            query.unsqueeze(0),
            torch.cat((context_key, key), dim=1).unsqueeze(0),
            torch.cat((context_value, value), dim=1).unsqueeze(0),
            attn_mask=mask,
        ).squeeze(0)

    monkeypatch.setattr(
        "online_draft.models.qwen3_dflash.context_attention", cuda_attention
    )

    torch.manual_seed(17)
    draft = Qwen3DFlashForCausalLM(Qwen3DFlashConfig(16, 32, 2, 1, 8, 2, 23, 2)).to(
        device="cuda", dtype=torch.bfloat16
    )
    target = nn.Module()
    target.embed_tokens, target.lm_head = draft.model.embed_tokens, draft.lm_head
    draft.model._fused_kv_weight = torch.cat(
        [
            layer.self_attn.qkv_proj.weight[layer.self_attn.q_size :]
            for layer in draft.model.layers
        ]
    )
    draft.model._k_norm_weights = torch.stack(
        [layer.self_attn.k_norm.weight for layer in draft.model.layers]
    )
    mutable = tuple(n for n, p in draft.named_parameters() if p.requires_grad)
    slots = DFlashWeightSlots(target, draft, mutable)
    installer = DraftWeightInstaller(
        slots.slots, mutable_names=mutable, prepare_slot=slots.prepare
    )
    initial = tuple((n, p.cpu().clone()) for n, p in installer.active_weights.items())
    features = torch.randn(5, 32, device="cuda", dtype=torch.bfloat16)
    context_positions = torch.arange(5, device="cuda")
    input_ids = torch.tensor([2, 22, 22, 22], device="cuda")
    positions = torch.arange(5, 9, device="cuda")

    def forward():
        context = draft.project_context(draft.model.fc(features), context_positions)
        return draft.lm_head(draft(input_ids, positions, context))

    original = forward().clone()
    graphs, outputs = [], []
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    for weights in slots.slots:
        slots.bind(weights)
        with torch.cuda.stream(stream):
            forward()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = forward()
        graphs.append(graph)
        outputs.append(output)
    torch.cuda.current_stream().wait_stream(stream)
    slots.bind(installer.active_weights)

    for version in (1, 2, 3):
        installer.stage(
            DraftWeightSnapshot.from_named_tensors(
                [(n, p + version * 0.05) for n, p in initial], version=version
            )
        )
        torch.accelerator.synchronize()
        assert installer.commit_if_ready() == version
        assert installer.active_slot == version % 2
        slots.bind(installer.active_weights)
        graphs[installer.active_slot].replay()
        torch.testing.assert_close(outputs[installer.active_slot], forward())
        assert not torch.allclose(outputs[installer.active_slot], original)

        installer.reset(initial)
        slots.bind(installer.active_weights)
        graphs[installer.active_slot].replay()
        torch.testing.assert_close(outputs[installer.active_slot], original)


def test_dflash_commit_and_reset_keep_derived_buffers_with_active_parameters():
    from online_draft.models.qwen3_dflash import (
        Qwen3DFlashConfig,
        Qwen3DFlashForCausalLM,
    )

    from vllm.v1.worker.gpu.spec_decode.dflash.online import DFlashWeightSlots

    draft = Qwen3DFlashForCausalLM(Qwen3DFlashConfig(8, 16, 2, 1, 4, 2, 23, 2))
    target = nn.Module()
    target.embed_tokens = draft.model.embed_tokens
    target.lm_head = draft.lm_head
    draft.model.layers[0].self_attn.qkv_proj.requires_grad_(False)

    def derived():
        return (
            torch.cat(
                [
                    layer.self_attn.qkv_proj.weight[layer.self_attn.q_size :]
                    for layer in draft.model.layers
                ]
            ),
            torch.stack(
                [layer.self_attn.k_norm.weight for layer in draft.model.layers]
            ),
        )

    with torch.no_grad():
        draft.model._fused_kv_weight, draft.model._k_norm_weights = derived()
    mutable = tuple(n for n, p in draft.named_parameters() if p.requires_grad)
    slots = DFlashWeightSlots(target, draft, mutable)
    installer = DraftWeightInstaller(
        slots.slots, mutable_names=mutable, prepare_slot=slots.prepare
    )
    initial = tuple((n, p.clone()) for n, p in installer.active_weights.items())
    for version in (1, 2):
        installer.stage(
            DraftWeightSnapshot.from_named_tensors(
                [(n, p + version) for n, p in initial],
                version=version,
                pin_memory=False,
            )
        )
        assert installer.commit_if_ready() == version
        slots.bind(installer.active_weights)
        expected_kv, expected_norm = derived()
        torch.testing.assert_close(draft.model._fused_kv_weight, expected_kv)
        torch.testing.assert_close(draft.model._k_norm_weights, expected_norm)
        assert (
            draft.model._hidden_norm_weight.data_ptr()
            == draft.model.hidden_norm.weight.data_ptr()
        )
    installer.reset(initial)
    slots.bind(installer.active_weights)
    expected_kv, expected_norm = derived()
    torch.testing.assert_close(draft.model._fused_kv_weight, expected_kv)
    torch.testing.assert_close(draft.model._k_norm_weights, expected_norm)


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
    speculator._log_online_draft_updates = True
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


def test_weight_install_skips_update_log_when_disabled() -> None:
    installed: list[str] = []
    speculator = object.__new__(EagleSpeculator)
    speculator.capture_adapter = SimpleNamespace()
    speculator.weight_handoff = SimpleNamespace(advance=lambda: 2)
    speculator._log_online_draft_updates = False
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
