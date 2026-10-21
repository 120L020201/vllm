# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import call, patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from online_draft.runtime.weights import DraftWeightInstaller, DraftWeightSnapshot

from vllm.v1.worker.gpu.spec_decode.eagle.online import logger
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator
from vllm.v1.worker.gpu.spec_decode.weight_slots import (
    DraftWeightSlots,
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
        installed = installer.advance(
            DraftWeightSnapshot.from_named_tensors(
                [(n, p + version * 0.05) for n, p in initial], version=version
            )
        )
        torch.accelerator.synchronize()
        if installed is None:
            installed = installer.advance(None)
        assert installed == version
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

    from vllm.v1.worker.gpu.spec_decode.dflash.online import DFlashOnlineState

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
    state = DFlashOnlineState(
        target, draft, mutable, profile_timeline=False, log_updates=False
    )
    installer = state.installer
    initial = tuple((n, p.clone()) for n, p in installer.active_weights.items())
    snapshots = iter(
        DraftWeightSnapshot.from_named_tensors(
            [(n, p + version) for n, p in initial],
            version=version,
            pin_memory=False,
        )
        for version in (1, 2)
    )
    state.poll_snapshot = lambda: next(snapshots, None)
    for version in (1, 2):
        assert state.advance("request-0") == version
        expected_kv, expected_norm = derived()
        torch.testing.assert_close(draft.model._fused_kv_weight, expected_kv)
        torch.testing.assert_close(draft.model._k_norm_weights, expected_norm)
        assert (
            draft.model._hidden_norm_weight.data_ptr()
            == draft.model.hidden_norm.weight.data_ptr()
        )
    assert state.advance("request-0") is None
    installer.reset(initial)
    state.slots.bind(installer.active_weights)
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
    slots = DraftWeightSlots.from_models(
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

    slots = DraftWeightSlots.from_models(
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


def _eagle_speculator(*, log_updates: bool = False) -> EagleSpeculator:
    draft = nn.Module()
    draft.projection = nn.Linear(2, 2)
    speculator = object.__new__(EagleSpeculator)
    speculator.model = draft
    speculator._online_target_model = nn.Module()
    speculator.capture_adapter = None
    speculator.configure_online_training(
        tuple(dict(draft.named_parameters())), log_updates=log_updates
    )
    return speculator


def test_prefill_captures_before_install_and_binds_before_closing_epoch() -> None:
    speculator = _eagle_speculator()
    installer = speculator.draft_weight_installer
    snapshot = DraftWeightSnapshot.from_named_tensors(
        [
            (name, torch.full_like(tensor, 3.0))
            for name, tensor in installer.active_weights.items()
        ],
        version=1,
        pin_memory=False,
    )
    events: list[tuple[str, int]] = []

    def close_epoch(request_id: str) -> None:
        assert request_id == "request-0"
        for name, parameter in speculator.model.named_parameters():
            assert parameter.data_ptr() == installer.active_weights[name].data_ptr()
        events.append(("close", installer.active_version))

    speculator.set_capture_adapter(
        SimpleNamespace(
            capture_prefill=lambda **kwargs: events.append(
                ("capture", installer.active_version)
            ),
            close_epoch=close_epoch,
        )
    )
    speculator.last_prefill_input_embeds = torch.zeros(1, 2)
    speculator.last_prefill_positions = torch.tensor([0])
    speculator.last_prefill_length = torch.tensor(1)
    snapshots = iter([snapshot])
    speculator.set_snapshot_source(lambda: next(snapshots, None))

    for _ in range(2):
        speculator.on_prefill_input_embeds_ready(
            num_reqs=1,
            input_batch=SimpleNamespace(
                req_ids=["request-0"], logits_indices=torch.tensor([0])
            ),
            aux_hidden_states=[torch.zeros(1, 2)],
            num_sampled=torch.tensor([1]),
            capture_enabled=True,
        )

    assert installer.active_version == 1
    assert events == [("capture", 0), ("close", 1), ("capture", 1)]


@pytest.mark.parametrize("configured", [False, True])
def test_prefill_without_snapshot_does_not_require_live_request(configured) -> None:
    if configured:
        speculator = _eagle_speculator()
        speculator.set_snapshot_source(lambda: None)
    else:
        speculator = object.__new__(EagleSpeculator)
        speculator.online_state = None
        speculator.capture_adapter = None

    speculator.on_prefill_input_embeds_ready(
        num_reqs=0,
        input_batch=SimpleNamespace(req_ids=[]),
        aux_hidden_states=None,
        num_sampled=torch.empty(0, dtype=torch.int32),
        capture_enabled=False,
    )


def test_finish_request_restores_initial_active_weights() -> None:
    speculator = _eagle_speculator()
    state = speculator.online_state
    installer = speculator.draft_weight_installer
    initial_weights = tuple(
        (name, tensor.clone()) for name, tensor in installer.active_weights.items()
    )
    snapshot = DraftWeightSnapshot.from_named_tensors(
        [
            (name, torch.full_like(tensor, 3.0))
            for name, tensor in installer.active_weights.items()
        ],
        version=1,
        pin_memory=False,
    )
    installer.stage(snapshot)
    assert installer.commit_if_ready() == 1
    state.slots.bind(installer.active_weights)

    finished: list[str] = []

    def finish_request(request_id: str) -> None:
        assert installer.active_version == 1
        finished.append(request_id)

    speculator.set_capture_adapter(SimpleNamespace(finish_request=finish_request))
    state._last_installed_round = 7

    speculator.finish_request("request-0", initial_weights)

    assert finished == ["request-0"]
    assert state._last_installed_round == 0
    assert installer.active_version == 0
    assert installer.staged_version is None
    parameters = dict(speculator.model.named_parameters())
    for name, expected in initial_weights:
        parameter = parameters[name]
        assert torch.equal(parameter, expected)
        assert parameter.data_ptr() == installer.active_weights[name].data_ptr()


@pytest.mark.parametrize("log_updates", [False, True])
def test_weight_install_logs_rounds_only_when_enabled(log_updates) -> None:
    installed: list[str] = []
    speculator = _eagle_speculator(log_updates=log_updates)
    installer = speculator.draft_weight_installer
    adapter = SimpleNamespace(completed_rounds=0, close_epoch=installed.append)
    speculator.set_capture_adapter(adapter)
    snapshots = iter(
        DraftWeightSnapshot.from_named_tensors(
            [
                (name, tensor + version)
                for name, tensor in installer.active_weights.items()
            ],
            version=version,
            pin_memory=False,
        )
        for version in (1, 2)
    )
    speculator.set_snapshot_source(lambda: next(snapshots, None))

    with patch.object(logger, "info") as log_info:
        for completed_rounds in (2, 5):
            adapter.completed_rounds = completed_rounds
            speculator.on_prefill_input_embeds_ready(
                num_reqs=1,
                input_batch=SimpleNamespace(req_ids=["request-0"]),
                aux_hidden_states=None,
                num_sampled=torch.tensor([1]),
                capture_enabled=False,
            )

    assert installed == ["request-0", "request-0"]
    if log_updates:
        assert log_info.call_args_list == [
            call(
                "Online draft installed weights: request_id=%s version=%d rounds=%d",
                "request-0",
                version,
                rounds,
            )
            for version, rounds in ((1, 2), (2, 3))
        ]
    else:
        log_info.assert_not_called()
