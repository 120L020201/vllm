# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.models import supports_multimodal_embeddings
from vllm.model_executor.models.exaone4_5_mtp import Exaone4_5_MTP
from vllm.model_executor.models.llama4_eagle import EagleLlama4ForCausalLM
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.model_executor.models.mistral_eagle import EagleMistralForCausalLM
from vllm.model_executor.models.mistral_large_3_eagle import (
    EagleMistralLarge3ForCausalLM,
)
from vllm.v1.attention.backends import flash_attn as flash_attn_module
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode import speculator as base_spec_module
from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as spec_module
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator import (
    MultiModuleMTPSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


class _TestSpeculator(AutoRegressiveSpeculator):
    def load_draft_model(self, target_model, target_attn_layer_names):
        return self.test_draft_model


class _DraftModel(torch.nn.Module):
    def __init__(self, output: torch.Tensor | tuple[torch.Tensor, torch.Tensor]):
        super().__init__()
        self.output = output

    def forward(self, **kwargs):
        return self.output


class _MultimodalDraftModel(torch.nn.Module):
    supports_multimodal_embeddings = True

    def embed_input_ids(
        self,
        input_ids,
        multimodal_embeddings=None,
        *,
        is_multimodal=None,
    ):
        raise AssertionError("embed_input_ids should not be called during loading")


class _TextOnlyDraftModel(torch.nn.Module):
    def embed_input_ids(
        self,
        input_ids,
        multimodal_embeddings=None,
        *,
        is_multimodal=None,
    ):
        raise AssertionError("embed_input_ids should not be called during loading")


def _mock_base_model_load(monkeypatch):
    monkeypatch.setattr(
        base_spec_module,
        "get_layers_from_vllm_config",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        DraftModelSpeculator,
        "_validate_local_argmax_reduction",
        lambda self: None,
    )


def _make_speculator(
    monkeypatch,
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> _TestSpeculator:
    monkeypatch.setattr(
        spec_module,
        "set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )

    speculator = object.__new__(_TestSpeculator)
    speculator.explicit_input_embeds = False
    speculator.supports_mm_inputs = False
    speculator.vllm_config = None
    speculator.input_buffers = SimpleNamespace(
        input_ids=torch.arange(4),
        positions=torch.arange(4),
    )
    speculator.hidden_states = torch.zeros(4, 3)
    speculator.model = _DraftModel(output)
    return speculator


def test_mm_support_configured_after_model_load(monkeypatch):
    target_model_config = object()
    draft_model_config = object()
    vllm_config = SimpleNamespace(model_config=target_model_config)
    draft_model = _MultimodalDraftModel()

    def init_base(speculator, vllm_config, device):
        speculator.vllm_config = vllm_config
        speculator.device = device
        speculator.max_num_tokens = 4
        speculator.max_num_reqs = 2
        speculator.hidden_size = 3
        speculator.dtype = torch.float32
        speculator.draft_model_config = draft_model_config
        speculator.supports_mm_inputs = False

    checked_configs = []

    def supports_multimodal_inputs(model_config):
        checked_configs.append(model_config)
        return True

    monkeypatch.setattr(DraftModelSpeculator, "__init__", init_base)
    _mock_base_model_load(monkeypatch)
    monkeypatch.setattr(
        base_spec_module.MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        supports_multimodal_inputs,
    )

    speculator = _TestSpeculator(vllm_config, torch.device("cpu"))

    assert checked_configs == []
    assert not speculator.supports_mm_inputs
    assert speculator.inputs_embeds is None

    speculator.test_draft_model = draft_model
    speculator.load_model(torch.nn.Module())

    assert checked_configs == [target_model_config]
    assert speculator.supports_mm_inputs
    assert speculator.inputs_embeds is not None
    assert speculator.inputs_embeds.shape == (4, 3)


def test_load_model_keeps_mm_support_for_capable_drafter(monkeypatch):
    speculator = object.__new__(_TestSpeculator)
    speculator.explicit_input_embeds = False
    speculator.supports_mm_inputs = False
    speculator.inputs_embeds = None
    speculator.vllm_config = SimpleNamespace(model_config=object())
    speculator.max_num_tokens = 4
    speculator.hidden_size = 3
    speculator.dtype = torch.float32
    speculator.device = torch.device("cpu")
    draft_model = _MultimodalDraftModel()
    speculator.test_draft_model = draft_model
    _mock_base_model_load(monkeypatch)
    monkeypatch.setattr(
        base_spec_module.MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        lambda model_config: True,
    )

    speculator.load_model(torch.nn.Module())

    assert speculator.supports_mm_inputs
    assert speculator.inputs_embeds is not None


def test_load_model_disables_mm_support_for_text_only_drafter(monkeypatch):
    speculator = object.__new__(_TestSpeculator)
    speculator.explicit_input_embeds = False
    speculator.supports_mm_inputs = False
    speculator.inputs_embeds = None
    speculator.vllm_config = SimpleNamespace(model_config=object())
    draft_model = _TextOnlyDraftModel()
    speculator.test_draft_model = draft_model
    warning_messages = []
    _mock_base_model_load(monkeypatch)
    monkeypatch.setattr(
        base_spec_module.MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        lambda model_config: True,
    )
    monkeypatch.setattr(
        base_spec_module.logger,
        "warning_once",
        lambda message, *args: warning_messages.append(message % args),
    )

    speculator.load_model(torch.nn.Module())

    assert not speculator.supports_mm_inputs
    assert warning_messages == [
        "Draft model _TextOnlyDraftModel does not support external multimodal "
        "embeddings. Embeddings from the target model will not be passed to the "
        "drafter; using text-only draft inputs instead."
    ]


def test_multi_module_mm_support_configured_after_model_load(monkeypatch):
    speculator = object.__new__(MultiModuleMTPSpeculator)
    speculator.supports_mm_inputs = False
    speculator.inputs_embeds = None
    speculator.cached_draft_input_embeds = None
    speculator.vllm_config = SimpleNamespace(model_config=object())
    speculator.max_num_tokens = 4
    speculator.max_num_reqs = 2
    speculator.num_speculative_steps = 3
    speculator.hidden_size = 3
    speculator.dtype = torch.float32
    speculator.device = torch.device("cpu")
    draft_model = _MultimodalDraftModel()
    _mock_base_model_load(monkeypatch)
    monkeypatch.setattr(
        MultiModuleMTPSpeculator,
        "load_draft_model",
        lambda self, target_model, target_attn_layer_names: draft_model,
    )
    monkeypatch.setattr(
        base_spec_module.MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        lambda model_config: True,
    )

    speculator.load_model(torch.nn.Module())

    assert speculator.supports_mm_inputs
    assert speculator.inputs_embeds is not None
    assert speculator.inputs_embeds.shape == (4, 3)
    assert speculator.cached_draft_input_embeds is not None
    assert speculator.cached_draft_input_embeds.shape == (2, 2, 3)


@pytest.mark.parametrize(
    ("model_cls", "expected"),
    [
        (EagleLlama4ForCausalLM, True),
        (EagleMistralForCausalLM, True),
        (EagleMistralLarge3ForCausalLM, True),
        (Exaone4_5_MTP, True),
        (Eagle3LlamaForCausalLM, False),
    ],
)
def test_draft_model_multimodal_embedding_capability(model_cls, expected):
    assert supports_multimodal_embeddings(model_cls) is expected


def test_run_model_unpacks_tuple_return_for_mtp(monkeypatch):
    logits_hidden = torch.full((4, 3), 1.0)
    feedback_hidden = torch.full((4, 3), 2.0)
    speculator = _make_speculator(monkeypatch, (logits_hidden, feedback_hidden))

    actual_logits_hidden, actual_feedback_hidden = speculator._run_model(
        4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert actual_logits_hidden is logits_hidden
    assert actual_feedback_hidden is feedback_hidden


def test_run_model_reuses_tensor_return_for_mtp(monkeypatch):
    hidden = torch.full((4, 3), 1.0)
    speculator = _make_speculator(monkeypatch, hidden)

    actual_logits_hidden, actual_feedback_hidden = speculator._run_model(
        4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert actual_logits_hidden is hidden
    assert actual_feedback_hidden is hidden


@pytest.mark.parametrize(
    (
        "method_name",
        "cg_mode",
        "expected_eager_calls",
        "expected_graph_replays",
    ),
    [
        ("_multi_step_decode", CUDAGraphMode.NONE, 3, 0),
        ("_multi_step_decode", CUDAGraphMode.FULL, 0, 3),
        ("_fused_multi_step_decode", CUDAGraphMode.NONE, 3, 0),
        ("_fused_multi_step_decode", CUDAGraphMode.FULL, 0, 1),
    ],
)
def test_multi_step_decode_replays_captured_graph_as_expected(
    method_name,
    cg_mode,
    expected_eager_calls,
    expected_graph_replays,
):
    speculator = object.__new__(_TestSpeculator)
    speculator.explicit_input_embeds = False
    speculator.num_speculative_steps = 4
    speculator.current_draft_step = torch.tensor(0)
    speculator.input_buffers = SimpleNamespace(
        positions=torch.arange(2),
        query_start_loc=torch.arange(3),
    )
    speculator.idx_mapping = torch.arange(2)
    generate_draft = Mock()
    speculator._generate_draft = generate_draft
    run_fullgraph = Mock()
    speculator.decode_cudagraph_manager = SimpleNamespace(run_fullgraph=run_fullgraph)
    batch_desc = BatchExecutionDescriptor(
        cg_mode=cg_mode,
        num_tokens=2,
        num_reqs=2,
    )

    getattr(speculator, method_name)(
        num_reqs=2,
        skip_attn=True,
        batch_desc=batch_desc,
        seq_lens_cpu_upper_bound=None,
        num_tokens_across_dp=None,
    )

    assert generate_draft.call_count == expected_eager_calls
    assert run_fullgraph.call_count == expected_graph_replays


def test_update_draft_decode_metadata_updates_fa3_scheduler_metadata(
    monkeypatch,
):
    builder = object.__new__(flash_attn_module.FlashAttentionMetadataBuilder)
    builder.aot_schedule = True
    builder.use_full_cuda_graph = True
    builder.scheduler_metadata = torch.zeros(8, dtype=torch.int32)
    builder.cache_config = SimpleNamespace(cache_dtype="bfloat16")
    builder.kv_cache_dtype = torch.bfloat16
    builder.num_heads_q = 2
    builder.num_heads_kv = 1
    builder.headdim = 128
    builder.block_size = 16
    builder.dcp_world_size = 1
    builder.dcp_rank = 0
    builder.cp_kv_cache_interleave_size = 1
    builder.aot_sliding_window = None

    expected = torch.tensor([7, 8, 9], dtype=torch.int32)

    def fake_get_scheduler_metadata(**kwargs):
        return expected

    monkeypatch.setattr(builder, "_get_scheduler_metadata", fake_get_scheduler_metadata)

    metadata = FlashAttentionMetadata(
        num_actual_tokens=3,
        max_query_len=2,
        query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
        max_seq_len=8,
        seq_lens=torch.tensor([5, 6], dtype=torch.int32),
        block_table=torch.zeros((2, 1), dtype=torch.int32),
        slot_mapping=torch.zeros(3, dtype=torch.int32),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        max_dcp_context_kv_len=None,
        dcp_context_kv_lens=None,
        num_decode_reqs=2,
        num_prefill_reqs=0,
        num_decode_tokens=3,
        num_prefill_tokens=0,
        scheduler_metadata=torch.tensor([-1, -1, -1], dtype=torch.int32),
        prefix_scheduler_metadata=None,
        max_num_splits=4,
        causal=True,
        mm_prefix_query_range_tensor=None,
        rswa_prefix_lens=None,
        rswa_window=None,
        rswa_window_tensor=None,
    )

    builder.update_draft_decode_metadata(metadata)

    assert torch.equal(metadata.scheduler_metadata, expected)
    assert torch.equal(builder.scheduler_metadata[:3], expected)


def test_update_draft_decode_metadata_skips_without_scheduler_metadata(monkeypatch):
    builder = object.__new__(flash_attn_module.FlashAttentionMetadataBuilder)
    builder.aot_schedule = True
    builder.use_full_cuda_graph = True
    builder.scheduler_metadata = torch.zeros(4, dtype=torch.int32)

    called = False

    def fake_get_scheduler_metadata(**kwargs):
        nonlocal called
        called = True
        return torch.tensor([1], dtype=torch.int32)

    monkeypatch.setattr(builder, "_get_scheduler_metadata", fake_get_scheduler_metadata)

    metadata = FlashAttentionMetadata(
        num_actual_tokens=1,
        max_query_len=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        max_seq_len=1,
        seq_lens=torch.tensor([1], dtype=torch.int32),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        slot_mapping=torch.zeros(1, dtype=torch.int32),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        max_dcp_context_kv_len=None,
        dcp_context_kv_lens=None,
        num_decode_reqs=1,
        num_prefill_reqs=0,
        num_decode_tokens=1,
        num_prefill_tokens=0,
        scheduler_metadata=None,
        prefix_scheduler_metadata=None,
        max_num_splits=1,
        causal=True,
        mm_prefix_query_range_tensor=None,
        rswa_prefix_lens=None,
        rswa_window=None,
        rswa_window_tensor=None,
    )

    builder.update_draft_decode_metadata(metadata)

    assert not called
    assert metadata.scheduler_metadata is None


def _decode_capture_speculator(monkeypatch, device: str = "cpu"):
    """Draft decode stub whose forward updates its input hidden in place."""
    hidden_size = 4
    speculator = object.__new__(_TestSpeculator)
    speculator.explicit_input_embeds = True
    speculator.hidden_size = hidden_size
    speculator.num_speculative_steps = 4
    speculator.dtype = torch.float32
    speculator.device = torch.device(device)
    speculator._draft_capture_input_embeds = None
    speculator._draft_capture_input_hidden_states = None
    speculator._ensure_draft_capture_buffers(1)
    speculator.hidden_states = torch.zeros(2, hidden_size, device=device)
    speculator.idx_mapping = torch.zeros(1, dtype=torch.long, device=device)
    speculator.input_buffers = SimpleNamespace(
        positions=torch.tensor([5], device=device)
    )
    speculator._prepare_eplb_forward = lambda num_tokens: None
    speculator.current_draft_step = torch.tensor(0, device=device)
    speculator.temperature = speculator.seeds = speculator.draft_logits = None
    speculator.draft_tokens = torch.zeros(1, 4, dtype=torch.long, device=device)
    speculator.max_model_len = 64
    speculator.sample_draft = lambda hidden, *args: hidden[:, 0].long()

    def run_model(*args, **kwargs):
        output = speculator.hidden_states[:1] + 1.0
        speculator.last_input_embeds = -output
        # Fused add-RMSNorm updates the first layer's residual in place.
        speculator.hidden_states[:1].add_(100.0)
        return output, output

    def update_draft_inputs(
        draft_tokens, step, hidden, tokens, next_hidden, *args, **kwargs
    ):
        next_hidden[:1].copy_(hidden[:1])

    speculator._run_model = run_model
    monkeypatch.setattr(spec_module, "update_draft_inputs", update_draft_inputs)
    return speculator


def test_draft_decode_captures_the_hidden_state_each_step_consumes(monkeypatch):
    """Online training replays decode step j from its inputs, so the captured
    hidden must be the one the step read, not its output or a buffer the
    forward has since updated in place."""
    speculator = _decode_capture_speculator(monkeypatch)
    # _prefill leaves the anchor's output hidden (h4) as the first decode input.
    speculator.hidden_states[0] = 4.0

    for step in range(1, speculator.num_speculative_steps):
        speculator.current_draft_step.fill_(step)
        speculator._generate_draft(1, 1, None, None, None)
    speculator._draft_capture_length = speculator.num_speculative_steps - 1

    snapshot = speculator.get_proposal_snapshot()
    assert snapshot.draft_input_hidden_states[:, 0].tolist() == [4.0, 5.0, 6.0]
    assert snapshot.draft_token_input_embeds[:, 0].tolist() == [-5.0, -6.0, -7.0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graphs")
def test_one_step_graph_replayed_per_step_captures_every_step(monkeypatch):
    """Backends without fused decode replay one step graph per draft step;
    each replay must record that step's rows, not the rows of the capture."""
    speculator = _decode_capture_speculator(monkeypatch, "cuda")

    def generate():
        speculator._generate_draft(1, 1, None, None, None)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        generate()
    graph = torch.cuda.CUDAGraph()
    # Capture runs with the step counter at 0, as in speculator capture.
    speculator.current_draft_step.fill_(0)
    with torch.cuda.graph(graph, stream=stream):
        generate()
    torch.cuda.current_stream().wait_stream(stream)

    speculator._draft_capture_input_hidden_states.zero_()
    speculator._draft_capture_input_embeds.zero_()
    speculator.hidden_states[0] = 4.0
    for step in range(1, speculator.num_speculative_steps):
        speculator.current_draft_step.fill_(step)
        graph.replay()
    speculator._draft_capture_length = speculator.num_speculative_steps - 1

    snapshot = speculator.get_proposal_snapshot()
    assert snapshot.draft_input_hidden_states[:, 0].tolist() == [4.0, 5.0, 6.0]
    assert snapshot.draft_token_input_embeds[:, 0].tolist() == [-5.0, -6.0, -7.0]


class _FakeGraphManager:
    def __init__(self, vllm_config, device, cudagraph_mode, *args, **kwargs):
        self.mode = cudagraph_mode


@pytest.mark.parametrize(
    ("online", "expected_slots", "expected_prefill_mode"),
    [
        (False, 1, CUDAGraphMode.FULL_AND_PIECEWISE),
        (True, 2, CUDAGraphMode.FULL_DECODE_ONLY),
    ],
)
def test_online_draft_gets_full_graphs_for_each_weight_slot(
    monkeypatch, online, expected_slots, expected_prefill_mode
):
    """PIECEWISE graphs live in the compiled model and keep one slot's weights."""
    monkeypatch.setattr(spec_module, "SpeculatorCudaGraphManager", _FakeGraphManager)
    speculator = object.__new__(_TestSpeculator)
    speculator.vllm_config = speculator.device = None
    speculator.num_speculative_steps = 3
    speculator.explicit_input_embeds = online
    speculator.num_weight_slots = 2 if online else 1

    speculator.init_cudagraph_manager(CUDAGraphMode.FULL_AND_PIECEWISE)

    assert len(speculator.cudagraph_managers) == expected_slots
    managers = {id(m) for pair in speculator.cudagraph_managers for m in pair}
    assert len(managers) == 2 * expected_slots
    for prefill, decode in speculator.cudagraph_managers:
        assert prefill.mode == expected_prefill_mode
        assert decode.mode == CUDAGraphMode.FULL_DECODE_ONLY
    assert (
        speculator.prefill_cudagraph_manager,
        speculator.decode_cudagraph_manager,
    ) == speculator.cudagraph_managers[0]


def test_online_draft_captures_each_slot_with_its_weights_bound():
    """A graph replays the parameter pointers bound while it was captured."""
    from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator

    bound: list[str] = []
    captured: list[tuple[str, object]] = []
    speculator = object.__new__(EagleSpeculator)
    speculator.online_state = SimpleNamespace(
        slots=SimpleNamespace(slots=("slot-0", "slot-1"), bind=bound.append),
        installer=SimpleNamespace(active_slot=1),
    )
    speculator.cudagraph_managers = (
        ("prefill-0", "decode-0"),
        ("prefill-1", "decode-1"),
    )
    speculator._capture_slot = lambda: captured.append(
        (bound[-1], speculator.prefill_cudagraph_manager)
    )

    speculator.capture()

    assert captured == [("slot-0", "prefill-0"), ("slot-1", "prefill-1")]
    assert bound[-1] == "slot-1"
    assert speculator.prefill_cudagraph_manager == "prefill-1"
    assert speculator.decode_cudagraph_manager == "decode-1"
