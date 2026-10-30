# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import json
import math
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.dflash.adapter import (
    DflashOnlineAdapter,
    DflashPolicyBridge,
    _Proposal,
    _Verification,
)
from methods.dflash.config import (
    default_draft_directory,
    resolve_draft_method,
    resolve_speculative_tokens,
)
from methods.dflash.factory import _clone_training_model
from methods.dflash.training import (
    DflashTrainingBatch,
    create_dflash_trainer,
    draft_logits,
    train_dflash_round,
)
from methods.prepare import MODELS
from methods.run import build_parser, configure

ROOT = Path(__file__).resolve().parents[2]


class _RMSNorm(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.variance_epsilon = 1e-6


class _Rotary(nn.Module):
    def __init__(self, head_dim: int, max_positions: int = 64) -> None:
        super().__init__()
        inverse = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_positions).float()
        frequencies = torch.outer(positions, inverse)
        self.register_buffer(
            "cos_sin_cache",
            torch.cat((frequencies.cos(), frequencies.sin()), dim=-1),
        )
        self.is_neox_style = True


class _Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        causal: bool = False,
        sliding_window: int | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.scaling = head_dim**-0.5
        self.qkv_proj = nn.Linear(
            hidden_size, self.q_size + 2 * self.kv_size, bias=False
        )
        self.o_proj = nn.Linear(self.q_size, hidden_size, bias=False)
        self.q_norm = _RMSNorm(head_dim)
        self.k_norm = _RMSNorm(head_dim)
        self.rotary_emb = _Rotary(head_dim)
        self.causal = causal
        self.sliding_window = sliding_window


class _NonCopyableRuntime(nn.Module):
    def __deepcopy__(self, memo):
        raise AssertionError("serving attention runtime must not be copied")


class _NonCopyableParameter(nn.Parameter):
    def __deepcopy__(self, memo):
        raise AssertionError("vLLM parameter subclasses must be cloned explicitly")


class _MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)


class _Layer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        *,
        causal: bool = False,
        sliding_window: int | None = None,
    ) -> None:
        super().__init__()
        self.self_attn = _Attention(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            causal=causal,
            sliding_window=sliding_window,
        )
        self.mlp = _MLP(hidden_size, intermediate_size)
        self.input_layernorm = _RMSNorm(hidden_size)
        self.post_attention_layernorm = _RMSNorm(hidden_size)


class _ModelConfig:
    def __init__(self, draft_vocab_size: int) -> None:
        self.rms_norm_eps = 1e-6
        self.draft_vocab_size = draft_vocab_size
        self.logit_scale = 1.0


class _DFlashModel(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int = 8,
        num_heads: int = 2,
        num_kv_heads: int = 1,
        head_dim: int = 4,
        intermediate_size: int = 16,
        num_layers: int = 2,
        draft_vocab_size: int = 11,
        aux_size: int = 12,
        layer_modes: tuple[tuple[bool, int | None], ...] = (),
    ) -> None:
        super().__init__()
        self.config = _ModelConfig(draft_vocab_size)
        self.use_aux_hidden_state = True
        self.fc = nn.Linear(aux_size, hidden_size, bias=False)
        self.hidden_norm = _RMSNorm(hidden_size)
        self.layers = nn.ModuleList(
            [
                _Layer(
                    hidden_size,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    intermediate_size,
                    causal=layer_modes[index][0] if index < len(layer_modes) else False,
                    sliding_window=layer_modes[index][1]
                    if index < len(layer_modes)
                    else None,
                )
                for index in range(num_layers)
            ]
        )
        self.norm = _RMSNorm(hidden_size)
        self.embed_tokens = nn.Embedding(32, hidden_size)
        self.rebuild_calls = 0

    def _build_fused_kv_buffers(self) -> None:
        self.rebuild_calls += 1


class _Draft(nn.Module):
    def __init__(self, model: _DFlashModel) -> None:
        super().__init__()
        self.model = model
        self.config = model.config
        self.lm_head = nn.Linear(
            model.layers[0].input_layernorm.weight.numel(),
            model.config.draft_vocab_size,
            bias=False,
        )
        self.draft_id_to_target_id = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)


def _reference_rms(x, weight, eps):
    source = x.double()
    normalized = source * torch.rsqrt(source.square().mean(dim=-1, keepdim=True) + eps)
    return normalized * weight.double()


def _reference_rope(x, cos, sin):
    cos = cos.double().unsqueeze(-2)
    sin = sin.double().unsqueeze(-2)
    first, second = torch.chunk(x.double(), 2, dim=-1)
    return torch.cat((first * cos - second * sin, second * cos + first * sin), -1)


def _reference_attention(query, key, value, scale, mask):
    queries = query.double()
    keys = key.double()
    values = value.double()
    heads = queries.shape[1]
    kv_heads = keys.shape[1]
    repeats = heads // kv_heads
    outputs = []
    for head in range(heads):
        kv_head = head // repeats
        scores = queries[:, head] @ keys[:, kv_head].transpose(0, 1) * scale
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        outputs.append(weights @ values[:, kv_head])
    return torch.stack(outputs, dim=1).float()


def _reference_logits(draft, context, context_positions, ids, query_positions):
    model = draft.model
    eps = model.config.rms_norm_eps
    context_states = _reference_rms(
        context.double() @ model.fc.weight.double().T,
        model.hidden_norm.weight,
        eps,
    )
    hidden = draft.model.embed_tokens(ids).double()
    for layer in model.layers:
        attention = layer.self_attn
        normalized = _reference_rms(hidden, layer.input_layernorm.weight, eps)
        qkv = normalized @ attention.qkv_proj.weight.double().T
        query, key, value = qkv.split(
            (attention.q_size, attention.kv_size, attention.kv_size), dim=-1
        )
        shape_q = (normalized.shape[0], attention.num_heads, attention.head_dim)
        shape_kv = (normalized.shape[0], attention.num_kv_heads, attention.head_dim)
        query = query.view(shape_q)
        key = key.view(shape_kv)
        value = value.view(shape_kv)
        query = _reference_rms(query, attention.q_norm.weight, eps)
        key = _reference_rms(key, attention.k_norm.weight, eps)
        cache = attention.rotary_emb.cos_sin_cache
        query_cos, query_sin = cache.index_select(0, query_positions).chunk(2, -1)
        context_cos, context_sin = cache.index_select(0, context_positions).chunk(2, -1)
        query = _reference_rope(query, query_cos, query_sin)

        context_qkv = context_states @ attention.qkv_proj.weight.double().T
        _, context_key, context_value = context_qkv.split(
            (attention.q_size, attention.kv_size, attention.kv_size), dim=-1
        )
        context_key = context_key.view(
            context_states.shape[0], attention.num_kv_heads, attention.head_dim
        )
        context_value = context_value.view(
            context_states.shape[0], attention.num_kv_heads, attention.head_dim
        )
        context_key = _reference_rms(context_key, attention.k_norm.weight, eps)
        context_key = _reference_rope(context_key, context_cos, context_sin)
        key = _reference_rope(key, query_cos, query_sin)

        key = torch.cat((context_key, key))
        value = torch.cat((context_value, value))
        key_positions = torch.cat((context_positions, query_positions))
        mask = None
        if attention.causal or attention.sliding_window is not None:
            rows = torch.arange(query_positions.numel())
            allowed = torch.ones(
                (query_positions.numel(), key_positions.numel()), dtype=torch.bool
            )
            if attention.sliding_window is not None:
                allowed &= key_positions.unsqueeze(0) > (
                    query_positions.unsqueeze(1) - attention.sliding_window
                )
            if attention.causal:
                allowed[:, context_positions.numel() :] &= rows.unsqueeze(
                    1
                ) >= rows.unsqueeze(0)
            mask = allowed
        attended = _reference_attention(query, key, value, attention.scaling, mask)
        hidden = hidden + attended.double().reshape(hidden.shape[0], -1) @ (
            attention.o_proj.weight.double().T
        )
        mlp_input = _reference_rms(hidden, layer.post_attention_layernorm.weight, eps)
        gate_up = mlp_input @ layer.mlp.gate_up_proj.weight.double().T
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = hidden + (F.silu(gate) * up) @ (layer.mlp.down_proj.weight.double().T)
    hidden = _reference_rms(hidden, model.norm.weight, eps)
    return (hidden @ draft.lm_head.weight.double().T).float()


def _make_batch(
    *,
    context_length: int = 4,
    depth: int = 3,
    target_count: int = 2,
    rejection_position: int | None = 1,
    draft_vocab_size: int = 11,
) -> DflashTrainingBatch:
    context_positions = torch.arange(context_length, dtype=torch.long)
    query_positions = torch.arange(
        context_length, context_length + depth + 1, dtype=torch.long
    )
    teacher = torch.softmax(torch.randn(target_count, draft_vocab_size), dim=-1)
    return DflashTrainingBatch(
        context_hidden_states=torch.randn(context_length, 12),
        context_positions=context_positions,
        query_input_ids=torch.tensor([3, 4, 5, 6], dtype=torch.long),
        query_positions=query_positions,
        teacher_probabilities=teacher,
        teacher_coverage=torch.ones(target_count),
        rejection_position=rejection_position,
    )


class DFlashTrainingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _check_forward(self, layer_modes=()):
        model = _DFlashModel(layer_modes=layer_modes)
        draft = _Draft(model)
        context = torch.randn(5, 12)
        context_positions = torch.arange(5, dtype=torch.long)
        ids = torch.tensor([3, 4, 5, 6, 7], dtype=torch.long)
        query_positions = torch.arange(5, 10, dtype=torch.long)
        actual = draft_logits(draft, context, context_positions, ids, query_positions)
        expected = _reference_logits(
            draft, context, context_positions, ids, query_positions
        )
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)

    def test_block_forward_matches_explicit_reference(self):
        self._check_forward()

    def test_causal_and_sliding_layers_match_reference(self):
        self._check_forward(((True, None), (True, 2)))

    def test_forward_reaches_every_draft_weight(self):
        model = _DFlashModel()
        draft = _Draft(model)
        batch = _make_batch()
        logits = draft_logits(
            draft,
            batch.context_hidden_states,
            batch.context_positions,
            batch.query_input_ids,
            batch.query_positions,
        )
        logits.sum().backward()
        for parameter in (
            model.fc.weight,
            model.hidden_norm.weight,
            model.layers[0].self_attn.qkv_proj.weight,
            model.layers[0].mlp.gate_up_proj.weight,
            model.norm.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_batch_validation_rejects_misaligned_inputs(self):
        valid = _make_batch()
        valid.validate(draft_vocab_size=11, num_speculative_steps=3)
        with self.assertRaisesRegex(ValueError, "query block"):
            DflashTrainingBatch(
                context_hidden_states=valid.context_hidden_states,
                context_positions=valid.context_positions,
                query_input_ids=valid.query_input_ids,
                query_positions=valid.query_positions + 1,
                teacher_probabilities=valid.teacher_probabilities,
                teacher_coverage=valid.teacher_coverage,
                rejection_position=valid.rejection_position,
            ).validate(draft_vocab_size=11, num_speculative_steps=3)
        with self.assertRaisesRegex(ValueError, "between 1 and the draft depth"):
            DflashTrainingBatch(
                context_hidden_states=valid.context_hidden_states,
                context_positions=valid.context_positions,
                query_input_ids=valid.query_input_ids,
                query_positions=valid.query_positions,
                teacher_probabilities=valid.teacher_probabilities[:1].repeat(4, 1),
                teacher_coverage=torch.ones(4),
                rejection_position=3,
            ).validate(draft_vocab_size=11, num_speculative_steps=3)
        DflashTrainingBatch(
            context_hidden_states=valid.context_hidden_states,
            context_positions=valid.context_positions,
            query_input_ids=valid.query_input_ids,
            query_positions=valid.query_positions,
            teacher_probabilities=valid.teacher_probabilities,
            teacher_coverage=valid.teacher_coverage,
            rejection_position=0,
        ).validate(draft_vocab_size=11, num_speculative_steps=3)
        with self.assertRaisesRegex(ValueError, "supervised row"):
            DflashTrainingBatch(
                context_hidden_states=valid.context_hidden_states,
                context_positions=valid.context_positions,
                query_input_ids=valid.query_input_ids,
                query_positions=valid.query_positions,
                teacher_probabilities=valid.teacher_probabilities,
                teacher_coverage=valid.teacher_coverage,
                rejection_position=2,
            ).validate(draft_vocab_size=11, num_speculative_steps=3)

    def test_train_round_updates_live_weights_and_rebuilds_buffers(self):
        model = _DFlashModel()
        draft = _Draft(model)
        batch = _make_batch()
        before = model.fc.weight.detach().clone()
        with patch.dict(
            "os.environ",
            {"OSD_FUSED_OPTIMIZER": "0", "OSD_KEEP_WEIGHTS": "0"},
            clear=False,
        ):
            trainer = create_dflash_trainer(draft, learning_rate=1e-2)
            loss = train_dflash_round(trainer, draft, batch)
        self.assertTrue(math.isfinite(loss))
        self.assertEqual(trainer.version, 1)
        self.assertEqual(model.rebuild_calls, 0)
        self.assertFalse(torch.equal(before, model.fc.weight.detach()))

    def test_trainer_freezes_shared_embedding_and_lm_head(self):
        draft = _Draft(_DFlashModel())
        with patch.dict(
            "os.environ",
            {"OSD_FUSED_OPTIMIZER": "0"},
            clear=False,
        ):
            trainer = create_dflash_trainer(draft, learning_rate=1e-2)
        names = trainer.trainable_parameter_names
        self.assertIn("model.fc.weight", names)
        self.assertNotIn("lm_head.weight", names)
        self.assertNotIn("model.embed_tokens.weight", names)
        self.assertAlmostEqual(trainer.loss_config.position_decay_rho, math.exp(-1 / 7))

    def test_train_round_preserves_the_actual_rejection_index(self):
        draft = _Draft(_DFlashModel())
        batch = _make_batch(rejection_position=0)
        captured = {}

        def loss(**kwargs):
            captured.update(kwargs)
            return kwargs["student_logits"].square().mean()

        with (
            patch.dict("os.environ", {"OSD_FUSED_OPTIMIZER": "0"}, clear=False),
            patch("methods.dflash.training.window_forward_kl_loss", side_effect=loss),
        ):
            trainer = create_dflash_trainer(draft, learning_rate=1e-2)
            train_dflash_round(trainer, draft, batch)
        self.assertEqual(
            captured["proposal_depths"].tolist(),
            [0, 1],
        )
        self.assertAlmostEqual(captured["position_decay_rho"], math.exp(-1 / 7))

    def test_training_copy_shares_runtime_but_clones_trainable_weights(self):
        draft = _Draft(_DFlashModel())
        draft.model.fc.weight = _NonCopyableParameter(
            draft.model.fc.weight.detach(), requires_grad=False
        )
        runtime = _NonCopyableRuntime()
        for layer in draft.model.layers:
            layer.self_attn.attn = runtime

        clone = _clone_training_model(draft)

        self.assertIs(clone.lm_head, draft.lm_head)
        self.assertIs(clone.model.embed_tokens, draft.model.embed_tokens)
        self.assertIs(clone.model.layers[0].self_attn.attn, runtime)
        self.assertIsNot(clone.model.fc.weight, draft.model.fc.weight)
        self.assertIs(type(clone.model.fc.weight), nn.Parameter)
        self.assertTrue(clone.model.fc.weight.requires_grad)
        torch.testing.assert_close(clone.model.fc.weight, draft.model.fc.weight)


class DflashAdapterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _adapter(self, depth=3):
        draft = _Draft(_DFlashModel(draft_vocab_size=11))
        bridge = DflashPolicyBridge.__new__(DflashPolicyBridge)
        return DflashOnlineAdapter(
            bridge,
            draft,
            num_speculative_steps=depth,
        )

    def test_rejection_row_tracks_accepted_count(self):
        adapter = self._adapter()
        proposal = _Proposal(
            request_id="req",
            context_hidden_states=torch.randn(4, 12),
            context_positions=torch.arange(4),
            query_input_ids=torch.tensor([3, 4, 5, 6]),
            query_positions=torch.arange(4, 8),
        )
        adapter._pending = proposal
        adapter._verification = _Verification(
            teacher_probabilities=torch.softmax(torch.randn(3, 11), -1),
            teacher_coverage=torch.ones(3),
            num_sampled=4,
        )
        adapter._request_id = "req"
        all_accepted = adapter._build_batch()
        self.assertIsNone(all_accepted.rejection_position)

        adapter._verification = _Verification(
            teacher_probabilities=torch.softmax(torch.randn(3, 11), -1),
            teacher_coverage=torch.ones(3),
            num_sampled=2,
        )
        rejected = adapter._build_batch()
        self.assertEqual(rejected.rejection_position, 1)
        self.assertEqual(rejected.target_count, 3)

    def test_verification_keeps_teacher_rows_for_the_full_canvas(self):
        adapter = self._adapter()
        adapter._pending = _Proposal(
            request_id="req",
            context_hidden_states=torch.randn(4, 12),
            context_positions=torch.arange(4),
            query_input_ids=torch.tensor([3, 4, 5, 6]),
            query_positions=torch.arange(4, 8),
        )
        adapter._request_id = "req"
        logits = torch.randn(4, 13)
        input_batch = types.SimpleNamespace(req_ids=["req"], num_reqs=1)
        result = types.SimpleNamespace(num_sampled=torch.tensor([2]))
        adapter.capture_verification(logits, input_batch, result)
        verification = adapter._verification
        self.assertIsNotNone(verification)
        self.assertEqual(verification.num_sampled, 2)
        self.assertEqual(tuple(verification.teacher_probabilities.shape), (3, 11))
        self.assertEqual(tuple(verification.teacher_coverage.shape), (3,))

    def test_capture_keeps_only_the_newest_context_tokens(self):
        draft = _Draft(_DFlashModel(draft_vocab_size=11))
        adapter = DflashOnlineAdapter(
            DflashPolicyBridge.__new__(DflashPolicyBridge),
            draft,
            num_speculative_steps=3,
            max_context_tokens=2,
            max_model_len=8,
            dtype=torch.float32,
        )
        speculator = types.SimpleNamespace(
            model=types.SimpleNamespace(model=draft.model),
            context_positions=torch.arange(5),
            input_buffers=types.SimpleNamespace(
                input_ids=torch.tensor([9, 10, 11, 12], dtype=torch.int32),
                positions=torch.arange(5, 9, dtype=torch.int32),
            ),
            num_query_per_req=4,
        )
        input_batch = types.SimpleNamespace(
            num_reqs=1,
            num_tokens=5,
            req_ids=["req"],
        )
        adapter.capture_proposal(
            speculator,
            input_batch,
            aux_hidden_states=[torch.randn(5, 12)],
            last_hidden_states=torch.randn(5, 8),
            num_rejected=torch.tensor([0]),
        )
        pending = adapter._pending
        self.assertIsNotNone(pending)
        self.assertEqual(tuple(pending.context_hidden_states.shape), (2, 12))
        self.assertEqual(pending.context_positions.tolist(), [3, 4])
        self.assertEqual(pending.query_input_ids.tolist(), [9, 10, 11, 12])
        self.assertEqual(pending.query_input_ids.dtype, torch.long)
        self.assertEqual(pending.query_positions.dtype, torch.long)

    def test_context_accumulates_across_proposal_rounds(self):
        draft = _Draft(_DFlashModel(draft_vocab_size=11))
        adapter = DflashOnlineAdapter(
            DflashPolicyBridge.__new__(DflashPolicyBridge),
            draft,
            num_speculative_steps=2,
            max_model_len=16,
            dtype=torch.float32,
        )

        def capture(num_tokens, positions, query_base):
            speculator = types.SimpleNamespace(
                model=types.SimpleNamespace(model=draft.model),
                context_positions=torch.tensor(positions),
                input_buffers=types.SimpleNamespace(
                    input_ids=torch.arange(query_base, query_base + 3),
                    positions=torch.arange(query_base, query_base + 3),
                ),
                num_query_per_req=3,
            )
            input_batch = types.SimpleNamespace(
                num_reqs=1,
                num_tokens=num_tokens,
                req_ids=["req"],
            )
            adapter.capture_proposal(
                speculator,
                input_batch,
                aux_hidden_states=[torch.randn(num_tokens, 12)],
                last_hidden_states=torch.randn(num_tokens, 8),
                num_rejected=torch.tensor([0]),
            )
            return adapter._pending

        first = capture(3, [0, 1, 2], 3)
        self.assertEqual(tuple(first.context_hidden_states.shape), (3, 12))
        adapter._pending = None
        second = capture(2, [3, 4], 5)
        self.assertEqual(tuple(second.context_hidden_states.shape), (5, 12))
        self.assertEqual(second.context_positions.tolist(), [0, 1, 2, 3, 4])

    def test_context_cache_drops_oldest_rows_and_compacts(self):
        from methods.dflash.adapter import _ContextCache

        cache = _ContextCache(3, 2, dtype=torch.float32, device=torch.device("cpu"))
        cache.append(
            torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
            torch.tensor([0, 1, 2]),
        )
        cache.append(torch.tensor([[3.0, 3.0]]), torch.tensor([3]))
        self.assertEqual(cache.positions.tolist(), [1, 2, 3])
        self.assertEqual(cache.features[:, 0].tolist(), [1.0, 2.0, 3.0])

    def test_unexpected_verification_rows_skip_the_round(self):
        adapter = self._adapter()
        adapter._pending = _Proposal(
            request_id="req",
            context_hidden_states=torch.randn(4, 12),
            context_positions=torch.arange(4),
            query_input_ids=torch.tensor([3, 4, 5, 6]),
            query_positions=torch.arange(4, 8),
        )
        adapter._request_id = "req"
        input_batch = types.SimpleNamespace(req_ids=["req"], num_reqs=1)
        result = types.SimpleNamespace(num_sampled=torch.tensor([2]))
        adapter.capture_verification(torch.randn(3, 13), input_batch, result)
        self.assertIsNone(adapter._pending)
        self.assertIsNone(adapter._verification)


class DflashBridgeTest(unittest.TestCase):
    def test_stride_controls_updates_and_reset_restores_weights(self):
        trainer = MagicMock()
        trainer.version = 1
        model = object()
        batch = object()
        with (
            patch.dict(
                "os.environ",
                {"OSD_UPDATE_STRIDE": "2", "OSD_KEEP_WEIGHTS": "0"},
            ),
            patch(
                "methods.dflash.adapter.train_dflash_round",
                return_value=0.5,
            ) as train_round,
            patch("methods.dflash.adapter.rebuild_fused_context_buffers") as rebuild,
        ):
            bridge = DflashPolicyBridge(trainer, model, method="tts")
            self.assertEqual(bridge.observe("req", 0, batch), 0)
            self.assertEqual(bridge.observe("req", 1, batch), 1)
            train_round.assert_called_once()
            self.assertEqual(bridge.stats.updates, 1)
            self.assertEqual(bridge.stats.weight_installs, 1)
            bridge.reset_request("req")
            trainer.reset.assert_called_once()
            rebuild.assert_called_once_with(model)

    def test_random_sampling_gate_can_skip_updates(self):
        with (
            patch.dict(
                "os.environ",
                {"OSD_PROBABILITY": "0", "OSD_UPDATE_STRIDE": "1"},
            ),
            patch("methods.dflash.adapter.train_dflash_round") as train_round,
            patch("methods.dflash.adapter.rebuild_fused_context_buffers"),
        ):
            bridge = DflashPolicyBridge(MagicMock(), object(), method="random_sampling")
            self.assertEqual(bridge.observe("req", 0, object()), 0)
            train_round.assert_not_called()
            self.assertEqual(bridge.stats.updates, 0)


class DflashLaunchTest(unittest.TestCase):
    def _command(self, *args):
        parsed = build_parser().parse_args((*args, "--dry-run"))
        return configure(parsed)

    def test_dflash_selects_native_speculation_without_hooks(self):
        command, env = self._command("dflash")
        self.assertEqual(command[command.index("--spec-method") + 1], "dflash")
        self.assertEqual(command[command.index("--spec-tokens") + 1], "15")
        self.assertTrue(command[command.index("--spec-model") + 1].endswith("-dflash"))
        self.assertEqual(env["OSD_METHOD"], "dflash")
        self.assertEqual(env["OSD_DRAFT_METHOD"], "dflash")
        self.assertEqual(env["OSD_DFLASH_MAX_CONTEXT_TOKENS"], "0")

    def test_tts_and_random_sampling_accept_dflash_drafts(self):
        for method in ("tts", "random_sampling"):
            with self.subTest(method=method):
                command, env = self._command(method, "--draft-method", "dflash")
                self.assertEqual(command[command.index("--spec-method") + 1], "dflash")
                self.assertEqual(command[command.index("--spec-tokens") + 1], "15")
                self.assertEqual(env["OSD_DRAFT_METHOD"], "dflash")
                self.assertEqual(env["OSD_METHOD"], method)

    def test_tts_keeps_eagle3_defaults(self):
        command, env = self._command("tts")
        self.assertEqual(command[command.index("--spec-method") + 1], "eagle3")
        self.assertEqual(command[command.index("--spec-tokens") + 1], "7")
        self.assertEqual(env["OSD_DRAFT_METHOD"], "eagle3")
        self.assertTrue(command[command.index("--spec-model") + 1].endswith("-eagle3"))

    def test_explicit_tokens_and_draft_override_defaults(self):
        command, _ = self._command(
            "tts", "--draft-method", "dflash", "--spec-tokens", "9", "--draft", "/tmp/x"
        )
        self.assertEqual(command[command.index("--spec-tokens") + 1], "9")
        self.assertEqual(command[command.index("--spec-model") + 1], "/tmp/x")

    def test_unsupported_combinations_are_rejected(self):
        for method in ("eagle", "ospec"):
            with (
                self.subTest(method=method),
                self.assertRaisesRegex(ValueError, "eagle3"),
            ):
                self._command(method, "--draft-method", "dflash")
        with self.assertRaisesRegex(ValueError, "requires a dflash"):
            self._command("dflash", "--draft-method", "eagle3")

    def test_dflash_tts_rejects_chunked_prefill(self):
        with self.assertRaisesRegex(ValueError, "chunked prefill"):
            self._command("tts", "--draft-method", "dflash", "--enable-chunked-prefill")

    def test_dflash_tts_requires_unbounded_prefill_budget(self):
        with self.assertRaisesRegex(ValueError, "max-num-batched-tokens"):
            self._command(
                "tts",
                "--draft-method",
                "dflash",
                "--max-num-batched-tokens",
                "1024",
            )

    def test_nonnegative_dflash_context_limit(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            self._command(
                "tts", "--draft-method", "dflash", "--dflash-max-context-tokens", "-1"
            )

    def test_resolvers_and_default_directory(self):
        self.assertEqual(resolve_draft_method("dflash", None), "dflash")
        self.assertEqual(resolve_draft_method("tts", None), "eagle3")
        self.assertEqual(resolve_draft_method("tts", "dflash"), "dflash")
        self.assertEqual(resolve_draft_method("tts", "mtp"), "mtp")
        self.assertEqual(resolve_speculative_tokens("dflash", None), 15)
        self.assertEqual(resolve_speculative_tokens("eagle3", None), 7)
        self.assertEqual(resolve_speculative_tokens("mtp", None), 3)
        self.assertEqual(
            default_draft_directory(Path("/root"), "4b", "dflash"),
            Path("/root/models/qwen3-4b-dflash"),
        )

    def test_mtp_uses_target_embedded_weights(self):
        command, env = self._command("mtp")
        self.assertEqual(command[command.index("--spec-method") + 1], "mtp")
        self.assertEqual(command[command.index("--spec-tokens") + 1], "3")
        self.assertNotIn("--spec-model", command)
        self.assertEqual(env["OSD_DRAFT_MODEL"], "")
        self.assertEqual(env["VLLM_ONLINE_DRAFT_TRAIN"], "0")

        command, env = self._command(
            "online_draft", "--draft-method", "mtp", "--online-draft-loss", "entropy"
        )
        config = json.loads(command[command.index("--additional-config") + 1])
        self.assertEqual(config["online_draft"]["loss_mode"], "entropy")
        self.assertEqual(env["VLLM_ONLINE_DRAFT_TRAIN"], "1")

    def test_prepare_includes_dflash_drafts(self):
        drafts = {spec.name: spec for spec in MODELS if spec.draft}
        self.assertEqual(drafts["qwen3-4b-dflash"].repo_id, "z-lab/Qwen3-4B-DFlash-b16")
        self.assertEqual(drafts["qwen3-8b-dflash"].repo_id, "z-lab/Qwen3-8B-DFlash-b16")
        self.assertTrue(drafts["qwen3-8b-dflash"].revision)


class DflashBootstrapTest(unittest.TestCase):
    def _install(self, method: str, draft_method: str):
        calls = []
        modules = {}
        for name, label in (
            ("methods.dflash.factory", "dflash"),
            ("methods.utils.gpu_factory", "eagle3"),
            ("methods.utils.cpu_stats", "cpu"),
        ):
            module = types.ModuleType(name)
            module.install = lambda label=label: calls.append(label)
            modules[name] = module
        path = ROOT / "methods" / "utils" / "bootstrap" / "sitecustomize.py"
        spec = importlib.util.spec_from_file_location("_dflash_bootstrap", path)
        module = importlib.util.module_from_spec(spec)
        with (
            patch.dict(sys.modules, modules),
            patch.dict(
                "os.environ",
                {"OSD_METHOD": method, "OSD_DRAFT_METHOD": draft_method},
            ),
        ):
            spec.loader.exec_module(module)
        return calls

    def test_dflash_tts_uses_the_dflash_factory(self):
        self.assertEqual(self._install("tts", "dflash"), ["dflash"])
        self.assertEqual(self._install("random_sampling", "dflash"), ["dflash"])

    def test_eagle3_tts_and_cpu_entries_keep_their_factories(self):
        self.assertEqual(self._install("tts", "eagle3"), ["eagle3"])
        self.assertEqual(self._install("tts", "mtp"), ["eagle3"])
        self.assertEqual(self._install("online_draft", "eagle3"), ["cpu"])
        self.assertEqual(self._install("online_draft", "dflash"), ["cpu"])
        self.assertEqual(self._install("dflash", "dflash"), [])


if __name__ == "__main__":
    unittest.main()
