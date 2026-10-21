# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import torch
from online_draft.models.cpu_attention import (
    context_attention,
    load_attention_backward,
)
from online_draft.models.qwen3_dflash import (
    DFlashAttention,
    Qwen3DFlashConfig,
    Qwen3DFlashForCausalLM,
)
from online_draft.training.dflash.window import (
    DFlashBatch,
    append_context,
    dflash_kl_loss,
    forward_dflash_window,
    position_weights,
    train_dflash_window,
)
from online_draft.training.trainer import DraftTrainer, TrainerConfig
from torch.nn import functional as F


def tiny_model(
    attention_blocks=8, grouped_query_attention=False, selective_last_layer=False
):
    torch.manual_seed(17)
    return Qwen3DFlashForCausalLM(
        Qwen3DFlashConfig(
            8,
            16,
            2,
            1,
            4,
            2,
            23,
            2,
            attention_blocks_per_chunk=attention_blocks,
            grouped_query_attention=grouped_query_attention,
            selective_last_layer=selective_last_layer,
        )
    )


def make_batch(anchor, accepted, *, prompt=False):
    count = min(accepted + 1, 3)
    return DFlashBatch(
        input_ids=torch.tensor([2, 22, 22, 22]),
        positions=torch.arange(anchor, anchor + 4),
        prompt_fc=torch.randn(anchor, 8) if prompt else None,
        confirmed_fc=torch.randn(accepted + 1, 8),
        teacher_probabilities=torch.randn(count, 23).softmax(-1),
        rejection_position=accepted if accepted < 3 else None,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("additive_mask", [False, True])
@pytest.mark.parametrize(
    "context_length,query_length", [(0, 7), (509, 140), (1031, 261)]
)
def test_context_attention_preserves_query_gradients_across_tile_boundaries(
    dtype, additive_mask, context_length, query_length
):
    """Detached history still affects dQ; only the trainable suffix needs dK/dV."""
    load_attention_backward()
    torch.manual_seed(17)
    query = torch.randn(2, 16, query_length, dtype=dtype).transpose(1, 2)
    query.requires_grad_()
    key, value = (
        torch.randn(2, 17, 16, dtype=dtype, requires_grad=True) for _ in range(2)
    )
    context = tuple(torch.randn(2, context_length, 16, dtype=dtype) for _ in range(2))
    mask = torch.rand(query_length, context_length + 17) > 0.2
    mask[0] = False
    if additive_mask:
        mask = torch.randn(mask.shape).masked_fill_(~mask, float("-inf"))
    output = context_attention(query, *context, key, value, mask)
    native = F.scaled_dot_product_attention(
        query.contiguous().unsqueeze(0),
        torch.cat((context[0], key), 1).unsqueeze(0),
        torch.cat((context[1], value), 1).unsqueeze(0),
        attn_mask=mask,
    ).squeeze(0)
    torch.testing.assert_close(output, native, rtol=0, atol=0)
    reference_inputs = tuple(
        tensor.detach().float().requires_grad_() for tensor in (query, key, value)
    )
    ref_q, ref_k, ref_v = reference_inputs
    reference = F.scaled_dot_product_attention(
        ref_q.unsqueeze(0),
        torch.cat((context[0].float(), ref_k), 1).unsqueeze(0),
        torch.cat((context[1].float(), ref_v), 1).unsqueeze(0),
        attn_mask=mask,
    ).squeeze(0)
    grad_output = torch.randn_like(output)
    actual = torch.autograd.grad(output, (query, key, value), grad_output)
    expected = torch.autograd.grad(reference, reference_inputs, grad_output.float())
    for result, target in zip(actual, expected, strict=True):
        assert torch.isfinite(result).all()
        if dtype == torch.float32:
            torch.testing.assert_close(result, target, rtol=2e-5, atol=2e-6)
        else:
            relative_error = (result.float() - target).norm() / target.norm()
            assert relative_error < 0.008
    torch.testing.assert_close(actual[0][:, 0], torch.zeros_like(actual[0][:, 0]))


@pytest.mark.parametrize("kv_heads", [1, 2])
@pytest.mark.parametrize("repeats", [2, 4])
@pytest.mark.parametrize("grouped", [False, True])
def test_attention_matches_explicit_full_block_reference(kv_heads, repeats, grouped):
    """Q/K normalization, absolute RoPE and full-block GQA match direct math."""
    hidden_size = kv_heads * repeats * 4
    attn = DFlashAttention(
        Qwen3DFlashConfig(
            hidden_size,
            32,
            kv_heads * repeats,
            kv_heads,
            4,
            1,
            23,
            2,
            grouped_query_attention=grouped,
        )
    )
    hidden = torch.randn(4, hidden_size)
    positions = torch.arange(7, 11)
    context = tuple(torch.randn(kv_heads, 7, 4, requires_grad=True) for _ in range(2))
    actual = attn(hidden, positions, context)
    q, k, v = F.linear(hidden, attn.qkv_proj.weight).split(
        (hidden_size, kv_heads * 4, kv_heads * 4), -1
    )

    def norm_rope(x, norm):
        x = x.reshape(4, -1, 4).transpose(0, 1)
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        x = x * norm.weight
        angles = positions[:, None] * torch.tensor([1.0, 0.001])
        first, second = x.chunk(2, -1)
        return torch.cat(
            (
                first * angles.cos() - second * angles.sin(),
                second * angles.cos() + first * angles.sin(),
            ),
            -1,
        )

    q = norm_rope(q, attn.q_norm)
    k = torch.cat((context[0], norm_rope(k, attn.k_norm)), 1).repeat_interleave(
        repeats, 0
    )
    v = torch.cat((context[1], v.reshape(4, kv_heads, 4).transpose(0, 1)), 1)
    v = v.repeat_interleave(repeats, 0)
    result = (q @ k.transpose(-1, -2) / 2).softmax(-1) @ v
    expected = F.linear(
        result.transpose(0, 1).reshape(4, hidden_size), attn.o_proj.weight
    )
    torch.testing.assert_close(actual, expected)
    parameters = (*attn.parameters(), *context)
    actual_grads = torch.autograd.grad(actual.square().sum(), parameters)
    expected_grads = torch.autograd.grad(expected.square().sum(), parameters)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("accepted", [0, 1, 3])
def test_bootstrap_freezes_context_parameters_and_preserves_supervision(accepted):
    model = tiny_model()
    trainer = DraftTrainer(
        model, TrainerConfig(learning_rate=1e-3, check_gradients=False)
    )
    batch = make_batch(5, accepted, prompt=True)
    original_fc = model.model.fc.weight.detach().clone()
    original_embed = model.model.embed_tokens.weight.detach().clone()
    original_head = model.lm_head.weight.detach().clone()

    cache, result = train_dflash_window(trainer, (batch,), None)

    assert result.target_count == min(accepted + 1, 3)
    assert result.model_version == 1
    assert cache.length == 5 + accepted + 1
    assert cache.segments == ((cache.length, 1),)
    torch.testing.assert_close(model.model.fc.weight, original_fc, rtol=0, atol=0)
    assert "model.fc.weight" not in trainer.trainable_parameter_names
    assert "model.hidden_norm.weight" not in trainer.trainable_parameter_names
    torch.testing.assert_close(model.model.embed_tokens.weight, original_embed)
    torch.testing.assert_close(model.lm_head.weight, original_head)
    assert all(not k.requires_grad and not v.requires_grad for k, v in cache.layers)


def test_window_preserves_old_kv_and_reprojects_increment_with_updated_weights():
    model = tiny_model()
    trainer = DraftTrainer(
        model, TrainerConfig(learning_rate=1e-3, check_gradients=False)
    )
    old, _ = train_dflash_window(trainer, (make_batch(5, 0, prompt=True),), None)
    rounds = (make_batch(6, 1), make_batch(8, 0))
    features = torch.cat([r.confirmed_fc for r in rounds])
    before_step = model.project_context(features, torch.arange(6, 9))
    updated, result = train_dflash_window(trainer, rounds, old)

    assert result.model_version == 2
    assert result.target_count == 3
    assert updated.length == 9
    assert updated.segments == ((6, 1), (9, 2))
    with torch.no_grad():
        expected = model.project_context(
            torch.cat([r.confirmed_fc for r in rounds]), torch.arange(6, 9)
        )
    assert any(
        not torch.equal(before, after)
        for old_layer, new_layer in zip(before_step, expected, strict=True)
        for before, after in zip(old_layer, new_layer, strict=True)
    )
    assert all(not tensor.requires_grad for layer in updated.layers for tensor in layer)
    for before, after, suffix in zip(old.layers, updated.layers, expected):
        for left, right, new in zip(before, after, suffix):
            torch.testing.assert_close(left, right[:, :6], rtol=0, atol=0)
            torch.testing.assert_close(new, right[:, 6:])


def test_masked_suffix_remains_part_of_noncausal_forward():
    model = tiny_model()
    batch = make_batch(5, 0, prompt=True)
    context = model.project_context(batch.prompt_fc, torch.arange(5))
    complete = model(batch.input_ids, batch.positions, context)
    truncated = model(batch.input_ids[:2], batch.positions[:2], context)
    assert not torch.allclose(complete[1], truncated[1])


@pytest.mark.parametrize("attention_blocks", [0, 1, 2])
@pytest.mark.parametrize("selective", [False, True])
def test_packed_window_matches_sequential_loss_and_parameter_gradients(
    attention_blocks,
    selective,
):
    """Packing cannot expose future context or queries belonging to another round."""
    model = tiny_model(
        attention_blocks, grouped_query_attention=True, selective_last_layer=selective
    )
    reference = tiny_model()
    config = TrainerConfig(check_gradients=False)
    packed_trainer, sequential_trainer = (
        DraftTrainer(model, config),
        DraftTrainer(reference, config),
    )
    rounds = (make_batch(5, 1, prompt=True), make_batch(7, 0), make_batch(8, 3))
    context, end = None, 0
    logits, teachers = [], []
    for index, batch in enumerate(rounds):
        features = batch.prompt_fc if index == 0 else rounds[index - 1].confirmed_fc
        context = append_context(
            context,
            reference.project_context(features, torch.arange(end, batch.anchor)),
        )
        end = batch.anchor
        hidden = reference(batch.input_ids, batch.positions, context)
        count = len(batch.teacher_probabilities)
        logits.append(reference.lm_head(hidden[1 : count + 1]))
        teachers.append(batch.teacher_probabilities)
    loss = dflash_kl_loss(
        torch.cat(logits), torch.cat(teachers), position_weights(rounds)
    )
    expected_loss = sequential_trainer.backward_and_step(loss)
    _, actual = train_dflash_window(packed_trainer, rounds, None)
    assert actual.loss == pytest.approx(expected_loss, rel=1e-6)
    for p, ref in zip(model.parameters(), reference.parameters()):
        if ref in sequential_trainer.optimizer.state:
            # Adam's first moment preserves each parameter's gradient from this step.
            torch.testing.assert_close(
                packed_trainer.optimizer.state[p]["exp_avg"],
                sequential_trainer.optimizer.state[ref]["exp_avg"],
                rtol=2e-5,
                atol=2e-7,
            )


def test_position_decay_restarts_per_block_and_weights_kl():
    rounds = (make_batch(5, 1, prompt=True), make_batch(7, 3))
    weights = position_weights(rounds)
    expected = torch.exp(-torch.tensor([0, 1, 0, 1, 2]) / 7)
    torch.testing.assert_close(weights, expected)
    logits = torch.randn(5, 23, requires_grad=True)
    teacher = torch.cat([batch.teacher_probabilities for batch in rounds])
    kl = (teacher * (teacher.log() - logits.log_softmax(-1))).sum(-1)
    torch.testing.assert_close(
        dflash_kl_loss(logits, teacher, weights), (kl * expected).sum() / expected.sum()
    )


def test_training_fewer_blocks_preserves_context_and_gradient_boundary():
    """Detached context keeps logits/history and trains shared KV through queries."""
    model = tiny_model()
    rounds = (make_batch(5, 1, prompt=True), make_batch(7, 0), make_batch(8, 3))
    rounds[0].prompt_fc.requires_grad_(True)
    rounds[0].confirmed_fc.requires_grad_(True)
    complete, _ = forward_dflash_window(model, rounds, None)
    selected, teacher = forward_dflash_window(model, rounds, None, 1)
    torch.testing.assert_close(selected, complete[-3:])
    torch.testing.assert_close(teacher, rounds[-1].teacher_probabilities)
    selected.sum().backward()
    for tensor in (
        rounds[0].prompt_fc,
        rounds[0].confirmed_fc,
        model.model.fc.weight,
        model.model.hidden_norm.weight,
    ):
        assert tensor.grad is None
    for layer in model.model.layers:
        attention = layer.self_attn
        key_grad, value_grad = attention.qkv_proj.weight.grad[attention.q_size :].chunk(
            2
        )
        assert key_grad.count_nonzero()
        assert value_grad.count_nonzero()
        assert attention.k_norm.weight.grad.count_nonzero()
    trainer = DraftTrainer(model, TrainerConfig(check_gradients=False))
    cache, result = train_dflash_window(trainer, rounds, None, max_train_rounds=1)
    assert result.round_count == 1
    assert result.target_count == 3
    assert cache.length == 12
    assert cache.segments == ((12, 1),)
