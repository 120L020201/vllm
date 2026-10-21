# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Startup-only comparison against the loaded vLLM draft's real GPU layers."""

import torch
from torch import nn
from torch.nn import functional as F


def check_dflash_math(worker):
    from online_draft.runtime.weights import DraftWeightSnapshot

    trainer = worker._online_draft_trainer
    state = worker._online_draft_speculator.online_state
    target_storage = {
        p.untyped_storage().data_ptr() for p in worker.model_runner.model.parameters()
    }
    draft_storage = {
        tensor.untyped_storage().data_ptr()
        for slot in state.slots.slots
        for tensor in slot.values()
    }
    assert target_storage.isdisjoint(draft_storage)
    result = {
        "target_storage_isolated": True,
        "w0": _compare_dflash_math(worker),
    }
    parameters = dict(trainer.model.named_parameters())
    try:
        with torch.inference_mode():
            for name in trainer.trainable_parameter_names:
                parameters[name].mul_(1.01)
            snapshot = DraftWeightSnapshot.from_named_tensors(
                [
                    (name, parameters[name])
                    for name in trainer.trainable_parameter_names
                ],
                version=1,
            )
            state.installer.stage(snapshot)
            # This startup-only diagnostic deliberately waits; no request is active.
            torch.accelerator.synchronize()
            assert state.installer.commit_if_ready() == 1
            state.slots.bind(state.installer.active_weights)
            layers = trainer.model.model.layers
            expected = torch.cat(
                [
                    layer.self_attn.qkv_proj.weight[layer.self_attn.q_size :]
                    for layer in layers
                ]
            )
            torch.testing.assert_close(
                state.slots.model._fused_kv_weight.cpu(), expected, rtol=0, atol=0
            )
            result["installed_nonzero_update"] = _compare_dflash_math(worker)
    finally:
        with torch.inference_mode():
            trainer.reset()
            state.installer.reset(trainer.initial_weights)
            state.slots.bind(state.installer.active_weights)
    return result


def _compare_dflash_math(worker):
    from vllm import _custom_ops as ops

    cpu = worker._online_draft_trainer.model
    gpu = worker.get_draft_model()
    model = gpu.model
    config = cpu.config
    generator = torch.Generator().manual_seed(17)
    features = torch.randn(
        8,
        config.hidden_size * config.num_aux_hidden_states,
        generator=generator,
        dtype=torch.bfloat16,
    )
    positions = torch.arange(8)
    query_positions = torch.arange(8, 24)
    input_ids = torch.tensor(
        [100] + [worker.model_runner.speculator.parallel_drafting_token_id] * 15
    )

    class ContextAttention(nn.Module):
        def __init__(self, key, value):
            super().__init__()
            self.key, self.value = key, value

        def forward(self, q, k, v):
            def heads(x):
                return x.view(16, -1, config.head_dim).transpose(0, 1)

            key = torch.cat((self.key, heads(k)), dim=1)
            value = torch.cat((self.value, heads(v)), dim=1)
            repeats = config.num_attention_heads // config.num_key_value_heads
            output = F.scaled_dot_product_attention(
                heads(q).unsqueeze(0),
                key.repeat_interleave(repeats, 0).unsqueeze(0),
                value.repeat_interleave(repeats, 0).unsqueeze(0),
                is_causal=False,
            )
            return output.squeeze(0).transpose(0, 1).reshape(16, -1)

    with torch.inference_mode():
        projected = gpu.combine_hidden_states(features.cuda())
        cpu_context = cpu.project_context(projected.cpu(), positions)
        key, value = model._project_context_kv(
            projected,
            8,
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim,
        )
        key = model._normalize_context_k(key)
        flat = key.view(config.num_hidden_layers * 8, -1)
        ops.rotary_embedding(
            positions.cuda().repeat(config.num_hidden_layers),
            flat,
            None,
            model._rope_head_size,
            model._rope_cos_sin_cache.to(flat.dtype),
            model._rope_is_neox,
        )
        gpu_context = tuple(
            (k.transpose(0, 1), v.transpose(0, 1)) for k, v in zip(key, value)
        )
        original = [layer.self_attn.attn for layer in model.layers]
        try:
            for layer, (k, v) in zip(model.layers, gpu_context):
                layer.self_attn.attn = ContextAttention(k, v)
            gpu_hidden = gpu(input_ids.cuda(), query_positions.cuda())
            gpu_logits = F.linear(gpu_hidden, gpu.lm_head.weight).float().cpu()
        finally:
            for layer, attention in zip(model.layers, original):
                layer.self_attn.attn = attention
        output_rows = (
            torch.tensor([1, 2, 7, 15]) if config.selective_last_layer else None
        )
        cpu_logits = cpu.lm_head(
            cpu(input_ids, query_positions, cpu_context, output_rows=output_rows)
        ).float()
        if output_rows is not None:
            gpu_logits = gpu_logits[output_rows]

    def relative_rmse(actual, expected):
        actual, expected = actual.float(), expected.float()
        return (
            ((actual - expected).square().mean() / expected.square().mean())
            .sqrt()
            .item()
        )

    result = {
        "context_relative_rmse": max(
            relative_rmse(actual, expected.cpu())
            for c, g in zip(cpu_context, gpu_context)
            for actual, expected in zip(c, g)
        ),
        "logits_relative_rmse": relative_rmse(cpu_logits, gpu_logits),
        "logits_cosine": F.cosine_similarity(
            cpu_logits.flatten(), gpu_logits.flatten(), dim=0
        ).item(),
        "forward_kl": F.kl_div(
            cpu_logits.log_softmax(-1), gpu_logits.softmax(-1), reduction="batchmean"
        ).item(),
    }
    assert result["context_relative_rmse"] < 0.03, result
    assert result["logits_relative_rmse"] < 0.03, result
    assert result["forward_kl"] < 0.02, result
    return result
