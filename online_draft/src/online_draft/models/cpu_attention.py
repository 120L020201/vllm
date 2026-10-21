# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import cache
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable
from torch.utils.cpp_extension import load


@cache
def load_attention_backward() -> None:
    """Build the CPU extension during model initialization, before decoding."""
    load(
        name="online_draft_cpu_attention",
        sources=[str(Path(__file__).with_name("csrc") / "attention_backward.cpp")],
        extra_cflags=[
            "-O3",
            "-fopenmp",
            "-march=native",
            "-DCPU_CAPABILITY_AVX512",
            "-DCPU_CAPABILITY=AVX512",
        ],
        is_python_module=False,
    )


class _ContextAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, context_key, context_value, key, value, mask):
        query = query.contiguous()
        keys = torch.cat((context_key, key), dim=1)
        values = torch.cat((context_value, value), dim=1)
        if mask is not None and mask.dtype == torch.bool:
            mask = torch.zeros_like(mask, dtype=query.dtype).masked_fill_(
                ~mask, float("-inf")
            )
        if mask is not None:
            mask = mask.contiguous()
        output, logsumexp = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
            query.unsqueeze(0),
            keys.unsqueeze(0),
            values.unsqueeze(0),
            attn_mask=mask,
        )
        output = output.squeeze(0)
        ctx.context_length = context_key.shape[1]
        ctx.context_grad = context_key.requires_grad or context_value.requires_grad
        ctx.save_for_backward(
            query,
            keys,
            values,
            output.contiguous(),
            logsumexp.squeeze(0).contiguous(),
            mask if mask is not None else torch.empty(0),
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        query, keys, values, output, logsumexp, mask = ctx.saved_tensors
        start = 0 if ctx.context_grad else ctx.context_length
        grad_query, grad_key, grad_value = (
            torch.ops.online_draft_cpu.attention_backward(
                query,
                keys,
                values,
                output,
                logsumexp,
                grad_output.contiguous(),
                mask,
                start,
            )
        )
        if ctx.context_grad:
            end = ctx.context_length
            return (
                grad_query,
                grad_key[:, :end],
                grad_value[:, :end],
                grad_key[:, end:],
                grad_value[:, end:],
                None,
            )
        return grad_query, None, None, grad_key, grad_value, None


context_attention = _ContextAttention.apply
