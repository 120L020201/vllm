# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class _LinearWithPersistentGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, module):  # type: ignore[override]
        ctx.save_for_backward(inputs, weight)
        ctx.module = module
        return F.linear(inputs, weight)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        inputs, weight = ctx.saved_tensors
        module = ctx.module
        grad_input = grad_output.mm(weight) if ctx.needs_input_grad[0] else None
        parameter = module.weight
        if parameter.grad is None:
            # Allocated on first backward, which never runs in inference mode.
            if module.grad_buffer is None:
                module.grad_buffer = torch.empty_like(
                    parameter, memory_format=torch.contiguous_format
                )
            torch.mm(grad_output.t(), inputs, out=module.grad_buffer)
            parameter.grad = module.grad_buffer
        else:
            parameter.grad.addmm_(grad_output.t(), inputs)
        return grad_input, None, None


class PersistentGradLinear(nn.Linear):
    """Bias-free linear layer that reuses one weight-gradient buffer."""

    grad_buffer: torch.Tensor | None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2 and torch.is_grad_enabled():
            return _LinearWithPersistentGrad.apply(inputs, self.weight, self)
        return F.linear(inputs, self.weight)


def install_persistent_weight_grads(
    model: nn.Module,
    trainable_names: Iterable[str],
) -> tuple[str, ...]:
    """Write bias-free linear weight gradients into reused buffers.

    Fresh weight-sized CPU gradients are page-faulted on every backward and
    unmapped on every zero_grad. Each patched layer instead computes its
    weight gradient straight into a buffer it owns, so gradients still
    follow set_to_none semantics while their storage is never freed.

    Args:
        model: Model whose 2-D-input linear layers should be patched.
        trainable_names: Parameter names the trainer updates.

    Returns:
        Names of the weights whose gradients now use persistent buffers.
    """
    trainable = set(trainable_names)
    patched: list[str] = []
    for module_name, module in model.named_modules():
        weight_name = f"{module_name}.weight" if module_name else "weight"
        if (
            type(module) is not nn.Linear
            or module.bias is not None
            or weight_name not in trainable
        ):
            continue
        module.__class__ = PersistentGradLinear
        module.grad_buffer = None
        patched.append(weight_name)
    return tuple(patched)
