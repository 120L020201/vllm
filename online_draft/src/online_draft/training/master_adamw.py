# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import logging
from collections.abc import Iterable
from types import ModuleType

import torch

logger = logging.getLogger(__name__)

# One memory pass per element: read BF16 grad and FP32 master/moments, write
# them back plus the rounded BF16 parameter. The binding releases the GIL so
# the serving thread keeps running while the CPU trainer steps.
_NATIVE_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <cmath>
#include <cstring>

static inline float bf16_to_float(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

static inline uint16_t float_to_bf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t rounded = (bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16;
  const uint32_t nan =
      0u - static_cast<uint32_t>((bits & 0x7FFFFFFFu) > 0x7F800000u);
  return static_cast<uint16_t>((rounded & ~nan) | (0x7FC0u & nan));
}

void bf16_master_adamw_step(
    const std::vector<at::Tensor>& params,
    const std::vector<at::Tensor>& grads,
    const std::vector<at::Tensor>& masters,
    const std::vector<at::Tensor>& exp_avgs,
    const std::vector<at::Tensor>& exp_avg_sqs,
    const std::vector<int64_t>& steps,
    double lr, double beta1, double beta2, double eps, double weight_decay) {
  for (size_t t = 0; t < params.size(); ++t) {
    const float decay = static_cast<float>(1.0 - lr * weight_decay);
    const float step_size =
        static_cast<float>(lr / (1.0 - std::pow(beta1, steps[t])));
    const float bias_correction2_sqrt =
        static_cast<float>(std::sqrt(1.0 - std::pow(beta2, steps[t])));
    const float b1 = static_cast<float>(beta1);
    const float b2 = static_cast<float>(beta2);
    const float e = static_cast<float>(eps);
    uint16_t* __restrict__ param =
        reinterpret_cast<uint16_t*>(params[t].data_ptr());
    const uint16_t* __restrict__ grad =
        reinterpret_cast<const uint16_t*>(grads[t].data_ptr());
    float* __restrict__ master = masters[t].data_ptr<float>();
    float* __restrict__ m = exp_avgs[t].data_ptr<float>();
    float* __restrict__ v = exp_avg_sqs[t].data_ptr<float>();
    at::parallel_for(0, params[t].numel(), 1 << 16, [=](int64_t begin, int64_t end) {
      #pragma omp simd
      for (int64_t i = begin; i < end; ++i) {
        const float g = bf16_to_float(grad[i]);
        const float mi = b1 * m[i] + (1.0f - b1) * g;
        const float vi = b2 * v[i] + (1.0f - b2) * g * g;
        const float denom = std::sqrt(vi) / bias_correction2_sqrt + e;
        const float w = master[i] * decay - step_size * mi / denom;
        m[i] = mi;
        v[i] = vi;
        master[i] = w;
        param[i] = float_to_bf16(w);
      }
    });
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "bf16_master_adamw_step",
      &bf16_master_adamw_step,
      pybind11::call_guard<pybind11::gil_scoped_release>());
}
"""


@functools.cache
def _native_kernel() -> ModuleType | None:
    try:
        from torch.utils.cpp_extension import load_inline

        return load_inline(
            name="online_draft_master_adamw",
            cpp_sources=_NATIVE_SOURCE,
            extra_cflags=["-O3", "-march=native", "-fopenmp", "-fno-math-errno"],
            extra_ldflags=["-fopenmp"],
        )
    except Exception as error:  # noqa: BLE001 - any build failure falls back
        logger.warning(
            "native master AdamW unavailable, using FP32 master copies: %s", error
        )
        return None


def _is_native_parameter(param: torch.Tensor) -> bool:
    return (
        param.device.type == "cpu"
        and param.dtype == torch.bfloat16
        and param.is_contiguous()
    )


def supports_master_weight_adamw(params: Iterable[torch.Tensor]) -> bool:
    """Whether ``MasterWeightAdamW`` can update every parameter natively.

    Builds the native kernel on first use when all parameters are contiguous
    CPU BF16 tensors.
    """
    return all(_is_native_parameter(param) for param in params) and (
        _native_kernel() is not None
    )


class MasterWeightAdamW(torch.optim.Optimizer):
    """AdamW with FP32 master weights and moments for BF16 CPU parameters.

    BF16 round-to-nearest discards any update smaller than half an ULP, so
    a 1e-5 Adam step leaves most BF16 weights unchanged. The model keeps its
    BF16 copy for forward and backward; each step updates the FP32 master and
    writes back its rounded value in one native pass that releases the GIL.
    Check ``supports_master_weight_adamw`` before constructing it.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        *,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(
            params,
            dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay),
        )
        parameters = [p for group in self.param_groups for p in group["params"]]
        if not supports_master_weight_adamw(parameters):
            raise ValueError(
                "MasterWeightAdamW requires the native kernel and contiguous "
                "CPU BF16 parameters"
            )
        kernel = _native_kernel()
        assert kernel is not None
        self._kernel = kernel

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            states = [self._state_for(param) for param in params]
            for state in states:
                state["step"] += 1
            beta1, beta2 = group["betas"]
            self._kernel.bf16_master_adamw_step(
                params,
                [p.grad.contiguous() for p in params],
                [s["master"] for s in states],
                [s["exp_avg"] for s in states],
                [s["exp_avg_sq"] for s in states],
                [s["step"] for s in states],
                group["lr"],
                beta1,
                beta2,
                group["eps"],
                group["weight_decay"],
            )
        return loss

    def _state_for(self, param: torch.Tensor) -> dict:
        state = self.state[param]
        if not state:
            master = param.detach().to(dtype=torch.float32, copy=True)
            state["step"] = 0
            state["master"] = master
            state["exp_avg"] = torch.zeros_like(master)
            state["exp_avg_sq"] = torch.zeros_like(master)
        return state
