# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Qwen3 target that ignores embedded ``mtp.*`` draft weights."""

from collections.abc import Iterable

import torch

from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM as _Qwen3ForCausalLM


class Qwen3ForCausalLM(_Qwen3ForCausalLM):
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(
            (name, weight) for name, weight in weights if not name.startswith("mtp.")
        )
