# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .qwen3_eagle3 import (
    Eagle3ForwardOutput,
    Eagle3KVCache,
    Qwen3Eagle3Config,
    Qwen3Eagle3ForCausalLM,
    convert_angelslim_eagle3_state_dict,
    load_qwen3_eagle3_checkpoint,
)

__all__ = [
    "Eagle3ForwardOutput",
    "Eagle3KVCache",
    "Qwen3Eagle3Config",
    "Qwen3Eagle3ForCausalLM",
    "convert_angelslim_eagle3_state_dict",
    "load_qwen3_eagle3_checkpoint",
]
