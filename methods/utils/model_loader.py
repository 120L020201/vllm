# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Load the CPU training model used by online draft methods."""

import json
from collections.abc import Mapping
from pathlib import Path

import torch
from online_draft.models.qwen3_eagle3 import (
    Qwen3Eagle3Config,
    Qwen3Eagle3ForCausalLM,
    convert_angelslim_eagle3_state_dict,
    load_qwen3_eagle3_checkpoint,
)
from safetensors.torch import load_file as load_safetensors_file


def load_cpu_training_model(
    model_directory: str | Path,
    *,
    dtype: torch.dtype = torch.float32,
) -> Qwen3Eagle3ForCausalLM:
    """Load an AngelSlim Qwen3 EAGLE3 checkpoint on CPU.

    Args:
        model_directory: Directory containing config.json and model weights.
        dtype: Floating-point dtype used by the loaded model.

    Returns:
        A CPU model with converted checkpoint weights.
    """
    model_path = Path(model_directory)
    pytorch_checkpoint_path = model_path / "pytorch_model.bin"
    safetensors_checkpoint_path = model_path / "model.safetensors"

    if pytorch_checkpoint_path.is_file():
        return load_qwen3_eagle3_checkpoint(model_path, dtype=dtype)
    if not safetensors_checkpoint_path.is_file():
        raise FileNotFoundError(
            "checkpoint directory must contain pytorch_model.bin or model.safetensors"
        )

    with (model_path / "config.json").open(encoding="utf-8") as config_file:
        raw_config = json.load(config_file)
    if not isinstance(raw_config, Mapping):
        raise TypeError("config.json must contain a JSON object")

    model = Qwen3Eagle3ForCausalLM(Qwen3Eagle3Config.from_dict(raw_config))
    model.to(device="cpu", dtype=dtype)

    source_state_dict = load_safetensors_file(safetensors_checkpoint_path)
    converted_state_dict = convert_angelslim_eagle3_state_dict(source_state_dict)
    model.load_state_dict(converted_state_dict, strict=True)
    return model
