# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from online_eagle3.config import Qwen3Eagle3TrainerConfig
from online_eagle3.data import DistillationBatch


def test_training_without_vllm_imports():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys

class BlockVllm(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "vllm" or fullname.startswith("vllm."):
            raise AssertionError("Training must not import vLLM: " + fullname)

sys.meta_path.insert(0, BlockVllm())
import runpy
import torch
import online_eagle3

torch.set_num_threads(2)
tests = runpy.run_path("tests/standalone_tests/test_online_eagle3_distillation.py")
tests["test_append_uses_updated_weights_and_preserves_history"](3, 2)
assert not any(n == "vllm" or n.startswith("vllm.") for n in sys.modules)
print("standalone training, append and reset passed")
""",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "standalone training, append and reset passed" in result.stdout


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lr": float("nan")},
        {"lr": 0},
        {"weight_decay": -1},
        {"dtype": torch.float16},
        {"torch_threads": 0},
        {"update_interval": 2},
    ],
)
def test_invalid_training_config(kwargs):
    with pytest.raises(ValueError):
        Qwen3Eagle3TrainerConfig(**kwargs)


def test_batch_preserves_integer_and_loss_precision():
    from dataclasses import fields

    payload = {item.name: torch.ones(2, 8) for item in fields(DistillationBatch)}
    for key in payload:
        if key.endswith("positions"):
            payload[key] = torch.arange(2)
    batch = DistillationBatch.from_payload(payload, torch.bfloat16)
    assert batch.proposal_positions.dtype == torch.long
    assert batch.teacher_probs.dtype == torch.float32
    assert batch.proposal_hidden_states.dtype == torch.bfloat16
    assert batch.proposal_hidden_states.grad_fn is None
