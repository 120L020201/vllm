# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from online_eagle3 import factory as factory_module
from online_eagle3.sync_bridge import Qwen3Eagle3LazySyncBridge


class _ToyEagle3Module(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(8, 4)
        self.model.fc = nn.Linear(4, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.draft_id_to_target_id = nn.Parameter(
            torch.zeros(8, dtype=torch.long), requires_grad=False
        )


def _make_vllm_config(
    *,
    max_num_seqs: int = 1,
    method: str = "eagle3",
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    data_parallel_size: int = 1,
):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method=method,
            draft_model_config=SimpleNamespace(model="/draft/from/config"),
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            data_parallel_size=data_parallel_size,
        ),
    )


def test_factory_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_ONLINE_EAGLE3", raising=False)

    bridge = factory_module.maybe_create_qwen3_eagle3_sync_bridge(_make_vllm_config())

    assert bridge is None


def test_factory_creates_lazy_sync_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded: dict[str, object] = {}

    def _load_model(path: str, *, dtype: torch.dtype) -> _ToyEagle3Module:
        loaded["path"] = path
        loaded["dtype"] = dtype
        return _ToyEagle3Module()

    monkeypatch.setenv("VLLM_ONLINE_EAGLE3", "1")
    monkeypatch.setenv("VLLM_ONLINE_EAGLE3_DRAFT_MODEL", "/draft/from/env")
    monkeypatch.setenv("VLLM_ONLINE_EAGLE3_UPDATE_INTERVAL", "2")
    monkeypatch.setenv("VLLM_ONLINE_EAGLE3_LR", "0.25")
    monkeypatch.setenv("VLLM_ONLINE_EAGLE3_WEIGHT_DECAY", "0.5")
    monkeypatch.setenv("VLLM_ONLINE_EAGLE3_DTYPE", "bf16")
    monkeypatch.setattr(factory_module, "load_torch_eagle3_model", _load_model)

    bridge = factory_module.maybe_create_qwen3_eagle3_sync_bridge(_make_vllm_config())

    assert isinstance(bridge, Qwen3Eagle3LazySyncBridge)
    assert not bridge.is_loaded
    trainer = bridge.trainer
    assert bridge.is_loaded
    assert trainer.config.lr == 0.25
    assert trainer.config.weight_decay == 0.5
    assert loaded == {"path": "/draft/from/env", "dtype": torch.bfloat16}


def test_factory_writes_cpu_runtime_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    def _load_model(path: str, *, dtype: torch.dtype) -> _ToyEagle3Module:
        return _ToyEagle3Module().to(dtype=dtype)

    metadata_path = tmp_path / "online_eagle3_cpu_runtime.json"
    monkeypatch.setenv("VLLM_ONLINE_EAGLE3", "1")
    monkeypatch.setenv("VLLM_ONLINE_EAGLE3_DRAFT_MODEL", "/draft/from/env")
    monkeypatch.setenv("VLLM_ONLINE_EAGLE3_DTYPE", "bf16")
    monkeypatch.setattr(factory_module, "load_torch_eagle3_model", _load_model)

    config = _make_vllm_config()
    config.profiler_config = SimpleNamespace(
        torch_profiler_dir=str(tmp_path),
    )
    bridge = factory_module.maybe_create_qwen3_eagle3_sync_bridge(config)
    assert bridge is not None
    trainer = bridge.trainer
    assert trainer.config.lr == 1e-5

    metadata = json.loads(metadata_path.read_text())
    assert metadata["cpu_model"]
    assert metadata["torch_num_threads"] == torch.get_num_threads()
    assert metadata["cpu_draft_dtype"] == "BF16"


def test_factory_rejects_non_eagle3(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_ONLINE_EAGLE3", "1")

    with pytest.raises(ValueError, match="eagle3"):
        factory_module.maybe_create_qwen3_eagle3_sync_bridge(
            _make_vllm_config(method="ngram")
        )


def test_factory_rejects_multi_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_ONLINE_EAGLE3", "1")

    with pytest.raises(ValueError, match="max-num-seqs 1"):
        factory_module.maybe_create_qwen3_eagle3_sync_bridge(
            _make_vllm_config(max_num_seqs=2)
        )


def test_factory_rejects_tensor_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_ONLINE_EAGLE3", "1")

    with pytest.raises(ValueError, match="tensor_parallel_size=1"):
        factory_module.maybe_create_qwen3_eagle3_sync_bridge(
            _make_vllm_config(tensor_parallel_size=2)
        )
