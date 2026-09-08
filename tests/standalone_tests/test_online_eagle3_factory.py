from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from online_eagle3.factory import maybe_create_qwen3_eagle3_bridge
from tests.standalone_tests.test_online_eagle3_torch_model import (
    _make_checkpoint_state,
    _make_config_dict,
)


def _make_vllm_config(model_path, method: str = "eagle3"):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method=method,
            draft_model_config=SimpleNamespace(model=str(model_path)),
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        ),
    )


def _write_model(tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_make_config_dict()))
    torch.save(_make_checkpoint_state(), tmp_path / "pytorch_model.bin")


def test_factory_returns_none_when_disabled(monkeypatch, tmp_path) -> None:
    _write_model(tmp_path)
    monkeypatch.delenv("VLLM_ONLINE_EAGLE3", raising=False)

    bridge = maybe_create_qwen3_eagle3_bridge(_make_vllm_config(tmp_path))

    assert bridge is None


def test_factory_creates_bridge_when_enabled(monkeypatch, tmp_path) -> None:
    _write_model(tmp_path)
    monkeypatch.setenv("VLLM_ONLINE_EAGLE3", "1")
    monkeypatch.setenv("VLLM_ONLINE_EAGLE3_UPDATE_INTERVAL", "2")
    monkeypatch.setenv("VLLM_ONLINE_EAGLE3_LR", "2e-5")

    bridge = maybe_create_qwen3_eagle3_bridge(_make_vllm_config(tmp_path))

    assert bridge is not None
    assert not bridge.is_loaded
    assert bridge.trainer.config.lr == 2e-5
    assert bridge.is_loaded

    bridge.shutdown()


def test_factory_loads_normal_tensors_inside_inference_mode(
    monkeypatch,
    tmp_path,
) -> None:
    _write_model(tmp_path)
    monkeypatch.setenv("VLLM_ONLINE_EAGLE3", "1")

    with torch.inference_mode():
        bridge = maybe_create_qwen3_eagle3_bridge(_make_vllm_config(tmp_path))
        assert bridge is not None
        trainer = bridge.trainer

    parameter = next(trainer.model.parameters())
    assert not parameter.is_inference()

    bridge.shutdown()


def test_factory_rejects_non_eagle3_method(monkeypatch, tmp_path) -> None:
    _write_model(tmp_path)
    monkeypatch.setenv("VLLM_ONLINE_EAGLE3", "1")

    with pytest.raises(ValueError, match="requires EAGLE3"):
        maybe_create_qwen3_eagle3_bridge(
            _make_vllm_config(tmp_path, method="draft_model")
        )


def test_factory_rejects_multi_rank(monkeypatch, tmp_path) -> None:
    _write_model(tmp_path)
    monkeypatch.setenv("VLLM_ONLINE_EAGLE3", "1")
    vllm_config = _make_vllm_config(tmp_path)
    vllm_config.parallel_config.tensor_parallel_size = 2

    with pytest.raises(ValueError, match="TP=PP=DP=1"):
        maybe_create_qwen3_eagle3_bridge(vllm_config)
