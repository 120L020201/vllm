# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

import vllm.v1.worker.gpu_worker as gpu_worker
from vllm.v1.worker.gpu_worker import Worker


def test_shutdown_closes_online_draft_before_model_runner(monkeypatch) -> None:
    closed: list[str] = []
    worker = object.__new__(Worker)
    worker._online_draft_capture_adapter = SimpleNamespace(
        close=lambda: closed.append("capture")
    )
    worker._online_draft_pipeline = SimpleNamespace(
        close=lambda: closed.append("pipeline")
    )
    worker._online_draft_trainer = object()
    worker._online_draft_speculator = object()
    worker.profiler = None
    worker.model_runner = SimpleNamespace(
        shutdown=lambda: closed.append("model_runner")
    )

    monkeypatch.setattr(gpu_worker.gc, "unfreeze", lambda: None)
    monkeypatch.setattr(gpu_worker, "ensure_kv_transfer_shutdown", None)
    monkeypatch.setattr(gpu_worker, "ensure_ec_transfer_shutdown", None)
    monkeypatch.setattr(
        gpu_worker.current_platform,
        "is_cuda_alike",
        lambda: False,
    )

    Worker.shutdown(worker)

    assert closed == ["capture", "pipeline", "model_runner"]
    assert worker._online_draft_capture_adapter is None
    assert worker._online_draft_pipeline is None
    assert worker._online_draft_trainer is None
    assert worker._online_draft_speculator is None


def test_online_draft_failure_stops_worker() -> None:
    def raise_if_failed() -> None:
        raise RuntimeError("training failed")

    worker = object.__new__(Worker)
    worker._online_draft_pipeline = SimpleNamespace(
        raise_if_failed=raise_if_failed,
    )

    with pytest.raises(RuntimeError, match="training failed"):
        Worker._raise_if_online_draft_failed(worker)
