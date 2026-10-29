# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import vllm.v1.worker.gpu_worker as gpu_worker
from vllm.v1.worker.gpu.online_draft import (
    OnlineDraftRuntime,
    _DFlashRecipe,
    _Eagle3Recipe,
    _MTPRecipe,
    _parse_online_draft_settings,
    _validate_online_draft_config,
)
from vllm.v1.worker.gpu_worker import Worker


def test_shutdown_closes_online_draft_before_model_runner(monkeypatch) -> None:
    closed: list[str] = []
    worker = object.__new__(Worker)
    worker._online_draft = OnlineDraftRuntime(
        trainer=Mock(),
        pipeline=SimpleNamespace(close=lambda: closed.append("pipeline")),
        capture_adapter=SimpleNamespace(close=lambda: closed.append("capture")),
        speculator=Mock(),
        inference_cpu=Mock(),
    )
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
    assert worker._online_draft is None


@pytest.mark.parametrize("component", ["pipeline", "capture_adapter"])
def test_online_draft_failure_stops_worker(component) -> None:
    def raise_if_failed() -> None:
        raise RuntimeError("online worker failed")

    worker = object.__new__(Worker)
    worker._online_draft = OnlineDraftRuntime(
        trainer=Mock(),
        pipeline=Mock(),
        capture_adapter=Mock(),
        speculator=Mock(),
        inference_cpu=Mock(),
    )
    setattr(
        worker._online_draft,
        component,
        SimpleNamespace(
            raise_if_failed=raise_if_failed,
        ),
    )

    with pytest.raises(RuntimeError, match="online worker failed"):
        Worker.execute_model(worker, SimpleNamespace(finished_req_ids={"request-0"}))

    worker._online_draft.speculator.finish_request.assert_not_called()


def test_capture_enable_applies_inference_cpu_settings_before_attachment() -> None:
    calls = []
    adapter = object()

    def attach(value):
        assert value is adapter
        calls.append("capture")

    runtime = OnlineDraftRuntime(
        trainer=Mock(),
        pipeline=Mock(),
        capture_adapter=adapter,
        speculator=SimpleNamespace(set_capture_adapter=attach),
        inference_cpu=SimpleNamespace(apply=lambda: calls.append("cpu")),
    )

    assert calls == []
    runtime.enable_capture()
    assert calls == ["cpu", "capture"]


def test_finish_uses_initial_weights_only_after_capture_is_enabled() -> None:
    initial_weights = object()
    speculator = Mock(capture_adapter=None)
    runtime = OnlineDraftRuntime(
        trainer=SimpleNamespace(initial_weights=initial_weights),
        pipeline=Mock(),
        capture_adapter=Mock(),
        speculator=speculator,
        inference_cpu=Mock(),
    )

    runtime.finish_requests({"request-0"})
    speculator.finish_request.assert_not_called()

    speculator.capture_adapter = runtime.capture_adapter
    runtime.finish_requests({"request-0"})
    speculator.finish_request.assert_called_once_with("request-0", initial_weights)


def _online_config(method: str, **overrides) -> SimpleNamespace:
    settings = {
        "max_num_seqs": 1,
        "enable_prefix_caching": False,
        "model_type": "qwen3",
        **overrides,
    }
    return SimpleNamespace(
        speculative_config=SimpleNamespace(method=method),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=settings["max_num_seqs"], enable_chunked_prefill=True
        ),
        cache_config=SimpleNamespace(
            enable_prefix_caching=settings["enable_prefix_caching"]
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type=settings["model_type"])
        ),
    )


@pytest.mark.parametrize(
    ("method", "name"),
    [("eagle3", "EAGLE3"), ("dflash", "DFlash")],
)
def test_online_draft_config_lists_unmet_requirements(method, name) -> None:
    """Both methods accept chunked prefill but need one contiguous request."""
    speculator = SimpleNamespace(_speculator_name="DFlash", supports_mm_inputs=False)
    _validate_online_draft_config(_online_config(method), speculator)

    config = _online_config(method, max_num_seqs=2, enable_prefix_caching=True)
    with pytest.raises(
        ValueError,
        match=f"online {name} requires max_num_seqs=1, enable_prefix_caching=False",
    ):
        _validate_online_draft_config(config, speculator)


def test_online_eagle3_rejects_other_speculative_methods() -> None:
    with pytest.raises(ValueError, match="online EAGLE3 requires method=eagle3"):
        _validate_online_draft_config(_online_config("eagle"), Mock())


def test_online_draft_rejects_multimodal_draft_inputs() -> None:
    """Capture embeds input ids only, so merged image embeddings would be lost."""
    speculator = SimpleNamespace(supports_mm_inputs=True)
    with pytest.raises(
        ValueError, match="online EAGLE3 requires text-only draft inputs"
    ):
        _validate_online_draft_config(_online_config("eagle3"), speculator)


def test_online_draft_settings_map_to_trainer_and_method() -> None:
    settings = _parse_online_draft_settings(
        {
            "fused_optimizer": False,
            "frozen_parameter_prefixes": ["model.fc"],
            "max_train_rounds": 4,
            "cpu_threads": {"training": 2},
        },
        _DFlashRecipe,
    )

    assert settings.trainer.fused is False
    assert settings.trainer.check_gradients is False
    assert settings.trainer.frozen_parameter_prefixes == ("model.fc",)
    assert settings.recipe == _DFlashRecipe(max_train_rounds=4)
    assert settings.cpu["training"].threads == 2
    assert settings.cpu["capture"].threads is None


@pytest.mark.parametrize(
    ("recipe", "values", "message"),
    [
        (_Eagle3Recipe, {"selective_last_layer": True}, "online EAGLE3 settings"),
        (_DFlashRecipe, {"attention_backend": "eager"}, "online DFlash settings"),
        (_DFlashRecipe, {"loss_decay_gamma": 7.0}, "settings: loss_decay_gamma"),
        (_Eagle3Recipe, {"cpu_threads": {"train": 2}}, "CPU roles: train"),
    ],
)
def test_online_draft_settings_reject_keys_the_method_ignores(
    recipe, values, message
) -> None:
    """A misspelled or other-method key would otherwise be silently ignored."""
    with pytest.raises(ValueError, match=message):
        _parse_online_draft_settings(values, recipe)


@pytest.mark.parametrize("recipe", [_Eagle3Recipe, _MTPRecipe, _DFlashRecipe])
def test_online_draft_max_train_rounds_defaults_to_every_round(recipe) -> None:
    """Without the key every method trains all rounds since the last install."""
    assert _parse_online_draft_settings({}, recipe).recipe.max_train_rounds == 0
    settings = _parse_online_draft_settings({"max_train_rounds": 4}, recipe)
    assert settings.recipe.max_train_rounds == 4
    with pytest.raises(ValueError, match="max_train_rounds must be >= 0"):
        _parse_online_draft_settings({"max_train_rounds": -1}, recipe)


def test_online_mtp_requires_a_qwen3_5_target() -> None:
    """The CPU MTP layer mirrors Qwen3.5's gated attention only."""
    speculator = SimpleNamespace(_speculator_name="MTP", supports_mm_inputs=False)
    _validate_online_draft_config(
        _online_config("mtp", model_type="qwen3_5"), speculator
    )

    with pytest.raises(
        ValueError, match=r"online MTP requires target model_type in \['qwen3_5'\]"
    ):
        _validate_online_draft_config(_online_config("mtp"), speculator)
