# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Install GPU-training callbacks for the experimental online policies."""

import os
from typing import Any

from methods.utils.gpu_adapter import GpuOnlineEagle3Adapter
from methods.utils.gpu_training import create_gpu_policy_bridge
from methods.utils.graph_capture import install_graph_safe_capture

_ADAPTER: GpuOnlineEagle3Adapter | None = None
_INSTALLED = False


def _require_value(config: Any, name: str, expected: int) -> None:
    actual = getattr(config, name, expected)
    if actual != expected:
        raise ValueError(f"online methods require {name}={expected}; got {actual}")


def _request_ids_to_reset(scheduler_output) -> set[str]:
    return set(scheduler_output.finished_req_ids or ()).union(
        scheduler_output.preempted_req_ids or ()
    )


def _is_intermediate_prefill_chunk(input_batch) -> bool:
    if not input_batch.has_prefill:
        return False
    return bool(
        (
            input_batch.num_computed_prefill_tokens_np
            + input_batch.num_scheduled_tokens
            < input_batch.prefill_len_np
        ).any()
    )


def _create_gpu_bridge(config, draft_model):
    speculative = getattr(config, "speculative_config", None)
    if speculative is None or speculative.method != "eagle3":
        raise ValueError("online methods require EAGLE3 speculative decoding")
    if getattr(speculative, "parallel_drafting", False):
        raise ValueError("online methods require autoregressive EAGLE3 drafting")
    _require_value(config.scheduler_config, "max_num_seqs", 1)
    if getattr(config.cache_config, "enable_prefix_caching", False):
        raise ValueError("online methods require prefix caching to be disabled")
    for name in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
    ):
        _require_value(config.parallel_config, name, 1)

    model_path = os.environ.get("OSD_DRAFT_MODEL")
    if not model_path:
        raise ValueError("OSD_DRAFT_MODEL is required")
    parameter = next(
        parameter
        for parameter in draft_model.parameters()
        if parameter.is_floating_point()
    )
    return create_gpu_policy_bridge(
        model_path,
        device=parameter.device,
        dtype=parameter.dtype,
        learning_rate=float(os.environ.get("OSD_LEARNING_RATE", "2e-5")),
        method=os.environ["OSD_METHOD"],
        inference_model=draft_model,
    )


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    install_graph_safe_capture()

    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
        AutoRegressiveSpeculator,
    )
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    original_load_model = AutoRegressiveSpeculator.load_model
    original_propose = AutoRegressiveSpeculator.propose
    original_rejection_call = RejectionSampler.__call__
    original_finish_requests = GPUModelRunner.finish_requests
    original_shutdown = GPUModelRunner.shutdown

    def load_gpu_model(self, target_model):
        global _ADAPTER
        self._methods_capture_enabled = True
        result = original_load_model(self, target_model)
        _ADAPTER = GpuOnlineEagle3Adapter(
            self.model,
            _create_gpu_bridge(self.vllm_config, self.model),
            self.num_speculative_steps,
        )
        return result

    def propose(self, input_batch, *args, **kwargs):
        if _ADAPTER is not None:
            auxiliary = args[3] if len(args) > 3 else kwargs.get("aux_hidden_states")
            _ADAPTER.observe_pending_verification(input_batch, auxiliary)
            _ADAPTER.apply_weights()
            skip = (
                kwargs.get("dummy_run", False)
                or kwargs.get("is_profile", False)
                or self.supports_mm_inputs
            )
            _ADAPTER.start_proposal(
                input_batch,
                auxiliary,
                skip=skip,
                defer=not skip and _is_intermediate_prefill_chunk(input_batch),
            )
        self._methods_capture_result_num_reqs = input_batch.num_reqs
        self._methods_capture_steps = self.num_speculative_steps - 1
        result = original_propose(self, input_batch, *args, **kwargs)
        if _ADAPTER is not None:
            _ADAPTER.finish_proposal(self.get_proposal_snapshot())
        return result

    def rejection_call(self, logits, input_batch, *args, **kwargs):
        result = original_rejection_call(self, logits, input_batch, *args, **kwargs)
        if _ADAPTER is not None:
            _ADAPTER.capture_verification(
                logits,
                result.sampled_token_ids,
                result.num_sampled,
            )
        return result

    def finish_requests(self, scheduler_output):
        if _ADAPTER is not None:
            for request_id in _request_ids_to_reset(scheduler_output):
                _ADAPTER.reset_request(request_id)
        return original_finish_requests(self, scheduler_output)

    def shutdown(self):
        global _ADAPTER
        if _ADAPTER is not None:
            _ADAPTER.shutdown()
            _ADAPTER = None
        return original_shutdown(self)

    AutoRegressiveSpeculator.load_model = load_gpu_model
    AutoRegressiveSpeculator.propose = propose
    RejectionSampler.__call__ = rejection_call
    GPUModelRunner.finish_requests = finish_requests
    GPUModelRunner.shutdown = shutdown
