# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Install opt-in vLLM callbacks for online draft policy adapters."""

import os
from typing import Any

import torch
from online_draft.models.qwen3_eagle3 import load_qwen3_eagle3_checkpoint
from online_draft.training.trainer import DraftTrainer, TrainerConfig

from methods.utils.adapter import OnlineEagle3Adapter
from methods.utils.bridge import PolicyBridge

_ADAPTER: OnlineEagle3Adapter | None = None
_INSTALLED = False


def _require_value(config: Any, name: str, expected: int) -> None:
    actual = getattr(config, name, expected)
    if actual != expected:
        raise ValueError(f"online methods require {name}={expected}; got {actual}")


def _request_ids_to_reset(scheduler_output) -> set[str]:
    return set(scheduler_output.finished_req_ids or ()).union(
        scheduler_output.preempted_req_ids or ()
    )


def create_bridge(config) -> PolicyBridge:
    speculative = getattr(config, "speculative_config", None)
    if speculative is None or speculative.method != "eagle3":
        raise ValueError("online methods require EAGLE3 speculative decoding")
    if getattr(speculative, "parallel_drafting", False):
        raise ValueError("online methods require autoregressive EAGLE3 drafting")
    _require_value(config.scheduler_config, "max_num_seqs", 1)
    if getattr(config.scheduler_config, "enable_chunked_prefill", False):
        raise ValueError("online methods require chunked prefill to be disabled")
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
    model = load_qwen3_eagle3_checkpoint(model_path, dtype=torch.bfloat16)
    trainer = DraftTrainer(
        model,
        TrainerConfig(
            learning_rate=float(os.environ.get("OSD_LEARNING_RATE", "2e-5")),
        ),
    )
    return PolicyBridge(trainer, method=os.environ["OSD_METHOD"])


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
        AutoRegressiveSpeculator,
    )
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    original_load_model = AutoRegressiveSpeculator.load_model
    original_propose = AutoRegressiveSpeculator.propose
    original_prefill = AutoRegressiveSpeculator._prefill
    original_generate_draft = AutoRegressiveSpeculator._generate_draft
    original_rejection_call = RejectionSampler.__call__
    original_finish_requests = GPUModelRunner.finish_requests
    original_shutdown = GPUModelRunner.shutdown

    def load_model(self, target_model):
        global _ADAPTER
        result = original_load_model(self, target_model)
        _ADAPTER = OnlineEagle3Adapter(
            self.model,
            create_bridge(self.vllm_config),
            self.num_speculative_steps,
        )
        return result

    def propose(self, input_batch, *args, **kwargs):
        if _ADAPTER is not None:
            auxiliary = args[3] if len(args) > 3 else kwargs.get("aux_hidden_states")
            _ADAPTER.observe_pending_verification(input_batch, auxiliary)
            _ADAPTER.apply_weights()
            _ADAPTER.start_proposal(
                input_batch,
                auxiliary,
                skip=kwargs.get("dummy_run", False)
                or kwargs.get("is_profile", False)
                or self.supports_mm_inputs,
            )
        result = original_propose(self, input_batch, *args, **kwargs)
        if _ADAPTER is not None:
            _ADAPTER.finish_proposal()
        return result

    def prefill(self, num_reqs, *args, **kwargs):
        if _ADAPTER is not None:
            _ADAPTER.record_prefill(
                self.input_buffers.input_ids,
                self.input_buffers.positions,
                self.last_token_indices[:num_reqs],
            )
        return original_prefill(self, num_reqs, *args, **kwargs)

    def generate_draft(self, num_reqs, *args, **kwargs):
        if _ADAPTER is not None:
            _ADAPTER.record_draft_step(
                self.input_buffers.input_ids,
                self.input_buffers.positions,
                self.hidden_states,
                self.idx_mapping.new_tensor(range(num_reqs), dtype=torch.long),
            )
        return original_generate_draft(self, num_reqs, *args, **kwargs)

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

    AutoRegressiveSpeculator.load_model = load_model
    AutoRegressiveSpeculator.propose = propose
    AutoRegressiveSpeculator._prefill = prefill
    AutoRegressiveSpeculator._generate_draft = generate_draft
    RejectionSampler.__call__ = rejection_call
    GPUModelRunner.finish_requests = finish_requests
    GPUModelRunner.shutdown = shutdown
