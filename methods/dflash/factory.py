# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Install DFlash TTS and random-sampling adaptation policies."""

import copy
import os

import torch

from methods.dflash.adapter import DflashOnlineAdapter, DflashPolicyBridge
from methods.dflash.training import create_dflash_trainer

_ADAPTER: DflashOnlineAdapter | None = None
_INSTALLED = False


def _require_value(config, name: str, expected: int) -> None:
    actual = getattr(config, name, expected)
    if actual != expected:
        raise ValueError(f"DFlash methods require {name}={expected}; got {actual}")


def _request_ids_to_reset(scheduler_output) -> set[str]:
    return set(scheduler_output.finished_req_ids or ()).union(
        scheduler_output.preempted_req_ids or ()
    )


def _is_frozen_parameter(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def _clone_module_tree(
    module: torch.nn.Module,
    *,
    prefix: str,
    frozen_prefixes: tuple[str, ...],
    shared_module_ids: set[int],
    memo: dict[int, torch.nn.Module],
) -> torch.nn.Module:
    if id(module) in shared_module_ids or _is_frozen_parameter(prefix, frozen_prefixes):
        return module
    if id(module) in memo:
        return memo[id(module)]

    clone = copy.copy(module)
    memo[id(module)] = clone
    clone._parameters = {}
    for name, parameter in module._parameters.items():
        if parameter is None:
            clone._parameters[name] = None
            continue
        qualified_name = f"{prefix}.{name}" if prefix else name
        if _is_frozen_parameter(qualified_name, frozen_prefixes):
            clone._parameters[name] = parameter
            continue
        with torch.inference_mode(False):
            clone._parameters[name] = torch.nn.Parameter(
                parameter.detach().clone(),
                requires_grad=parameter.is_floating_point(),
            )

    clone._buffers = module._buffers.copy()
    clone._modules = {}
    for name, child in module._modules.items():
        qualified_name = f"{prefix}.{name}" if prefix else name
        clone._modules[name] = (
            None
            if child is None
            else _clone_module_tree(
                child,
                prefix=qualified_name,
                frozen_prefixes=frozen_prefixes,
                shared_module_ids=shared_module_ids,
                memo=memo,
            )
        )
    return clone


def _clone_training_model(model: torch.nn.Module) -> torch.nn.Module:
    """Clone trainable weights without copying serving attention runtimes."""
    frozen_prefixes = (
        "lm_head",
        "model.embed_tokens",
        "model.mask_embedding",
        "draft_id_to_target_id",
        *(
            prefix.strip()
            for prefix in os.environ.get("OSD_FROZEN_PARAMETER_PREFIXES", "").split(",")
            if prefix.strip()
        ),
    )
    shared_module_ids = set()
    for module_name in ("lm_head", "logits_processor"):
        module = getattr(model, module_name, None)
        if isinstance(module, torch.nn.Module):
            shared_module_ids.add(id(module))
    embed_tokens = getattr(getattr(model, "model", None), "embed_tokens", None)
    if isinstance(embed_tokens, torch.nn.Module):
        shared_module_ids.add(id(embed_tokens))
    for layer in getattr(getattr(model, "model", None), "layers", ()):
        runtime = getattr(getattr(layer, "self_attn", None), "attn", None)
        if isinstance(runtime, torch.nn.Module):
            shared_module_ids.add(id(runtime))
    training_model = _clone_module_tree(
        model,
        prefix="",
        frozen_prefixes=frozen_prefixes,
        shared_module_ids=shared_module_ids,
        memo={},
    )
    serving_parameters = dict(model.named_parameters())
    training_parameters = dict(training_model.named_parameters())
    if serving_parameters.keys() != training_parameters.keys():
        raise RuntimeError("DFlash training copy changed the parameter layout")
    for name, parameter in training_parameters.items():
        serving_parameter = serving_parameters[name]
        shared = parameter is serving_parameter
        if _is_frozen_parameter(name, frozen_prefixes):
            if not shared:
                raise RuntimeError(f"frozen DFlash weight was copied: {name}")
        elif shared:
            raise RuntimeError(f"trainable DFlash weight was not copied: {name}")
        elif parameter.is_floating_point() and not parameter.requires_grad:
            raise RuntimeError(f"DFlash training weight is frozen: {name}")
    return training_model


def _create_adapter(speculator) -> DflashOnlineAdapter:
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM

    vllm_config = speculator.vllm_config
    speculative = vllm_config.speculative_config
    if speculative is None or speculative.method != "dflash":
        raise ValueError("DFlash methods require DFlash speculative decoding")
    if "DFlash2DraftModel" in speculative.draft_model_config.architectures:
        raise NotImplementedError("DFlash2 selectors are not supported by TTS")
    _require_value(vllm_config.scheduler_config, "max_num_seqs", 1)
    if getattr(vllm_config.cache_config, "enable_prefix_caching", False):
        raise ValueError("DFlash methods require prefix caching to be disabled")
    if getattr(vllm_config.scheduler_config, "enable_chunked_prefill", False):
        raise ValueError("DFlash methods require chunked prefill to be disabled")
    for name in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
    ):
        _require_value(vllm_config.parallel_config, name, 1)
    if speculator.supports_mm_inputs:
        raise ValueError("DFlash methods do not support multimodal drafts")
    model = speculator.model
    if not isinstance(model, DFlashQwen3ForCausalLM) or "DFlash2" in (
        type(model).__name__
    ):
        raise NotImplementedError(
            "DFlash methods currently support the Qwen3 DFlash drafter; got "
            f"{type(model).__name__}"
        )
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    if getattr(model.model, "quant_config", None) is not None:
        raise ValueError("DFlash draft updates require an unquantized drafter")
    if not isinstance(model.lm_head.quant_method, UnquantizedEmbeddingMethod):
        raise ValueError("DFlash draft updates require an unquantized LM head")
    draft_dtype = model.lm_head.weight.dtype
    if draft_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            f"DFlash draft updates require BF16 or FP16; got {draft_dtype}"
        )

    method = os.environ["OSD_METHOD"]
    learning_rate = float(os.environ.get("OSD_LEARNING_RATE", "1e-5"))
    max_context_tokens = int(os.environ.get("OSD_DFLASH_MAX_CONTEXT_TOKENS", "0"))
    training_model = model
    if method == "tts":
        training_model = _clone_training_model(model)
    trainer = create_dflash_trainer(training_model, learning_rate=learning_rate)
    bridge = DflashPolicyBridge(
        trainer,
        model,
        method=method,
        max_context_tokens=max_context_tokens,
    )
    return DflashOnlineAdapter(
        bridge,
        model,
        num_speculative_steps=speculator.num_speculative_steps,
        max_context_tokens=max_context_tokens,
        max_model_len=vllm_config.model_config.max_model_len,
        dtype=vllm_config.model_config.dtype,
        device=speculator.device,
    )


def install() -> None:
    """Wrap the DFlash speculator and rejection sampler once per worker."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    original_load_model = DFlashSpeculator.load_model
    original_propose = DFlashSpeculator.propose
    original_rejection_call = RejectionSampler.__call__
    original_finish_requests = GPUModelRunner.finish_requests
    original_shutdown = GPUModelRunner.shutdown

    def load_gpu_model(self, target_model):
        global _ADAPTER
        result = original_load_model(self, target_model)
        _ADAPTER = _create_adapter(self)
        return result

    def propose(
        self,
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        num_tokens_across_dp=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        mm_inputs=None,
        is_profile=False,
    ):
        skip = dummy_run or is_profile or self.supports_mm_inputs
        if _ADAPTER is not None and not skip:
            _ADAPTER.before_propose(input_batch)
        result = original_propose(
            self,
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            num_tokens_across_dp,
            dummy_run,
            skip_attn_for_dummy_run,
            mm_inputs,
            is_profile,
        )
        if _ADAPTER is not None and not skip:
            _ADAPTER.capture_proposal(
                self,
                input_batch,
                aux_hidden_states=aux_hidden_states,
                last_hidden_states=last_hidden_states,
                num_rejected=num_rejected,
            )
        return result

    def rejection_call(self, logits, input_batch, *args, **kwargs):
        result = original_rejection_call(self, logits, input_batch, *args, **kwargs)
        if _ADAPTER is not None:
            _ADAPTER.capture_verification(logits, input_batch, result)
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

    DFlashSpeculator.load_model = load_gpu_model
    DFlashSpeculator.propose = propose
    RejectionSampler.__call__ = rejection_call
    GPUModelRunner.finish_requests = finish_requests
    GPUModelRunner.shutdown = shutdown
