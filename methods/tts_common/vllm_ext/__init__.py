# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Out-of-tree vLLM support for Qwen3 checkpoints with embedded MTP layers.

vLLM has no built-in Qwen3 dense MTP draft. ``install()`` adds one without
editing vLLM: the draft model is registered through ``ModelRegistry`` and the
target loader skips ``mtp.*`` weights. Draft-config recognition has no public
hook, so it extends three module-level tables at runtime. Call ``install()``
in every process that builds a ``VllmConfig`` or loads models.
"""

from typing import Literal, get_args

_INSTALLED = False
_QWEN3_MTP = "qwen3_mtp"


def _qwen3_mtp_hf_config_override(hf_config):
    # Unpickled draft configs may call this before install() in a new process.
    install()
    hf_config = _BASE_HF_CONFIG_OVERRIDE(hf_config)
    if (
        hf_config.model_type == "qwen3"
        and hf_config.architectures[0] == "Qwen3ForCausalLM"
        and getattr(hf_config, "num_nextn_predict_layers", 0) > 0
    ):
        hf_config.model_type = _QWEN3_MTP
    if hf_config.model_type == _QWEN3_MTP:
        hf_config.update(
            {
                "n_predict": hf_config.num_nextn_predict_layers,
                "architectures": ["Qwen3MTP"],
            }
        )
    return hf_config


_BASE_HF_CONFIG_OVERRIDE = None


def install() -> None:
    """Register Qwen3 dense MTP with vLLM; idempotent."""
    global _INSTALLED, _BASE_HF_CONFIG_OVERRIDE
    if _INSTALLED:
        return
    _INSTALLED = True

    import vllm.config.speculative as speculative
    from vllm import ModelRegistry
    from vllm.transformers_utils import model_arch_config_convertor as convertors

    ModelRegistry.register_model(
        "Qwen3MTP", "methods.tts_common.vllm_ext.qwen3_mtp:Qwen3MTP"
    )
    ModelRegistry.register_model(
        "Qwen3ForCausalLM",
        "methods.tts_common.vllm_ext.qwen3_target:Qwen3ForCausalLM",
    )

    mtp_types = get_args(speculative.MTPModelTypes)
    if _QWEN3_MTP not in mtp_types:
        speculative.MTPModelTypes = Literal[(*mtp_types, _QWEN3_MTP)]
    convertors.MODEL_ARCH_CONFIG_CONVERTORS.setdefault(
        _QWEN3_MTP, convertors.Qwen3NextMTPModelArchConfigConvertor
    )
    _BASE_HF_CONFIG_OVERRIDE = speculative.SpeculativeConfig.hf_config_override
    speculative.SpeculativeConfig.hf_config_override = staticmethod(
        _qwen3_mtp_hf_config_override
    )
