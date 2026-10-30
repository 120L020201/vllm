# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for Qwen3 MTP speculative decoding configuration."""

from transformers import PretrainedConfig

from vllm.config.speculative import SpeculativeConfig


def test_qwen3_mtp_override_uses_embedded_predictor_layers():
    cfg = SpeculativeConfig.hf_config_override(
        PretrainedConfig(
            model_type="qwen3",
            architectures=["Qwen3ForCausalLM"],
            num_nextn_predict_layers=1,
        )
    )

    assert cfg.model_type == "qwen3_mtp"
    assert cfg.architectures == ["Qwen3MTP"]
    assert cfg.n_predict == 1


def test_qwen3_without_mtp_layers_is_not_converted():
    cfg = SpeculativeConfig.hf_config_override(
        PretrainedConfig(
            model_type="qwen3",
            architectures=["Qwen3ForCausalLM"],
        )
    )

    assert cfg.model_type == "qwen3"
    assert cfg.architectures == ["Qwen3ForCausalLM"]
