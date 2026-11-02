# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Out-of-tree Qwen3 MTP support must work without vLLM source changes."""

import unittest
from typing import get_args

from transformers import PretrainedConfig


class Qwen3MTPExtensionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from methods.tts_common.vllm_ext import install

        install()

    def _override(self, **config):
        from vllm.config.speculative import SpeculativeConfig

        return SpeculativeConfig.hf_config_override(PretrainedConfig(**config))

    def test_embedded_mtp_layers_select_the_qwen3_mtp_draft(self):
        config = self._override(
            model_type="qwen3",
            architectures=["Qwen3ForCausalLM"],
            num_nextn_predict_layers=1,
        )

        self.assertEqual(config.model_type, "qwen3_mtp")
        self.assertEqual(config.architectures, ["Qwen3MTP"])
        self.assertEqual(config.n_predict, 1)

    def test_qwen3_without_mtp_layers_is_unchanged(self):
        config = self._override(model_type="qwen3", architectures=["Qwen3ForCausalLM"])

        self.assertEqual(config.model_type, "qwen3")
        self.assertEqual(config.architectures, ["Qwen3ForCausalLM"])

    def test_draft_detection_and_layer_count_recognize_qwen3_mtp(self):
        import vllm.config.speculative as speculative
        from vllm.transformers_utils.model_arch_config_convertor import (
            MODEL_ARCH_CONFIG_CONVERTORS,
            Qwen3NextMTPModelArchConfigConvertor,
        )

        self.assertIn("qwen3_mtp", get_args(speculative.MTPModelTypes))
        self.assertIs(
            MODEL_ARCH_CONFIG_CONVERTORS["qwen3_mtp"],
            Qwen3NextMTPModelArchConfigConvertor,
        )

    def test_models_resolve_to_out_of_tree_classes(self):
        from methods.tts_common.vllm_ext.qwen3_mtp import Qwen3MTP
        from methods.tts_common.vllm_ext.qwen3_target import Qwen3ForCausalLM
        from vllm import ModelRegistry

        self.assertIs(ModelRegistry._try_load_model_cls("Qwen3MTP"), Qwen3MTP)
        self.assertIs(
            ModelRegistry._try_load_model_cls("Qwen3ForCausalLM"),
            Qwen3ForCausalLM,
        )

    def test_target_loader_drops_embedded_mtp_weights(self):
        from unittest.mock import patch

        from methods.tts_common.vllm_ext.qwen3_target import Qwen3ForCausalLM
        from vllm.model_executor.models.qwen3 import (
            Qwen3ForCausalLM as BaseQwen3ForCausalLM,
        )

        seen = []
        with patch.object(
            BaseQwen3ForCausalLM,
            "load_weights",
            lambda self, weights: seen.extend(name for name, _ in weights) or set(),
        ):
            Qwen3ForCausalLM.load_weights(
                object.__new__(Qwen3ForCausalLM),
                [("model.embed_tokens.weight", None), ("mtp.fc.weight", None)],
            )

        self.assertEqual(seen, ["model.embed_tokens.weight"])


if __name__ == "__main__":
    unittest.main()
