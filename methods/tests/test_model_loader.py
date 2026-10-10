# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import tempfile
import unittest
from pathlib import Path

import torch
from online_draft.models.qwen3_eagle3 import (
    convert_angelslim_eagle3_state_dict,
)
from safetensors.torch import save_file as save_safetensors_file

from methods.utils.model_loader import load_cpu_training_model


def _make_source_state_dict() -> dict[str, torch.Tensor]:
    return {
        "d2t": torch.tensor([0, 1, 2, 3, 4, 4]),
        "t2d": torch.zeros(10, dtype=torch.bool),
        "midlayer.self_attn.q_proj.weight": torch.randn(4, 8),
        "midlayer.self_attn.k_proj.weight": torch.randn(2, 8),
        "midlayer.self_attn.v_proj.weight": torch.randn(2, 8),
        "midlayer.self_attn.o_proj.weight": torch.randn(4, 4),
        "midlayer.mlp.gate_proj.weight": torch.randn(8, 4),
        "midlayer.mlp.up_proj.weight": torch.randn(8, 4),
        "midlayer.mlp.down_proj.weight": torch.randn(4, 8),
        "midlayer.hidden_norm.weight": torch.randn(4),
        "midlayer.input_layernorm.weight": torch.randn(4),
        "midlayer.post_attention_layernorm.weight": torch.randn(4),
        "norm.weight": torch.randn(4),
        "fc.weight": torch.randn(4, 12),
        "lm_head.weight": torch.randn(6, 4),
    }


class CpuModelLoaderTest(unittest.TestCase):
    def test_loads_supported_checkpoint_formats_on_cpu(self) -> None:
        raw_config = {
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "num_hidden_layers": 1,
            "vocab_size": 10,
            "draft_vocab_size": 6,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
        }
        source_state_dict = _make_source_state_dict()
        expected_state_dict = convert_angelslim_eagle3_state_dict(source_state_dict)

        for checkpoint_format in ("pytorch", "safetensors"):
            with (
                self.subTest(checkpoint_format=checkpoint_format),
                tempfile.TemporaryDirectory() as directory,
            ):
                model_path = Path(directory)
                (model_path / "config.json").write_text(
                    json.dumps(raw_config),
                    encoding="utf-8",
                )
                if checkpoint_format == "pytorch":
                    torch.save(
                        source_state_dict,
                        model_path / "pytorch_model.bin",
                    )
                else:
                    save_safetensors_file(
                        source_state_dict,
                        model_path / "model.safetensors",
                    )

                model = load_cpu_training_model(model_path)
                loaded_state_dict = model.state_dict()

                self.assertEqual(
                    set(loaded_state_dict),
                    set(expected_state_dict),
                )
                self.assertTrue(
                    all(
                        tensor.device.type == "cpu"
                        for tensor in loaded_state_dict.values()
                    )
                )
                for name, expected_tensor in expected_state_dict.items():
                    torch.testing.assert_close(
                        loaded_state_dict[name],
                        expected_tensor,
                    )


if __name__ == "__main__":
    unittest.main()
