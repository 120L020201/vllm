# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class TtsStandaloneImportTest(unittest.TestCase):
    def test_tts_import_graph_does_not_load_online_draft(self) -> None:
        script = textwrap.dedent(
            """
            import importlib.abc
            import sys


            class BlockOnlineDraft(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "online_draft" or fullname.startswith(
                        "online_draft."
                    ):
                        raise ImportError(f"blocked dependency: {fullname}")
                    return None


            sys.meta_path.insert(0, BlockOnlineDraft())

            import methods.dflash.factory
            import methods.tts_common.batch
            import methods.tts_common.loss
            import methods.tts_common.qwen3_5_mtp
            import methods.tts_common.qwen3_mtp
            import methods.tts_common.step
            import methods.tts_common.window
            import methods.utils.gpu_factory
            import methods.utils.gpu_training

            loaded = sorted(
                name for name in sys.modules if name.startswith("online_draft")
            )
            if loaded:
                raise AssertionError(f"TTS loaded online_draft modules: {loaded}")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
