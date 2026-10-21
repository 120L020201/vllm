# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import psutil
import torch
from online_draft.models.qwen3_eagle3 import (
    Qwen3Eagle3Config,
    Qwen3Eagle3ForCausalLM,
)
from online_draft.training.eagle3_batch import Eagle3DistillationBatch
from online_draft.training.trainer import DraftTrainer, TrainerConfig

from experiments import prepare_datasets as dataset_preparation
from experiments import prepare_models as model_preparation
from experiments.benchmark import (
    _acceptance_length,
    _completion_tokens,
    _fetch_spec_metrics,
    _output_token_limit,
    _parse_benchmark_specs,
    _read_worker_stats,
    _request_body,
    _select_benchmark_records,
    _training_metrics,
    _wait_for_worker,
)
from experiments.prepare_datasets import _download_dataset_file, _line_count
from experiments.prepare_models import MODEL_SPECS, _validate_model
from experiments.server_guard import (
    _owns_listener,
    check_port_available,
    server_url,
    wait_for_server,
)
from methods.prepare import (
    DATASETS,
    MODELS,
    DatasetSpec,
    ModelSpec,
    prepare_datasets,
    prepare_models,
)
from methods.prepare import (
    main as prepare_main,
)
from methods.run import (
    _capped_gpu_memory_utilization,
    build_parser,
    configure,
)
from methods.utils.cpu_stats import attach_cpu_stats
from methods.utils.gpu_factory import (
    _is_intermediate_prefill_chunk,
    _request_ids_to_reset,
)
from methods.utils.stats import WorkerStats
from vllm.config import AttentionConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _make_trainer(learning_rate=1e-2, device="cpu"):
    model = Qwen3Eagle3ForCausalLM(
        Qwen3Eagle3Config(
            hidden_size=4,
            intermediate_size=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=2,
            num_hidden_layers=1,
            target_vocab_size=8,
            draft_vocab_size=5,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            num_aux_hidden_states=2,
        )
    )
    model.to(device=device)
    return DraftTrainer(model, TrainerConfig(learning_rate=learning_rate))


def _make_batch(anchor_position=2, device="cpu"):
    hidden_size = 4
    auxiliary_size = 8
    draft_length = 2
    return Eagle3DistillationBatch(
        prefill_positions=torch.arange(
            anchor_position + 1,
            dtype=torch.long,
            device=device,
        ),
        prefill_input_embeds=torch.randn(
            anchor_position + 1,
            hidden_size,
            device=device,
        ),
        prefill_aux_hidden_states=torch.randn(
            anchor_position + 1,
            auxiliary_size,
            device=device,
        ),
        proposal_positions=torch.arange(
            anchor_position,
            anchor_position + draft_length,
            dtype=torch.long,
            device=device,
        ),
        draft_token_input_embeds=torch.randn(
            draft_length - 1,
            hidden_size,
            device=device,
        ),
        draft_recurrent_hidden_states=torch.randn(
            draft_length - 1,
            hidden_size,
            device=device,
        ),
        teacher_probabilities=torch.randn(
            draft_length,
            5,
            device=device,
        ).softmax(dim=-1),
        confirmed_positions=torch.tensor([anchor_position + 1], device=device),
        confirmed_input_embeds=torch.randn(1, hidden_size, device=device),
        confirmed_aux_hidden_states=torch.randn(1, auxiliary_size, device=device),
        rejection_position=0,
    )


class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(
            patch.dict(
                os.environ,
                {"VLLM_ONLINE_DRAFT_TRAIN": "", "VLLM_ONLINE_DRAFT_PROFILE": "0"},
            )
        )

    @staticmethod
    def _online_draft_config(command):
        value = command[command.index("--additional-config") + 1]
        return json.loads(value)["online_draft"]

    def test_same_inference_args_for_every_method(self):
        commands = []
        for method in ("eagle", "online_draft", "tts", "random_sampling", "ospec"):
            args = build_parser().parse_args((method, "--dry-run"))
            command, env = configure(args)
            config_index = command.index("--additional-config")
            commands.append(command[:config_index] + command[config_index + 2 :])
            self.assertEqual(env["OSD_METHOD"], method)
            self.assertEqual(env["VLLM_USE_V2_MODEL_RUNNER"], "1")
            self.assertEqual(
                env["VLLM_ONLINE_DRAFT_TRAIN"], str(int(method == "online_draft"))
            )
            self.assertEqual(env["OSD_LOG_UPDATES"], str(int(method == "online_draft")))
            self.assertIn("--max-num-seqs", command)
            self.assertIn("1", command)
            self.assertIn("-cc.cudagraph_mode=FULL_AND_PIECEWISE", command)
            self.assertNotIn("--enforce-eager", command)
            self.assertIn("--no-enable-chunked-prefill", command)
            self.assertIn("--trust-remote-code", command)
            self.assertNotIn("--torch-threads", command)
            self.assertEqual(command[command.index("--seed") + 1], "0")
            self.assertEqual(
                command[command.index("--max-num-batched-tokens") + 1],
                "32768",
            )
            self.assertEqual(
                command[command.index("--gpu-memory-utilization") + 1],
                "0.8",
            )
        self.assertTrue(all(command == commands[0] for command in commands))

    def test_conflicting_cpu_training_environment_is_rejected(self):
        for method in ("eagle", "online_draft", "tts", "random_sampling", "ospec"):
            with (
                self.subTest(method=method),
                patch.dict(
                    os.environ,
                    {"VLLM_ONLINE_DRAFT_TRAIN": str(int(method != "online_draft"))},
                ),
            ):
                args = build_parser().parse_args((method, "--dry-run"))
                with self.assertRaisesRegex(ValueError, "conflicts"):
                    configure(args)

    def test_gpu_attention_is_explicit_for_target_and_draft(self):
        parser = make_arg_parser(FlexibleArgumentParser())
        for flags, backend, autotune in (
            ((), "FLASHINFER", False),
            (
                ("--gpu-attention-backend", "FLASH_ATTN", "--flashinfer-autotune"),
                "FLASH_ATTN",
                True,
            ),
        ):
            with self.subTest(backend=backend, autotune=autotune):
                args = build_parser().parse_args(("eagle", "--dry-run", *flags))
                command, _ = configure(args)
                engine_args = EngineArgs.from_cli_args(
                    parser.parse_args(["--model", command[4], *command[5:]])
                )

                self.assertEqual(
                    AttentionConfig(backend=engine_args.attention_backend).backend,
                    AttentionBackendEnum[backend],
                )
                self.assertEqual(
                    engine_args.speculative_config["attention_backend"], backend
                )
                self.assertEqual(engine_args.enable_flashinfer_autotune, autotune)
                self.assertEqual(
                    self._online_draft_config(command)["attention_backend"],
                    "flash_attention",
                )

    def test_cpu_online_trainer_defaults_match_artifact_configuration(self):
        args = build_parser().parse_args(("online_draft", "--dry-run"))
        with patch.dict(os.environ):
            os.environ.pop("OMP_NUM_THREADS", None)
            os.environ.pop("MKL_NUM_THREADS", None)
            command, env = configure(args)

        self.assertEqual(
            self._online_draft_config(command),
            {
                "attention_backend": "flash_attention",
                "learning_rate": 1e-5,
                "fused_optimizer": True,
                "frozen_parameter_prefixes": [],
                "check_gradients": False,
                "log_updates": True,
            },
        )
        self.assertEqual(env["OMP_NUM_THREADS"], "20")
        self.assertEqual(env["MKL_NUM_THREADS"], "20")

    def test_cpu_online_trainer_options_are_configurable(self):
        args = build_parser().parse_args(
            (
                "online_draft",
                "--dry-run",
                "--check-gradients",
                "--no-fused-optimizer",
                "--no-log-updates",
                "--frozen-parameter-prefixes",
                "model.embed_tokens, lm_head",
            )
        )
        with patch.dict(os.environ, {"OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "4"}):
            command, env = configure(args)
        config = self._online_draft_config(command)

        self.assertTrue(config["check_gradients"])
        self.assertFalse(config["fused_optimizer"])
        self.assertFalse(config["log_updates"])
        self.assertEqual(env["OMP_NUM_THREADS"], "8")
        self.assertEqual(env["MKL_NUM_THREADS"], "4")
        self.assertEqual(
            config["frozen_parameter_prefixes"],
            ["model.embed_tokens", "lm_head"],
        )

    def test_invalid_options_fail_before_start(self):
        for flags in (
            ("--probability", "1.1"),
            ("--chunk-size", "0"),
            ("--ensemble-lrs", "1e-5,2e-5"),
        ):
            args = build_parser().parse_args(("tts", "--dry-run", *flags))
            with self.assertRaises(ValueError):
                configure(args)

    def test_model_size_selects_matching_target_and_draft(self):
        args = build_parser().parse_args(("eagle", "--model-size", "4b", "--dry-run"))
        command, env = configure(args)

        self.assertTrue(any(value.endswith("qwen3-4b") for value in command))
        self.assertTrue(any(value.endswith("qwen3-4b-eagle3") for value in command))
        self.assertTrue(env["OSD_DRAFT_MODEL"].endswith("-eagle3"))

    def test_gpu_memory_is_capped_at_24_gib(self):
        properties = SimpleNamespace(total_memory=48 * 2**30)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties", return_value=properties),
        ):
            utilization = _capped_gpu_memory_utilization(
                0.9,
                24.0,
                require_gpu=True,
            )

        self.assertEqual(utilization, 22 / 48)

    def test_memory_limit_is_only_applied_when_explicit(self):
        properties = SimpleNamespace(total_memory=48 * 2**30)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties", return_value=properties),
        ):
            default_args = build_parser().parse_args(
                ("eagle", "--model-size", "4b", "--dry-run")
            )
            capped_args = build_parser().parse_args(
                (
                    "eagle",
                    "--model-size",
                    "8b",
                    "--max-gpu-memory-gib",
                    "32",
                    "--dry-run",
                )
            )
            default_command, _ = configure(default_args)
            capped_command, _ = configure(capped_args)

        default_index = default_command.index("--gpu-memory-utilization") + 1
        capped_index = capped_command.index("--gpu-memory-utilization") + 1
        self.assertEqual(float(default_command[default_index]), 0.8)
        self.assertEqual(float(capped_command[capped_index]), 30 / 48)

    def test_warmup_with_no_preemptions_has_no_resets(self):
        scheduler_output = SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=None,
        )

        self.assertEqual(_request_ids_to_reset(scheduler_output), set())

    def test_only_intermediate_prefill_chunks_defer_training_capture(self):
        intermediate = SimpleNamespace(
            has_prefill=True,
            num_scheduled_tokens=torch.tensor([512]),
            num_computed_prefill_tokens_np=torch.tensor([0]),
            prefill_len_np=torch.tensor([700]),
        )
        final = SimpleNamespace(
            has_prefill=True,
            num_scheduled_tokens=torch.tensor([200]),
            num_computed_prefill_tokens_np=torch.tensor([500]),
            prefill_len_np=torch.tensor([700]),
        )
        exact_final = SimpleNamespace(
            has_prefill=True,
            num_scheduled_tokens=torch.tensor([512]),
            num_computed_prefill_tokens_np=torch.tensor([0]),
            prefill_len_np=torch.tensor([512]),
        )
        decode = SimpleNamespace(
            has_prefill=False,
            num_scheduled_tokens=torch.tensor([512]),
            num_computed_prefill_tokens_np=torch.tensor([700]),
            prefill_len_np=torch.tensor([700]),
        )

        self.assertTrue(_is_intermediate_prefill_chunk(intermediate))
        self.assertFalse(_is_intermediate_prefill_chunk(final))
        self.assertFalse(_is_intermediate_prefill_chunk(exact_final))
        self.assertFalse(_is_intermediate_prefill_chunk(decode))


class RunScriptTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[2]
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.env = {
            "PATH": os.defpath,
            "PYTHON_BIN": sys.executable,
            "MODEL": str(self.directory),
            "SPEC_MODEL": str(self.directory),
            "DATA_DIR": str(self.directory),
            "OUTPUT_DIR": str(self.directory / "results"),
            "BENCHMARKS": "aime2026[14]",
            "DRY_RUN": "1",
        }

    def _run(self, method, **overrides):
        return subprocess.run(
            ["bash", str(self.root / "experiments" / "run.sh"), method],
            env={**self.env, **overrides},
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_method_selects_training_and_resume_defaults(self):
        for method in ("base", "eagle3", "online_draft", "tts", "ospec"):
            with self.subTest(method=method):
                result = self._run(method)
                resume = "question" if method in ("base", "eagle3") else "none"
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"--resume-mode {resume}", result.stdout)
                self.assertIn(
                    f"training: cpu_online={int(method == 'online_draft')}",
                    result.stdout,
                )
                if method == "online_draft":
                    self.assertIn("-m methods.run online_draft", result.stdout)
                    self.assertIn("--no-check-gradients", result.stdout)
                    self.assertIn("--log-updates", result.stdout)
                elif method in ("tts", "ospec"):
                    self.assertIn("--no-log-updates", result.stdout)

    def test_training_resume_is_rejected_before_creating_outputs(self):
        for method in ("online_draft", "tts", "ospec"):
            with self.subTest(method=method):
                result = self._run(method, RESUME="1")
                self.assertEqual(result.returncode, 2)
                self.assertIn("does not support RESUME=1", result.stderr)
                self.assertFalse((self.directory / "results").exists())

    def test_occupied_port_is_rejected_before_changing_existing_outputs(self):
        files = {
            name: self.directory / name.lower()
            for name in ("RUN_LOG", "SERVER_LOG", "STATS_FILE", "OUTPUT_FILE")
        }
        for path in files.values():
            path.write_text("keep existing run\n", encoding="utf-8")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            result = self._run(
                "base",
                DRY_RUN="0",
                RESUME="0",
                HOST="127.0.0.1",
                PORT=str(listener.getsockname()[1]),
                **{name: str(path) for name, path in files.items()},
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("cannot bind", result.stderr)
            self.assertTrue(
                listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN)
            )
        self.assertFalse((self.directory / "results").exists())
        for path in files.values():
            self.assertEqual(path.read_text(encoding="utf-8"), "keep existing run\n")

    def test_readiness_accepts_a_real_server_owned_by_the_launched_process(self):
        program = """\
from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

server = HTTPServer(("127.0.0.1", 0), Handler)
print(server.server_port, flush=True)
server.serve_forever()
"""
        server = subprocess.Popen(
            [sys.executable, "-c", program],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.addCleanup(server.communicate, timeout=5)
        self.addCleanup(server.terminate)
        port = server.stdout.readline().strip()
        result = subprocess.run(
            [
                sys.executable,
                str(self.root / "experiments" / "server_guard.py"),
                "wait",
                "--host",
                "127.0.0.1",
                "--port",
                port,
                "--pid",
                str(server.pid),
                "--timeout",
                "2",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_stats_path_is_exported_to_every_training_method(self):
        path = str(self.directory / "stats.json")
        for method in ("online_draft", "tts", "ospec"):
            with self.subTest(method=method):
                result = subprocess.run(
                    [
                        "bash",
                        "-c",
                        'trap \'printenv OSD_STATS_FILE\' EXIT; source "$1" "$2"',
                        "test",
                        str(self.root / "experiments" / "run.sh"),
                        method,
                    ],
                    env={**self.env, "STATS_FILE": path},
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines()[-1], path)

    def test_inference_resume_remains_configurable(self):
        for method in ("base", "eagle3"):
            for resume, mode in (("0", "none"), ("1", "question")):
                with self.subTest(method=method, resume=resume):
                    result = self._run(method, RESUME=resume)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(f"--resume-mode {mode}", result.stdout)

    def test_training_flag_cannot_override_method(self):
        for method in ("base", "eagle3", "online_draft", "tts", "ospec"):
            with self.subTest(method=method):
                result = self._run(
                    method,
                    VLLM_ONLINE_DRAFT_TRAIN=str(int(method != "online_draft")),
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("online_draft", result.stderr)
                self.assertFalse((self.directory / "results").exists())

    def test_profiling_requires_cpu_online_method(self):
        for method in ("base", "eagle3", "online_draft", "tts", "ospec"):
            with self.subTest(method=method):
                result = self._run(method, VLLM_ONLINE_DRAFT_PROFILE="1")
                if method == "online_draft":
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("cpu_online=1 profile=1", result.stdout)
                else:
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("requires the online_draft entry", result.stderr)


class ServerGuardTest(unittest.TestCase):
    def setUp(self):
        self.process = MagicMock(pid=123)
        self.process.is_running.return_value = True
        self.process.status.return_value = psutil.STATUS_RUNNING
        self.enterContext(
            patch("experiments.server_guard.psutil.Process", return_value=self.process)
        )
        self.opener = MagicMock()
        response = self.opener.open.return_value.__enter__.return_value
        response.status = 200
        response.geturl.return_value = "http://127.0.0.1:8000/health"
        self.enterContext(
            patch(
                "experiments.server_guard.urllib.request.build_opener",
                return_value=self.opener,
            )
        )

    def test_another_healthy_server_cannot_satisfy_readiness(self):
        with (
            patch("experiments.server_guard._owns_listener", return_value=False),
            patch("experiments.server_guard.time.monotonic", side_effect=[0, 0, 1, 2]),
            patch("experiments.server_guard.time.sleep"),
            self.assertRaises(TimeoutError),
        ):
            wait_for_server("127.0.0.1", 8000, 123, timeout=1)
        self.opener.open.assert_not_called()

    def test_exited_launcher_is_rejected_before_health_probe(self):
        self.process.is_running.return_value = False
        with self.assertRaisesRegex(RuntimeError, "exited"):
            wait_for_server("127.0.0.1", 8000, 123)
        self.opener.open.assert_not_called()

    def test_process_exit_during_successful_health_probe_is_rejected(self):
        self.process.is_running.side_effect = [True, False]
        with (
            patch("experiments.server_guard._owns_listener", return_value=True),
            self.assertRaisesRegex(RuntimeError, "exited"),
        ):
            wait_for_server("127.0.0.1", 8000, 123)

    def test_ready_owned_server_uses_a_bounded_health_probe(self):
        with patch(
            "experiments.server_guard._owns_listener", return_value=True
        ) as owned:
            wait_for_server("127.0.0.1", 8000, 123)
        self.opener.open.assert_called_once_with(
            "http://127.0.0.1:8000/health", timeout=2.0
        )
        self.assertEqual(owned.call_count, 2)

    def test_listener_must_match_address_and_port_in_the_process_tree(self):
        self.process.net_connections.return_value = []
        child = MagicMock()
        self.process.children.return_value = [child]
        child.net_connections.return_value = [
            SimpleNamespace(
                status=psutil.CONN_LISTEN,
                laddr=SimpleNamespace(ip="127.0.0.1", port=8000),
            )
        ]
        self.assertTrue(_owns_listener(self.process, {"127.0.0.1"}, 8000))
        self.assertFalse(_owns_listener(self.process, {"127.0.0.2"}, 8000))
        self.assertFalse(_owns_listener(self.process, {"127.0.0.1"}, 8001))

    def test_bind_probe_rejects_a_listener_and_accepts_a_released_port(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            listener.listen()
            with self.assertRaisesRegex(RuntimeError, "cannot bind"):
                check_port_available("127.0.0.1", port)
        check_port_available("127.0.0.1", port)

    def test_wildcard_listener_uses_a_local_client_address(self):
        self.assertEqual(server_url("0.0.0.0", 8000), "http://127.0.0.1:8000")
        self.assertEqual(server_url("::", 8000), "http://[::1]:8000")
        self.assertEqual(server_url("::1", 8000), "http://[::1]:8000")


class PreparationTest(unittest.TestCase):
    def _prepare_script(self, *args, **env):
        root = Path(__file__).resolve().parents[2]
        return subprocess.run(
            ["bash", str(root / "experiments" / "prepare.sh"), *args],
            env={"PATH": os.defpath, "PYTHON_BIN": sys.executable, **env},
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_unselected_assets_do_not_start_python_or_create_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            for selection in ({}, {"DATASETS": "", "MODEL_SIZES": " \t"}):
                with self.subTest(selection=selection):
                    result = self._prepare_script(
                        PYTHON_BIN=f"{directory}/missing-python",
                        DATA_DIR=f"{directory}/datasets",
                        MODEL_DIR=f"{directory}/models",
                        **selection,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("nothing selected", result.stdout)
                    self.assertEqual(list(Path(directory).iterdir()), [])

    def test_only_explicitly_selected_asset_classes_are_scheduled(self):
        cases = (
            ((), {"DATASETS": "aime2026"}, True, False),
            ((), {"MODEL_SIZES": "8b"}, False, True),
            ((), {"DATASETS": "aime2026", "MODEL_SIZES": "8b"}, True, True),
            (("--dataset", "aime2026"), {}, True, False),
            (("--model-size", "8b"), {}, False, True),
            (("--dataset", "aime2026", "--model-size", "8b"), {}, True, True),
        )
        for args, env, datasets, models in cases:
            with self.subTest(args=args, env=env):
                result = self._prepare_script(*args, DRY_RUN="1", **env)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual("datasets:" in result.stdout, datasets)
                self.assertEqual("models:" in result.stdout, models)
                if datasets:
                    self.assertIn("--dataset aime2026", result.stdout)
                if models:
                    self.assertIn("--model-size 8b", result.stdout)

    def test_repeated_cli_selections_override_only_the_matching_environment(self):
        result = self._prepare_script(
            "--dataset",
            "aime2026",
            "--dataset",
            "gpqa_diamond",
            DATASETS="LongBench-v2",
            MODEL_SIZES="8b",
            DRY_RUN="1",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--dataset aime2026 --dataset gpqa_diamond", result.stdout)
        self.assertNotIn("LongBench-v2", result.stdout)
        self.assertIn("--model-size 8b", result.stdout)

    def test_invalid_prepare_arguments_fail_before_downloading(self):
        for args in (("--unknown",), ("--dataset",), ("--model-size", "--dataset")):
            with self.subTest(args=args):
                result = self._prepare_script(
                    *args, PYTHON_BIN="/missing-python", DATASETS="aime2026"
                )
                self.assertEqual(result.returncode, 2)
                self.assertNotIn("PYTHON_BIN", result.stderr)
                self.assertNotIn("starting", result.stdout)

    def test_python_downloaders_require_explicit_selections(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("builtins.print"),
            patch.object(dataset_preparation, "_load_dataset_rows") as load_rows,
            patch.object(dataset_preparation, "_write_dataset") as write_dataset,
            patch.object(model_preparation, "_prepare_model") as prepare_model,
        ):
            output = Path(directory) / "assets"
            argv = ["prepare", "--output-dir", str(output)]
            with patch.object(sys, "argv", argv):
                dataset_preparation.main()
                model_preparation.main()
            load_rows.assert_not_called()
            write_dataset.assert_not_called()
            prepare_model.assert_not_called()
            self.assertFalse(output.exists())

    def test_python_downloaders_prepare_only_requested_assets(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(dataset_preparation, "_load_dataset_rows") as load_rows,
            patch.object(dataset_preparation, "_write_dataset"),
            patch.object(model_preparation, "_prepare_model") as prepare_model,
        ):
            argv = ["prepare", "--output-dir", directory]
            with patch.object(sys, "argv", [*argv, "--dataset", "aime2026"]):
                dataset_preparation.main()
            with patch.object(sys, "argv", [*argv, "--model-size", "8b"]):
                model_preparation.main()
            self.assertEqual(
                [call.args[0] for call in load_rows.call_args_list], ["aime2026"]
            )
            self.assertEqual(
                [call.args[1] for call in prepare_model.call_args_list],
                list(MODEL_SPECS["8b"]),
            )

    def test_manifest_contains_every_requested_artifact(self):
        self.assertEqual(
            {spec.name for spec in DATASETS},
            {
                "aime2026",
                "gpqa_diamond",
                "mmlu_pro",
                "computer_science",
                "livecodebench_lite",
                "longbench_v2",
            },
        )
        self.assertEqual(
            {(spec.size, spec.draft) for spec in MODELS},
            {("4b", False), ("4b", True), ("8b", False), ("8b", True)},
        )

    def test_dry_run_does_not_import_download_dependencies(self):
        with patch("builtins.print") as print_mock:
            prepare_main(("all", "--dry-run"))

        output = "\n".join(call.args[0] for call in print_mock.call_args_list)
        self.assertIn("dataset aime2026", output)
        self.assertIn("model qwen3-8b-eagle3", output)

    def test_computer_science_is_filtered_and_written_as_jsonl(self):
        class FakeDataset:
            def __init__(self, rows):
                self.rows = rows

            def __len__(self):
                return len(self.rows)

            def filter(self, predicate):
                return FakeDataset([row for row in self.rows if predicate(row)])

            def to_json(self, path, **kwargs):
                del kwargs
                Path(path).write_text(
                    "".join(f"{row!r}\n" for row in self.rows),
                    encoding="utf-8",
                )

        calls = []

        def load_dataset(**kwargs):
            calls.append(kwargs)
            return FakeDataset(
                [
                    {"category": "computer science", "question": "kept"},
                    {"category": "physics", "question": "removed"},
                ]
            )

        spec = DatasetSpec(
            name="computer_science",
            repo_id="example/mmlu",
            revision="abc",
            split="test",
            category="computer science",
        )
        fake_module = SimpleNamespace(load_dataset=load_dataset)
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(sys.modules, {"datasets": fake_module}):
                prepare_datasets(Path(directory), (spec,))
            output = Path(directory) / "datasets" / spec.name
            manifest = (output / "manifest.json").read_text(encoding="utf-8")
            data = (output / "data.jsonl").read_text(encoding="utf-8")

        self.assertEqual(calls[0]["revision"], "abc")
        self.assertIn('"rows": 1', manifest)
        self.assertIn("kept", data)
        self.assertNotIn("removed", data)

    def test_model_download_validates_and_records_revision(self):
        calls = []

        def snapshot_download(**kwargs):
            calls.append(kwargs)
            destination = Path(kwargs["local_dir"])
            destination.mkdir(parents=True)
            (destination / "config.json").write_text("{}", encoding="utf-8")
            (destination / "model.safetensors").write_bytes(b"weights")

        spec = ModelSpec(
            name="qwen3-test",
            repo_id="example/model",
            revision="def",
            size="4b",
        )
        fake_module = SimpleNamespace(snapshot_download=snapshot_download)
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(sys.modules, {"huggingface_hub": fake_module}):
                prepare_models(Path(directory), (spec,), max_workers=2)
            output = Path(directory) / "models" / spec.name
            manifest = (output / "manifest.json").read_text(encoding="utf-8")

        self.assertEqual(calls[0]["revision"], "def")
        self.assertEqual(calls[0]["max_workers"], 2)
        self.assertIn('"repo_id": "example/model"', manifest)

    def test_complete_flat_dataset_is_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.jsonl"
            path.write_text('{"id": 1}\n\n{"id": 2}\n', encoding="utf-8")

            self.assertEqual(_line_count(path), 2)

    def test_dataset_download_uses_explicit_file_and_endpoint(self):
        response = MagicMock()
        response.status = 200
        response.read.side_effect = [b'{"id": 1}\n', b""]
        context = MagicMock()
        context.__enter__.return_value = response
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("experiments.prepare_datasets.get_token", return_value=None),
            patch("urllib.request.urlopen", return_value=context) as urlopen,
        ):
            path = _download_dataset_file(
                "https://huggingface.co",
                "math-ai/aime26",
                "revision",
                "aime2026.jsonl",
                Path(directory),
            )
            request = urlopen.call_args.args[0]

            self.assertEqual(path.read_bytes(), b'{"id": 1}\n')
            self.assertEqual(
                request.full_url,
                "https://huggingface.co/datasets/math-ai/aime26/"
                "resolve/revision/aime2026.jsonl",
            )

    def test_existing_qwen_model_is_validated_without_download(self):
        spec = MODEL_SPECS["4b"][0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(
                json.dumps({"hidden_size": spec.hidden_size}),
                encoding="utf-8",
            )
            (path / "model.safetensors").write_bytes(b"weights")

            _validate_model(path, spec)


class BenchmarkLoggingTest(unittest.TestCase):
    def test_parentheses_select_a_seeded_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aime2026.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps({"id": index, "prompt": f"Question {index}"})
                    for index in range(1, 13)
                )
                + "\n",
                encoding="utf-8",
            )
            spec = _parse_benchmark_specs("aime2026(3)")[0]

            first = _select_benchmark_records(path, spec, seed=7)
            repeated = _select_benchmark_records(path, spec, seed=7)

        self.assertEqual(len(first), 3)
        self.assertEqual(first, repeated)

    def test_brackets_select_one_based_question_number(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aime2026.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps({"id": index, "prompt": f"Question {index}"})
                    for index in range(1, 13)
                )
                + "\n",
                encoding="utf-8",
            )
            spec = _parse_benchmark_specs("aime2026[10]")[0]

            selected = _select_benchmark_records(path, spec, seed=7)

        self.assertEqual(
            selected,
            [(10, {"id": 10, "prompt": "Question 10"})],
        )

    def test_warmup_overrides_only_the_output_token_limit(self):
        args = SimpleNamespace(
            model="model",
            enable_thinking=True,
            max_output_tokens=1024,
            max_prompt_tokens=2048,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repetition_penalty=1.0,
            stop_token_ids="151645,151643",
            seed=0,
        )

        body = _request_body(args, "Warm up.", max_tokens=16)

        self.assertEqual(body["max_tokens"], 16)
        self.assertEqual(body["messages"][0]["content"], "Warm up.")
        self.assertEqual(body["presence_penalty"], 0.0)
        self.assertEqual(body["stop_token_ids"], [151645, 151643])

    def test_output_limit_uses_remaining_full_context_by_default(self):
        self.assertEqual(_output_token_limit(32768, 768, 0), 32000)
        self.assertEqual(_output_token_limit(32768, 768, 16384), 16384)
        self.assertEqual(_output_token_limit(32768, 9000, 0, 8192), 24576)
        with self.assertRaises(ValueError):
            _output_token_limit(32768, 32768, 0)

    def test_acceptance_length_uses_counter_delta(self):
        before = {"drafts": 10, "draft_tokens": 70, "accepted_tokens": 20}
        after = {"drafts": 18, "draft_tokens": 126, "accepted_tokens": 35}

        self.assertEqual(_acceptance_length(before, after), 2.875)
        self.assertEqual(_acceptance_length(None, None), 1.0)

    def test_completion_tokens_uses_api_usage(self):
        response = {"usage": {"completion_tokens": 24987}}

        self.assertEqual(_completion_tokens(response), 24987)

    def test_worker_stats_are_written_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            with patch.dict(os.environ, {"OSD_STATS_FILE": str(path)}):
                stats = WorkerStats()
                stats.begin_request()
                stats.enqueue()
                stats.complete(3)
                stats.record_weight_install()
                stats.record_cpu_train(4)
                stats.end_request()
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload,
            {
                "active_requests": 0,
                "pending": 0,
                "updates": 3,
                "weight_installs": 1,
                "cpu_train_calls": 1,
                "cpu_train_rounds": 4,
            },
        )

    def test_missing_or_invalid_stats_do_not_report_zero_installs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            self.assertIsNone(_read_worker_stats(path))
            for payload in ("{", "{}", '{"updates": 3}'):
                path.write_text(payload, encoding="utf-8")
                self.assertIsNone(_read_worker_stats(path))
        self.assertEqual(_training_metrics(None, None, "tts"), (None, None))
        self.assertEqual(_training_metrics(None, None, "base"), (0, None))

    def test_training_metrics_use_completed_request_deltas(self):
        before = {
            "active_requests": 0,
            "pending": 0,
            "weight_installs": 3,
            "cpu_train_calls": 4,
            "cpu_train_rounds": 10,
        }
        after = {
            **before,
            "weight_installs": 5,
            "cpu_train_calls": 7,
            "cpu_train_rounds": 22,
        }
        self.assertEqual(_training_metrics(before, after, "online_draft"), (2, 4.0))
        self.assertEqual(_training_metrics(before, before, "online_draft"), (0, None))
        self.assertEqual(_training_metrics(after, before, "online_draft"), (None, None))

    def test_busy_worker_timeout_is_not_a_complete_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            with patch.dict(os.environ, {"OSD_STATS_FILE": str(path)}):
                stats = WorkerStats()
                stats.begin_request()
                self.assertIsNone(_wait_for_worker(path, timeout=0))
                stats.end_request()
                self.assertEqual(
                    _wait_for_worker(path, timeout=0)["weight_installs"], 0
                )

    def test_metrics_parser_ignores_per_position_counters(self):
        metrics = """\
vllm:spec_decode_num_drafts_total{engine="0"} 8
vllm:spec_decode_num_draft_tokens_total{engine="0"} 56
vllm:spec_decode_num_accepted_tokens_total{engine="0"} 15
vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 7
"""
        response = MagicMock()
        response.__enter__.return_value.read.return_value = metrics.encode()

        with patch("urllib.request.urlopen", return_value=response):
            parsed = _fetch_spec_metrics("http://example")

        self.assertEqual(
            parsed,
            {"drafts": 8, "draft_tokens": 56, "accepted_tokens": 15},
        )


class PolicyTest(unittest.TestCase):
    def test_cpu_stats_count_training_windows_and_installs_separately(self):
        speculator = SimpleNamespace(
            propose=MagicMock(),
            install_committed_weights=MagicMock(),
            finish_request=MagicMock(),
        )
        pipeline = SimpleNamespace(_train_window=MagicMock())
        stats = attach_cpu_stats(speculator, pipeline)
        speculator.propose(SimpleNamespace(req_ids=["question"]))
        for rounds in (1, 5):
            pipeline._train_window(None, SimpleNamespace(rounds=[None] * rounds))
        speculator.install_committed_weights("question")
        speculator.finish_request("question")

        self.assertEqual(stats.cpu_train_calls, 2)
        self.assertEqual(stats.cpu_train_rounds, 6)
        self.assertEqual(stats.weight_installs, 1)
        self.assertEqual(stats.active_requests, 0)

    def test_cpu_warmup_is_drained_but_not_counted(self):
        speculator = SimpleNamespace(
            propose=MagicMock(),
            install_committed_weights=MagicMock(),
            finish_request=MagicMock(),
        )
        pipeline = SimpleNamespace(_train_window=MagicMock())
        stats = attach_cpu_stats(speculator, pipeline)
        request_id = "chatcmpl-_warmup_run_sh"
        speculator.propose(SimpleNamespace(req_ids=[request_id]))
        self.assertEqual(stats.active_requests, 1)
        pipeline._train_window(None, SimpleNamespace(rounds=[None]))
        speculator.install_committed_weights(request_id)
        speculator.finish_request(request_id)
        self.assertEqual((stats.cpu_train_calls, stats.weight_installs), (0, 0))
        self.assertEqual(stats.active_requests, 0)

    def test_failed_cpu_training_or_install_is_not_counted(self):
        speculator = SimpleNamespace(
            propose=MagicMock(),
            install_committed_weights=MagicMock(side_effect=RuntimeError("install")),
            finish_request=MagicMock(),
        )
        pipeline = SimpleNamespace(
            _train_window=MagicMock(side_effect=ValueError("train"))
        )
        stats = attach_cpu_stats(speculator, pipeline)
        speculator.propose(SimpleNamespace(req_ids=["question"]))
        with self.assertRaises(ValueError):
            pipeline._train_window(None, SimpleNamespace(rounds=[None]))
        with self.assertRaises(RuntimeError):
            speculator.install_committed_weights("question")
        self.assertEqual((stats.cpu_train_calls, stats.weight_installs), (0, 0))

    def test_gpu_installs_exclude_initial_weight_restoration(self):
        from methods.utils.gpu_adapter import GpuOnlineEagle3Adapter

        stats = WorkerStats()
        bridge = SimpleNamespace(
            stats=stats,
            maybe_apply_pending_weights=MagicMock(
                side_effect=[
                    SimpleNamespace(version=1),
                    SimpleNamespace(version=0),
                    None,
                ]
            ),
        )
        adapter = GpuOnlineEagle3Adapter(None, bridge, depth=2)
        with patch.object(adapter, "_apply_snapshot"):
            for _ in range(3):
                adapter.apply_weights()
        self.assertEqual(stats.weight_installs, 1)

    def test_gpu_request_is_not_idle_until_final_weights_are_installed(self):
        from methods.utils.gpu_adapter import GpuOnlineEagle3Adapter

        stats = WorkerStats()
        stats.begin_request()
        bridge = SimpleNamespace(
            stats=stats,
            reset_request=lambda request_id: stats.end_request(),
            maybe_apply_pending_weights=lambda: SimpleNamespace(version=1),
        )
        adapter = GpuOnlineEagle3Adapter(None, bridge, depth=2)
        with patch.object(
            adapter,
            "_apply_snapshot",
            side_effect=lambda snapshot: self.assertEqual(stats.pending, 1),
        ):
            adapter.reset_request("question")
        self.assertEqual((stats.pending, stats.active_requests), (0, 0))
        self.assertEqual(stats.weight_installs, 1)

    def test_adapter_ignores_http_warmup_request(self):
        from methods.utils.gpu_adapter import GpuOnlineEagle3Adapter

        adapter = GpuOnlineEagle3Adapter(None, None, depth=2)
        proposal = SimpleNamespace(
            req_ids=["chatcmpl-_warmup_run_sh"],
            num_reqs=1,
        )

        adapter.start_proposal(proposal, None)
        adapter.reset_request("chatcmpl-_warmup_run_sh")

        self.assertIsNone(adapter._trace)
        self.assertIsNone(adapter._request_id)

    def test_gate_is_deterministic_without_global_rng_use(self):
        from methods.random_sampling.gating import BernoulliGate

        torch.manual_seed(47)
        expected = torch.rand(())
        torch.manual_seed(47)
        a = BernoulliGate(0.5, 7, 2)
        b = BernoulliGate(0.5, 7, 2)
        self.assertEqual(
            [a.select("a") for _ in range(12)], [b.select("a") for _ in range(12)]
        )
        self.assertEqual(torch.rand(()), expected)
        a.reset("a")
        self.assertEqual(a.rounds, {})

    def test_adapter_owns_captured_buffers(self):
        from methods.utils.adapter import OnlineEagle3Adapter

        class FakeGpuModel:
            def __init__(self):
                self.embedding = torch.nn.Embedding(8, 4)

            def embed_input_ids(self, input_ids):
                return self.embedding(input_ids.long())

        class FakeBridge:
            def __init__(self):
                self.trainer = _make_trainer()
                self.observations = []

            def observe_step(self, request_id, step_id, batch):
                self.observations.append((request_id, step_id, batch))

            def maybe_apply_pending_weights(self):
                return None

            def reset_request(self, request_id):
                del request_id

            def shutdown(self):
                pass

        bridge = FakeBridge()
        adapter = OnlineEagle3Adapter(FakeGpuModel(), bridge, depth=2)
        positions = torch.tensor([0, 1, 2])
        proposal = SimpleNamespace(
            req_ids=["request"],
            num_reqs=1,
        )
        adapter.start_proposal(
            proposal,
            [torch.randn(3, 4), torch.randn(3, 4)],
        )
        adapter.record_prefill(
            torch.tensor([0, 1, 2]),
            positions,
            torch.tensor([2]),
        )
        positions.fill_(99)
        adapter.record_draft_step(
            torch.tensor([3]),
            torch.tensor([3]),
            torch.randn(1, 4),
            torch.tensor([0]),
        )
        adapter.finish_proposal()
        adapter.capture_verification(
            torch.randn(3, 8),
            torch.tensor([[4, -1, -1]]),
            torch.tensor([1]),
        )
        verification = SimpleNamespace(
            req_ids=["request"],
            input_ids=torch.tensor([2, 3, 4]),
            positions=torch.tensor([3, 4, 5]),
        )
        adapter.observe_pending_verification(
            verification,
            [torch.randn(3, 4), torch.randn(3, 4)],
        )

        batch = bridge.observations[0][2]
        torch.testing.assert_close(batch.prefill_positions, torch.tensor([0, 1, 2]))
        batch.validate(
            hidden_size=4,
            num_aux_hidden_states=2,
            draft_vocab_size=5,
            feature_dtype=torch.float32,
        )

    def test_distillation_step_updates_trainable_weights(self):
        from methods.tts_common.step import train_batches

        trainer = _make_trainer()
        initial = trainer.model.model.fc.weight.detach().clone()
        _, losses = train_batches(trainer, (_make_batch(),))

        self.assertEqual(len(losses), 1)
        self.assertEqual(trainer.version, 1)
        self.assertFalse(torch.equal(initial, trainer.model.model.fc.weight))

    def test_reset_drains_queue_and_restores_baseline(self):
        from methods.ospec_common.ensemble import export_trainable_state
        from methods.utils.bridge import PolicyBridge

        trainer = _make_trainer()
        with patch.dict(os.environ, {"OSD_KEEP_WEIGHTS": "0"}):
            bridge = PolicyBridge(trainer, method="tts")
            try:
                original = export_trainable_state(trainer)
                bridge.observe_step("one", 0, _make_batch())
                bridge.reset_request("one")
                self.assertTrue(
                    all(
                        torch.equal(value, export_trainable_state(trainer)[key])
                        for key, value in original.items()
                    )
                )
                self.assertEqual(bridge.maybe_apply_pending_weights().version, 0)
            finally:
                bridge.shutdown()

    def test_random_probability_zero_never_trains(self):
        from methods.utils.bridge import PolicyBridge

        with patch.dict(os.environ, {"OSD_PROBABILITY": "0"}):
            bridge = PolicyBridge(_make_trainer(), method="random_sampling")
            try:
                bridge.observe_step("one", 0, _make_batch())
                bridge.reset_request("one")
                self.assertEqual(bridge.trainer.version, 0)
            finally:
                bridge.shutdown()

    def test_ospec_merges_three_cpu_learners(self):
        from methods.ospec_common.ensemble import (
            ChunkEnsemble,
            export_trainable_state,
        )

        ensemble = ChunkEnsemble(_make_trainer(), (1e-5, 2e-5, 3e-5), 0.1)
        snapshot = ensemble.update(((_make_batch(),),))
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.version, 1)
        self.assertEqual(len(ensemble.learners), 3)
        self.assertEqual(
            len(snapshot.state_dict), len(export_trainable_state(ensemble.learners[0]))
        )

    def test_ospec_publishes_only_at_chunk_boundary(self):
        from methods.utils.bridge import PolicyBridge

        with patch.dict(os.environ, {"OSD_CHUNK_SIZE": "2"}):
            bridge = PolicyBridge(_make_trainer(), method="ospec")
            try:
                bridge.observe_step("first", 0, _make_batch())
                bridge.reset_request("first")
                self.assertIsNone(bridge.maybe_apply_pending_weights())
                bridge.observe_step("second", 0, _make_batch())
                bridge.reset_request("second")
                self.assertEqual(bridge.maybe_apply_pending_weights().version, 1)
                self.assertIsNone(bridge.maybe_apply_pending_weights())
            finally:
                bridge.shutdown()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class GpuPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from methods.utils.gpu_training import install_cuda_training_validation

        install_cuda_training_validation()

    @classmethod
    def tearDownClass(cls):
        from methods.utils.gpu_training import uninstall_cuda_training_validation

        uninstall_cuda_training_validation()

    def test_tts_training_batch_and_snapshot_stay_on_cuda(self):
        from methods.utils.gpu_training import GpuPolicyBridge

        trainer = _make_trainer(device="cuda")
        bridge = GpuPolicyBridge(trainer, method="tts")
        initial = trainer.model.model.fc.weight.detach().clone()

        bridge.observe_step("request", 0, _make_batch(device="cuda"))
        snapshot = bridge.maybe_apply_pending_weights()

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(trainer.version, 1)
        self.assertFalse(torch.equal(initial, trainer.model.model.fc.weight))
        self.assertTrue(
            all(tensor.device.type == "cuda" for tensor in snapshot.state_dict.values())
        )

    def test_chunked_prefill_is_merged_into_trainable_cuda_tensors(self):
        from methods.utils.gpu_adapter import GpuOnlineEagle3Adapter

        adapter = GpuOnlineEagle3Adapter(None, None, depth=2)
        input_batch = SimpleNamespace(req_ids=["request"], num_reqs=1)

        def snapshot(start, length):
            return SimpleNamespace(
                prefill_length=torch.tensor(length, device="cuda"),
                prefill_positions=torch.arange(
                    start,
                    start + length,
                    device="cuda",
                ),
                prefill_input_embeds=torch.randn(length, 4, device="cuda"),
                draft_token_input_embeds=torch.randn(1, 4, device="cuda"),
                draft_recurrent_hidden_states=torch.randn(1, 4, device="cuda"),
            )

        with torch.inference_mode():
            adapter.start_proposal(
                input_batch,
                [torch.randn(2, 4, device="cuda") for _ in range(2)],
                defer=True,
            )
            adapter.finish_proposal(snapshot(0, 2))
            adapter.start_proposal(
                input_batch,
                [torch.randn(1, 4, device="cuda") for _ in range(2)],
            )
            adapter.finish_proposal(snapshot(2, 1))

        assert adapter._pending is not None
        torch.testing.assert_close(
            adapter._pending["prefill_positions"],
            torch.arange(3, device="cuda"),
        )
        self.assertTrue(
            all(tensor.device.type == "cuda" for tensor in adapter._pending.values())
        )
        self.assertTrue(
            all(not tensor.is_inference() for tensor in adapter._pending.values())
        )

    def test_ospec_keeps_three_learners_and_merge_on_cuda(self):
        from methods.utils.gpu_training import GpuPolicyBridge

        with patch.dict(os.environ, {"OSD_CHUNK_SIZE": "1"}):
            bridge = GpuPolicyBridge(_make_trainer(device="cuda"), method="ospec")
            assert bridge.ensemble is not None
            bridge.observe_step("request", 0, _make_batch(device="cuda"))
            bridge.reset_request("request")
            snapshot = bridge.maybe_apply_pending_weights()

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(len(bridge.ensemble.learners), 3)
        self.assertTrue(
            all(not learner.config.fused for learner in bridge.ensemble.learners)
        )
        self.assertEqual(bridge.stats.updates, 3)
        self.assertTrue(
            all(
                parameter.device.type == "cuda"
                for learner in bridge.ensemble.learners
                for parameter in learner.model.parameters()
            )
        )
        self.assertTrue(
            all(tensor.device.type == "cuda" for tensor in snapshot.state_dict.values())
        )


if __name__ == "__main__":
    unittest.main()
