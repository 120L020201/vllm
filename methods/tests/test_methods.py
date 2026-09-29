# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from online_draft.models.qwen3_eagle3 import (
    Qwen3Eagle3Config,
    Qwen3Eagle3ForCausalLM,
)
from online_draft.training.eagle3_batch import Eagle3DistillationBatch
from online_draft.training.trainer import DraftTrainer, TrainerConfig

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
from methods.utils.factory import _request_ids_to_reset
from methods.utils.stats import WorkerStats
from script.benchmark import (
    _acceptance_length,
    _completion_tokens,
    _fetch_spec_metrics,
)


def _make_trainer(learning_rate=1e-2):
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
    return DraftTrainer(model, TrainerConfig(learning_rate=learning_rate))


def _make_batch(anchor_position=2):
    hidden_size = 4
    auxiliary_size = 8
    draft_length = 2
    return Eagle3DistillationBatch(
        prefill_positions=torch.arange(anchor_position + 1, dtype=torch.long),
        prefill_input_embeds=torch.randn(anchor_position + 1, hidden_size),
        prefill_aux_hidden_states=torch.randn(anchor_position + 1, auxiliary_size),
        proposal_positions=torch.arange(
            anchor_position,
            anchor_position + draft_length,
            dtype=torch.long,
        ),
        draft_token_input_embeds=torch.randn(draft_length - 1, hidden_size),
        draft_recurrent_hidden_states=torch.randn(draft_length - 1, hidden_size),
        teacher_probabilities=torch.randn(draft_length, 5).softmax(dim=-1),
        confirmed_positions=torch.tensor([anchor_position + 1]),
        confirmed_input_embeds=torch.randn(1, hidden_size),
        confirmed_aux_hidden_states=torch.randn(1, auxiliary_size),
        rejection_position=0,
    )


class LauncherTest(unittest.TestCase):
    def test_same_inference_args_for_every_method(self):
        commands = []
        for method in ("eagle", "tts", "random_sampling", "ospec"):
            args = build_parser().parse_args((method, "--dry-run"))
            command, env = configure(args)
            commands.append(command)
            self.assertEqual(env["OSD_METHOD"], method)
            self.assertEqual(env["VLLM_USE_V2_MODEL_RUNNER"], "1")
            self.assertIn("--max-num-seqs", command)
            self.assertIn("1", command)
        self.assertTrue(all(command == commands[0] for command in commands))

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

    def test_warmup_with_no_preemptions_has_no_resets(self):
        scheduler_output = SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=None,
        )

        self.assertEqual(_request_ids_to_reset(scheduler_output), set())


class PreparationTest(unittest.TestCase):
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


class BenchmarkLoggingTest(unittest.TestCase):
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
                stats.enqueue()
                stats.complete(3)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload, {"pending": 0, "updates": 3})

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


if __name__ == "__main__":
    unittest.main()
