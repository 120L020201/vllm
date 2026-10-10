# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from argparse import Namespace
from pathlib import Path

from experiments.benchmark import (
    BenchmarkSpec,
    _parse_benchmark_specs,
    _prepare_resume,
    _request_body,
    _request_config,
    _select_benchmark_records,
)
from experiments.prepare_datasets import (
    DATASET_REVISIONS,
    EXPECTED_ROWS,
    _format_longwriter,
)


def _write_dataset(path: Path, prompts: list[str]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for prompt in prompts:
            output.write(json.dumps({"prompt": prompt}) + "\n")


def _result(
    benchmark: str,
    line: int,
    prompt: str,
    *,
    successful: bool = True,
) -> dict:
    return {
        "benchmark": benchmark,
        "line": line,
        "prompt": prompt,
        "response": {"usage": {"completion_tokens": 1}} if successful else None,
        "error": None if successful else "interrupted",
    }


def _read_results(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_request_uses_qwen_chat_template() -> None:
    """Benchmark requests ask the server to render Qwen's thinking template."""
    args = Namespace(
        model="Qwen3-4B",
        max_output_tokens=32768,
        max_prompt_tokens=16384,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        presence_penalty=1.5,
        enable_thinking=True,
        seed=7,
        benchmarks="aime2026[15] gpqa_diamond",
    )

    body = _request_body(args, "Solve this problem")

    assert body["messages"] == [{"role": "user", "content": "Solve this problem"}]
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["presence_penalty"] == 1.5
    assert body["seed"] == 7
    assert "prompt" not in body
    assert _request_config(args)["endpoint"] == "/v1/chat/completions"


def test_benchmark_selection_is_seeded_per_dataset(tmp_path: Path) -> None:
    """Bracket syntax selects a reproducible subset without limiting others."""
    dataset_path = tmp_path / "aime2026.jsonl"
    _write_dataset(dataset_path, [f"question-{index}" for index in range(30)])

    specs = _parse_benchmark_specs("aime2026[15] gpqa_diamond")
    first = _select_benchmark_records(dataset_path, specs[0], seed=7)
    repeated = _select_benchmark_records(dataset_path, specs[0], seed=7)
    changed = _select_benchmark_records(dataset_path, specs[0], seed=8)

    assert specs == [BenchmarkSpec("aime2026", 15), BenchmarkSpec("gpqa_diamond")]
    assert len(first) == 15
    assert first == repeated
    assert first != changed


def test_sampled_resume_keeps_only_selected_questions(tmp_path: Path) -> None:
    """Resume state is scoped to the seeded subset, not the full dataset."""
    _write_dataset(tmp_path / "first.jsonl", ["one", "two", "three", "four"])
    output_path = tmp_path / "results.jsonl"
    output_path.write_text(
        "".join(
            json.dumps(_result("first", line, prompt)) + "\n"
            for line, prompt in enumerate(
                ("one", "two", "three", "four"),
                start=1,
            )
        ),
        encoding="utf-8",
    )

    completed, restart, discarded = _prepare_resume(
        output_path,
        tmp_path,
        ["first"],
        "question",
        selected_lines={"first": {1, 3}},
    )

    assert completed == {("first", 1), ("first", 3)}
    assert restart is None
    assert discarded == 2


def test_longwriter_formatter_keeps_only_the_generation_prompt() -> None:
    """LongWriter preparation omits the very large reference completion."""
    formatted = _format_longwriter(
        {
            "messages": [
                {"role": "user", "content": "Write a long report."},
                {"role": "assistant", "content": "large reference output"},
            ]
        },
        2,
    )

    assert DATASET_REVISIONS["LongWriter-6k"] == (
        "0db15c0624f19d63e2efe1021595af933cc5b6cc"
    )
    assert EXPECTED_ROWS["LongWriter-6k"] == 6000
    assert formatted == {
        "id": "LongWriter-6k-3",
        "benchmark": "LongWriter-6k",
        "prompt": "Write a long report.",
    }


def test_question_resume_keeps_successful_questions(tmp_path: Path) -> None:
    """Independent methods retry only failed or missing questions."""
    _write_dataset(tmp_path / "first.jsonl", ["one", "two"])
    _write_dataset(tmp_path / "second.jsonl", ["three", "four"])
    output_path = tmp_path / "results.jsonl"
    existing = [
        _result("first", 1, "one"),
        _result("first", 2, "two", successful=False),
        _result("second", 1, "three"),
    ]
    output_path.write_text(
        "".join(json.dumps(result) + "\n" for result in existing) + "incomplete",
        encoding="utf-8",
    )

    completed, restart, discarded = _prepare_resume(
        output_path,
        tmp_path,
        ["first", "second"],
        "question",
    )

    assert completed == {("first", 1), ("second", 1)}
    assert restart == "first"
    assert discarded == 2
    assert [
        (result["benchmark"], result["line"]) for result in _read_results(output_path)
    ] == [("first", 1), ("second", 1)]


def test_dataset_resume_restarts_first_incomplete_dataset(tmp_path: Path) -> None:
    """Stateful methods discard partial datasets before resuming."""
    _write_dataset(tmp_path / "first.jsonl", ["one", "two"])
    _write_dataset(tmp_path / "second.jsonl", ["three", "four"])
    output_path = tmp_path / "results.jsonl"
    existing = [
        _result("first", 1, "one"),
        _result("first", 2, "two"),
        _result("second", 1, "three"),
    ]
    output_path.write_text(
        "".join(json.dumps(result) + "\n" for result in existing),
        encoding="utf-8",
    )

    completed, restart, discarded = _prepare_resume(
        output_path,
        tmp_path,
        ["first", "second"],
        "dataset",
    )

    assert completed == {("first", 1), ("first", 2)}
    assert restart == "second"
    assert discarded == 1
    assert [
        (result["benchmark"], result["line"]) for result in _read_results(output_path)
    ] == [("first", 1), ("first", 2)]


def test_resume_retries_results_from_old_request_format(tmp_path: Path) -> None:
    """Raw-completion results are not mixed with chat-template results."""
    _write_dataset(tmp_path / "first.jsonl", ["one"])
    output_path = tmp_path / "results.jsonl"
    old_result = _result("first", 1, "one")
    old_result["request_config"] = {"endpoint": "/v1/completions"}
    output_path.write_text(json.dumps(old_result) + "\n", encoding="utf-8")

    completed, restart, discarded = _prepare_resume(
        output_path,
        tmp_path,
        ["first"],
        "question",
        {"endpoint": "/v1/chat/completions"},
    )

    assert completed == set()
    assert restart == "first"
    assert discarded == 1
    assert output_path.read_text() == ""
