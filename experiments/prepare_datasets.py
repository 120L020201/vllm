# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prepare complete benchmark JSONL files consumed by experiments/run.sh."""

import argparse
import csv
import json
import os
import random
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import get_token

DATASET_REVISIONS = {
    "aime2026": "79037aebdb6580008fb960d17cb21fd3099083e3",
    "gpqa_diamond": "83022cefff930aea54f654c0b282e74b9eeda5c6",
    "mmlu-pro-computer_science": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
    "livecodebench-lite": "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
    "LongBench-v2": "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9",
    "LongWriter-6k": "0db15c0624f19d63e2efe1021595af933cc5b6cc",
}
EXPECTED_ROWS = {
    "aime2026": 30,
    "gpqa_diamond": 198,
    "mmlu-pro-computer_science": 410,
    "livecodebench-lite": 1055,
    "LongBench-v2": 503,
    "LongWriter-6k": 6000,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/srv/Datasets"))
    parser.add_argument(
        "--hf-endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=tuple(EXPECTED_ROWS),
        dest="datasets",
        help="prepare only this dataset; may be repeated",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _choices(options: Iterable[str]) -> str:
    return "\n".join(
        f"{chr(ord('A') + index)}. {option}" for index, option in enumerate(options)
    )


def _format_aime(row: dict[str, Any], index: int) -> dict[str, Any]:
    problem = row["problem"]
    return {
        "id": f"aime2026-{index + 1}",
        "benchmark": "aime2026",
        "prompt": (
            "Solve the following AIME problem. Reason step by step, and put "
            "the final integer answer in \\boxed{} only at the end.\n\n"
            f"Problem:\n{problem}"
        ),
        "answer": row["answer"],
        "source": row,
    }


def _format_gpqa(row: dict[str, Any], index: int) -> dict[str, Any]:
    correct_answer = row["Correct Answer"]
    options = [correct_answer, *(row[f"Incorrect Answer {i}"] for i in range(1, 4))]
    random.Random(f"gpqa-diamond-{index}").shuffle(options)
    answer_label = chr(ord("A") + options.index(correct_answer))
    return {
        "id": f"gpqa-diamond-{index + 1}",
        "benchmark": "gpqa_diamond",
        "prompt": (
            "Answer the following graduate-level science multiple-choice "
            "question. Reason step by step, then give the final choice letter "
            "in \\boxed{} only at the end.\n\n"
            f"Question:\n{row['Question']}\n\nChoices:\n{_choices(options)}"
        ),
        "answer": answer_label,
        "source": row,
    }


def _format_mmlu(row: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "id": f"mmlu-pro-computer_science-{index + 1}",
        "benchmark": "mmlu-pro-computer_science",
        "prompt": (
            "Answer the following computer science multiple-choice question. "
            "Reason step by step, then give the final choice letter in "
            "\\boxed{} only at the end.\n\n"
            f"Question:\n{row['question']}\n\n"
            f"Choices:\n{_choices(row['options'])}"
        ),
        "answer": row["answer"],
        "source": row,
    }


def _format_livecodebench(row: dict[str, Any], index: int) -> dict[str, Any]:
    starter_code = row.get("starter_code") or ""
    starter = f"\n\nStarter code:\n{starter_code}" if starter_code else ""
    return {
        "id": f"livecodebench-lite-{index + 1}",
        "benchmark": "livecodebench-lite",
        "prompt": (
            "Solve the following programming problem. Reason step by step, "
            "then provide the final accepted solution code.\n\n"
            f"Problem:\n{row['question_content']}{starter}"
        ),
        "source": row,
    }


def _format_longbench(row: dict[str, Any], index: int) -> dict[str, Any]:
    options = [row[f"choice_{label}"] for label in "ABCD"]
    return {
        "id": f"LongBench-v2-{index + 1}",
        "benchmark": "LongBench-v2",
        "prompt": (
            "Read the complete context and answer the multiple-choice question. "
            "Reason step by step, then give the final choice letter in "
            "\\boxed{} only at the end.\n\n"
            f"Context:\n{row['context']}\n\nQuestion:\n{row['question']}\n\n"
            f"Choices:\n{_choices(options)}"
        ),
        "answer": row["answer"],
        "source": row,
    }


def _format_longwriter(row: dict[str, Any], index: int) -> dict[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise TypeError("LongWriter-6k row must contain a messages list")
    prompt = next(
        (
            message.get("content")
            for message in messages
            if isinstance(message, dict) and message.get("role") == "user"
        ),
        None,
    )
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("LongWriter-6k row must contain a user prompt")
    return {
        "id": f"LongWriter-6k-{index + 1}",
        "benchmark": "LongWriter-6k",
        "prompt": prompt,
    }


def _download_dataset_file(
    endpoint: str,
    repo_id: str,
    revision: str,
    filename: str,
    cache_dir: Path,
) -> Path:
    destination = cache_dir / repo_id.replace("/", "--") / revision / filename
    if destination.is_file():
        print(f"cached {destination}", flush=True)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    url = (
        f"{endpoint.rstrip('/')}/datasets/{repo_id}/resolve/{revision}/"
        f"{urllib.parse.quote(filename)}"
    )
    token = get_token()
    timeout = float(os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT", "7200"))
    for attempt in range(1, 6):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            print(
                f"downloading {repo_id}/{filename} "
                f"(attempt {attempt}/5, resume={offset} bytes)",
                flush=True,
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                append = offset > 0 and response.status == 206
                with partial.open("ab" if append else "wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            partial.replace(destination)
            return destination
        except urllib.error.HTTPError as error:
            if error.code in (401, 403):
                raise PermissionError(
                    f"access denied for {repo_id}/{filename}; run `hf auth login`"
                ) from error
            if attempt == 5:
                raise
        except (TimeoutError, urllib.error.URLError):
            if attempt == 5:
                raise
        time.sleep(2**attempt)
    raise AssertionError("download retry loop did not return")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _load_dataset_rows(
    name: str,
    endpoint: str,
    cache_dir: Path,
) -> list[dict[str, Any]]:
    revision = DATASET_REVISIONS[name]
    if name == "aime2026":
        path = _download_dataset_file(
            endpoint,
            "math-ai/aime26",
            revision,
            "aime2026.jsonl",
            cache_dir,
        )
        return _read_jsonl(path)
    if name == "gpqa_diamond":
        path = _download_dataset_file(
            endpoint,
            "Idavidrein/gpqa",
            revision,
            "gpqa_diamond.csv",
            cache_dir,
        )
        with path.open(encoding="utf-8-sig", newline="") as source:
            return list(csv.DictReader(source))
    if name == "mmlu-pro-computer_science":
        path = _download_dataset_file(
            endpoint,
            "TIGER-Lab/MMLU-Pro",
            revision,
            "data/test-00000-of-00001.parquet",
            cache_dir,
        )
        rows = pq.read_table(path).to_pylist()
        return [row for row in rows if row["category"] == "computer science"]
    if name == "livecodebench-lite":
        rows = []
        for suffix in ("", "2", "3", "4", "5", "6"):
            path = _download_dataset_file(
                endpoint,
                "livecodebench/code_generation_lite",
                revision,
                f"test{suffix}.jsonl",
                cache_dir,
            )
            rows.extend(_read_jsonl(path))
        return rows
    if name == "LongBench-v2":
        path = _download_dataset_file(
            endpoint,
            "THUDM/LongBench-v2",
            revision,
            "data.json",
            cache_dir,
        )
        return list(load_dataset("json", data_files=str(path), split="train"))
    if name == "LongWriter-6k":
        path = _download_dataset_file(
            endpoint,
            "zai-org/LongWriter-6k",
            revision,
            "long.jsonl",
            cache_dir,
        )
        return _read_jsonl(path)
    raise ValueError(f"unsupported dataset: {name}")


def _line_count(path: Path) -> int:
    with path.open("rb") as source:
        return sum(1 for line in source if line.strip())


def _write_dataset(
    output_dir: Path,
    name: str,
    rows: list[dict[str, Any]],
    formatter: Callable[[dict[str, Any], int], dict[str, Any]],
    *,
    force: bool,
) -> None:
    expected = EXPECTED_ROWS[name]
    if len(rows) != expected:
        raise ValueError(f"{name} expected {expected} rows, got {len(rows)}")
    destination = output_dir / f"{name}.jsonl"
    if destination.exists() and not force:
        actual = _line_count(destination)
        if actual == expected:
            print(f"ready {destination}: {expected} rows", flush=True)
            return
        raise FileExistsError(
            f"{destination} has {actual} rows, expected {expected}; pass --force"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_dir,
        prefix=f".{name}-",
        suffix=".jsonl",
        delete=False,
    ) as output:
        temporary = Path(output.name)
        try:
            for index, row in enumerate(rows):
                formatted = formatter(row, index)
                output.write(json.dumps(formatted, ensure_ascii=False) + "\n")
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(destination)
    print(f"prepared {destination}: {expected} rows", flush=True)


def main() -> None:
    args = _parse_args()
    selected = tuple(args.datasets or EXPECTED_ROWS)
    formatters = {
        "aime2026": _format_aime,
        "gpqa_diamond": _format_gpqa,
        "mmlu-pro-computer_science": _format_mmlu,
        "livecodebench-lite": _format_livecodebench,
        "LongBench-v2": _format_longbench,
        "LongWriter-6k": _format_longwriter,
    }
    missing = []
    for name in selected:
        destination = args.output_dir / f"{name}.jsonl"
        if not args.force and destination.is_file():
            actual = _line_count(destination)
            if actual == EXPECTED_ROWS[name]:
                print(f"ready {destination}: {actual} rows", flush=True)
                continue
        missing.append(name)
    cache_dir = args.output_dir / ".downloads"
    for name in missing:
        rows = _load_dataset_rows(name, args.hf_endpoint, cache_dir)
        _write_dataset(
            args.output_dir,
            name,
            rows,
            formatters[name],
            force=args.force,
        )


if __name__ == "__main__":
    main()
