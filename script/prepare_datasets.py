# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prepare complete benchmark JSONL files consumed by script/run.sh."""

import argparse
import json
import os
import random
import tempfile
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from datasets import config as datasets_config
from datasets import load_dataset
from huggingface_hub import constants as hub_constants

DATASET_REVISIONS = {
    "aime2026": "79037aebdb6580008fb960d17cb21fd3099083e3",
    "gpqa_diamond": "83022cefff930aea54f654c0b282e74b9eeda5c6",
    "mmlu-pro-computer_science": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
    "livecodebench-lite": "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
    "LongBench-v2": "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9",
}
EXPECTED_ROWS = {
    "aime2026": 30,
    "gpqa_diamond": 198,
    "mmlu-pro-computer_science": 410,
    "livecodebench-lite": 1055,
    "LongBench-v2": 503,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/srv/Datasets/TTS"))
    parser.add_argument(
        "--hf-endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
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


def _load_livecodebench(endpoint: str) -> list[dict[str, Any]]:
    revision = DATASET_REVISIONS["livecodebench-lite"]
    rows: list[dict[str, Any]] = []
    for suffix in ("", "2", "3", "4", "5", "6"):
        filename = f"test{suffix}.jsonl"
        url = (
            f"{endpoint.rstrip('/')}/datasets/livecodebench/"
            f"code_generation_lite/resolve/{revision}/{filename}"
        )
        with urllib.request.urlopen(url, timeout=7200) as response:
            for raw_line in response:
                if raw_line.strip():
                    rows.append(json.loads(raw_line))
    return rows


def _load_all(endpoint: str) -> dict[str, list[dict[str, Any]]]:
    aime = load_dataset(
        "math-ai/aime26",
        revision=DATASET_REVISIONS["aime2026"],
        split="test",
    )
    gpqa = load_dataset(
        "Idavidrein/gpqa",
        "gpqa_diamond",
        revision=DATASET_REVISIONS["gpqa_diamond"],
        split="train",
    )
    mmlu = load_dataset(
        "TIGER-Lab/MMLU-Pro",
        revision=DATASET_REVISIONS["mmlu-pro-computer_science"],
        split="test",
    ).filter(lambda row: row["category"] == "computer science")
    longbench = load_dataset(
        "THUDM/LongBench-v2",
        revision=DATASET_REVISIONS["LongBench-v2"],
        split="train",
    )
    return {
        "aime2026": list(aime),
        "gpqa_diamond": list(gpqa),
        "mmlu-pro-computer_science": list(mmlu),
        "livecodebench-lite": _load_livecodebench(endpoint),
        "LongBench-v2": list(longbench),
    }


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
        raise FileExistsError(f"refusing to overwrite {destination}; pass --force")
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
    print(f"prepared {destination}: {expected} rows")


def main() -> None:
    args = _parse_args()
    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    datasets_config.HF_ENDPOINT = args.hf_endpoint
    hub_constants.ENDPOINT = args.hf_endpoint
    datasets = _load_all(args.hf_endpoint)
    formatters = {
        "aime2026": _format_aime,
        "gpqa_diamond": _format_gpqa,
        "mmlu-pro-computer_science": _format_mmlu,
        "livecodebench-lite": _format_livecodebench,
        "LongBench-v2": _format_longbench,
    }
    for name, rows in datasets.items():
        _write_dataset(
            args.output_dir,
            name,
            rows,
            formatters[name],
            force=args.force,
        )


if __name__ == "__main__":
    main()
