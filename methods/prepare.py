# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prepare reproducible benchmark datasets and Qwen3 checkpoints."""

import argparse
import importlib
import json
import math
import os
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any

METHODS_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_ROOT = METHODS_ROOT / "artifacts"


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    name: str
    repo_id: str
    revision: str
    split: str
    config: str | None = None
    load_kwargs: tuple[tuple[str, Any], ...] = ()
    category: str | None = None


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    repo_id: str
    revision: str
    size: str
    draft: bool = False


DATASETS = (
    DatasetSpec(
        name="aime2026",
        repo_id="math-ai/aime26",
        revision="79037aebdb6580008fb960d17cb21fd3099083e3",
        split="train",
    ),
    DatasetSpec(
        name="gpqa_diamond",
        repo_id="Idavidrein/gpqa",
        revision="83022cefff930aea54f654c0b282e74b9eeda5c6",
        config="gpqa_diamond",
        split="train",
    ),
    DatasetSpec(
        name="mmlu_pro",
        repo_id="TIGER-Lab/MMLU-Pro",
        revision="b189ec765aa7ed75c8acfea42df31fdae71f97be",
        split="test",
    ),
    DatasetSpec(
        name="computer_science",
        repo_id="TIGER-Lab/MMLU-Pro",
        revision="b189ec765aa7ed75c8acfea42df31fdae71f97be",
        split="test",
        category="computer science",
    ),
    DatasetSpec(
        name="livecodebench_lite",
        repo_id="livecodebench/code_generation_lite",
        revision="0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
        split="test",
        load_kwargs=(
            ("trust_remote_code", True),
            ("version_tag", "release_v6"),
        ),
    ),
    DatasetSpec(
        name="longbench_v2",
        repo_id="THUDM/LongBench-v2",
        revision="2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9",
        split="train",
    ),
    DatasetSpec(
        name="longwriter_6k",
        repo_id="zai-org/LongWriter-6k",
        revision="0db15c0624f19d63e2efe1021595af933cc5b6cc",
        split="train",
    ),
)

MODELS = (
    ModelSpec(
        name="qwen3-4b",
        repo_id="Qwen/Qwen3-4B",
        revision="1cfa9a7208912126459214e8b04321603b3df60c",
        size="4b",
    ),
    ModelSpec(
        name="qwen3-4b-eagle3",
        repo_id="AngelSlim/Qwen3-4B_eagle3",
        revision="fd331e59626c8e95c392381a16ee59d518727fbb",
        size="4b",
        draft=True,
    ),
    ModelSpec(
        name="qwen3-8b",
        repo_id="Qwen/Qwen3-8B",
        revision="b968826d9c46dd6066d109eabc6255188de91218",
        size="8b",
    ),
    ModelSpec(
        name="qwen3-8b-eagle3",
        repo_id="AngelSlim/Qwen3-8B_eagle3",
        revision="9629dfce7a4a10564dd48d3e5485c3976095653c",
        size="8b",
        draft=True,
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("all", "datasets", "models"))
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="output root (default: methods/artifacts)",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=tuple(spec.name for spec in DATASETS),
        dest="datasets",
        help="prepare only this dataset; may be repeated",
    )
    parser.add_argument(
        "--model-size",
        action="append",
        choices=("4b", "8b"),
        dest="model_sizes",
        help="prepare only this target/draft pair; may be repeated",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="parallel Hugging Face model downloads",
    )
    parser.add_argument(
        "--hf-endpoint",
        help="optional Hugging Face endpoint, also available as HF_ENDPOINT",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _select_datasets(names: list[str] | None) -> tuple[DatasetSpec, ...]:
    selected = set(names or ())
    return tuple(spec for spec in DATASETS if not selected or spec.name in selected)


def _select_models(sizes: list[str] | None) -> tuple[ModelSpec, ...]:
    selected = set(sizes or ())
    return tuple(spec for spec in MODELS if not selected or spec.size in selected)


def _load_optional_module(name: str, requirement: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise RuntimeError(
            f"{name} is required; run `uv pip install -r {requirement}`"
        ) from error


def _manifest(spec: DatasetSpec | ModelSpec, rows: int | None = None) -> dict:
    manifest = asdict(spec)
    if rows is not None:
        manifest["rows"] = rows
    return manifest


def _existing_artifact_matches(destination: Path, manifest: dict) -> bool:
    manifest_path = destination / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return all(existing.get(key) == value for key, value in manifest.items())


def _ensure_destination_available(destination: Path, manifest: dict) -> bool:
    if not destination.exists():
        return True
    if destination.is_dir() and _existing_artifact_matches(destination, manifest):
        print(f"ready: {destination}")
        return False
    raise FileExistsError(
        f"refusing to replace existing artifact with unknown provenance: {destination}"
    )


def _write_dataset(dataset: Any, destination: Path, manifest: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-",
        dir=destination.parent,
    ) as temporary_directory:
        payload = Path(temporary_directory) / "payload"
        payload.mkdir()
        dataset.to_json(
            payload / "data.jsonl",
            orient="records",
            lines=True,
            force_ascii=False,
        )
        (payload / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        payload.rename(destination)


def _is_selected_category(row: dict, category: str) -> bool:
    value = str(row.get("category", "")).replace("_", " ").casefold()
    return value == category.casefold()


def prepare_datasets(
    root: Path,
    specs: Iterable[DatasetSpec],
) -> None:
    datasets = _load_optional_module("datasets", "methods/requirements.txt")
    for spec in specs:
        destination = root / "datasets" / spec.name
        base_manifest = _manifest(spec)
        if not _ensure_destination_available(destination, base_manifest):
            if not (destination / "data.jsonl").is_file():
                raise FileNotFoundError(f"missing data.jsonl in {destination}")
            continue

        kwargs = dict(spec.load_kwargs)
        kwargs.update(
            {
                "path": spec.repo_id,
                "revision": spec.revision,
                "split": spec.split,
            }
        )
        if spec.config is not None:
            kwargs["name"] = spec.config
        dataset = datasets.load_dataset(**kwargs)
        if spec.category is not None:
            dataset = dataset.filter(
                partial(_is_selected_category, category=spec.category)
            )

        rows = len(dataset)
        if rows == 0:
            raise ValueError(f"dataset selection is empty: {spec.name}")
        manifest = _manifest(spec, rows=rows)
        _write_dataset(dataset, destination, manifest)
        print(f"prepared: {destination} ({rows} rows)")


def _validate_model_directory(destination: Path, spec: ModelSpec) -> None:
    if not (destination / "config.json").is_file():
        raise FileNotFoundError(f"missing config.json in {destination}")
    weight_names = (
        ("pytorch_model.bin", "model.safetensors")
        if spec.draft
        else (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    )
    if not any((destination / name).is_file() for name in weight_names):
        raise FileNotFoundError(f"missing model weights in {destination}")


def prepare_models(
    root: Path,
    specs: Iterable[ModelSpec],
    *,
    max_workers: int,
) -> None:
    huggingface_hub = _load_optional_module(
        "huggingface_hub", "methods/requirements.txt"
    )
    for spec in specs:
        destination = root / "models" / spec.name
        manifest = _manifest(spec)
        if not _ensure_destination_available(destination, manifest):
            _validate_model_directory(destination, spec)
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{destination.name}-",
            dir=destination.parent,
        ) as temporary_directory:
            payload = Path(temporary_directory) / "payload"
            huggingface_hub.snapshot_download(
                repo_id=spec.repo_id,
                revision=spec.revision,
                local_dir=payload,
                max_workers=max_workers,
            )
            _validate_model_directory(payload, spec)
            (payload / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            payload.rename(destination)
        print(f"prepared: {destination}")


def _print_plan(
    root: Path,
    datasets: Iterable[DatasetSpec],
    models: Iterable[ModelSpec],
) -> None:
    for spec in datasets:
        print(
            f"dataset {spec.name}: {spec.repo_id}@{spec.revision} "
            f"-> {root / 'datasets' / spec.name}"
        )
    for spec in models:
        print(
            f"model {spec.name}: {spec.repo_id}@{spec.revision} "
            f"-> {root / 'models' / spec.name}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_workers < 1 or not math.isfinite(args.max_workers):
        parser.error("max-workers must be positive")
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    dataset_specs = (
        _select_datasets(args.datasets) if args.kind in ("all", "datasets") else ()
    )
    model_specs = (
        _select_models(args.model_sizes) if args.kind in ("all", "models") else ()
    )
    if args.dry_run:
        _print_plan(args.root, dataset_specs, model_specs)
        return
    if dataset_specs:
        prepare_datasets(args.root, dataset_specs)
    if model_specs:
        prepare_models(args.root, model_specs, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
