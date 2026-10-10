# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Download selected Qwen3 target and matching EAGLE3 draft checkpoints."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from vllm.transformers_utils.repo_utils import hf_api


@dataclass(frozen=True, slots=True)
class ModelSpec:
    directory: str
    repo_id: str
    revision: str
    hidden_size: int
    draft: bool


MODEL_SPECS = {
    "4b": (
        ModelSpec(
            "Qwen3-4B",
            "Qwen/Qwen3-4B",
            "1cfa9a7208912126459214e8b04321603b3df60c",
            2560,
            False,
        ),
        ModelSpec(
            "Qwen3-4B_eagle3",
            "AngelSlim/Qwen3-4B_eagle3",
            "fd331e59626c8e95c392381a16ee59d518727fbb",
            2560,
            True,
        ),
    ),
    "8b": (
        ModelSpec(
            "Qwen3-8B",
            "Qwen/Qwen3-8B",
            "b968826d9c46dd6066d109eabc6255188de91218",
            4096,
            False,
        ),
        ModelSpec(
            "Qwen3-8B_eagle3",
            "AngelSlim/Qwen3-8B_eagle3",
            "9629dfce7a4a10564dd48d3e5485c3976095653c",
            4096,
            True,
        ),
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/srv/Models"))
    parser.add_argument(
        "--model-size",
        action="append",
        choices=tuple(MODEL_SPECS),
        dest="model_sizes",
        help="download this target/draft pair; may be repeated",
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _validate_model(path: Path, spec: ModelSpec) -> None:
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config.get("hidden_size", 0)) != spec.hidden_size:
        raise ValueError(f"unexpected hidden_size in {config_path}")

    index_paths = tuple(path.glob("*.index.json"))
    if index_paths:
        for index_path in index_paths:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            for filename in set(index.get("weight_map", {}).values()):
                if not (path / filename).is_file():
                    raise FileNotFoundError(f"missing weight shard {path / filename}")
        return
    weights = tuple(path.glob("*.safetensors")) + tuple(path.glob("*.bin"))
    if not weights:
        raise FileNotFoundError(f"missing model weights in {path}")


def _prepare_model(
    output_dir: Path,
    spec: ModelSpec,
    *,
    max_workers: int,
    force: bool,
) -> None:
    destination = output_dir / spec.directory
    if destination.exists() and not force:
        _validate_model(destination, spec)
        print(f"ready {destination}", flush=True)
        return
    if destination.exists():
        raise FileExistsError(
            f"refusing to replace {destination}; move it aside before --force"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = output_dir / f".{spec.directory}.download"
    payload.mkdir(exist_ok=True)
    print(f"staging {spec.repo_id} in {payload}", flush=True)
    try:
        hf_api().snapshot_download(
            repo_id=spec.repo_id,
            revision=spec.revision,
            local_dir=payload,
            max_workers=max_workers,
        )
        _validate_model(payload, spec)
        (payload / ".osd-source.json").write_text(
            json.dumps(
                {"repo_id": spec.repo_id, "revision": spec.revision},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        payload.rename(destination)
    except Exception:
        print(
            f"incomplete files retained in {payload}; rerun to resume",
            flush=True,
        )
        raise
    print(f"prepared {destination}", flush=True)


def main() -> None:
    args = _parse_args()
    if args.max_workers < 1:
        raise ValueError("max-workers must be positive")
    for size in args.model_sizes or MODEL_SPECS:
        for spec in MODEL_SPECS[size]:
            _prepare_model(
                args.output_dir,
                spec,
                max_workers=args.max_workers,
                force=args.force,
            )


if __name__ == "__main__":
    main()
