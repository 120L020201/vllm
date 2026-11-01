# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Download selected Qwen3 targets and matching external draft checkpoints."""

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
    requires_mtp: bool = False


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
    "qwen3.5-4b": (
        ModelSpec(
            "Qwen3.5-4B",
            "Qwen/Qwen3.5-4B",
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            2560,
            False,
            True,
        ),
    ),
}

MTP_TARGET_MODEL_SPECS = {
    "8b": ModelSpec(
        "Qwen3-8B-MTP",
        "AngelSlim/Qwen3-8B-MTP-TTT3",
        "098ace0fd1b858c81e3bdbb58d4796400e154582",
        4096,
        False,
        True,
    ),
    "qwen3.5-4b": MODEL_SPECS["qwen3.5-4b"][0],
}

DRAFT_MODEL_SPECS = {
    "4b": {
        "eagle3": MODEL_SPECS["4b"][1],
        "dflash": ModelSpec(
            "Qwen3-4B_dflash",
            "z-lab/Qwen3-4B-DFlash-b16",
            "b74e3a329c4d963783143b1e970d95b002be72bd",
            2560,
            True,
        ),
    },
    "8b": {
        "eagle3": MODEL_SPECS["8b"][1],
        "dflash": ModelSpec(
            "Qwen3-8B_dflash",
            "z-lab/Qwen3-8B-DFlash-b16",
            "9b41424b7109f9c5413454f481b09a82b85333f4",
            4096,
            True,
        ),
    },
    "qwen3.5-4b": {},
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/srv/Models"))
    parser.add_argument(
        "--model-size",
        action="append",
        choices=tuple(MODEL_SPECS),
        dest="model_sizes",
        help="download this target and selected draft(s); may be repeated",
    )
    parser.add_argument(
        "--draft-method",
        action="append",
        choices=("eagle3", "dflash", "mtp"),
        dest="draft_methods",
        help=(
            "select a draft type; may be repeated; MTP uses target-embedded "
            "weights and has no separate download"
        ),
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _validate_model(path: Path, spec: ModelSpec) -> None:
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_config = config.get("text_config", config)
    if int(model_config.get("hidden_size", 0)) != spec.hidden_size:
        raise ValueError(f"unexpected hidden_size in {config_path}")

    if spec.requires_mtp:
        model_type = model_config.get("model_type", config.get("model_type"))
        if model_type == "qwen3":
            mtp_layers = model_config.get(
                "num_nextn_predict_layers",
                config.get("num_nextn_predict_layers", 0),
            )
        elif model_type in {
            "qwen3_5",
            "qwen3_5_text",
            "qwen3_5_moe",
            "qwen3_5_moe_text",
        }:
            mtp_layers = model_config.get(
                "mtp_num_hidden_layers",
                config.get("mtp_num_hidden_layers", 0),
            )
        else:
            raise ValueError(f"{path} is not a target with embedded MTP weights")
        if int(mtp_layers or 0) < 1:
            raise ValueError(f"{path} has no embedded MTP layers")

        index_paths = tuple(path.glob("*.index.json"))
        if index_paths and not any(
            name.startswith("mtp.") or ".mtp." in name
            for index_path in index_paths
            for name in json.loads(index_path.read_text(encoding="utf-8")).get(
                "weight_map", {}
            )
        ):
            raise ValueError(f"{path} has no embedded mtp.* weights")

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
    if not args.model_sizes:
        print("No models selected; specify --model-size to download.", flush=True)
        return
    if args.max_workers < 1:
        raise ValueError("max-workers must be positive")
    draft_methods = tuple(dict.fromkeys(args.draft_methods or ("eagle3",)))
    for size in args.model_sizes:
        target_spec = MODEL_SPECS[size][0]
        use_mtp = "mtp" in draft_methods
        if use_mtp and size not in MTP_TARGET_MODEL_SPECS:
            raise ValueError(
                f"{target_spec.repo_id} has no prepared checkpoint with embedded "
                "MTP weights; select --model-size 8b or qwen3.5-4b"
            )
        external_drafts = tuple(method for method in draft_methods if method != "mtp")
        unavailable = set(external_drafts) - DRAFT_MODEL_SPECS[size].keys()
        if unavailable:
            raise ValueError(
                f"{size} has no external draft download for "
                f"{', '.join(sorted(unavailable))}"
            )
        specs = [] if draft_methods == ("mtp",) else [target_spec]
        specs.extend(DRAFT_MODEL_SPECS[size][method] for method in external_drafts)
        if use_mtp:
            specs.append(MTP_TARGET_MODEL_SPECS[size])
        for spec in specs:
            _prepare_model(
                args.output_dir,
                spec,
                max_workers=args.max_workers,
                force=args.force,
            )
        if use_mtp:
            mtp_target = args.output_dir / MTP_TARGET_MODEL_SPECS[size].directory
            print(
                f"MTP weights are embedded in {mtp_target}; no separate drafter "
                f"was downloaded. Use MODEL={mtp_target} for MTP runs.",
                flush=True,
            )


if __name__ == "__main__":
    main()
