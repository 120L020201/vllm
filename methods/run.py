# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Launch frozen, CPU-online, and GPU-online speculative methods."""

import argparse
import json
import math
import os
import shlex
import sys
from pathlib import Path

import torch

from methods.dflash.config import (
    default_draft_directory,
    resolve_draft_method,
    resolve_speculative_tokens,
)

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "methods" / "utils" / "bootstrap"
DEFAULT_ARTIFACT_ROOT = ROOT / "methods" / "artifacts"
GPU_MEMORY_HEADROOM_GIB = 2.0


def _validate_mtp_target(path: str) -> None:
    config_path = Path(path) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    model_type = text_config.get("model_type", config.get("model_type"))
    if model_type == "qwen3":
        mtp_layers = text_config.get(
            "num_nextn_predict_layers",
            config.get("num_nextn_predict_layers", 0),
        )
    elif model_type in {
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    }:
        mtp_layers = text_config.get(
            "mtp_num_hidden_layers", config.get("mtp_num_hidden_layers", 0)
        )
    else:
        mtp_layers = 0
    if int(mtp_layers or 0) < 1:
        raise ValueError(
            "MTP requires a Qwen3-8B or Qwen3.5 target with embedded MTP weights; "
            f"{path} is not compatible"
        )

    index_paths = tuple(Path(path).glob("*.index.json"))
    if index_paths and not any(
        name.startswith("mtp.") or ".mtp." in name
        for index_path in index_paths
        for name in json.loads(index_path.read_text(encoding="utf-8")).get(
            "weight_map", {}
        )
    ):
        raise ValueError(f"{path} has no embedded mtp.* weights")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "method",
        choices=(
            "eagle",
            "dflash",
            "mtp",
            "online_draft",
            "tts",
            "ospec",
            "random_sampling",
        ),
    )
    parser.add_argument(
        "--model-size", choices=("4b", "8b", "qwen3.5-4b"), default="8b"
    )
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--target")
    parser.add_argument("--draft")
    parser.add_argument(
        "--draft-method",
        choices=("eagle3", "dflash", "mtp"),
        default=None,
        help="draft architecture; MTP is embedded in a compatible target",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--spec-tokens",
        type=int,
        default=None,
        help="draft tokens per step; defaults: EAGLE3=7, DFlash=15, MTP=3",
    )
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument(
        "--enable-chunked-prefill",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--language-model-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable multimodal inputs and load only the language model",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--gpu-attention-backend",
        type=str.upper,
        default="FLASHINFER",
        help="vLLM attention backend for both target and draft GPU inference",
    )
    parser.add_argument(
        "--flashinfer-autotune",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--max-gpu-memory-gib", type=float)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--update-stride", type=int, default=1)
    parser.add_argument("--probability", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--ensemble-lrs", default="1e-5,2e-5,3e-5")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument(
        "--dflash-max-context-tokens",
        type=int,
        default=0,
        help="DFlash TTS replays only the newest N context tokens; "
        "0 keeps the full context",
    )
    parser.add_argument("--dflash-max-train-rounds", type=int, default=16)
    parser.add_argument(
        "--dflash-loss-gamma",
        type=float,
        default=None,
        help="DFlash position decay exp(-depth/gamma); default keeps rho=0.9",
    )
    parser.add_argument(
        "--dflash-loss-normalization",
        choices=("mean", "weights", "block"),
        default="mean",
        help="normalize DFlash loss by token, weight sum, or equally per block",
    )
    parser.add_argument(
        "--dflash-loss-rejection-weight",
        type=float,
        default=1.0,
        help="extra weight on each block's first rejected prediction",
    )
    parser.add_argument(
        "--dflash-loss-objective",
        choices=("kl", "overlap", "ce", "kl_overlap"),
        default="kl",
        help="DFlash objective: full KL, acceptance overlap, top-1 CE, or KL+overlap",
    )
    parser.add_argument(
        "--dflash-loss-overlap-weight",
        type=float,
        default=1.0,
        help="overlap coefficient for the kl_overlap objective",
    )
    parser.add_argument(
        "--dflash-loss-temperature",
        type=float,
        default=1.0,
        help="temperature shared by the DFlash student and teacher distributions",
    )
    parser.add_argument(
        "--dflash-loss-top-k",
        type=int,
        default=0,
        help="teacher sampler top-k used by the DFlash objective; 0 keeps all",
    )
    parser.add_argument(
        "--dflash-loss-top-p",
        type=float,
        default=1.0,
        help="teacher sampler top-p used by the DFlash objective",
    )
    parser.add_argument(
        "--dflash-loss-reference-weight",
        type=float,
        default=0.0,
        help="KL weight toward the frozen initial DFlash draft",
    )
    parser.add_argument(
        "--dflash-loss-reference-decay-tokens",
        type=int,
        default=0,
        help="linearly release the DFlash reference KL by this sequence "
        "length; 0 keeps it",
    )
    parser.add_argument(
        "--online-draft-loss",
        choices=("uniform", "position", "entropy"),
        default="uniform",
        help="CPU online loss; position uses rho=0.9 and entropy uses gamma=2",
    )
    parser.add_argument(
        "--check-gradients",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="check gradients in the online draft CPU trainer",
    )
    parser.add_argument(
        "--fused-optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--master-weights",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="keep FP32 optimizer master weights for online CPU training",
    )
    parser.add_argument(
        "--persistent-grads",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="reuse CPU gradient buffers during online training",
    )
    parser.add_argument(
        "--continuous-training",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="train queued rounds without waiting for the next weight install",
    )
    parser.add_argument(
        "--online-draft-train-schedule",
        default="",
        help="cumulative completed rounds at which to train, e.g. 63,111,151",
    )
    parser.add_argument(
        "--lr-reference-rows",
        type=float,
        default=None,
        help="scale the online learning rate by target_rows / reference_rows",
    )
    parser.add_argument(
        "--log-updates",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="log weight updates (enabled by default for online_draft)",
    )
    parser.add_argument("--frozen-parameter-prefixes", default="")
    parser.add_argument(
        "--cpu-training-attention-backend",
        default="flash_attention",
    )
    parser.add_argument("--keep-weights", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def configure(args):
    draft_method = resolve_draft_method(args.method, args.draft_method)
    if args.model_size == "qwen3.5-4b" and draft_method != "mtp":
        raise ValueError("qwen3.5-4b is currently prepared only for MTP")
    spec_tokens = resolve_speculative_tokens(draft_method, args.spec_tokens)
    cpu_online = str(int(args.method == "online_draft"))
    log_updates = args.log_updates
    if log_updates is None:
        log_updates = args.method == "online_draft"
    master_weights = bool(args.master_weights)
    persistent_grads = bool(args.persistent_grads)
    continuous_training = bool(args.continuous_training)
    train_schedule = tuple(
        int(value)
        for value in args.online_draft_train_schedule.replace(",", " ").split()
    )
    if train_schedule and (
        any(value < 1 for value in train_schedule)
        or any(left >= right for left, right in zip(train_schedule, train_schedule[1:]))
    ):
        raise ValueError("train schedule must be strictly increasing rounds")
    if os.environ.get("VLLM_ONLINE_DRAFT_TRAIN") not in (None, "", cpu_online):
        raise ValueError(
            "VLLM_ONLINE_DRAFT_TRAIN conflicts with the selected method; "
            "VLLM_ONLINE_DRAFT_TRAIN=1 is required for online_draft and "
            "disabled for other methods"
        )
    profile = os.environ.get("VLLM_ONLINE_DRAFT_PROFILE", "0")
    if profile not in ("0", "1"):
        raise ValueError("VLLM_ONLINE_DRAFT_PROFILE must be 0 or 1")
    if profile == "1" and cpu_online != "1":
        raise ValueError("VLLM_ONLINE_DRAFT_PROFILE requires online_draft")
    if (
        spec_tokens < 1
        or args.max_model_len < spec_tokens + 2
        or args.max_num_batched_tokens < 1
    ):
        raise ValueError("context length must exceed the draft token count")
    if args.update_stride < 1 or args.chunk_size < 1 or args.seed < 0:
        raise ValueError(
            "stride and chunk size must be positive; seed must be nonnegative"
        )
    if args.dflash_max_context_tokens < 0:
        raise ValueError("dflash max context tokens must be nonnegative")
    if args.dflash_max_train_rounds < 0:
        raise ValueError("DFlash max train rounds must be nonnegative")
    if draft_method == "dflash" and args.method in {"tts", "random_sampling"}:
        if args.enable_chunked_prefill:
            raise ValueError("DFlash TTS requires chunked prefill to be off")
        if args.max_num_batched_tokens < args.max_model_len:
            raise ValueError(
                "DFlash TTS requires max-num-batched-tokens >= "
                "max-model-len so prompts are not split into prefill chunks"
            )
    if not 0 <= args.probability <= 1:
        raise ValueError("probability must be in [0,1]")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning rate must be finite and positive")
    if not math.isfinite(args.epsilon) or args.epsilon < 0:
        raise ValueError("learning rate must be positive and epsilon nonnegative")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("gpu memory utilization must be in (0,1]")
    if args.max_gpu_memory_gib is not None and (
        not math.isfinite(args.max_gpu_memory_gib) or args.max_gpu_memory_gib <= 0
    ):
        raise ValueError("max GPU memory must be finite and positive")
    rates = [float(rate) for rate in args.ensemble_lrs.split(",")]
    if len(rates) != 3 or any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("ensemble-lrs must specify three positive values")
    prefixes = tuple(
        prefix.strip()
        for prefix in args.frozen_parameter_prefixes.split(",")
        if prefix.strip()
    )
    target_name = (
        "Qwen3.5-4B" if args.model_size == "qwen3.5-4b" else f"qwen3-{args.model_size}"
    )
    target = args.target or str(args.artifact_root / "models" / target_name)
    if draft_method == "mtp":
        if args.draft:
            raise ValueError("MTP uses weights embedded in the target; omit --draft")
        draft = None
    else:
        draft = args.draft or str(
            default_draft_directory(
                args.artifact_root,
                args.model_size,
                draft_method,
            )
        )
    if not args.dry_run:
        for path in (target, draft):
            if path is None:
                continue
            if not Path(path).is_dir():
                raise FileNotFoundError(path)
        if draft_method == "mtp":
            _validate_mtp_target(target)
    gpu_memory_utilization = args.gpu_memory_utilization
    if args.max_gpu_memory_gib is not None:
        gpu_memory_utilization = _capped_gpu_memory_utilization(
            gpu_memory_utilization,
            args.max_gpu_memory_gib,
            require_gpu=not args.dry_run,
        )
    env = os.environ.copy()
    if args.method == "online_draft":
        omp_threads = env.setdefault("OMP_NUM_THREADS", "20")
        env.setdefault("MKL_NUM_THREADS", omp_threads)
        if int(omp_threads) < 1:
            raise ValueError("OMP_NUM_THREADS must be a positive integer")
        env.setdefault("OSD_TORCH_THREADS", omp_threads)
    env.update(
        {
            # The opt-in bootstrap installs observation hooks on the V2 runner.
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_ONLINE_DRAFT_TRAIN": cpu_online,
            "VLLM_ONLINE_DRAFT_PROFILE": profile,
            "OSD_METHOD": args.method,
            "OSD_DRAFT_METHOD": draft_method,
            "OSD_DRAFT_MODEL": draft or "",
            "OSD_DFLASH_MAX_CONTEXT_TOKENS": str(args.dflash_max_context_tokens),
            "OSD_LEARNING_RATE": str(args.learning_rate),
            "OSD_UPDATE_STRIDE": str(args.update_stride),
            "OSD_PROBABILITY": str(args.probability),
            "OSD_SEED": str(args.seed),
            "OSD_CHUNK_SIZE": str(args.chunk_size),
            "OSD_ENSEMBLE_LRS": args.ensemble_lrs,
            "OSD_EPSILON": str(args.epsilon),
            "OSD_KEEP_WEIGHTS": str(int(args.keep_weights)),
            "OSD_FUSED_OPTIMIZER": str(int(args.fused_optimizer)),
            "OSD_LOG_UPDATES": str(int(log_updates)),
            "OSD_MASTER_WEIGHTS": str(int(master_weights)),
            "OSD_PERSISTENT_GRADS": str(int(persistent_grads)),
            "OSD_CONTINUOUS_TRAINING": str(int(continuous_training)),
            "OSD_FROZEN_PARAMETER_PREFIXES": ",".join(prefixes),
            "PYTHONPATH": os.pathsep.join(
                (
                    str(BOOTSTRAP),
                    str(ROOT),
                    str(ROOT / "online_draft" / "src"),
                    env.get("PYTHONPATH", ""),
                )
            ),
        }
    )
    online_draft_config = {
        "learning_rate": args.learning_rate,
        "fused_optimizer": args.fused_optimizer,
        "frozen_parameter_prefixes": prefixes,
        "check_gradients": args.check_gradients,
        "log_updates": log_updates,
    }
    if args.method == "online_draft":
        online_draft_config["loss_mode"] = args.online_draft_loss
    if master_weights:
        online_draft_config["master_weights"] = True
    if persistent_grads:
        online_draft_config["persistent_grads"] = True
    if continuous_training:
        online_draft_config["continuous_training"] = True
    if train_schedule:
        online_draft_config["train_schedule"] = list(train_schedule)
    if args.lr_reference_rows is not None:
        online_draft_config["lr_reference_rows"] = args.lr_reference_rows
    if draft_method == "dflash":
        online_draft_config["max_train_rounds"] = args.dflash_max_train_rounds
        online_draft_config["loss_gamma"] = args.dflash_loss_gamma
        online_draft_config["loss_normalization"] = args.dflash_loss_normalization
        online_draft_config["loss_rejection_weight"] = args.dflash_loss_rejection_weight
        online_draft_config["loss_objective"] = args.dflash_loss_objective
        online_draft_config["loss_overlap_weight"] = args.dflash_loss_overlap_weight
        online_draft_config["loss_temperature"] = args.dflash_loss_temperature
        online_draft_config["loss_top_k"] = args.dflash_loss_top_k
        online_draft_config["loss_top_p"] = args.dflash_loss_top_p
        online_draft_config["loss_reference_weight"] = args.dflash_loss_reference_weight
        online_draft_config["loss_reference_decay_tokens"] = (
            args.dflash_loss_reference_decay_tokens
        )
    else:
        online_draft_config["attention_backend"] = args.cpu_training_attention_backend

    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        target,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        "1",
        "--pipeline-parallel-size",
        "1",
        "--max-num-seqs",
        "1",
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--attention-backend",
        args.gpu_attention_backend,
        (
            "--enable-flashinfer-autotune"
            if args.flashinfer_autotune
            else "--no-enable-flashinfer-autotune"
        ),
        "--generation-config",
        "vllm",
        "--dtype",
        "bfloat16",
        "--seed",
        str(args.seed),
        "-cc.cudagraph_mode=FULL_AND_PIECEWISE",
        "--no-enable-prefix-caching",
        (
            "--enable-chunked-prefill"
            if args.enable_chunked_prefill
            else "--no-enable-chunked-prefill"
        ),
        "--additional-config",
        json.dumps(
            {"online_draft": online_draft_config},
            separators=(",", ":"),
        ),
        "--spec-method",
        draft_method,
        "--spec-tokens",
        str(spec_tokens),
        "--speculative-config",
        json.dumps({"attention_backend": args.gpu_attention_backend}),
    ]
    if draft is not None:
        command.extend(("--spec-model", draft))
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    if args.language_model_only:
        command.append("--language-model-only")
    return command, env


def _capped_gpu_memory_utilization(
    requested: float,
    max_gpu_memory_gib: float,
    *,
    require_gpu: bool,
) -> float:
    vllm_budget_gib = max_gpu_memory_gib - GPU_MEMORY_HEADROOM_GIB
    if vllm_budget_gib <= 0:
        raise ValueError(
            f"max GPU memory must exceed {GPU_MEMORY_HEADROOM_GIB} GiB headroom"
        )
    if not torch.cuda.is_available():
        if require_gpu:
            raise RuntimeError("CUDA is required to enforce the GPU memory limit")
        return requested
    total_memory = torch.cuda.get_device_properties(0).total_memory
    cap = vllm_budget_gib * 2**30 / total_memory
    return min(requested, cap)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        command, env = configure(args)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(shlex.join(command))
        print(
            f"method={env['OSD_METHOD']} draft_method={env['OSD_DRAFT_METHOD']} "
            f"online={int(args.method not in {'eagle', 'dflash', 'mtp'})} "
            f"cpu_online={env['VLLM_ONLINE_DRAFT_TRAIN']} "
            f"master_weights={env['OSD_MASTER_WEIGHTS']} "
            f"persistent_grads={env['OSD_PERSISTENT_GRADS']} "
            f"continuous_training={env['OSD_CONTINUOUS_TRAINING']}"
        )
        return
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
