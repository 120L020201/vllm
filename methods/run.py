# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Launch isolated EAGLE-3 methods with a common vLLM inference engine."""

import argparse
import json
import math
import os
import shlex
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "methods" / "utils" / "bootstrap"
DEFAULT_ARTIFACT_ROOT = ROOT / "methods" / "artifacts"
GPU_MEMORY_HEADROOM_GIB = 2.0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "method", choices=("eagle", "online_draft", "tts", "ospec", "random_sampling")
    )
    parser.add_argument("--model-size", choices=("4b", "8b"), default="8b")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--target")
    parser.add_argument("--draft")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--spec-tokens", type=int, default=7)
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
        "--check-gradients",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="check gradients in the optional built-in CPU online trainer",
    )
    parser.add_argument(
        "--fused-optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--log-updates",
        action=argparse.BooleanOptionalAction,
        default=False,
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
    cpu_online = str(int(args.method == "online_draft"))
    if os.environ.get("VLLM_ONLINE_DRAFT_TRAIN") not in (None, "", cpu_online):
        raise ValueError(
            "VLLM_ONLINE_DRAFT_TRAIN conflicts with the selected method; "
            "use online_draft for CPU training and leave this variable unset"
        )
    profile = os.environ.get("VLLM_ONLINE_DRAFT_PROFILE", "0")
    if profile not in ("0", "1"):
        raise ValueError("VLLM_ONLINE_DRAFT_PROFILE must be 0 or 1")
    if profile == "1" and cpu_online != "1":
        raise ValueError("VLLM_ONLINE_DRAFT_PROFILE requires online_draft")
    if (
        args.spec_tokens < 1
        or args.max_model_len < args.spec_tokens + 2
        or args.max_num_batched_tokens < 1
    ):
        raise ValueError("context length must exceed the draft token count")
    if args.update_stride < 1 or args.chunk_size < 1 or args.seed < 0:
        raise ValueError(
            "stride and chunk size must be positive; seed must be nonnegative"
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
    target = args.target or str(
        args.artifact_root / "models" / f"qwen3-{args.model_size}"
    )
    draft = args.draft or str(
        args.artifact_root / "models" / f"qwen3-{args.model_size}-eagle3"
    )
    if not args.dry_run:
        for path in (target, draft):
            if not Path(path).is_dir():
                raise FileNotFoundError(path)
    gpu_memory_utilization = args.gpu_memory_utilization
    if args.max_gpu_memory_gib is not None:
        gpu_memory_utilization = _capped_gpu_memory_utilization(
            gpu_memory_utilization,
            args.max_gpu_memory_gib,
            require_gpu=not args.dry_run,
        )
    env = os.environ.copy()
    env.update(
        {
            # The opt-in bootstrap installs observation hooks on the V2 runner.
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_ONLINE_DRAFT_TRAIN": cpu_online,
            "VLLM_ONLINE_DRAFT_PROFILE": profile,
            "OSD_METHOD": args.method,
            "OSD_DRAFT_MODEL": draft,
            "OSD_LEARNING_RATE": str(args.learning_rate),
            "OSD_UPDATE_STRIDE": str(args.update_stride),
            "OSD_PROBABILITY": str(args.probability),
            "OSD_SEED": str(args.seed),
            "OSD_CHUNK_SIZE": str(args.chunk_size),
            "OSD_ENSEMBLE_LRS": args.ensemble_lrs,
            "OSD_EPSILON": str(args.epsilon),
            "OSD_KEEP_WEIGHTS": str(int(args.keep_weights)),
            "OSD_FUSED_OPTIMIZER": str(int(args.fused_optimizer)),
            "OSD_LOG_UPDATES": str(int(args.log_updates)),
            "PYTHONPATH": os.pathsep.join(
                (str(BOOTSTRAP), str(ROOT), env.get("PYTHONPATH", ""))
            ),
        }
    )
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
            {
                "online_draft": {
                    "attention_backend": args.cpu_training_attention_backend,
                    "learning_rate": args.learning_rate,
                    "fused_optimizer": args.fused_optimizer,
                    "frozen_parameter_prefixes": prefixes,
                    "check_gradients": args.check_gradients,
                    "log_updates": args.log_updates,
                }
            },
            separators=(",", ":"),
        ),
        "--spec-method",
        "eagle3",
        "--spec-model",
        draft,
        "--spec-tokens",
        str(args.spec_tokens),
        "--speculative-config",
        json.dumps({"attention_backend": args.gpu_attention_backend}),
    ]
    if args.trust_remote_code:
        command.append("--trust-remote-code")
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
            f"method={env['OSD_METHOD']} online={int(args.method != 'eagle')} "
            f"cpu_online={env['VLLM_ONLINE_DRAFT_TRAIN']}"
        )
        return
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
