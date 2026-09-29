# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Launch isolated EAGLE-3 methods with a common vLLM inference engine."""

import argparse
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
DEFAULT_TORCH_THREADS = min(
    20,
    len(os.sched_getaffinity(0))
    if hasattr(os, "sched_getaffinity")
    else os.cpu_count() or 1,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("eagle", "tts", "ospec", "random_sampling"))
    parser.add_argument("--model-size", choices=("4b", "8b"), default="8b")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--target")
    parser.add_argument("--draft")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--spec-tokens", type=int, default=7)
    parser.add_argument("--max-model-len", type=int, default=4176)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-gpu-memory-gib", type=float)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--torch-threads", type=int, default=DEFAULT_TORCH_THREADS)
    parser.add_argument("--update-stride", type=int, default=1)
    parser.add_argument("--probability", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--ensemble-lrs", default="1e-5,2e-5,3e-5")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--keep-weights", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def configure(args):
    if args.spec_tokens < 1 or args.max_model_len < args.spec_tokens + 2:
        raise ValueError("context length must exceed the draft token count")
    if (
        args.update_stride < 1
        or args.chunk_size < 1
        or args.seed < 0
        or args.torch_threads < 1
    ):
        raise ValueError(
            "stride, chunk size, and torch threads must be positive; "
            "seed must be nonnegative"
        )
    if not 0 <= args.probability <= 1:
        raise ValueError("probability must be in [0,1]")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning rate must be finite and positive")
    if not math.isfinite(args.epsilon) or args.epsilon < 0:
        raise ValueError("learning rate must be positive and epsilon nonnegative")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("gpu memory utilization must be in (0,1]")
    max_gpu_memory_gib = args.max_gpu_memory_gib or (
        24.0 if args.model_size == "4b" else 32.0
    )
    if not math.isfinite(max_gpu_memory_gib) or max_gpu_memory_gib <= 0:
        raise ValueError("max GPU memory must be finite and positive")
    rates = [float(rate) for rate in args.ensemble_lrs.split(",")]
    if len(rates) != 3 or any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("ensemble-lrs must specify three positive values")
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
    gpu_memory_utilization = _capped_gpu_memory_utilization(
        args.gpu_memory_utilization,
        max_gpu_memory_gib,
        require_gpu=not args.dry_run,
    )
    env = os.environ.copy()
    env.update(
        {
            # The opt-in bootstrap installs observation hooks on the V2 runner.
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "OSD_METHOD": args.method,
            "OSD_DRAFT_MODEL": draft,
            "OSD_LEARNING_RATE": str(args.learning_rate),
            "OSD_TORCH_THREADS": str(args.torch_threads),
            "OSD_UPDATE_STRIDE": str(args.update_stride),
            "OSD_PROBABILITY": str(args.probability),
            "OSD_SEED": str(args.seed),
            "OSD_CHUNK_SIZE": str(args.chunk_size),
            "OSD_ENSEMBLE_LRS": args.ensemble_lrs,
            "OSD_EPSILON": str(args.epsilon),
            "OSD_KEEP_WEIGHTS": str(int(args.keep_weights)),
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
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--generation-config",
        "vllm",
        "--dtype",
        "bfloat16",
        "--enforce-eager",
        "--no-enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--spec-method",
        "eagle3",
        "--spec-model",
        draft,
        "--spec-tokens",
        str(args.spec_tokens),
    ]
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
        print(f"method={env['OSD_METHOD']} online={int(args.method != 'eagle')}")
        return
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
