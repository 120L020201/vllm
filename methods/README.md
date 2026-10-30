# vLLM EAGLE-3 and DFlash method adapters

All entries use vLLM GPU target/draft inference and matching Qwen3 checkpoints.
`dflash` and `mtp` are frozen native entries; other entries use EAGLE-3 drafts
by default. All three `online_draft` recipes use vLLM's asynchronous CPU
trainer. TTS and OSpec train on the GPU, and TTS supports EAGLE-3, DFlash, and
target-embedded MTP drafts.
The opt-in `methods/utils/bootstrap/sitecustomize.py` installs observation and
weight-publication callbacks in each vLLM worker process; it is enabled only by
`methods.run`. TTS and OSpec use ordinary PyTorch eager training on CUDA.
For `eagle`, `dflash`, and the GPU-training entries, target inference, target
verification, and the EAGLE-3/DFlash draft run with
`FULL_AND_PIECEWISE` CUDA Graph mode. TTS trains separate EAGLE-3 and DFlash
copies on a CUDA side stream and uses a CPU-side barrier before graph replay;
it overlaps the next `S-1` speculation rounds. `online_draft+dflash` uses the
remote DFlash recipe and shared CPU pipeline. `random_sampling` updates the
resident DFlash drafter synchronously. The built-in `online_draft+eagle3` path
still forces GPU draft prefill/decode to eager at runtime, despite the requested
graph configuration. EAGLE-3 and DFlash async paths train separate copies so
inference reads stable weights.
Chunked prefill and prefix caching are disabled by default.

`online_draft+eagle3` uses the native `online_draft.runtime` capture, pinned D2H
transport, and CPU trainer from `artifacts/scripts/benchmark_aime_2026_online.py`.
Its defaults match that script: gradient checks off, weight-update logging on,
and `OMP_NUM_THREADS=20` with `MKL_NUM_THREADS` following it unless overridden.
`online_draft+dflash` uses the remote DFlash recipe and shared asynchronous
CPU training pipeline, while target and draft inference remain on GPU.

```bash
.venv/bin/python -m methods.run eagle --port 8000
.venv/bin/python -m methods.run dflash --port 8005
.venv/bin/python -m methods.run mtp --target /path/to/Qwen3-8B-MTP --port 8009
.venv/bin/python -m methods.run online_draft --port 8004
.venv/bin/python -m methods.run online_draft --draft-method dflash --port 8008
.venv/bin/python -m methods.run online_draft --draft-method mtp \
  --target /path/to/Qwen3-8B-MTP --port 8010
.venv/bin/python -m methods.run tts --port 8001
.venv/bin/python -m methods.run tts --draft-method dflash --port 8006
.venv/bin/python -m methods.run tts --draft-method mtp \
  --target /path/to/Qwen3-8B-MTP --port 8011
.venv/bin/python -m methods.run random_sampling --port 8002 --probability 0.5
.venv/bin/python -m methods.run random_sampling --draft-method dflash \
  --probability 0.5 --port 8007
.venv/bin/python -m methods.run ospec --port 8003 --chunk-size 5
```

Each command runs one server in the foreground. Use the same OpenAI-compatible
`/v1/chat/completions` requests, chat template, and sampling parameters for
comparisons.
Use `--model-size 4b` or `--model-size 8b` to select a prepared target/draft
pair. `--target` and `--draft` override either checkpoint location. Defaults
are the 8B pair under `methods/artifacts`, BF16, context length 32768, chunked
prefill disabled, 0.8 GPU memory utilization, and one concurrent request.
EAGLE-3 uses seven draft tokens; DFlash drafts use 15 (`b16` blocks). Select a
DFlash draft explicitly with `--draft-method dflash`, or implicitly with the
`dflash` entry; `--spec-tokens` overrides either default. Remote model code is
trusted by default to match the artifact scripts. There is no model-size-based
memory cap by default.
MTP defaults to three tokens and uses weights embedded in the target, so it
does not accept `--draft`. The online recipe supports Qwen3 checkpoints with
`num_nextn_predict_layers`, such as `Qwen3-8B-MTP`, and Qwen3.5 MTP checkpoints;
standard `Qwen/Qwen3-8B` has no embedded MTP layers.
The TTS recipe trains a copy of the embedded MTP layer on its CUDA side stream;
the target embedding and LM head remain shared and frozen.
Use `--language-model-only` for text benchmarks on multimodal Qwen3.5 targets;
the online MTP path requires text-only draft inputs.
Target and draft GPU inference both explicitly select `FLASHINFER`, with
FlashInfer autotuning disabled to match the artifact scripts and avoid its
long-prefill startup memory peak. Override these with `--gpu-attention-backend`
and `--flashinfer-autotune`. The CPU trainer's attention setting is independent.
Set `--max-gpu-memory-gib` only when a hard cap is required; the cap reserves
2 GiB for CUDA/runtime overhead. `--dry-run` displays the command without
starting the server. For the GPU-training entries, training batches, optimizer
state, and published draft weights remain on the serving CUDA device.

## Prepare datasets and models

Install the preparation dependency with `uv`, then inspect or execute the
pinned download plan:

```bash
uv pip install -r methods/requirements.txt
.venv/bin/python -m methods.prepare all --dry-run
.venv/bin/python -m methods.prepare all
```

The default output is `methods/artifacts/`, which is ignored by Git. A shared
location can be selected with `--root`; pass the same path to the launcher:

```bash
.venv/bin/python -m methods.prepare all --root /srv/vllm-osd
.venv/bin/python -m methods.run tts --model-size 4b \
  --artifact-root /srv/vllm-osd
```

The model plan contains both target checkpoints and the matching EAGLE-3 and
DFlash draft checkpoints:

| Selection | Target | EAGLE-3 draft | DFlash draft |
| --- | --- | --- | --- |
| `4b` | `Qwen/Qwen3-4B` | `AngelSlim/Qwen3-4B_eagle3` | `z-lab/Qwen3-4B-DFlash-b16` |
| `8b` | `Qwen/Qwen3-8B` | `AngelSlim/Qwen3-8B_eagle3` | `z-lab/Qwen3-8B-DFlash-b16` |

The dataset plan writes one normalized `data.jsonl` plus `manifest.json` per
selection. Revisions are pinned so comparisons can be reproduced.

| Output | Hugging Face source | Selection |
| --- | --- | --- |
| `aime2026` | `math-ai/aime26` | `train` (30 AIME 2026 questions) |
| `gpqa_diamond` | `Idavidrein/gpqa` | `gpqa_diamond`, `train` |
| `mmlu_pro` | `TIGER-Lab/MMLU-Pro` | `test` |
| `computer_science` | `TIGER-Lab/MMLU-Pro` | `test`, computer science only |
| `livecodebench_lite` | `livecodebench/code_generation_lite` | `release_v6` |
| `longbench_v2` | `THUDM/LongBench-v2` | `train` |

## Run the artifact AIME scripts directly

The three scripts under `methods/artifacts/scripts` now run without required
arguments and use the prepared 8B target/draft pair under
`methods/artifacts/models`:

```bash
.venv/bin/python methods/artifacts/scripts/benchmark_aime_2026.py
.venv/bin/python methods/artifacts/scripts/benchmark_aime_2026_online.py
.venv/bin/python \
  methods/artifacts/scripts/benchmark_aime_2026_speculative_baseline.py
```

The first command defaults to `graph_speculative`; pass `--mode` to select one
of its other comparison modes. The online command enables CPU draft training,
keeps target inference under CUDA Graph, and defaults both `OMP_NUM_THREADS`
and `MKL_NUM_THREADS` to 20. All three commands use a 32768-token context,
disable chunked prefill, trust remote model code, and use the full prompt with
the remaining context available for generation.

On the local 32 GiB GPU these launchers default to
`GPU_MEMORY_UTILIZATION=0.9` and disable FlashInfer autotuning to avoid its
32768-token startup memory spike. Set `FLASHINFER_AUTOTUNE=1` only when enough
extra GPU memory is available. `MODEL`, `SPEC_MODEL`, `MAX_MODEL_LEN`,
`MAX_NUM_BATCHED_TOKENS`, `GPU_MEMORY_UTILIZATION`, `OMP_NUM_THREADS`, and
`MKL_NUM_THREADS` can be overridden through the environment.

Results default to
`results/<method>-aime2026-Qwen3-8B-<YYYYMMDD_HHMMSS>/results.json`.
Use `--output` to select an explicit file.

GPQA requires accepting its Hugging Face access terms and authenticating with
`hf auth login`. In environments that need a mirror, use `--hf-endpoint` or
set `HF_ENDPOINT`. Existing output with missing or different provenance is
never overwritten.

**Semantic limits:** TTS updates use the full proposal canvas and target
distribution captured during verification. EAGLE-3, DFlash, and MTP copy the
updated trainable weights into the serving draft at each stride boundary.
`random_sampling` gates complete updates using a separate seeded CPU generator
(probability 1 matches TTS). `ospec` trains three
GPU-resident draft learners at different learning rates after every N completed
requests, then merges trainable weights using softmax of cumulative KL losses.
It adapts the method policies to this repository; it does not reproduce the
papers' full replay or offline-training pipelines. OSpec persists learners
across request boundaries; TTS/random reset by default (`--keep-weights`
changes this). One request at a time is required for ordering and reset
semantics. Training data remain on CUDA, and OSpec chunk size increases GPU
memory use. Request completion drains pending updates before the next request.
`--update-stride` trades adaptation frequency for lower GPU training cost.

DFlash `tts` and `random_sampling` follow the same one-request,
reset-by-default contract. DFlash TTS trains a copy of the trainable DFlash
parameters on its side stream, then installs them at the stride boundary;
`random_sampling` retains its synchronous in-place update path. Each TTS update
uses all rows in the verified proposal canvas. The embedding table and LM head
remain shared and frozen.
Accepted auxiliary hidden states are cached per request and replayed each
round, so long prompts dominate training memory. The cache reserves
`N * auxiliary_width` at load, where `N` is `--max-model-len` unless
`--dflash-max-context-tokens` caps training to the newest `N` context tokens.
`ospec` remains EAGLE-3 only. See [`dflash/README.md`](dflash/README.md).

MTP TTS trains a CUDA-resident copy of the embedded MTP layer and keeps the
target embedding and LM head shared and frozen. It therefore needs no separate
draft checkpoint; `--draft-method mtp` with a target containing `mtp.*` weights
selects it. The same verified proposal canvas is used for its full-vocabulary
forward KL loss.

Tests: `.venv/bin/python -m pytest methods/tests -v`.

## `experiments/run.sh` compatibility

See [`experiments/README.md`](../experiments/README.md) for the complete Chinese setup,
download, launch, output, and troubleshooting guide.

The repository also includes `experiments/run.sh` for the long-running `nohup env ...`
workflow. It accepts frozen EAGLE3/DFlash/MTP, all three online draft recipes,
and EAGLE3/DFlash TTS. It starts the server,
waits for `/health`, sends every prompt in `BENCHMARKS` to the chat completions
API, and writes incremental JSONL output. The benchmark files are read as
`$DATA_DIR/<benchmark>.jsonl`.

Before launching, `run.sh` rejects an occupied `HOST:PORT` without changing
existing outputs or stopping the old service. Readiness requires the launched
process to remain alive, a matching listener owned by it or its descendants, and
a successful `/health` response. Probes time out after two seconds, with a
300-second startup deadline. Set another `PORT` if needed; `DRY_RUN=1` skips the
port probe. Wildcard binds use a loopback address for benchmark requests.

Prepare the complete benchmark files with:

```bash
HF_ENDPOINT=https://hf-mirror.com \
  .venv/bin/python experiments/prepare_datasets.py --output-dir /srv/Datasets \
  --dataset aime2026 --dataset gpqa_diamond \
  --dataset mmlu-pro-computer_science --dataset livecodebench-lite \
  --dataset LongBench-v2
```

The command validates the complete official row counts and writes
`aime2026.jsonl` (30), `gpqa_diamond.jsonl` (198),
`mmlu-pro-computer_science.jsonl` (410), `livecodebench-lite.jsonl` (1055), and
`LongBench-v2.jsonl` (503). It refuses to overwrite existing files unless
`--force` is supplied.

To prepare selected datasets and model pairs through one command, use:

```bash
DATASETS="aime2026 gpqa_diamond LongBench-v2" \
MODEL_SIZES="4b 8b" \
HF_ENDPOINT=https://hf-mirror.com \
bash experiments/prepare.sh
```

`experiments/prepare.sh` downloads nothing by default. Explicitly select datasets
with `DATASETS` or repeated `--dataset NAME` options, and model pairs with
`MODEL_SIZES` or repeated `--model-size 4b|8b` options. The experiments
downloader defaults to EAGLE-3 drafts; select `DRAFT_METHODS=dflash` for DFlash
or `DRAFT_METHODS="eagle3 dflash"` for both. `mtp` records that no separate
draft download exists. The corresponding repeated CLI
option is `--draft-method`. Unselected classes are skipped; an already exported
environment variable also counts as a selection. CLI options override the
corresponding environment variable. Dataset and model output directories still
default to `/srv/Datasets` and `/srv/Models`.
Existing complete files and model directories are validated and skipped.
Use `DRY_RUN=1` to inspect the resolved commands. The Python entry points
`experiments/prepare_datasets.py` and `experiments/prepare_models.py` also require
explicit `--dataset` / `--model-size` selections; omitting them downloads nothing.
GPQA requires prior acceptance of its Hugging Face access terms and an
authenticated token.

The default benchmark list is:

```text
aime2026 gpqa_diamond mmlu-pro-computer_science livecodebench-lite LongBench-v2
```

Append `(N)` to sample `N` deterministic questions from one dataset. Append
`[N]` or `[N,M,...]` to run only the selected questions, using one-based
numbering. Unsuffixed datasets run in full. Keep `BENCHMARKS` quoted because
parentheses are special to the shell. `SEED` controls both subset selection and
request generation, for example:

```bash
SEED=7 \
BENCHMARKS="aime2026(15) gpqa_diamond livecodebench-lite mmlu-pro-computer_science" \
bash experiments/run.sh base
```

After every request the foreground experiment log prints:

```text
aime2026-10: AL=2.825, weight_installs=120, cpu_rounds/train=8.25, tokens=24987, time=720.8s, tokens/s=34.67, prefill=410 tokens/0.112s/3660.71 tokens/s, decode=24986 tokens/287.400s/86.94 tokens/s
```

`AL` is `1 + accepted draft tokens / speculative rounds`, and is `1.000` for
the base method. `weight_installs` counts newly trained weight installations into
the GPU inference draft, excluding initial-weight restoration. It is not an
optimizer-step count: one OSpec chunk merge counts as one installation.
`cpu_rounds/train` is the total proposal rounds in successfully trained CPU
windows divided by the number of those training calls. It excludes warmup and
untrained trailing rounds, and is `N/A` for GPU training or no completed CPU
training. The JSONL fields are `weight_installs` and `cpu_rounds_per_train`;
they replace the old `updates` field. Missing, invalid or incomplete statistics
produce `N/A` / JSON `null`, not a misleading zero.

These counters work with `LOG_UPDATES=0`. CPU statistics are attached by
`methods/utils/cpu_stats.py` without modifying vLLM sources or artifacts scripts.
The launcher exports `OSD_STATS_FILE` for every method. `tokens` is the completion
token count; time includes the HTTP request, its background cleanup and weight
installation wait, and the following server-metrics read. Warmup is excluded.
Prefill and decode use vLLM's server-side
request phase timers. The decode count excludes the first output token because
vLLM attributes that token to prefill.

After each dataset, the log prints its round-weighted speculative acceptance
length as `dataset_AL=1+sum(accepted_tokens)/sum(drafts)`. Raw per-request
speculative counters and phase timings are stored in the JSONL result. Aggregate
dataset metrics are written to the adjacent `<output>.summary.json` file.

The run script defaults to a 32768-token context, keeps the complete prompt,
and gives each request the remaining context as its output limit. A positive
`MAX_OUTPUT_TOKENS` adds an output cap; a positive `MAX_PROMPT_TOKENS` opts into
left truncation. Chunked prefill can be enabled explicitly with
`ENABLE_CHUNKED_PREFILL=1` and `MAX_NUM_BATCHED_TOKENS`.

The script maps `TTS_UPDATE_STRIDE`, `TTS_LEARNING_RATE` (default `1e-5`),
`TTS_RESET_PER_REQUEST`, and `CHUNK_SIZE` to the method launcher. Online
training is intentionally single-request ordered.

Every `experiments/run.sh` mode defaults to `GPU_MEMORY_UTILIZATION=0.9` without
a model-size-based hard cap. `GPU_ATTENTION_BACKEND=FLASHINFER` explicitly sets
both target and draft inference (target only for Base), and
`FLASHINFER_AUTOTUNE=0` disables startup kernel autotuning. Neither option changes
the CPU trainer's attention backend. A FlashInfer sampling log does not identify
the attention backend; check the attention configuration and backend log.
Set `MAX_GPU_MEMORY_GIB` explicitly to enable a
cap; `GPU_MEMORY_HEADROOM_GIB=2` is then reserved for CUDA/runtime overhead.
Each run defaults to
`results/<method>-<datasets>-<model>-<YYYYMMDD_HHMMSS>/`, with stable filenames
such as `results.jsonl`, `server.log`, and `monitor.csv` inside. `MONITOR=1`
writes host CPU/memory, server-process-tree CPU/RSS, and GPU samples every five
seconds. Use `DRY_RUN=1` to inspect the resolved server command without starting
it.

The built-in CPU online trainer is selected by all three `online_draft` aliases,
which automatically set `VLLM_ONLINE_DRAFT_TRAIN=1`.
Other entries set it to `0`. `CHECK_GRADIENTS`, `FROZEN_PARAMETER_PREFIXES`,
`FUSED_OPTIMIZER`, and `LOG_UPDATES` configure the shared trainer;
`VLLM_ONLINE_DRAFT_PROFILE` applies to all recipes. `ONLINE_DRAFT_LOSS` selects
`uniform`, `position`, or `entropy`; DFlash also accepts
`DFLASH_MAX_TRAIN_ROUNDS`.

Base, EAGLE3, DFlash, and MTP support resume and default to `RESUME=1`, skipping successful
questions individually. `online_draft`, TTS, and OSpec default to `RESUME=0` and
reject `RESUME=1` before starting the server or changing outputs. These training
methods always restart from the checkpoint and the beginning of the selected
questions because complete training state is not restored. The default timestamp
creates a new run directory; explicitly reusing output paths with `RESUME=0`
overwrites the old results and logs. Set `RUN_LOG` to manage the foreground log
the same way.
