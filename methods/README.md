# vLLM EAGLE-3 method adapters

All entries use vLLM EAGLE-3 GPU target/draft inference and matching Qwen3
checkpoints. `online_draft` trains on the CPU; TTS and OSpec train on the GPU.
The opt-in `methods/utils/bootstrap/sitecustomize.py` installs observation and
weight-publication callbacks in each vLLM worker process; it is enabled only by
`methods.run`. Draft training uses ordinary PyTorch eager execution on CUDA.
For `eagle` and the GPU-training entries, target inference, target verification,
and the GPU EAGLE3 draft run with
`FULL_AND_PIECEWISE` CUDA Graph mode. A small opt-in speculator path keeps the
training snapshots graph-safe; it is inactive for launchers that do not install
the online callbacks. The built-in `online_draft` path still forces GPU draft
prefill/decode to eager at runtime, despite the requested graph configuration.
Chunked prefill and prefix caching are disabled by default.

```bash
.venv/bin/python -m methods.run eagle --port 8000
.venv/bin/python -m methods.run online_draft --port 8004
.venv/bin/python -m methods.run tts --port 8001
.venv/bin/python -m methods.run random_sampling --port 8002 --probability 0.5
.venv/bin/python -m methods.run ospec --port 8003 --chunk-size 5
```

Each command runs one server in the foreground. Use the same OpenAI-compatible
`/v1/chat/completions` requests, chat template, and sampling parameters for
comparisons.
Use `--model-size 4b` or `--model-size 8b` to select a prepared target/draft
pair. `--target` and `--draft` override either checkpoint location. Defaults
are the 8B pair under `methods/artifacts`, BF16, seven draft tokens, context
length 32768, chunked prefill disabled, 0.8 GPU memory utilization, and one
concurrent request. Remote model code is trusted by default to match the
artifact scripts. There is no model-size-based memory cap by default.
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

The model plan contains both target checkpoints and the matching EAGLE3 draft
checkpoints required by every method:

| Selection | Target | Draft |
| --- | --- | --- |
| `4b` | `Qwen/Qwen3-4B` | `AngelSlim/Qwen3-4B_eagle3` |
| `8b` | `Qwen/Qwen3-8B` | `AngelSlim/Qwen3-8B_eagle3` |

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
| `longwriter_6k` | `zai-org/LongWriter-6k` | `train` |

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

**Semantic limits:** Each eligible update uses the current `online_draft`
confirmed-path forward-KL objective, including the target teacher distribution
captured during verification. `random_sampling` gates complete updates using a
separate seeded CPU generator (probability 1 matches TTS). `ospec` trains three
GPU-resident draft learners at different learning rates after every N completed
requests, then merges trainable weights using softmax of cumulative KL losses.
It adapts the method policies to this repository; it does not reproduce the
papers' full replay or offline-training pipelines. OSpec persists learners
across request boundaries; TTS/random reset by default (`--keep-weights`
changes this). One request at a time is required for ordering and reset
semantics. Training data remain on CUDA, and OSpec chunk size increases GPU
memory use. Request completion runs pending GPU updates synchronously before
the next request. `--update-stride` trades adaptation frequency for lower GPU
training cost.

Tests: `.venv/bin/python -m pytest methods/tests -v`.

## `experiments/run.sh` compatibility

See [`experiments/README.md`](../experiments/README.md) for the complete Chinese setup,
download, launch, output, and troubleshooting guide.

The repository also includes `experiments/run.sh` for the long-running `nohup env ...`
workflow. It accepts `base`, `eagle3`, `online_draft`, `tts`, and `ospec`, starts the server,
waits for `/health`, sends every prompt in `BENCHMARKS` to the chat completions
API, and writes incremental JSONL output. The benchmark files are read as
`$DATA_DIR/<benchmark>.jsonl`.

Prepare the complete benchmark files with:

```bash
HF_ENDPOINT=https://hf-mirror.com \
  .venv/bin/python experiments/prepare_datasets.py --output-dir /srv/Datasets
```

The command validates the complete official row counts and writes
`aime2026.jsonl` (30), `gpqa_diamond.jsonl` (198),
`mmlu-pro-computer_science.jsonl` (410), `livecodebench-lite.jsonl` (1055),
`LongBench-v2.jsonl` (503), and `LongWriter-6k.jsonl` (6000). It refuses to
overwrite existing files unless
`--force` is supplied.

To prepare selected datasets and model pairs through one command, use:

```bash
DATASETS="aime2026 gpqa_diamond LongBench-v2" \
MODEL_SIZES="4b 8b" \
HF_ENDPOINT=https://hf-mirror.com \
bash experiments/prepare.sh
```

`experiments/prepare.sh` defaults to all six complete datasets under
`/srv/Datasets` and the Qwen3 4B/8B target plus matching EAGLE3 checkpoints
under `/srv/Models`. Existing complete files and model directories are
validated and skipped. Use `DATASETS=""` or `MODEL_SIZES=""` to skip either
class, and `DRY_RUN=1` to inspect the resolved commands. GPQA requires prior
acceptance of its Hugging Face access terms and an authenticated token.

The default benchmark list is:

```text
aime2026 gpqa_diamond mmlu-pro-computer_science livecodebench-lite LongBench-v2
```

Append `(N)` to sample `N` deterministic questions from one dataset. Append
`[N]` to run only its Nth question, using one-based numbering. Unsuffixed
datasets run in full. Keep `BENCHMARKS` quoted because parentheses are special
to the shell. `SEED` controls both subset selection and request generation, for
example:

```bash
SEED=7 \
BENCHMARKS="aime2026(15) gpqa_diamond livecodebench-lite mmlu-pro-computer_science" \
bash experiments/run.sh base
```

LongWriter-6k contains 6000 long-generation prompts, so use
`LongWriter-6k(N)` unless a full run is intentional. For example,
`LongWriter-6k[10]` runs only its tenth prompt.

After every request the foreground experiment log prints:

```text
aime2026-10: AL=2.825, updates=1105, tokens=24987, time=720.8s, tokens/s=34.67, prefill=410 tokens/0.112s/3660.71 tokens/s, decode=24986 tokens/287.400s/86.94 tokens/s
```

`AL` is `1 + accepted draft tokens / speculative rounds`, and is `1.000` for
the base method. `updates` is the number of completed GPU optimizer steps,
`tokens` is the completion token count, and time includes synchronous GPU
updates belonging to the request. Prefill and decode use vLLM's server-side
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
`TTS_RESET_PER_REQUEST`, and `CHUNK_SIZE` to the method launcher. `D_TEMPERATURE`
is accepted for command compatibility but EAGLE3 drafts remain greedy. Online
training is intentionally single-request ordered; `BATCH_SIZE=2` is recorded as
a warning rather than enabling two concurrent requests. `OSPEC_TRAIN_MAX_LEN`
and `UPDATE_DELAY` are also accepted with warnings because the current runtime
does not truncate training prompts or defer updates.

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

The built-in CPU online trainer is exposed through the `online_draft` entry,
which automatically sets `VLLM_ONLINE_DRAFT_TRAIN=1`. All other entries set it to
`0`. The old `eagle3` plus `VLLM_ONLINE_DRAFT_TRAIN=1` combination is rejected;
remove the environment override and select `online_draft` instead. Its `CHECK_GRADIENTS`,
`FROZEN_PARAMETER_PREFIXES`, `FUSED_OPTIMIZER`, `LOG_UPDATES`, and
`VLLM_ONLINE_DRAFT_PROFILE` settings are configurable from `run.sh`. TTS and
OSpec continue to use GPU eager training; CPU-only gradient and freezing
settings do not alter them. `VLLM_ONLINE_DRAFT_PROFILE=1` requires `online_draft`.
The same method-controlled CPU training switch applies to `methods.run`.

Only Base and EAGLE3 support resume and default to `RESUME=1`, skipping successful
questions individually. `online_draft`, TTS, and OSpec default to `RESUME=0` and
reject `RESUME=1` before starting the server or changing outputs. These training
methods always restart from the checkpoint and the beginning of the selected
questions because complete training state is not restored. The default timestamp
creates a new run directory; explicitly reusing output paths with `RESUME=0`
overwrites the old results and logs. Set `RUN_LOG` to manage the foreground log
the same way.
