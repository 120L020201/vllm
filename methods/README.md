# vLLM EAGLE-3 method adapters

All four entries use the **same vLLM EAGLE-3 GPU target/draft inference path**
and matching Qwen3 checkpoints. Only the CPU draft-training policy changes.
The opt-in `methods/utils/bootstrap/sitecustomize.py` installs observation and
weight-publication callbacks in each vLLM worker process; it is enabled only by
`methods.run`. No vLLM source is modified. The runner selects the V2 GPU runner
and eager execution because the callback capture path is intentionally outside
CUDA graphs. Chunked prefill and prefix caching are disabled so the CPU trainer
always receives complete request-local prompt state.

```bash
.venv/bin/python -m methods.run eagle --port 8000
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
length 4176, and one concurrent request. The launcher caps Qwen3-4B at 24 GiB
and Qwen3-8B at 32 GiB by default, even if `--gpu-memory-utilization` requests
more. The vLLM budget reserves 2 GiB below either cap for CUDA runtime overhead;
lower the cap with `--max-gpu-memory-gib` when needed. `--dry-run` displays the
command without starting the server. For training, CPU snapshots are published
to the GPU draft between decoding steps. CPU training uses up to 20 PyTorch
threads by default; override this with `--torch-threads`.

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

GPQA requires accepting its Hugging Face access terms and authenticating with
`hf auth login`. In environments that need a mirror, use `--hf-endpoint` or
set `HF_ENDPOINT`. Existing output with missing or different provenance is
never overwritten.

**Semantic limits:** Each eligible update uses the current `online_draft`
confirmed-path forward-KL objective, including the target teacher distribution
captured during verification. `random_sampling` gates complete updates using a
separate seeded CPU generator (probability 1 matches TTS). `ospec` trains three
CPU draft learners at different learning rates after every N completed
requests, then merges trainable weights using softmax of cumulative KL losses.
It adapts the method policies to this repository; it does not reproduce the
papers' full replay or offline-training pipelines. OSpec persists learners
across request boundaries; TTS/random reset by default (`--keep-weights`
changes this). One request at a time is required for ordering and reset
semantics. Training data are copied from GPU to CPU, and OSpec chunk size
increases CPU memory use. Request completion drains pending CPU updates before
the next request; on this machine the 8B draft can add tens of seconds at that
boundary. `--update-stride` trades adaptation frequency for lower CPU cost.

Tests: `.venv/bin/python -m pytest methods/tests -v`.

## `experiments/run.sh` compatibility

See [`experiments/README.md`](../experiments/README.md) for the complete Chinese setup,
download, launch, output, and troubleshooting guide.

The repository also includes `experiments/run.sh` for the long-running `nohup env ...`
workflow. It accepts `base`, `eagle3`, `tts`, and `ospec`, starts the server,
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

Append `[N]` to sample `N` deterministic questions from one dataset while
running unsuffixed datasets in full. `SEED` controls both subset selection and
request generation, for example:

```bash
SEED=7 \
BENCHMARKS="aime2026[15] gpqa_diamond livecodebench-lite mmlu-pro-computer_science" \
bash experiments/run.sh base
```

LongWriter-6k contains 6000 long-generation prompts, so use
`LongWriter-6k[N]` unless a full run is intentional.

After every request the foreground experiment log prints:

```text
aime2026-10: AL=2.825, updates=1105, tokens=24987, time=720.8s, tokens/s=34.67, prefill=410 tokens/0.112s/3660.71 tokens/s, decode=24986 tokens/287.400s/86.94 tokens/s
```

`AL` is `1 + accepted draft tokens / speculative rounds`, and is `1.000` for
the base method. `updates` is the number of completed CPU optimizer steps,
`tokens` is the completion token count, and time includes waiting for queued
CPU updates belonging to the request. Prefill and decode use vLLM's server-side
request phase timers. The decode count excludes the first output token because
vLLM attributes that token to prefill.

After each dataset, the log prints its round-weighted speculative acceptance
length as `dataset_AL=1+sum(accepted_tokens)/sum(drafts)`. Raw per-request
speculative counters and phase timings are stored in the JSONL result. Aggregate
dataset metrics are written to the adjacent `<output>.summary.json` file.

With the default `MAX_OUTPUT_TOKENS=32768`, the run script uses a 49152-token
context and left-truncates prompts to 16384 tokens. Qwen3 natively declares
40960 tokens, so the extra 8192 tokens use vLLM's explicit long-context override
and RoPE extrapolation. This evaluates all 503 LongBench v2 examples; quality in
the extrapolated region must be confirmed by the benchmark results.

The script maps `TTS_UPDATE_STRIDE`, `TTS_LEARNING_RATE`,
`TTS_RESET_PER_REQUEST`, and `CHUNK_SIZE` to the method launcher. `D_TEMPERATURE`
is accepted for command compatibility but EAGLE3 drafts remain greedy. Online
training is intentionally single-request ordered; `BATCH_SIZE=2` is recorded as
a warning rather than enabling two concurrent requests. `OSPEC_TRAIN_MAX_LEN`
and `UPDATE_DELAY` are also accepted with warnings because the current runtime
does not truncate training prompts or defer updates.

Every `experiments/run.sh` mode enforces a model-sized GPU policy: Qwen3-4B defaults
to a 24 GiB hard limit and Qwen3-8B defaults to 32 GiB. In both cases,
`GPU_MEMORY_HEADROOM_GIB=2` is reserved for CUDA/runtime overhead. Override
`MAX_GPU_MEMORY_GIB` explicitly for nonstandard model directory names.
`MONITOR=1` writes periodic `nvidia-smi` samples to
`<output>.gpu.csv`; `NSYS` is accepted but profiler wrapping is not implemented.
Use `DRY_RUN=1` to inspect the resolved server command without starting it.

Runs resume by default. Base and EAGLE3 skip successful questions individually;
TTS and OSpec retain complete datasets but restart the first incomplete dataset
from its first question. Set `RUN_LOG` to let the script append it for resumed
runs and overwrite it together with the result files when `RESUME=0`.
