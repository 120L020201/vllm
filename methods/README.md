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
`/v1/completions` requests and identical sampling parameters for comparisons.
Use `--model-size 4b` or `--model-size 8b` to select a prepared target/draft
pair. `--target` and `--draft` override either checkpoint location. Defaults
are the 8B pair under `methods/artifacts`, BF16, seven draft tokens, context
length 4176, and one concurrent request. The launcher caps vLLM at 24 GiB of
the first visible GPU by default, even if `--gpu-memory-utilization` requests
more. The vLLM budget reserves 2 GiB below that cap for CUDA runtime overhead;
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

## `script/run.sh` compatibility

The repository also includes `script/run.sh` for the long-running `nohup env ...`
workflow. It accepts `base`, `eagle3`, `tts`, and `ospec`, starts the server,
waits for `/health`, sends every prompt in `BENCHMARKS` to the completions API,
and writes incremental JSONL output. The benchmark files are read as
`$DATA_DIR/<benchmark>.jsonl`.

Prepare the complete benchmark files with:

```bash
HF_ENDPOINT=https://hf-mirror.com \
  .venv/bin/python script/prepare_datasets.py --output-dir /srv/Datasets/TTS
```

The command validates the complete official row counts and writes
`aime2026.jsonl` (30), `gpqa_diamond.jsonl` (198),
`mmlu-pro-computer_science.jsonl` (410), `livecodebench-lite.jsonl` (1055),
and `LongBench-v2.jsonl` (503). It refuses to overwrite existing files unless
`--force` is supplied.

The default benchmark list is:

```text
aime2026 gpqa_diamond mmlu-pro-computer_science livecodebench-lite LongBench-v2
```

After every request the foreground experiment log prints:

```text
aime2026-10: AL=2.825, updates=1105, tokens=24987, time=720.8s, tokens/s=34.67
```

`AL` is `1 + accepted draft tokens / speculative rounds`, and is `1.000` for
the base method. `updates` is the number of completed CPU optimizer steps,
`tokens` is the completion token count, and time includes waiting for queued
CPU updates belonging to the request.

With the default `MAX_OUTPUT_TOKENS=32768`, the run script uses Qwen3's 40960
token context and left-truncates prompts to 8192 tokens. This still evaluates
all 503 LongBench v2 examples; increase `MAX_MODEL_LEN`/`MAX_PROMPT_TOKENS` only
when the selected model and the 24 GiB memory limit permit it.

The script maps `TTS_UPDATE_STRIDE`, `TTS_LEARNING_RATE`,
`TTS_RESET_PER_REQUEST`, and `CHUNK_SIZE` to the method launcher. `D_TEMPERATURE`
is accepted for command compatibility but EAGLE3 drafts remain greedy. Online
training is intentionally single-request ordered; `BATCH_SIZE=2` is recorded as
a warning rather than enabling two concurrent requests. `OSPEC_TRAIN_MAX_LEN`
and `UPDATE_DELAY` are also accepted with warnings because the current runtime
does not truncate training prompts or defer updates.

Every `script/run.sh` mode enforces the same GPU policy: `MAX_GPU_MEMORY_GIB`
defaults to 24, with `GPU_MEMORY_HEADROOM_GIB=2` reserved for CUDA/runtime
overhead. `MONITOR=1` writes periodic `nvidia-smi` samples to
`<output>.gpu.csv`; `NSYS` is accepted but profiler wrapping is not implemented.
Use `DRY_RUN=1` to inspect the resolved server command without starting it.
