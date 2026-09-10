# S=1 CPU Online Distillation

This branch implements synchronous request-local updates for the local,
single-layer Llama-style Qwen3-8B EAGLE3 checkpoint. The older CE step and async
bridge remain available as historical components; the runtime factory selects
`qwen3_eagle3_distillation_step`.

## Module Boundaries

The training package imports only PyTorch and the Python standard library. It
does not import vLLM, inspect serving configuration, or read vLLM environment
variables. It can be installed separately with `uv pip install -e ./online_eagle3`
in a CPU-only environment; a compatible PyTorch installation is required.

```text
online_eagle3/
  config.py          Validated immutable training configuration
  data.py            Observation, typed CPU batch, bridge/step interfaces
  factory.py         Lazy standalone trainer construction and dependency injection
  qwen3_trainer.py   Model, optimizer, request state and weight snapshots
  distillation.py    S=1 fixed-hidden forward/backward/update/append ordering
  loss.py            FP32 teacher projection and forward KL
  cache.py           Temporary forward cache and append-only persistent history
  torch_eagle3.py    Pure PyTorch draft architecture
  checkpoint.py      Checkpoint conversion and loading
  weights.py         Frozen parameter selection, snapshot export/application
  sync_bridge.py    Observe/update/publish/reset lifecycle
  runtime.py         CPU and PyTorch metadata for trace comparisons
  ce_step.py         Legacy CE step (not selected by the current runtime)
  observations.py    Legacy CE label helpers
  async_bridge.py    Legacy queue experiment (not the S=1 runtime)

vllm/v1/worker/gpu/spec_decode/
  online_eagle3_config.py  Serving guards and environment-to-config translation
  online_eagle3.py         GPU capture, verify alignment and weight application
```

`TrainObservation` and `Qwen3Eagle3StepFn` now live in `data.py`; checkpoint loading
now lives in `checkpoint.py`; loss functions now live in `loss.py`. The package
root exports the main public classes/functions. The old vLLM factory entry point
moved to `vllm/.../online_eagle3_config.py`.

To construct training without vLLM:

```python
import torch
from online_eagle3 import Qwen3Eagle3TrainerConfig, create_cpu_bridge

config = Qwen3Eagle3TrainerConfig(
    dtype=torch.bfloat16, torch_threads=8, lr=1e-5,
    weight_decay=0.0, check_gradients=True,
)
bridge = create_cpu_bridge("/path/to/eagle3", config, trace_dir="/tmp/cpu-trace")
# Feed each complete canvas using the payload contract documented below.
# bridge.observe_step(request_id, step_id, payload)
# snapshot = bridge.maybe_apply_pending_weights()
# bridge.reset_request(request_id)
bridge.shutdown()
```

`create_cpu_bridge(..., step_fn=custom_step)` replaces the update strategy without
changing vLLM callbacks. `Qwen3Eagle3CpuTrainer(..., optimizer_cls=...)` replaces the
optimizer when constructing a trainer directly. Direct trainer construction owns
neither model loading nor thread settings; its caller supplies the initialized
model. `create_cpu_bridge` applies those settings. S is deliberately restricted
to 1 for this distillation strategy.

## Configuration

Shell launchers map `ONLINE_EAGLE3_*` variables to the runtime variables below.
Direct `vllm serve` calls use the full `VLLM_ONLINE_EAGLE3_*` names.

| Runtime variable | Config field | Default |
| --- | --- | --- |
| `VLLM_ONLINE_EAGLE3` | Enable adapter | Off (trace/smoke enable it) |
| `VLLM_ONLINE_EAGLE3_DRAFT_MODEL` | Checkpoint path | Serving draft path |
| `VLLM_ONLINE_EAGLE3_DTYPE` | `dtype` | FP32; accepts `fp32`, `float32`, `bf16`, `bfloat16` |
| `VLLM_ONLINE_EAGLE3_LR` | `lr` | `1e-5` |
| `VLLM_ONLINE_EAGLE3_WEIGHT_DECAY` | `weight_decay` | `0` |
| `VLLM_ONLINE_EAGLE3_TORCH_THREADS` | `torch_threads` | PyTorch default; launchers use 8 |
| `VLLM_ONLINE_EAGLE3_UPDATE_INTERVAL` | `update_interval` | `1`, other values rejected |
| `VLLM_ONLINE_EAGLE3_CHECK_GRADIENTS` | `check_gradients` | True |

BF16 selects BF16 CPU weights, gradients, optimizer moments and KV. Teacher
probabilities and KL stay FP32; IDs/positions stay integer. There are no FP32
master weights in BF16 mode. Small updates can be lost to BF16 rounding. Disabling
gradient checks is an explicit benchmark option, not the default; keep it equal
across comparisons and use it only after validating numerical stability.

## vLLM Hooks

The runner only constructs the bridge after loading the draft, captures raw
teacher logits before sampling transforms, submits verification tensors, applies
pending weights before the next proposal, and forwards reset/shutdown events.
It no longer assembles training payload dictionaries.

The autoregressive speculator forwards proposal start, prepared prefill inputs,
each decode input, and proposal completion to `OnlineEagle3Adapter`. Capture is
outside CUDA graph replay. The adapter owns pending canvas state, request/step
alignment, teacher projection and confirmed-token shifting. Internal warmup
requests are skipped. Other speculators inherit no-op hooks.

The adapter passes only complete observations to the CPU bridge. The bridge
publishes a snapshot only after the update and confirmed-history append succeed.
Errors propagate. The GPU receives full trainable snapshots; its native KV
strategy is unchanged. Request completion/preemption restores baseline weights
and clears CPU optimizer/history state before the next request.

## Execution Order

1. GPU EAGLE3 proposes a complete canvas and the target verifies it.
2. Capture teacher distributions before sampling transforms, proposal inputs,
   and target auxiliary features for the confirmed path.
3. CPU evaluates the canvas in one causal forward using fixed GPU recurrent
   inputs and its own detached historical KV.
4. Compute forward KL and perform one AdamW step.
5. With the updated weights and no gradients, append newly confirmed positions
   to CPU history. Bootstrap the prompt here on the first update.
6. Publish the weight snapshot, apply it on GPU, then propose again.

GPU KV refresh and rejection handling are unchanged. CPU persistent KV entries
are never recalculated or overwritten during a request. Concatenating storage
may copy their bytes, but does not change their values. Different entries can
therefore originate from different weight versions. Temporary rollout KV never
becomes persistent history, even for accepted candidates.

The persistent history includes the latest confirmed draft input (target feature
plus the next known token embedding). On the next training step, this boundary
query is recomputed in a temporary cache using the prefix *before* that position,
so it has gradients without attending to itself twice. Its persistent entry
remains unchanged. Earlier history has no gradient. Later canvas positions use
detached `proposal_hidden_states` captured on GPU, not CPU recurrent outputs.
The first position still uses the trainable CPU target-feature fusion. On the
first round, the prompt and candidate suffix share one forward; subsequent rounds
evaluate the boundary and candidate suffix together. Only the last K outputs
contribute to the loss.

This removes cross-step recurrent gradients but retains causal attention
gradients through temporary K/V within the canvas. It is a different training
objective from differentiable autoregressive replay, not an exact acceleration
of that objective. Parallel here refers to canvas positions in a CPU forward;
CPU training and GPU inference still execute synchronously at S=1.

## Loss and Data

For K candidate positions, the loss is the mean of `KL(p_teacher || q_cpu)`.
The teacher is conditioned on the draft vocabulary by gathering the mapped
target logits and applying softmax at temperature 1. Vocabulary coverage is also
captured. All verified canvas positions contribute, including rejected branches;
bonus logits do not. Position weights are uniform and the old-draft regularizer
is disabled (lambda=0). This is the initial functional baseline, not an exact
reproduction of all TTS hyperparameters.

The payload contains:

- `prefill_positions`, `prefill_input_embeds`, `prefill_aux_hidden_states`:
  the complete valid GPU proposal prefill, including the prompt on the first round.
- `proposal_positions`, `proposal_input_embeds`, `proposal_hidden_states`:
  all K rollout inputs, recorded outside CUDA graph replay. Later recurrent
  hidden inputs are fixed; the first is recomputed from target auxiliary features.
- `teacher_probs`: K distributions over the draft vocabulary.
- `confirmed_positions`, `confirmed_input_embeds`, `confirmed_aux_hidden_states`:
  target-feature inputs shifted by one token, ending with the correction or bonus
  token. Only newly covered positions enter persistent CPU history.

S is fixed to 1. Chunked prefill, prefix caching, parallel drafting and multiple
requests/workers are unsupported in this baseline. A missing or inconsistent
observation raises an error rather than continuing with incomplete CPU history.
Request reset restores baseline parameters, clears optimizer state and CPU KV.
This prototype is not a long-context CPU memory optimization: attention and KV
concatenation still use ordinary PyTorch operations.

## Verification

```bash
.venv/bin/python -m pytest tests/standalone_tests/test_online_eagle3_*.py -q
PORT=8107 MAX_MODEL_LEN=512 MAX_NUM_BATCHED_TOKENS=512 \
  SMOKE_MAX_TOKENS=16 bash run/online_eagle3_smoke.sh
```

The shared experiment launcher disables prefix caching and chunked prefill for
both online and baseline runs. To launch manually, set
`VLLM_ONLINE_EAGLE3_UPDATE_INTERVAL=1` and pass `--no-enable-prefix-caching`,
`--no-enable-chunked-prefill` and `--max-num-seqs 1`.

Tests cover cached/full forward equivalence, KL gradients, post-update KV
construction, immutable historical entries, rejection boundaries, request reset
and S=1 validation. GPU smoke testing is required to exercise the serving and
CUDA graph capture paths in addition to these CPU tests.

## Draft Length and Context Traces

```bash
PORT=8107 NUM_SPECULATIVE_TOKENS=8 \
  MAX_MODEL_LEN=512 MAX_NUM_BATCHED_TOKENS=512 \
  TRACE_MAX_TOKENS=16 TRACE_MAX_ITERATIONS=32 \
  bash run/online_eagle3_trace.sh

PORT=8107 NUM_SPECULATIVE_TOKENS=8 TRACE_CONTEXT_TOKENS=4096 \
  MAX_MODEL_LEN=4608 MAX_NUM_BATCHED_TOKENS=4608 \
  TRACE_MAX_TOKENS=16 TRACE_MAX_ITERATIONS=32 \
  bash run/online_eagle3_trace.sh
```

`TRACE_CONTEXT_TOKENS=0` keeps the original text prompt. A positive value builds
an exact-length token-ID prompt from repeated reference notes followed by the
question. This is a synthetic timing workload, not a quality benchmark. Each run
saves `request.json`, `trace_config.json`, and server-reported token counts in
`response.json`. Explicit `SPECULATIVE_CONFIG` overrides `NUM_SPECULATIVE_TOKENS`.
Compare the first update separately: it includes full CPU prompt computation,
whereas later updates use detached historical KV.

## Moving to an AMX Host

Install this branch into the target machine's vLLM environment using the repo's
`uv` setup instructions. Do not copy the source machine's `.venv` or compilation
cache. The full serving trace requires a supported GPU as well as the AMX CPU.
Only the standalone training package can run without a GPU/vLLM installation.

Run the same workload in both precisions, supplying local model paths:

```bash
MODEL=/models/Qwen3-8B DRAFT=/models/Qwen3-8B_eagle3 \
  PORT=8107 ONLINE_EAGLE3_DTYPE=float32 ONLINE_EAGLE3_TORCH_THREADS=8 \
  NUM_SPECULATIVE_TOKENS=4 MAX_MODEL_LEN=512 MAX_NUM_BATCHED_TOKENS=512 \
  TRACE_MAX_TOKENS=16 TRACE_MAX_ITERATIONS=32 \
  bash run/online_eagle3_trace.sh

MODEL=/models/Qwen3-8B DRAFT=/models/Qwen3-8B_eagle3 \
  PORT=8107 ONLINE_EAGLE3_DTYPE=bf16 ONLINE_EAGLE3_TORCH_THREADS=8 \
  NUM_SPECULATIVE_TOKENS=4 MAX_MODEL_LEN=512 MAX_NUM_BATCHED_TOKENS=512 \
  TRACE_MAX_TOKENS=16 TRACE_MAX_ITERATIONS=32 \
  bash run/online_eagle3_trace.sh
```

`online_eagle3_cpu_runtime.json` records CPU flags, PyTorch version/build, thread
settings, effective training config and checkpoint path. AMX flags alone do not
prove an operator uses AMX. For a separate diagnostic run, `ONEDNN_VERBOSE=1` can
report oneDNN implementations when that backend is used; keep verbose diagnostics
off for timing comparisons. On multi-socket systems, keep CPU/NUMA placement and
thread counts controlled across runs.

Existing profiler labels are preserved. New labels separate `cpu_load_model`,
`cpu_prepare_batch`, `cpu_check_gradients` and `cpu_snapshot`. AdamW already has
its native optimizer label. `gpu_apply_weights` on the CPU track is host wall
time, not pure CUDA kernel time. CPU/GPU annotation tracks are reported separately.

This refactor does not switch attention backends or detach the first prompt's
training KV. Very long prompts still create quadratic CPU attention tensors;
AMX accelerates suitable compute but does not remove that memory requirement.
