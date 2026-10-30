# DFlash methods

`dflash` is vLLM's native block-diffusion drafter: one bonus token plus
`num_speculative_tokens` mask positions are proposed in a single non-causal
forward pass. [`methods/eagle`](../eagle/README.md) is the frozen EAGLE-3
baseline; this package follows the same non-invasive rule and keeps every
change outside `vllm/`. The vLLM checkout already contains the DFlash
speculator, so the frozen entry only selects the native implementation.

```bash
.venv/bin/python -m methods.run dflash --port 8005
.venv/bin/python -m methods.run tts --draft-method dflash --port 8001
.venv/bin/python -m methods.run online_draft --draft-method dflash --port 8008
.venv/bin/python -m methods.run random_sampling --draft-method dflash \
  --probability 0.5 --port 8002
```

`dflash` uses `--spec-method dflash`; `--spec-tokens` defaults to 15 for the
prepared `z-lab/Qwen3-*-DFlash-b16` drafts (block size 16). The frozen entry
installs no hooks. `online_draft`, `tts`, and `random_sampling` install the
optional worker bootstrap. `online_draft+dflash` uses vLLM's shared asynchronous
CPU pipeline and the remote DFlash recipe, configured by `max_train_rounds` and
the shared `uniform` / `position` / `entropy` loss preset. TTS sends updates to a side CUDA stream and
installs results after the next `S-1` rounds; `random_sampling` keeps its
synchronous update path.

TTS keeps its independent GPU training copy and CUDA side-stream. Its DFlash
loss uses the paper's block-position decay weights; it does not install the
online CPU adapter.

## Test-time draft updates

The DFlash TTS entry trains a separate copy of the drafter. Its trainable
block-diffusion parameters (`fc`, `hidden_norm`, `norm`, and decoder layers)
update on the side stream. Frozen embedding and LM-head parameters remain
shared with the serving drafter. At each stride boundary, TTS copies the
trainable weights into the serving model and rebuilds its fused context-KV
buffers. This keeps inference from reading weights while the optimizer changes
them.

One round is captured as follows:

1. At proposal time the adapter appends the accepted target auxiliary hidden
   states to a per-request context cache and records the block query tokens
   (bonus plus mask tokens) with their positions.
2. At verification time it projects the target next-token distribution of the
   same block onto the draft vocabulary and records how many tokens were
   accepted.
3. The training copy reruns the block forward from the captured context and
   applies forward KL over every row in the verified proposal canvas, including
   rows after a rejection.
4. It overlaps the next `S-1` proposal rounds. At the stride boundary, the
   adapter waits for the update, installs its trainable parameters, and rebuilds
   the serving model's fused context-KV buffers before the next proposal.

Training the full canvas follows the TTS objective, which distills the target
distribution at each drafted position. Rejected suffix rows remain part of the
supervision canvas. For the `b16` checkpoint, row `d` (zero based) uses the
original DFlash training weight `exp(-d / 7)`; there is no extra rejection-row
bonus. The bonus query row is not supervised: DFlash query rows `1..15` align
with target verification rows `0..14`.

The paper also writes a `KL(q_t || q)` regularizer, but evaluates its gradient
at `q = q_t` and takes exactly one step. That term has zero first derivative at
the evaluation point, so it is not materialized separately here; doing so would
not change the stated one-step update.

`--update-stride` controls TTS update cadence; `random_sampling` also gates
updates with `--probability`. Weights reset after each request unless
`--keep-weights` is set.
`ospec` remains EAGLE-3 only: its three independent learners would require
three drafter copies.

### Memory

The update keeps training activations, gradients, and optimizer state on the
serving GPU. TTS also keeps a training copy of trainable DFlash weights. Every
accepted token contributes its target auxiliary hidden states to a per-request
cache, and the block forward replays that cache from scratch each round. A
full 32k context with five auxiliary layers costs about
1.3 GiB in BF16 for an 8B target, and the cache capacity is reserved at model
load so vLLM's KV-cache budget accounts for it. Set
`--dflash-max-context-tokens N` to bound the cache and train on the most recent
`N` context tokens; the default, `0`, uses the full `--max-model-len`. The
prepared draft also ships no embedding table or LM head, so no target weights
are duplicated.

Gradients and AdamW state are allocated on the first update, after vLLM has
sized the KV cache, so leave several GiB of GPU headroom or lower
`--gpu-memory-utilization`. An 8B target plus the 1.2B DFlash drafter, its
trainable TTS copy, a full 32k context cache, gradients, and optimizer state
requires more memory than the frozen DFlash entry; the 4B pair is the
comfortable configuration on smaller devices.

## Preparation

`methods.prepare` downloads the DFlash drafts alongside the EAGLE-3 drafts:

| Selection | Draft |
| --- | --- |
| `4b` | `z-lab/Qwen3-4B-DFlash-b16` |
| `8b` | `z-lab/Qwen3-8B-DFlash-b16` |

```bash
.venv/bin/python -m methods.prepare models --model-size 8b
```

## Limitations

- One request per worker, tensor/pipeline/data parallelism 1, prefix caching
  and chunked prefill disabled, matching the other online entries.
  `--max-num-batched-tokens` must cover `--max-model-len` so a prompt is never
  split into prefill chunks.
- DFlash2 selectors and non-Qwen3 DFlash drafters are rejected.
- Quantized drafters are rejected; the prepared BF16 `z-lab` drafts are
  supported.
- `random_sampling` still updates the resident drafter synchronously. TTS uses
  a training copy so failed background updates do not expose partial weights
  to inference. The copy shares the frozen embedding, LM head, and serving
  attention/KV runtime, but clones every trainable draft weight.
- Older draft context K/V stays in the inference cache with the weights that
  produced it; the update rewrites only the accepted tokens' K/V. The training
  forward recomputes all context K/V from the cached features with the current
  weights. Per-step learning rates are small, so this staleness is bounded but
  not rebuilt.
