# Stage 3 real Qwen3-8B CPU training validation

## Status

The standalone CPU training path passed this validation on 2026-09-20.
The reduced-vocabulary teacher-coverage behavior described below is a deferred
optimization. It is not treated as a blocker for moving beyond stage 3, and no
coverage-based filtering or weighting is currently implemented.

This experiment validates a real target/draft checkpoint pair and a batch
captured from real GPU forwards. The batch was produced by a standalone capture
script, not by the future vLLM runtime adapter.

## Configuration

| Item | Value |
| --- | --- |
| Target checkpoint | `/data/models/Qwen3-8B` |
| Draft checkpoint | `/data/models/Qwen3-8B-eagle3` |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| PyTorch | 2.11.0+cu130 |
| Feature and model dtype | BF16 |
| Target auxiliary layers | 2, 18, 33 |
| Draft length | 3 |
| Window size | 2 rounds |
| Optimizer | Fused AdamW |
| CPU attention | PyTorch Flash SDPA |
| CPU threads | 20 |

The draft checkpoint contains 399,555,840 parameters, of which 399,523,840 are
trainable. Its reduced vocabulary contains 32,000 entries mapped into the
151,936-entry target vocabulary.

## Captured rounds

| Round | Anchor | Proposal target count | Rejection position | Confirmed count |
| --- | ---: | ---: | ---: | ---: |
| 0 | 19 | 3 | 0 | 1 |
| 1 | 20 | 3 | 2 | 3 |

The serialized batch was reloaded and validated for tensor shapes, BF16 feature
dtypes, position continuity, rejection boundaries, and detached CPU ownership.

## CPU training results

| Mode | Loss | Input rows | Loss rows | Training time |
| --- | ---: | ---: | ---: | ---: |
| `confirmed_path` | 2.9067557 | 4 | 4 | 0.484 s |
| `proposal_canvas` | 5.4180002 | 6 | 6 | 0.491 s |

Both modes completed exactly one optimizer update. The loss and optimizer state
were finite, the sampled parameter region changed by a maximum of 0.0010986,
and the rebuilt persistent KV cache was detached and ended at length 24. Peak
process RSS was approximately 4.8 GiB.

## Confirmed-path input revalidation

On 2026-09-21, `confirmed_path` was changed from one continuous true-token
forward to isolated per-round confirmed query branches. Only each branch anchor
uses auxiliary hidden states through FC. Later confirmed query rows use the
draft recurrent hidden states captured on GPU, and rejected suffix rows do not
participate in the forward.

The saved real capture was rerun after this change:

| Mode | Loss | Input rows | Loss rows | Training time |
| --- | ---: | ---: | ---: | ---: |
| `confirmed_path` | 3.5073674 | 4 | 4 | 0.556 s |
| `proposal_canvas` | 5.4180002 | 6 | 6 | 0.505 s |

Both losses and all optimizer states were finite. Both rebuilt caches were
detached and ended at length 24. The changed confirmed-path loss is expected
because its non-anchor rows now use captured draft recurrent hidden states
instead of recomputed FC outputs from confirmed auxiliary hidden states.

## Deferred optimization: low teacher coverage

The first round exposed a reduced-vocabulary edge case. The target model's
greedy token at the rejected position was `<think>` with target token ID
151667. That token is not represented by the draft checkpoint's 32K
draft-to-target mapping, whose largest mapped target ID is 148442.

Projecting the full target distribution onto the draft vocabulary retained only
approximately `4.95e-8` probability mass at this position. The current training
code renormalizes that retained mass and applies KL loss to the resulting
conditional distribution. Because this was also the rejection position, the
current rejection weighting gave it the configured higher weight.

This behavior is numerically valid and did not produce a nonfinite loss or
gradient. However, it cannot directly teach the draft model to emit the actual
target token because that token is outside the draft vocabulary. It may spend
gradient capacity optimizing a conditional distribution with little relevance
to acceptance rate.

No source change is being made for this observation yet. In particular,
`Eagle3DistillationBatch` does not currently retain teacher coverage, and the
loss does not filter or scale positions by coverage.

Candidate future experiments are:

1. Skip positions whose retained teacher mass is below a configured threshold.
2. Multiply the existing position weight by retained teacher coverage.
3. Compare both policies against the current conditional-KL baseline using
   acceptance length, end-to-end throughput, and training stability.
4. Treat unrepresentable correction tokens separately from low-coverage
   positions whose target argmax remains representable.

Before adopting a policy, measure the frequency and location of low-coverage
positions on representative prompts. The threshold and weighting rule should be
driven by that distribution rather than by this single observation.

## Artifacts

The local experiment produced these temporary artifacts:

| Artifact | Path | SHA-256 |
| --- | --- | --- |
| Captured batch | `/tmp/online_draft_stage3_qwen3_8b_capture.pt` | `c0aaf72e5401c405e6bb25d90eaed6d804d857d2ebe0d435547ea12aac2bd401` |
| Initial JSON report | `/tmp/online_draft_stage3_qwen3_8b_report.json` | `ba2a00617c7e514746539696ff4976831a571140b91added007ba5cfe68e95d8` |

Files under `/tmp` are not durable. The configuration, measured results, and
deferred optimization decision are therefore recorded in this document.
