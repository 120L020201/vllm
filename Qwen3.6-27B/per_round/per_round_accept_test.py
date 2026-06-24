# SPDX-License-Identifier: Apache-2.0
"""Capture the per-round (per draft-target step) accepted length for each prompt.

For every speculative-decoding round, the scheduler computes how many draft
tokens a request accepted. We hook ``Scheduler.make_spec_decoding_stats`` to
record (request_id -> [accepted_len per round]) without touching vLLM core.

This requires the scheduler to run IN-PROCESS, so launch with
``VLLM_ENABLE_V1_MULTIPROCESSING=0`` (see the command in the chat / header).

Run with a large --num-spec-tokens (e.g. 32) so the per-round accepted length
is never clipped by the draft budget. The max possible per-round accept is
num_spec_tokens; the bonus (always-correct) token is NOT counted here.
"""
import argparse
import json
import os
from collections import defaultdict

# --- Workaround for transformers 5.x LlamaConfig over-strict validation ---
try:
    from transformers.models.llama.configuration_llama import LlamaConfig

    def _patched_validate_architecture(self):
        if getattr(self, "head_dim", None):
            return
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of "
                f"the number of attention heads ({self.num_attention_heads})."
            )

    _cvs = getattr(LlamaConfig, "__class_validators__", None)
    if _cvs is not None:
        for _i, _v in enumerate(_cvs):
            if getattr(_v, "__name__", "") == "validate_architecture":
                _cvs[_i] = _patched_validate_architecture
    LlamaConfig.validate_architecture = _patched_validate_architecture
except Exception:
    pass
# --------------------------------------------------------------------------

from transformers import AutoTokenizer  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.v1.core.sched.scheduler import Scheduler  # noqa: E402

# request_id (str) -> list of accepted lengths, one entry per draft round.
ROUNDS: dict[str, list[int]] = defaultdict(list)

_ORIG_MAKE_STATS = Scheduler.make_spec_decoding_stats


def _patched_make_spec_decoding_stats(
    self, spec_decoding_stats, num_draft_tokens, num_accepted_tokens,
    num_invalid_spec_tokens, request_id,
):
    # Record this round's accepted length for this request before delegating.
    ROUNDS[request_id].append(int(num_accepted_tokens))
    return _ORIG_MAKE_STATS(
        self, spec_decoding_stats, num_draft_tokens, num_accepted_tokens,
        num_invalid_spec_tokens, request_id,
    )


Scheduler.make_spec_decoding_stats = _patched_make_spec_decoding_stats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--method", required=True, choices=["mtp", "eagle3", "dflash"])
    p.add_argument("--draft-model", default=None)
    p.add_argument("--num-spec-tokens", type=int, default=32)
    p.add_argument(
        "--dataset", required=True,
        help="JSONL with fields problem_idx, answer, prompt (one per line).",
    )
    p.add_argument("--num-prompts", type=int, default=5)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Max concurrent sequences (batch size). None = vLLM default. "
        "Set to 1 to measure pure single-sequence spec-decode speedup.",
    )
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--out-json", default=None)
    return p.parse_args()


def build_spec_config(args):
    cfg = {"method": args.method, "num_speculative_tokens": args.num_spec_tokens}
    if args.method in ("eagle3", "dflash"):
        if not args.draft_model:
            raise ValueError(f"--draft-model is required for method={args.method}")
        cfg["model"] = args.draft_model
    return cfg


def main(args):
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        raise SystemExit(
            "Set VLLM_ENABLE_V1_MULTIPROCESSING=0 so the scheduler runs "
            "in-process and the per-round hook can see the stats."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    recs = []
    with open(args.dataset) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    recs = recs[: args.num_prompts]

    # request_id is assigned in submission order as str(0..n-1), so build the
    # prompts in the same order to map request_id -> problem.
    prompts = [
        {
            "prompt": tokenizer.apply_chat_template(
                [{"role": "user", "content": r["prompt"]}],
                add_generation_prompt=True,
                tokenize=False,
            )
        }
        for r in recs
    ]

    # max_num_seqs=1 keeps requests sequential so a "round" maps cleanly to a
    # single problem (no batched mixing across problems within a step). With a
    # larger batch, per-problem rounds are still attributed correctly via
    # request_id, but rounds from different problems interleave in time.
    llm = LLM(
        model=args.model_dir,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        speculative_config=build_spec_config(args),
        disable_log_stats=False,
        max_model_len=args.max_model_len,
        language_model_only=True,
        **({"max_num_seqs": args.max_num_seqs} if args.max_num_seqs else {}),
    )

    sampling_params = SamplingParams(temperature=args.temp, max_tokens=args.output_len)
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    # Diagnostic: show exactly what request_ids the hook captured, so a
    # key-mismatch (empty per-round lists) is immediately visible.
    print("=" * 70)
    print(f"[hook] captured {len(ROUNDS)} request_id(s): "
          f"{ {k: len(v) for k, v in ROUNDS.items()} }")
    print("=" * 70)

    report = []
    print(f"method={args.method}  num_spec_tokens={args.num_spec_tokens}  "
          f"(per-round accept is capped at {args.num_spec_tokens}; bonus token "
          f"excluded)")
    print("=" * 70)
    # generate() returns outputs in submission order, so outputs[idx] <-> recs[idx].
    # The scheduler keys stats by an internal request_id that is the LLM-level
    # request_id with a "-<hash>" suffix appended (e.g. "0" -> "0-b72853f0").
    # Match by stripping that suffix so the rounds line up with each problem.
    def rounds_for(req_id):
        if req_id in ROUNDS:
            return ROUNDS[req_id]
        merged = []
        for k, v in ROUNDS.items():
            if k == req_id or k.startswith(req_id + "-"):
                merged.extend(v)
        return merged

    for idx, (r, o) in enumerate(zip(recs, outputs)):
        rid = o.request_id
        rounds = rounds_for(rid)
        out_tokens = len(o.outputs[0].token_ids)
        max_accept = max(rounds) if rounds else 0
        mean_accept = (sum(rounds) / len(rounds)) if rounds else 0.0
        print(f"\nproblem_idx={r['problem_idx']}  (request_id={rid})")
        print(f"  output_tokens:        {out_tokens}")
        print(f"  num_rounds:           {len(rounds)}")
        print(f"  MAX accepted/round:   {max_accept}")
        print(f"  mean accepted/round:  {mean_accept:.3f}")
        print(f"  per-round accepted:   {rounds}")
        report.append(
            {
                "problem_idx": r["problem_idx"],
                "request_id": rid,
                "output_tokens": out_tokens,
                "num_rounds": len(rounds),
                "max_accept_per_round": max_accept,
                "mean_accept_per_round": round(mean_accept, 3),
                "per_round_accept": rounds,
            }
        )
    print("=" * 70)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main(parse_args())
