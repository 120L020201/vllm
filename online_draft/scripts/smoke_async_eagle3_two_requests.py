# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from time import perf_counter

from vllm import LLM, SamplingParams

TARGET_MODEL = "/data/models/Qwen3-8B"
DRAFT_MODEL = "/data/models/Qwen3-8B-eagle3"

METRIC_NAMES = {
    "num_drafts": "vllm:spec_decode_num_drafts",
    "num_draft_tokens": "vllm:spec_decode_num_draft_tokens",
    "num_accepted_tokens": "vllm:spec_decode_num_accepted_tokens",
}


def read_spec_metrics(llm: LLM) -> dict[str, int]:
    metrics = {
        metric.name: metric.value
        for metric in llm.get_metrics()
        if hasattr(metric, "value")
    }
    return {
        name: metrics.get(metric_name, 0) for name, metric_name in METRIC_NAMES.items()
    }


llm = LLM(
    model=TARGET_MODEL,
    dtype="bfloat16",
    tensor_parallel_size=1,
    max_num_seqs=1,
    max_model_len=4096,
    gpu_memory_utilization=0.8,
    trust_remote_code=True,
    disable_log_stats=False,
    speculative_config={
        "method": "eagle3",
        "model": DRAFT_MODEL,
        "num_speculative_tokens": 3,
    },
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=512,
    ignore_eos=True,
)

prompts = [
    (
        "Write a detailed technical explanation of speculative decoding, "
        "including proposal generation, target verification, acceptance, "
        "and asynchronous online draft-model training."
    ),
    (
        "Explain how a CPU training pipeline can update a GPU draft model "
        "asynchronously while preserving uninterrupted speculative decoding."
    ),
]

previous_metrics = read_spec_metrics(llm)

try:
    for request_index, prompt in enumerate(prompts, start=1):
        print(f"request_{request_index}_start", flush=True)
        start = perf_counter()

        outputs = llm.generate([prompt], sampling_params=sampling_params)

        elapsed = perf_counter() - start
        generated = outputs[0].outputs[0]
        current_metrics = read_spec_metrics(llm)
        request_metrics = {
            name: current_metrics[name] - previous_metrics[name]
            for name in METRIC_NAMES
        }
        previous_metrics = current_metrics

        num_drafts = request_metrics["num_drafts"]
        num_accepted_tokens = request_metrics["num_accepted_tokens"]

        print(f"request_{request_index}_id={outputs[0].request_id}")
        print(f"request_{request_index}_generated_tokens={len(generated.token_ids)}")
        print(f"request_{request_index}_elapsed_seconds={elapsed:.3f}")
        print(
            f"request_{request_index}_tokens_per_second="
            f"{len(generated.token_ids) / elapsed:.3f}"
        )

        for name, value in request_metrics.items():
            print(f"request_{request_index}_{name}={value}")

        if num_drafts:
            acceptance_length = 1 + num_accepted_tokens / num_drafts
            print(
                f"request_{request_index}_mean_acceptance_length="
                f"{acceptance_length:.3f}"
            )

        print(f"request_{request_index}_end", flush=True)
finally:
    llm.llm_engine.engine_core.shutdown()
