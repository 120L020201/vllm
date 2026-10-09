# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, SamplingParams

TARGET_MODEL = "/data/models/Qwen3-8B"
DRAFT_MODEL = "/data/models/Qwen3-8B-eagle3"

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

prompt = (
    "Write a detailed technical explanation of speculative decoding, "
    "including proposal generation, target verification, acceptance, "
    "and asynchronous online draft-model training."
)

outputs = llm.generate(
    [prompt],
    sampling_params=sampling_params,
)

generated = outputs[0].outputs[0]
print(f"generated_tokens={len(generated.token_ids)}")
print(f"output_prefix={generated.text[:200]!r}")

metrics = {
    metric.name: metric.value
    for metric in llm.get_metrics()
    if hasattr(metric, "value")
}

num_drafts = metrics.get("vllm:spec_decode_num_drafts", 0)
num_draft_tokens = metrics.get("vllm:spec_decode_num_draft_tokens", 0)
num_accepted_tokens = metrics.get(
    "vllm:spec_decode_num_accepted_tokens",
    0,
)

print(f"num_drafts={num_drafts}")
print(f"num_draft_tokens={num_draft_tokens}")
print(f"num_accepted_tokens={num_accepted_tokens}")

if num_drafts:
    acceptance_length = 1 + num_accepted_tokens / num_drafts
    print(f"mean_acceptance_length={acceptance_length:.3f}")
