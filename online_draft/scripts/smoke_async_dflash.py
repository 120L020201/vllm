# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run repeatable DFlash baseline/online requests and save observable metrics."""

import argparse
import json
import os
import time
from pathlib import Path

import regex as re


def configure_baseline_cpu(worker, threads, cpus):
    from online_draft.runtime.cpu_worker import CPUWorkerConfig

    CPUWorkerConfig(threads=threads, cpus=tuple(cpus)).apply()


def inspect_inference_cpu(worker):
    import torch

    return {"cpus": sorted(os.sched_getaffinity(0)), "threads": torch.get_num_threads()}


def configure_queue_measurement(worker):
    """Measure host queue depth and residence time without per-round logging."""
    pipeline = worker._online_draft.pipeline
    adapter = worker._online_draft.capture_adapter
    builder = adapter._batch_builder
    submit, consume = builder._on_round, pipeline._handle_round
    submitted = {}

    def on_round(prepared):
        submitted[prepared.step_id] = time.perf_counter()
        submit(prepared)

    def handle_round(prepared):
        now = time.perf_counter()
        worker.queue_measurements["residence"].append(
            [prepared.step_id, now - submitted.pop(prepared.step_id)]
        )
        consume(prepared)

    def measure(name):
        return lambda size: worker.queue_measurements[name].append(
            [time.perf_counter() - worker.queue_measurement_start, size]
        )

    builder._on_round = on_round
    pipeline._handle_round = handle_round
    pipeline._on_queue_size = measure("train")
    adapter._queue._on_queue_size = measure("capture")


def begin_queue_measurement(worker):
    worker.queue_measurement_start = time.perf_counter()
    worker.queue_measurements = {"train": [], "capture": [], "residence": []}


def inspect_queue_measurement(worker):
    return worker.queue_measurements


def configure_eagle_stats(worker):
    speculator = worker._online_draft.speculator
    finish = speculator.finish_request

    def measured_finish(request_id, initial_weights):
        speculator.finished_request = {
            "request_id": request_id,
            "gpu_version": speculator.draft_weight_installer.active_version,
            "completed_rounds": speculator.capture_adapter.completed_rounds,
        }
        finish(request_id, initial_weights)

    speculator.finish_request = measured_finish


def inspect_online(worker, method="dflash"):
    runtime = worker._online_draft
    if runtime is None:
        return None
    pipeline = runtime.pipeline
    if runtime.speculator.capture_adapter is None:
        return None
    if method == "eagle3":
        speculator = runtime.speculator
        return {
            "cpu_version": runtime.trainer.version,
            "gpu_version": speculator.draft_weight_installer.active_version,
            "finished_cpu": pipeline.finished_request,
            "finished_gpu": speculator.finished_request,
        }
    speculator = runtime.speculator
    state = speculator.online_state
    result = {
        "draft_graphs_per_slot": [
            len(m.graphs) for m in speculator.query_cudagraph_managers
        ],
        "cpu_version": runtime.trainer.version,
        "gpu_version": state.installer.active_version,
        "completed_rounds": state.adapter.completed_rounds,
        "updates": list(pipeline.update_stats),
        "finished_cpu": pipeline.finished_request,
        "finished_gpu": state.finished_request,
        "snapshot_bytes": sum(
            tensor.numel() * tensor.element_size()
            for _, tensor in runtime.trainer.initial_weights
        ),
    }
    return result


def summarize_decode(timeline, prompt_tokens):
    """Measure complete output batches using engine-core timestamps."""
    start = timeline[0]
    segments = []
    first_boundary = (prompt_tokens + start[1]) // 4096 * 4096 + 4096
    for boundary in range(first_boundary, prompt_tokens + timeline[-1][1] + 4096, 4096):
        end = next(
            point
            for point in reversed(timeline)
            if prompt_tokens + point[1] <= boundary
        )
        if end == start:
            continue
        elapsed = end[0] - start[0]
        tokens = end[1] - start[1]
        segments.append(
            {
                "context_start": prompt_tokens + start[1],
                "context_end": prompt_tokens + end[1],
                "tokens": tokens,
                "elapsed_s": elapsed,
                "tokens_per_second": tokens / elapsed,
                "output_batches": end[2] - start[2],
            }
        )
        start = end
    return segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--method", choices=("dflash", "eagle3"), default="dflash")
    parser.add_argument("--speculative-tokens", type=int, default=15)
    parser.add_argument("--train-threads", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-train-rounds", type=int, default=16)
    parser.add_argument("--loss-decay-gamma", type=float, default=7.0)
    parser.add_argument("--grouped-query-attention", action="store_true")
    parser.add_argument("--selective-last-layer", action="store_true")
    parser.add_argument("--queue-metrics", action="store_true")
    parser.add_argument("--master-weights", action="store_true")
    parser.add_argument("--cpu-isolation", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--target-graphs", action="store_true")
    parser.add_argument("--check-math", action="store_true")
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--profile-delay-iterations", type=int, default=0)
    parser.add_argument("--profile-max-iterations", type=int, default=0)
    parser.add_argument("--gsm8k-questions", type=int, default=0)
    parser.add_argument("--aime-questions", type=int, default=0)
    parser.add_argument("--aime-offset", type=int, default=0)
    parser.add_argument("--context-tokens", type=int, default=0)
    parser.add_argument("--decode-timeline", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.gsm8k_questions and args.aime_questions:
        parser.error("select only one evaluation dataset")
    if args.cpu_isolation and not 1 <= args.train_threads <= 18:
        parser.error("this workstation's isolation requires 1..18 training threads")
    if args.queue_metrics and (not args.online or args.method != "dflash"):
        parser.error("queue metrics require online DFlash")
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["VLLM_ONLINE_DRAFT_TRAIN"] = str(int(args.online))
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    if args.profile_dir:
        os.environ["VLLM_ONLINE_DRAFT_PROFILE"] = "1"
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="/data/models/Qwen3-8B",
        dtype="bfloat16",
        max_num_seqs=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.65,
        enable_prefix_caching=False,
        enforce_eager=not args.target_graphs,
        compilation_config={
            "mode": 0,
            "cudagraph_mode": "FULL" if args.target_graphs else "NONE",
            "cudagraph_capture_sizes": [args.speculative_tokens + 1],
        },
        profiler_config=(
            {
                "profiler": "torch",
                "torch_profiler_dir": str(args.profile_dir),
                "torch_profiler_with_stack": False,
                "torch_profiler_all_threads": True,
                "delay_iterations": args.profile_delay_iterations,
                "max_iterations": args.profile_max_iterations,
                "ignore_frontend": True,
            }
            if args.profile_dir
            else {}
        ),
        disable_log_stats=False,
        attention_config={"backend": "FLASHINFER"},
        speculative_config={
            "method": args.method,
            "model": "/data/models/Qwen3-8B-"
            + ("DFlash-b16" if args.method == "dflash" else "eagle3"),
            "num_speculative_tokens": args.speculative_tokens,
            "attention_backend": "FLASHINFER",
        },
        additional_config={
            "online_draft": {
                "learning_rate": args.learning_rate,
                "max_train_rounds": args.max_train_rounds,
                "loss_decay_gamma": args.loss_decay_gamma,
                "grouped_query_attention": args.grouped_query_attention,
                "selective_last_layer": args.selective_last_layer,
                "master_weights": args.master_weights,
                "fused_optimizer": True,
                "check_gradients": False,
                "log_updates": True,
                "attention_backend": "flash_attention",
                "cpu_threads": (
                    {"inference": 1, "capture": 1, "training": args.train_threads}
                    if args.train_threads
                    else {}
                ),
                "cpu_affinity": (
                    {
                        "inference": [18],
                        "capture": [19],
                        "training": list(range(args.train_threads)),
                    }
                    if args.cpu_isolation
                    else {}
                ),
            }
        },
    )
    questions = [
        (
            "Explain how to compute the sum of the first n positive integers, "
            "derive the formula in two different ways, and work through n=100. "
            "Then explain how to verify the result with a Python program."
        )
    ] * args.requests
    labels = []
    results = []
    timeline = []
    if args.decode_timeline:
        process_outputs = llm.llm_engine.output_processor.process_outputs

        def measured_outputs(outputs, engine_core_timestamp, iteration_stats):
            for output in outputs:
                if output.new_token_ids:
                    count = timeline[-1][1] if timeline else 0
                    timeline.append(
                        (
                            engine_core_timestamp,
                            count + len(output.new_token_ids),
                            len(timeline) + 1,
                        )
                    )
            return process_outputs(outputs, engine_core_timestamp, iteration_stats)

        llm.llm_engine.output_processor.process_outputs = measured_outputs
    try:
        if args.cpu_isolation and not args.online:
            llm.collective_rpc(configure_baseline_cpu, args=(1, [18]))
        inference_cpu = llm.collective_rpc(inspect_inference_cpu)
        if args.queue_metrics:
            llm.collective_rpc(configure_queue_measurement)
        if args.online and args.method == "eagle3":
            llm.collective_rpc(configure_eagle_stats)
        if args.gsm8k_questions:
            from tests.evals.gsm8k.gsm8k_eval import (
                _build_gsm8k_prompts,
                get_answer_value,
            )

            questions, labels = _build_gsm8k_prompts(args.gsm8k_questions, num_shots=0)
        if args.aime_questions:
            problems = [
                json.loads(line)
                for line in Path("/data/benchmarks/aime2026.jsonl")
                .read_text()
                .splitlines()
            ][args.aime_offset : args.aime_offset + args.aime_questions]
            questions = [problem["prompt"] for problem in problems]
            labels = [problem["answer"] for problem in problems]
        if args.check_math:
            from check_dflash_math import check_dflash_math

            print("math:", llm.collective_rpc(check_dflash_math), flush=True)
        if args.profile_dir:
            llm.start_profile()
        for index, question in enumerate(questions):
            mode = "online" if args.online else "baseline"
            prompt = llm.get_tokenizer().apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=bool(args.aime_questions),
            )
            prompt_tokens = len(llm.get_tokenizer().encode(prompt))
            max_tokens = (
                args.context_tokens - prompt_tokens
                if args.context_tokens
                else args.max_tokens
            )
            timeline.clear()
            if args.queue_metrics:
                llm.collective_rpc(begin_queue_measurement)
            start = time.perf_counter()
            request_output = llm.generate(
                [prompt],
                SamplingParams(
                    temperature=0,
                    max_tokens=max_tokens,
                    ignore_eos=not (args.gsm8k_questions or args.aime_questions),
                    seed=17,
                ),
            )[0]
            elapsed = time.perf_counter() - start
            output = request_output.outputs[0]
            request_metrics = request_output.metrics
            decode_s = request_metrics.last_token_ts - request_metrics.first_token_ts
            decode_tokens = request_metrics.num_generation_tokens - 1
            metrics = {
                m.name: m.value for m in llm.get_metrics() if hasattr(m, "value")
            }
            reset_start = time.perf_counter()
            online = llm.collective_rpc(inspect_online, args=(args.method,))
            finished = time.perf_counter()
            record = {
                "request": index,
                "mode": mode,
                "method": args.method,
                "target_graphs": args.target_graphs,
                "profile": {
                    "directory": str(args.profile_dir) if args.profile_dir else None,
                    "delay_iterations": args.profile_delay_iterations,
                    "max_iterations": args.profile_max_iterations,
                },
                "speculative_tokens": args.speculative_tokens,
                "training_threads": args.train_threads,
                "learning_rate": args.learning_rate,
                "max_train_rounds": args.max_train_rounds,
                "loss_decay_gamma": args.loss_decay_gamma,
                "grouped_query_attention": args.grouped_query_attention,
                "selective_last_layer": args.selective_last_layer,
                "master_weights": args.master_weights,
                "cpu_isolation": args.cpu_isolation,
                "inference_cpu": inference_cpu,
                "cpu_allocator": {
                    name: os.environ.get(name)
                    for name in ("MALLOC_MMAP_THRESHOLD_", "MALLOC_TRIM_THRESHOLD_")
                },
                "finish_reason": output.finish_reason,
                "prompt_tokens": len(request_output.prompt_token_ids),
                "final_context_tokens": (
                    len(request_output.prompt_token_ids) + len(output.token_ids)
                ),
                "max_model_len": args.max_model_len,
                "max_output_tokens": max_tokens,
                "prefill_s": (
                    request_metrics.first_token_ts - request_metrics.scheduled_ts
                ),
                "decode_s": decode_s,
                "decode_tokens": decode_tokens,
                "decode_tokens_per_second": (
                    decode_tokens / decode_s if decode_tokens else None
                ),
                "elapsed_s": elapsed,
                "tokens_per_second": len(output.token_ids) / elapsed,
                "token_ids": output.token_ids,
                "text": output.text,
                "metrics": metrics,
                "online": online,
                "finish_barrier_s": finished - reset_start,
                "end_to_end_s": finished - start,
            }
            if args.decode_timeline:
                record["decode_segments"] = summarize_decode(
                    timeline, len(request_output.prompt_token_ids)
                )
                record["decode_timeline"] = [
                    [stamp - timeline[0][0], count, batch]
                    for stamp, count, batch in timeline
                ]
            if args.queue_metrics:
                record["queue_measurements"] = llm.collective_rpc(
                    inspect_queue_measurement
                )
            if args.gsm8k_questions:
                record["expected_answer"] = labels[index]
                record["predicted_answer"] = get_answer_value(output.text)
                record["correct"] = record["predicted_answer"] == labels[index]
            if args.aime_questions:
                answers = re.findall(r"\\boxed\{\s*(\d+)\s*\}", output.text)
                record["problem_id"] = problems[index]["id"]
                record["expected_answer"] = labels[index]
                record["predicted_answer"] = int(answers[-1]) if answers else None
                record["correct"] = record["predicted_answer"] == labels[index]
            results.append(record)
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in record.items()
                        if k
                        not in (
                            "token_ids",
                            "text",
                            "decode_timeline",
                            "queue_measurements",
                        )
                    }
                ),
                flush=True,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        if args.profile_dir:
            llm.stop_profile()
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
