# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Send local JSONL benchmark prompts to an OpenAI-compatible server."""

import argparse
import contextlib
import hashlib
import json
import os
import random
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import regex as re

ResultKey = tuple[str, int]
ResumeMode = Literal["none", "question", "dataset"]
RESULT_SCHEMA_VERSION = 2

_SPEC_METRIC_KEYS = ("drafts", "draft_tokens", "accepted_tokens")
_REQUEST_TIMING_METRICS = {
    "prefill": "vllm:request_prefill_time_seconds",
    "decode": "vllm:request_decode_time_seconds",
}
_BENCHMARK_PATTERN = re.compile(
    r"^(?P<name>[A-Za-z0-9_.-]+)(?:\[(?P<sample_size>[0-9]+)\])?$"
)


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    name: str
    sample_size: int | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--benchmarks", required=True)
    parser.add_argument("--max-output-tokens", type=int, required=True)
    parser.add_argument("--max-prompt-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--presence-penalty", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-timeout", type=float, default=7200)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--stats-file", type=Path)
    parser.add_argument(
        "--resume-mode",
        choices=("none", "question", "dataset"),
        default="none",
        help=(
            "question skips successful questions; dataset keeps only complete "
            "datasets and restarts the first incomplete dataset"
        ),
    )
    return parser.parse_args()


def _parse_benchmark_specs(value: str) -> list[BenchmarkSpec]:
    specs = []
    names = set()
    for item in value.split():
        match = _BENCHMARK_PATTERN.fullmatch(item)
        if match is None:
            raise ValueError(f"invalid benchmark selection: {item}")
        name = match.group("name")
        if name in names:
            raise ValueError(f"duplicate benchmark: {name}")
        raw_sample_size = match.group("sample_size")
        sample_size = int(raw_sample_size) if raw_sample_size is not None else None
        if sample_size is not None and sample_size < 1:
            raise ValueError(f"benchmark sample size must be positive: {item}")
        specs.append(BenchmarkSpec(name, sample_size))
        names.add(name)
    if not specs:
        raise ValueError("BENCHMARKS must not be empty")
    return specs


def _select_benchmark_records(
    path: Path,
    spec: BenchmarkSpec,
    seed: int,
) -> list[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"benchmark file does not exist: {path}")
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if line.strip():
                records.append((line_number, json.loads(line)))

    if spec.sample_size is None:
        return records
    if spec.sample_size > len(records):
        raise ValueError(
            f"{spec.name} requested {spec.sample_size} samples, "
            f"but only {len(records)} are available"
        )

    digest = hashlib.sha256(f"{seed}:{spec.name}".encode()).digest()
    generator = random.Random(int.from_bytes(digest, byteorder="big"))
    selected_indices = set(generator.sample(range(len(records)), spec.sample_size))
    return [record for index, record in enumerate(records) if index in selected_indices]


def _prompt(record: dict[str, Any]) -> str:
    for key in ("prompt", "question", "problem", "input"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    source = record.get("source")
    if isinstance(source, dict):
        for key in ("prompt", "Question", "question", "problem"):
            value = source.get(key)
            if isinstance(value, str) and value:
                return value
    raise ValueError("record has no supported prompt field")


def _request_body(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    return {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
        "max_tokens": args.max_output_tokens,
        "truncate_prompt_tokens": args.max_prompt_tokens,
        "truncation_side": "left",
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "presence_penalty": args.presence_penalty,
        "seed": args.seed,
        "stream": False,
    }


def _request_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "endpoint": "/v1/chat/completions",
        "model": args.model,
        "max_output_tokens": args.max_output_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "presence_penalty": args.presence_penalty,
        "enable_thinking": args.enable_thinking,
        "seed": args.seed,
        "benchmarks": args.benchmarks,
    }


def _request(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    body = _request_body(args, prompt)
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=args.request_timeout) as response:
        return json.load(response)


def _fetch_server_metrics(base_url: str) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(
            f"{base_url.rstrip('/')}/metrics",
            timeout=30,
        ) as response:
            content = response.read().decode("utf-8")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None

    spec_values = {key: 0 for key in _SPEC_METRIC_KEYS}
    timing_values = {
        f"{phase}_{suffix}": 0.0
        for phase in ("prefill", "decode")
        for suffix in ("seconds", "requests")
    }
    found_spec = False
    found_timing: set[str] = set()
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        metric_name = parts[0].split("{")[0]
        if metric_name.startswith("vllm:spec_decode"):
            if not metric_name.endswith("_total"):
                continue
            with contextlib.suppress(ValueError):
                value = int(float(parts[1]))
                if "num_accepted_tokens_per_pos" in metric_name:
                    continue
                if "num_drafts" in metric_name:
                    spec_values["drafts"] += value
                elif "num_draft_tokens" in metric_name:
                    spec_values["draft_tokens"] += value
                elif "num_accepted_tokens" in metric_name:
                    spec_values["accepted_tokens"] += value
                else:
                    continue
                found_spec = True
            continue

        for phase, metric_prefix in _REQUEST_TIMING_METRICS.items():
            if metric_name == f"{metric_prefix}_sum":
                with contextlib.suppress(ValueError):
                    timing_values[f"{phase}_seconds"] += float(parts[1])
                    found_timing.add(f"{phase}_seconds")
            elif metric_name == f"{metric_prefix}_count":
                with contextlib.suppress(ValueError):
                    timing_values[f"{phase}_requests"] += int(float(parts[1]))
                    found_timing.add(f"{phase}_requests")

    expected_timing = set(timing_values)
    return {
        "spec_decode": spec_values if found_spec else None,
        "request_timing": (timing_values if found_timing == expected_timing else None),
    }


def _fetch_spec_metrics(base_url: str) -> dict[str, int] | None:
    metrics = _fetch_server_metrics(base_url)
    if metrics is None:
        return None
    return metrics["spec_decode"]


def _read_worker_stats(path: Path | None) -> dict[str, int]:
    if path is None or not path.is_file():
        return {"active_requests": 0, "updates": 0, "pending": 0}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            "active_requests": int(payload.get("active_requests", 0)),
            "updates": int(payload.get("updates", 0)),
            "pending": int(payload.get("pending", 0)),
        }
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return {"active_requests": 0, "updates": 0, "pending": 0}


def _wait_for_worker(path: Path | None, timeout: float) -> dict[str, int]:
    deadline = time.monotonic() + timeout
    stats = _read_worker_stats(path)
    while (
        stats["pending"] > 0 or stats["active_requests"] > 0
    ) and time.monotonic() < deadline:
        time.sleep(0.05)
        stats = _read_worker_stats(path)
    return stats


def _acceptance_length(
    before: dict[str, int] | None,
    after: dict[str, int] | None,
) -> float:
    delta = _spec_metrics_delta(before, after)
    if delta is None:
        return 1.0
    drafts = delta["drafts"]
    accepted = delta["accepted_tokens"]
    return 1.0 + accepted / drafts if drafts > 0 else 1.0


def _spec_metrics_delta(
    before: dict[str, int] | None,
    after: dict[str, int] | None,
) -> dict[str, int] | None:
    if before is None or after is None:
        return None
    delta = {key: after[key] - before[key] for key in _SPEC_METRIC_KEYS}
    return delta if all(value >= 0 for value in delta.values()) else None


def _request_timing_delta(
    before: dict[str, float] | None,
    after: dict[str, float] | None,
) -> dict[str, float] | None:
    if before is None or after is None:
        return None
    prefill_requests = int(after["prefill_requests"] - before["prefill_requests"])
    decode_requests = int(after["decode_requests"] - before["decode_requests"])
    if prefill_requests != 1 or decode_requests != 1:
        return None
    timing = {
        "prefill_seconds": after["prefill_seconds"] - before["prefill_seconds"],
        "decode_seconds": after["decode_seconds"] - before["decode_seconds"],
    }
    return timing if all(value >= 0 for value in timing.values()) else None


def _completion_tokens(response: dict[str, Any] | None) -> int:
    if response is None:
        return 0
    usage = response.get("usage") or {}
    return int(usage.get("completion_tokens") or 0)


def _prompt_tokens(response: dict[str, Any] | None) -> int:
    if response is None:
        return 0
    usage = response.get("usage") or {}
    return int(usage.get("prompt_tokens") or 0)


def _phase_performance(
    response: dict[str, Any] | None,
    timing: dict[str, float] | None,
) -> dict[str, int | float | None]:
    prompt_tokens = _prompt_tokens(response)
    completion_tokens = _completion_tokens(response)
    decode_tokens = max(completion_tokens - 1, 0)
    prefill_seconds = timing["prefill_seconds"] if timing is not None else None
    decode_seconds = timing["decode_seconds"] if timing is not None else None
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "decode_tokens": decode_tokens,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "prefill_tokens_per_second": (
            prompt_tokens / prefill_seconds
            if prefill_seconds is not None and prefill_seconds > 0
            else None
        ),
        "decode_tokens_per_second": (
            decode_tokens / decode_seconds
            if decode_seconds is not None and decode_seconds > 0
            else None
        ),
    }


def _new_dataset_totals() -> dict[str, int | float]:
    return {
        "requests": 0,
        "spec_decode_requests": 0,
        "drafts": 0,
        "draft_tokens": 0,
        "accepted_tokens": 0,
        "prefill_requests": 0,
        "prefill_tokens": 0,
        "prefill_seconds": 0.0,
        "decode_requests": 0,
        "decode_tokens": 0,
        "decode_seconds": 0.0,
    }


def _update_dataset_totals(
    totals: dict[str, int | float],
    result: dict[str, Any],
) -> None:
    totals["requests"] += 1
    spec_decode = result.get("spec_decode")
    if isinstance(spec_decode, dict):
        totals["spec_decode_requests"] += 1
        for key in _SPEC_METRIC_KEYS:
            totals[key] += int(spec_decode.get(key) or 0)

    prefill_seconds = result.get("prefill_seconds")
    if isinstance(prefill_seconds, (int, float)) and prefill_seconds >= 0:
        totals["prefill_requests"] += 1
        totals["prefill_tokens"] += int(result.get("prompt_tokens") or 0)
        totals["prefill_seconds"] += prefill_seconds

    decode_seconds = result.get("decode_seconds")
    if isinstance(decode_seconds, (int, float)) and decode_seconds >= 0:
        totals["decode_requests"] += 1
        totals["decode_tokens"] += int(result.get("decode_tokens") or 0)
        totals["decode_seconds"] += decode_seconds


def _dataset_summary(totals: dict[str, int | float]) -> dict[str, Any]:
    requests = int(totals["requests"])
    spec_requests = int(totals["spec_decode_requests"])
    drafts = int(totals["drafts"])
    accepted_tokens = int(totals["accepted_tokens"])
    spec_decode = None
    if spec_requests:
        spec_decode = {
            "measured_requests": spec_requests,
            "complete": spec_requests == requests,
            "drafts": drafts,
            "draft_tokens": int(totals["draft_tokens"]),
            "accepted_tokens": accepted_tokens,
            "acceptance_length": (
                1.0 + accepted_tokens / drafts if drafts > 0 else 1.0
            ),
        }

    prefill_seconds = float(totals["prefill_seconds"])
    decode_seconds = float(totals["decode_seconds"])
    prefill_tokens = int(totals["prefill_tokens"])
    decode_tokens = int(totals["decode_tokens"])
    return {
        "requests": requests,
        "spec_decode": spec_decode,
        "prefill": {
            "measured_requests": int(totals["prefill_requests"]),
            "tokens": prefill_tokens,
            "seconds": prefill_seconds,
            "tokens_per_second": (
                prefill_tokens / prefill_seconds if prefill_seconds > 0 else None
            ),
        },
        "decode": {
            "measured_requests": int(totals["decode_requests"]),
            "tokens": decode_tokens,
            "seconds": decode_seconds,
            "tokens_per_second": (
                decode_tokens / decode_seconds if decode_seconds > 0 else None
            ),
        },
    }


def _write_summaries(
    path: Path,
    request_config: dict[str, Any],
    summaries: dict[str, dict[str, Any]],
) -> None:
    payload = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "request_config": request_config,
        "datasets": summaries,
    }
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _phase_log(
    name: str,
    tokens: int,
    seconds: float | None,
    tokens_per_second: float | None,
) -> str:
    if seconds is None or tokens_per_second is None:
        return f"{name}=N/A"
    return f"{name}={tokens} tokens/{seconds:.3f}s/{tokens_per_second:.2f} tokens/s"


def _successful_results(
    path: Path,
    request_config: dict[str, Any] | None = None,
) -> tuple[dict[ResultKey, dict[str, Any]], int]:
    results: dict[ResultKey, dict[str, Any]] = {}
    discarded = 0
    if not path.is_file():
        return results, discarded

    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            try:
                result = json.loads(line)
                key = (str(result["benchmark"]), int(result["line"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                discarded += 1
                continue
            successful = (
                result.get("error") is None
                and result.get("response") is not None
                and (
                    request_config is None
                    or result.get("request_config") == request_config
                )
            )
            if successful:
                if key in results:
                    discarded += 1
                results[key] = result
            else:
                discarded += 1
    return results, discarded


def _expected_keys(
    data_dir: Path,
    benchmarks: list[str],
    results: dict[ResultKey, dict[str, Any]],
    selected_lines: dict[str, set[int]] | None = None,
) -> dict[str, list[ResultKey]]:
    expected: dict[str, list[ResultKey]] = {}
    for benchmark in benchmarks:
        path = data_dir / f"{benchmark}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"benchmark file does not exist: {path}")
        keys = []
        benchmark_lines = (
            selected_lines.get(benchmark) if selected_lines is not None else None
        )
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                if benchmark_lines is not None and line_number not in benchmark_lines:
                    continue
                record = json.loads(line)
                key = (benchmark, line_number)
                keys.append(key)
                result = results.get(key)
                if result is not None and result.get("prompt") != _prompt(record):
                    del results[key]
        expected[benchmark] = keys
    return expected


def _rewrite_results(
    path: Path,
    results: dict[ResultKey, dict[str, Any]],
    retained_keys: set[ResultKey],
    benchmarks: list[str],
) -> None:
    benchmark_order = {benchmark: index for index, benchmark in enumerate(benchmarks)}
    ordered_keys = sorted(
        retained_keys,
        key=lambda key: (benchmark_order[key[0]], key[1]),
    )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            for key in ordered_keys:
                output.write(json.dumps(results[key], ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _prepare_resume(
    output_path: Path,
    data_dir: Path,
    benchmarks: list[str],
    mode: ResumeMode,
    request_config: dict[str, Any] | None = None,
    *,
    selected_lines: dict[str, set[int]] | None = None,
) -> tuple[set[ResultKey], str | None, int]:
    results, discarded = _successful_results(output_path, request_config)
    result_count = len(results)
    expected = _expected_keys(
        data_dir,
        benchmarks,
        results,
        selected_lines,
    )
    discarded += result_count - len(results)
    expected_keys = {key for keys in expected.values() for key in keys}
    discarded += len(results.keys() - expected_keys)
    results = {key: result for key, result in results.items() if key in expected_keys}

    retained_keys: set[ResultKey]
    if mode == "question":
        retained_keys = set(results)
        restart_benchmark = next(
            (
                benchmark
                for benchmark in benchmarks
                if any(key not in retained_keys for key in expected[benchmark])
            ),
            None,
        )
    else:
        retained_keys = set()
        restart_benchmark = None
        for benchmark in benchmarks:
            benchmark_keys = set(expected[benchmark])
            if benchmark_keys <= results.keys():
                retained_keys.update(benchmark_keys)
            else:
                restart_benchmark = benchmark
                break
        discarded += len(results.keys() - retained_keys)

    _rewrite_results(
        output_path,
        results,
        retained_keys,
        benchmarks,
    )
    return retained_keys, restart_benchmark, discarded


def main() -> None:
    args = _parse_args()
    if (
        args.max_output_tokens < 1
        or args.max_prompt_tokens < 1
        or args.top_k < -1
        or args.seed < 0
    ):
        raise ValueError(
            "max output and prompt tokens must be positive, top-k >= -1, "
            "and seed nonnegative"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    benchmark_specs = _parse_benchmark_specs(args.benchmarks)
    benchmark_records = {
        spec.name: _select_benchmark_records(
            args.data_dir / f"{spec.name}.jsonl",
            spec,
            args.seed,
        )
        for spec in benchmark_specs
    }
    benchmarks = [spec.name for spec in benchmark_specs]
    selected_lines = {
        benchmark: {line_number for line_number, _ in records}
        for benchmark, records in benchmark_records.items()
    }

    request_config = _request_config(args)
    completed_keys: set[ResultKey] = set()
    retained_results: dict[ResultKey, dict[str, Any]] = {}
    if args.resume_mode != "none":
        completed_keys, restart_benchmark, discarded = _prepare_resume(
            args.output,
            args.data_dir,
            benchmarks,
            args.resume_mode,
            request_config,
            selected_lines=selected_lines,
        )
        restart = restart_benchmark or "<complete>"
        print(
            f"resume={args.resume_mode}: kept {len(completed_keys)} successful "
            f"requests, discarded {discarded}, restart={restart}",
            flush=True,
        )
        retained_results, _ = _successful_results(args.output, request_config)

    written = len(completed_keys)
    output_mode = "a" if args.resume_mode != "none" else "w"
    summary_path = args.output.with_suffix(".summary.json")
    summaries: dict[str, dict[str, Any]] = {}
    _write_summaries(summary_path, request_config, summaries)
    with args.output.open(output_mode, encoding="utf-8") as output:
        for benchmark in benchmarks:
            dataset_totals = _new_dataset_totals()
            for key, result in retained_results.items():
                if key[0] == benchmark:
                    _update_dataset_totals(dataset_totals, result)
            for line_number, record in benchmark_records[benchmark]:
                if args.max_samples is not None and written >= args.max_samples:
                    break
                key = (benchmark, line_number)
                if key in completed_keys:
                    continue
                prompt = _prompt(record)
                metrics_before = _fetch_server_metrics(args.base_url)
                worker_before = _read_worker_stats(args.stats_file)
                started = time.monotonic()
                try:
                    response = _request(args, prompt)
                    error = None
                except (urllib.error.HTTPError, urllib.error.URLError) as exc:
                    response = None
                    error = f"{type(exc).__name__}: {exc}"
                worker_after = _wait_for_worker(
                    args.stats_file,
                    args.request_timeout,
                )
                metrics_after = _fetch_server_metrics(args.base_url)
                elapsed = time.monotonic() - started
                spec_before = (
                    metrics_before["spec_decode"]
                    if metrics_before is not None
                    else None
                )
                spec_after = (
                    metrics_after["spec_decode"] if metrics_after is not None else None
                )
                timing_before = (
                    metrics_before["request_timing"]
                    if metrics_before is not None
                    else None
                )
                timing_after = (
                    metrics_after["request_timing"]
                    if metrics_after is not None
                    else None
                )
                spec_decode = _spec_metrics_delta(spec_before, spec_after)
                timing = _request_timing_delta(timing_before, timing_after)
                phase_performance = _phase_performance(response, timing)
                tokens = int(phase_performance["completion_tokens"] or 0)
                updates = worker_after["updates"] - worker_before["updates"]
                acceptance_length = _acceptance_length(spec_before, spec_after)
                tokens_per_second = tokens / elapsed if elapsed > 0 else 0.0
                record_id = record.get("id") or f"{benchmark}-{line_number}"
                result = {
                    "result_schema_version": RESULT_SCHEMA_VERSION,
                    "benchmark": benchmark,
                    "line": line_number,
                    "id": record.get("id"),
                    "prompt": prompt,
                    "request_config": request_config,
                    "response": response,
                    "error": error,
                    "acceptance_length": acceptance_length,
                    "spec_decode": spec_decode,
                    "updates": updates,
                    "tokens": tokens,
                    "elapsed_seconds": elapsed,
                    "tokens_per_second": tokens_per_second,
                    **phase_performance,
                }
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
                _update_dataset_totals(dataset_totals, result)
                written += 1
                prefill_log = _phase_log(
                    "prefill",
                    int(phase_performance["prompt_tokens"] or 0),
                    phase_performance["prefill_seconds"],
                    phase_performance["prefill_tokens_per_second"],
                )
                decode_log = _phase_log(
                    "decode",
                    int(phase_performance["decode_tokens"] or 0),
                    phase_performance["decode_seconds"],
                    phase_performance["decode_tokens_per_second"],
                )
                print(
                    f"{record_id}: AL={acceptance_length:.3f}, "
                    f"updates={updates}, tokens={tokens}, time={elapsed:.1f}s, "
                    f"tokens/s={tokens_per_second:.2f}, {prefill_log}, "
                    f"{decode_log}",
                    flush=True,
                )
                if error is not None:
                    raise RuntimeError(
                        f"request failed for {benchmark}:{line_number}: {error}"
                    )
            summary = _dataset_summary(dataset_totals)
            summaries[benchmark] = summary
            _write_summaries(summary_path, request_config, summaries)
            spec_summary = summary["spec_decode"]
            if spec_summary is None:
                dataset_al = "N/A (speculative decoding inactive)"
            elif not spec_summary["complete"]:
                dataset_al = "N/A (incomplete metrics coverage)"
            else:
                dataset_al = f"{spec_summary['acceptance_length']:.3f}"
            print(
                f"{benchmark}: dataset_AL={dataset_al}, requests={summary['requests']}",
                flush=True,
            )
            if args.max_samples is not None and written >= args.max_samples:
                break
    print(f"completed {written} requests -> {args.output}; summaries -> {summary_path}")


if __name__ == "__main__":
    main()
