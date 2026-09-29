# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Send local JSONL benchmark prompts to an OpenAI-compatible server."""

import argparse
import contextlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-timeout", type=float, default=7200)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--stats-file", type=Path)
    return parser.parse_args()


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


def _request(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    body = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.max_output_tokens,
        "truncate_prompt_tokens": args.max_prompt_tokens,
        "truncation_side": "left",
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "stream": False,
    }
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/v1/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=args.request_timeout) as response:
        return json.load(response)


def _fetch_spec_metrics(base_url: str) -> dict[str, int] | None:
    try:
        with urllib.request.urlopen(
            f"{base_url.rstrip('/')}/metrics",
            timeout=30,
        ) as response:
            content = response.read().decode("utf-8")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None

    values = {"drafts": 0, "draft_tokens": 0, "accepted_tokens": 0}
    found = False
    for line in content.splitlines():
        line = line.strip()
        if not line.startswith("vllm:spec_decode"):
            continue
        parts = line.split(None, 1)
        metric_name = parts[0].split("{")[0]
        if not metric_name.endswith("_total"):
            continue
        with contextlib.suppress(ValueError, IndexError):
            value = int(float(parts[-1]))
            if "num_accepted_tokens_per_pos" in metric_name:
                continue
            if "num_drafts" in metric_name:
                values["drafts"] += value
            elif "num_draft_tokens" in metric_name:
                values["draft_tokens"] += value
            elif "num_accepted_tokens" in metric_name:
                values["accepted_tokens"] += value
            found = True
    return values if found else None


def _read_worker_stats(path: Path | None) -> dict[str, int]:
    if path is None or not path.is_file():
        return {"updates": 0, "pending": 0}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            "updates": int(payload.get("updates", 0)),
            "pending": int(payload.get("pending", 0)),
        }
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return {"updates": 0, "pending": 0}


def _wait_for_worker(path: Path | None, timeout: float) -> dict[str, int]:
    deadline = time.monotonic() + timeout
    stats = _read_worker_stats(path)
    while stats["pending"] > 0 and time.monotonic() < deadline:
        time.sleep(0.05)
        stats = _read_worker_stats(path)
    return stats


def _acceptance_length(
    before: dict[str, int] | None,
    after: dict[str, int] | None,
) -> float:
    if before is None or after is None:
        return 1.0
    drafts = after["drafts"] - before["drafts"]
    accepted = after["accepted_tokens"] - before["accepted_tokens"]
    return 1.0 + accepted / drafts if drafts > 0 else 1.0


def _completion_tokens(response: dict[str, Any] | None) -> int:
    if response is None:
        return 0
    usage = response.get("usage") or {}
    return int(usage.get("completion_tokens") or 0)


def main() -> None:
    args = _parse_args()
    if args.max_output_tokens < 1 or args.max_prompt_tokens < 1 or args.top_k < -1:
        raise ValueError(
            "max output and prompt tokens must be positive and top-k >= -1"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    benchmarks = [item for item in args.benchmarks.split() if item]
    if not benchmarks:
        raise ValueError("BENCHMARKS must not be empty")

    written = 0
    with args.output.open("w", encoding="utf-8") as output:
        for benchmark in benchmarks:
            path = args.data_dir / f"{benchmark}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(f"benchmark file does not exist: {path}")
            with path.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    if args.max_samples is not None and written >= args.max_samples:
                        break
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    prompt = _prompt(record)
                    spec_before = _fetch_spec_metrics(args.base_url)
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
                    spec_after = _fetch_spec_metrics(args.base_url)
                    elapsed = time.monotonic() - started
                    tokens = _completion_tokens(response)
                    updates = worker_after["updates"] - worker_before["updates"]
                    acceptance_length = _acceptance_length(spec_before, spec_after)
                    tokens_per_second = tokens / elapsed if elapsed > 0 else 0.0
                    record_id = record.get("id") or f"{benchmark}-{line_number}"
                    result = {
                        "benchmark": benchmark,
                        "line": line_number,
                        "id": record.get("id"),
                        "prompt": prompt,
                        "response": response,
                        "error": error,
                        "acceptance_length": acceptance_length,
                        "updates": updates,
                        "tokens": tokens,
                        "elapsed_seconds": elapsed,
                        "tokens_per_second": tokens_per_second,
                    }
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output.flush()
                    written += 1
                    print(
                        f"{record_id}: AL={acceptance_length:.3f}, "
                        f"updates={updates}, tokens={tokens}, time={elapsed:.1f}s, "
                        f"tokens/s={tokens_per_second:.2f}",
                        flush=True,
                    )
                    if error is not None:
                        raise RuntimeError(
                            f"request failed for {benchmark}:{line_number}: {error}"
                        )
            if args.max_samples is not None and written >= args.max_samples:
                break
    print(f"completed {written} requests -> {args.output}")


if __name__ == "__main__":
    main()
