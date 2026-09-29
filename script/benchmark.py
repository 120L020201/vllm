# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Send local JSONL benchmark prompts to an OpenAI-compatible server."""

import argparse
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
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-timeout", type=float, default=7200)
    parser.add_argument("--max-samples", type=int)
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


def main() -> None:
    args = _parse_args()
    if args.max_output_tokens < 1 or args.top_k < -1:
        raise ValueError("max output tokens must be positive and top-k >= -1")
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
                    started = time.monotonic()
                    try:
                        response = _request(args, prompt)
                        error = None
                    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
                        response = None
                        error = f"{type(exc).__name__}: {exc}"
                    result = {
                        "benchmark": benchmark,
                        "line": line_number,
                        "id": record.get("id"),
                        "prompt": prompt,
                        "response": response,
                        "error": error,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output.flush()
                    written += 1
                    if error is not None:
                        raise RuntimeError(
                            f"request failed for {benchmark}:{line_number}: {error}"
                        )
            if args.max_samples is not None and written >= args.max_samples:
                break
    print(f"completed {written} requests -> {args.output}")


if __name__ == "__main__":
    main()
