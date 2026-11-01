# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Plot per-round TTS acceptance length over generated context length."""

import argparse
import csv
import json
from collections import deque
from pathlib import Path
from typing import Any

import regex as re


def _read_result_records(
    experiment_dir: Path,
    benchmarks: set[str],
) -> dict[str, dict[str, Any]]:
    records_by_request_id = {}
    for path in sorted(experiment_dir.rglob("*.jsonl")):
        try:
            source = path.open(encoding="utf-8", errors="replace")
        except OSError:
            continue
        with source:
            for line in source:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    not isinstance(record, dict)
                    or record.get("benchmark") not in benchmarks
                    or record.get("error") is not None
                ):
                    continue
                response = record.get("response")
                request_id = response.get("id") if isinstance(response, dict) else None
                if request_id:
                    records_by_request_id[str(request_id)] = record
    return records_by_request_id


def _record_for_request_id(
    request_id: str,
    records_by_request_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    record = records_by_request_id.get(request_id)
    if record is not None:
        return record
    base_id, separator, suffix = request_id.rpartition("-")
    if separator and re.fullmatch(r"[0-9a-f]{8}", suffix):
        return records_by_request_id.get(base_id)
    return None


def _read_step_records(
    experiment_dir: Path,
    records_by_request_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    decoder = json.JSONDecoder()
    points = []
    malformed = 0
    unmatched = 0
    log_paths = sorted(
        path
        for path in experiment_dir.rglob("*.log")
        if path.name == "server.log" or path.name.endswith(".server.log")
    )
    for path in log_paths:
        with path.open(encoding="utf-8", errors="replace") as source:
            for line in source:
                _, marker, payload = line.partition("SPEC_DECODE_STEP ")
                if not marker:
                    continue
                try:
                    event, _ = decoder.raw_decode(payload.lstrip())
                    request_id = str(event["request_id"])
                    record = _record_for_request_id(
                        request_id,
                        records_by_request_id,
                    )
                    if record is None:
                        unmatched += 1
                        continue
                    accepted_tokens = int(event["accepted_tokens"])
                    point = {
                        "benchmark": record["benchmark"],
                        "line": int(record["line"]),
                        "question_id": record.get("id")
                        or f"{record['benchmark']}-{record['line']}",
                        "request_id": request_id,
                        "step": int(event["step"]),
                        "prompt_tokens": int(event["prompt_tokens"]),
                        "generated_tokens": int(event["generated_tokens"]),
                        "context_tokens": int(event["context_tokens"]),
                        "draft_tokens": int(event["draft_tokens"]),
                        "accepted_tokens": accepted_tokens,
                        "acceptance_length": 1 + accepted_tokens,
                    }
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    malformed += 1
                    continue
                points.append(point)
    return points, malformed, unmatched


def _window_summary(
    points: list[dict[str, Any]],
    *,
    x_key: str,
    max_context_tokens: int,
    window_tokens: int,
) -> list[dict[str, float | int | None]]:
    ordered = sorted(points, key=lambda point: point[x_key])
    x_values = [int(point[x_key]) for point in ordered]
    y_values = [float(point["acceptance_length"]) for point in ordered]
    radius = window_tokens / 2
    left = 0
    right = 0
    total = 0.0
    min_queue: deque[int] = deque()
    max_queue: deque[int] = deque()
    summary = []

    for center in range(max_context_tokens + 1):
        upper = center + radius
        while right < len(ordered) and x_values[right] <= upper:
            value = y_values[right]
            total += value
            while min_queue and y_values[min_queue[-1]] >= value:
                min_queue.pop()
            while max_queue and y_values[max_queue[-1]] <= value:
                max_queue.pop()
            min_queue.append(right)
            max_queue.append(right)
            right += 1

        lower = center - radius
        while left < right and x_values[left] < lower:
            total -= y_values[left]
            if min_queue and min_queue[0] == left:
                min_queue.popleft()
            if max_queue and max_queue[0] == left:
                max_queue.popleft()
            left += 1

        count = right - left
        if count:
            summary.append(
                {
                    "x": center,
                    "mean": total / count,
                    "minimum": y_values[min_queue[0]],
                    "maximum": y_values[max_queue[0]],
                    "points": count,
                }
            )
        else:
            summary.append(
                {
                    "x": center,
                    "mean": None,
                    "minimum": None,
                    "maximum": None,
                    "points": 0,
                }
            )
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--benchmark", default="aime2026")
    parser.add_argument(
        "--window-tokens",
        type=int,
        default=512,
        help="centered smoothing window width in tokens (default: 512)",
    )
    parser.add_argument("--max-context-tokens", type=int, default=32000)
    parser.add_argument("--max-acceptance-length", type=float, default=15)
    parser.add_argument(
        "--x-axis",
        choices=("generated", "context"),
        default="context",
        help="generated tokens, or prompt plus generated tokens",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    if not experiment_dir.is_dir():
        parser.error(f"experiment directory does not exist: {experiment_dir}")
    if args.window_tokens < 1:
        parser.error("--window-tokens must be positive")
    if args.max_context_tokens < 1:
        parser.error("--max-context-tokens must be positive")
    if args.max_acceptance_length <= 1:
        parser.error("--max-acceptance-length must be greater than 1")

    records = _read_result_records(experiment_dir, {args.benchmark})
    points, malformed, unmatched = _read_step_records(experiment_dir, records)
    if not points:
        raise SystemExit(
            "No matching per-round SPEC_DECODE_STEP records found. TTS runs "
            "started before VLLM_SPEC_DECODE_STEP_METRICS logging was enabled "
            "cannot produce this curve; rerun with the updated "
            "experiments/run.sh."
        )

    x_key = "generated_tokens" if args.x_axis == "generated" else "context_tokens"
    points = [point for point in points if 0 <= point[x_key] <= args.max_context_tokens]
    if not points:
        raise SystemExit("No step records fall within the selected context range.")
    points.sort(key=lambda point: (point["benchmark"], point["line"], point["step"]))

    output_dir = (args.output_dir or experiment_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    curve = _window_summary(
        points,
        x_key=x_key,
        max_context_tokens=args.max_context_tokens,
        window_tokens=args.window_tokens,
    )
    _write_csv(output_dir / "tts_step_metrics.csv", points)
    _write_csv(output_dir / "tts_window_summary.csv", curve)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.asarray([row["x"] for row in curve], dtype=float)
    mean = np.asarray(
        [np.nan if row["mean"] is None else row["mean"] for row in curve],
        dtype=float,
    )
    minimum = np.asarray(
        [np.nan if row["minimum"] is None else row["minimum"] for row in curve],
        dtype=float,
    )
    maximum = np.asarray(
        [np.nan if row["maximum"] is None else row["maximum"] for row in curve],
        dtype=float,
    )

    figure, axis = plt.subplots(figsize=(11, 6.5))
    axis.fill_between(
        x,
        minimum,
        maximum,
        color="#9CA3AF",
        alpha=0.35,
        label="Window min–max",
    )
    axis.plot(x, mean, color="black", linewidth=1.8, label="Window mean AL")
    x_label = (
        "Generated context length (tokens; prompt excluded)"
        if args.x_axis == "generated"
        else "Context length (prompt + generated tokens)"
    )
    axis.set_xlim(0, args.max_context_tokens)
    axis.set_ylim(1, args.max_acceptance_length)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Acceptance length (including bonus token)")
    axis.set_title(f"{args.benchmark}: TTS acceptance length by context length")
    axis.grid(axis="both", color="#E5E7EB", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "tts_figure1b.png", dpi=180)
    plt.close(figure)

    clipped = sum(
        point["acceptance_length"] > args.max_acceptance_length for point in points
    )
    print(
        f"plotted {len(points)} steps from {len(records)} completed requests; "
        f"window={args.window_tokens} tokens; output={output_dir}"
    )
    if malformed or unmatched:
        print(
            f"ignored malformed step lines={malformed}, unmatched requests={unmatched}"
        )
    if clipped:
        print(
            f"note: {clipped} AL values exceed the plotted y-limit "
            f"{args.max_acceptance_length}"
        )


if __name__ == "__main__":
    main()
