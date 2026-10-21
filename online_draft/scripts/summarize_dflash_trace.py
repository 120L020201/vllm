# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Measure online copies and their overlap with inference in a PyTorch trace."""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def overlap(first, second):
    index = 0
    total = 0
    for start, end in first:
        while index < len(second) and second[index][1] <= start:
            index += 1
        current = index
        while current < len(second) and second[current][0] < end:
            left, right = second[current]
            total += max(0, min(end, right) - max(start, left))
            current += 1
    return total


def event_summary(events):
    durations = sorted(e["dur"] / 1000 for e in events)
    return {
        "count": len(durations),
        "total_ms": sum(durations),
        "median_ms": median(durations),
        "p95_ms": durations[int(0.95 * (len(durations) - 1))],
        "max_ms": durations[-1],
    }


def summarize(path):
    raw = (
        gzip.decompress(path.read_bytes())
        if path.suffix == ".gz"
        else path.read_bytes()
    )
    events = json.loads(raw)["traceEvents"]
    kernels = defaultdict(list)
    for event in events:
        if event.get("cat") == "kernel":
            kernels[event["args"]["stream"]].append(
                (event["ts"], event["ts"] + event["dur"])
            )
    inference_stream = max(kernels, key=lambda s: sum(b - a for a, b in kernels[s]))
    inference = merge_intervals(kernels[inference_stream])
    runtime = [event for event in events if event.get("cat") == "cuda_runtime"]
    result = {
        "trace": str(path),
        "inference_stream": inference_stream,
        "cudaHostAlloc_count": sum(e["name"] == "cudaHostAlloc" for e in runtime),
    }
    by_kernel = defaultdict(list)
    by_range = defaultdict(list)
    by_sync = defaultdict(list)
    for event in events:
        if event.get("cat") == "kernel" and event["args"]["stream"] == inference_stream:
            by_kernel[event["name"]].append(event)
        if event.get("cat") == "user_annotation" and event.get("name", "").startswith(
            "online_draft."
        ):
            by_range[event["name"].split("[")[0]].append(event)
        if event.get("cat") == "cuda_runtime" and "Synchronize" in event["name"]:
            by_sync[f"{event['tid']}:{event['name']}"].append(event)
    busy = sum(end - start for start, end in inference)
    span = inference[-1][1] - inference[0][0]
    graph_launch = [e for e in runtime if e["name"] == "cudaGraphLaunch"]
    result["inference"] = {
        "span_ms": span / 1000,
        "kernel_busy_ms": busy / 1000,
        "kernel_busy_percent": 100 * busy / span,
        "graph_launch": event_summary(graph_launch) if graph_launch else None,
        "top_kernels": dict(
            sorted(
                ((name, event_summary(group)) for name, group in by_kernel.items()),
                key=lambda item: item[1]["total_ms"],
                reverse=True,
            )[:20]
        ),
    }
    result["cpu_ranges"] = {
        name: event_summary(group) for name, group in by_range.items()
    }
    training_threads = {e["tid"] for e in by_range["online_draft.cpu_train"]}
    training_ops = defaultdict(list)
    for event in events:
        if event.get("cat") == "cpu_op" and event["tid"] in training_threads:
            training_ops[event["name"]].append(event)
    result["training_operators_inclusive"] = dict(
        sorted(
            ((name, event_summary(group)) for name, group in training_ops.items()),
            key=lambda item: item[1]["total_ms"],
            reverse=True,
        )[:30]
    )
    result["synchronization"] = {
        name: event_summary(group) for name, group in by_sync.items()
    }
    for label, prefix in (
        ("h2d", "online_draft.h2d_stage"),
        ("d2h", "online_draft.d2h"),
    ):
        ranges = [
            e
            for e in events
            if e.get("cat") == "user_annotation"
            and e.get("name", "").startswith(prefix)
        ]
        calls = [
            event
            for event in runtime
            if any(
                event["tid"] == r["tid"] and r["ts"] <= event["ts"] < r["ts"] + r["dur"]
                for r in ranges
            )
        ]
        correlations = {event["args"]["correlation"] for event in calls}
        copies = [
            event
            for event in events
            if event.get("cat") == "gpu_memcpy"
            and event["args"]["correlation"] in correlations
        ]
        intervals = merge_intervals((e["ts"], e["ts"] + e["dur"]) for e in copies)
        duration = sum(end - start for start, end in intervals)
        result[label] = {
            "bytes": sum(e["args"]["bytes"] for e in copies),
            "copy_ms": duration / 1000,
            "overlap_percent": (
                100 * overlap(intervals, inference) / duration if duration else None
            ),
            "streams": sorted({e["args"]["stream"] for e in copies}),
            "cpu_range_ms": [e["dur"] / 1000 for e in ranges],
            "synchronize_calls": [
                e["name"] for e in calls if "Synchronize" in e["name"]
            ],
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.trace)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("h2d", "d2h")}))
    for label in ("h2d", "d2h"):
        print(
            label,
            {
                k: v
                for k, v in result[label].items()
                if k not in ("cpu_range_ms", "synchronize_calls")
            },
        )


if __name__ == "__main__":
    main()
