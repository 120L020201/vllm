#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: online_eagle3_trace_summary.py TRACE.pt.trace.json.gz")
        return 2

    trace_path = Path(sys.argv[1])
    with gzip.open(trace_path, "rt") as trace_file:
        trace = json.load(trace_file)

    stats: dict[tuple[str, str], list[float]] = defaultdict(list)
    for event in trace.get("traceEvents", []):
        name = event.get("name")
        duration_us = event.get("dur")
        category = event.get("cat")
        if category not in ("user_annotation", "gpu_user_annotation"):
            continue
        if not isinstance(name, str) or not (
            name.startswith("online_eagle3.") or name.startswith("Optimizer.step#")
        ):
            continue
        if event.get("ph") != "X" or not isinstance(duration_us, int | float):
            continue
        track = "cpu" if category == "user_annotation" else "gpu_range"
        stats[track, name].append(float(duration_us) / 1000.0)

    print(f"trace: {trace_path}")
    if not stats:
        print("No online_eagle3.* markers found.")
        return 0

    print("CPU ranges are host wall time; GPU ranges may include gaps between kernels.")
    print("track,marker,count,total_ms,mean_ms,max_ms")
    for track, name in sorted(stats):
        values = stats[track, name]
        total = sum(values)
        mean = total / len(values)
        max_value = max(values)
        print(f"{track},{name},{len(values)},{total:.3f},{mean:.3f},{max_value:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
