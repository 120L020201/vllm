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

    stats: dict[str, list[float]] = defaultdict(list)
    for event in trace.get("traceEvents", []):
        name = event.get("name")
        duration_us = event.get("dur")
        if not isinstance(name, str) or not name.startswith("online_eagle3."):
            continue
        if isinstance(duration_us, int | float):
            stats[name].append(float(duration_us) / 1000.0)
        else:
            stats[name].append(0.0)

    print(f"trace: {trace_path}")
    if not stats:
        print("No online_eagle3.* markers found.")
        return 0

    print("marker,count,total_ms,mean_ms,max_ms")
    for name in sorted(stats):
        values = stats[name]
        total = sum(values)
        mean = total / len(values)
        max_value = max(values)
        print(f"{name},{len(values)},{total:.3f},{mean:.3f},{max_value:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
