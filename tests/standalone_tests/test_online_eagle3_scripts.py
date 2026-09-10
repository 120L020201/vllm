# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gzip
import json
import subprocess
import sys
from pathlib import Path


def test_result_directory_names(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
source "$1/run/online_eagle3_common.sh"
ONLINE_EAGLE3_REPO_ROOT=$2
RESULT_ROOT=runs
date() { printf '20260910_120000\\n'; }
online_eagle3_result_dir online_eagle3_smoke
online_eagle3_result_dir online_eagle3_smoke
""",
            "bash",
            str(repo_root),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    paths = [Path(line) for line in result.stdout.splitlines()]
    assert paths == [
        tmp_path / "runs/online_eagle3_smoke_20260910_120000",
        tmp_path / "runs/online_eagle3_smoke_20260910_120000_2",
    ]
    assert all(path.is_dir() for path in paths)
    assert result.stderr == ""


def test_trace_summary_separates_cpu_and_gpu(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    event = {
        "name": "online_eagle3.cpu_update_step",
        "cat": "user_annotation",
        "ph": "X",
        "dur": 1000,
    }
    trace_path = tmp_path / "trace.json.gz"
    with gzip.open(trace_path, "wt", encoding="utf-8") as trace_file:
        json.dump(
            {
                "traceEvents": [
                    event,
                    {**event, "cat": "gpu_user_annotation", "dur": 100},
                    {**event, "ph": "B"},
                    {**event, "dur": None},
                    {**event, "cat": "cpu_op"},
                    {**event, "name": "Optimizer.step#AdamW.step", "dur": 500},
                ]
            },
            trace_file,
        )
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run/online_eagle3_trace_summary.py"),
            str(trace_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "cpu,online_eagle3.cpu_update_step,1,1.000,1.000,1.000" in result.stdout
    assert (
        "gpu_range,online_eagle3.cpu_update_step,1,0.100,0.100,0.100" in result.stdout
    )
    assert "cpu,Optimizer.step#AdamW.step,1,0.500,0.500,0.500" in result.stdout
