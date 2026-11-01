# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot saved interval AL logs for five AIME2026 question 23 experiments."""

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import regex as re
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("baseline", "Baseline", "online_draft-prefix-retest-20261021/baseline"),
    ("position09", "Position 0.9", "online_draft-prefix-retest-20261021/position09"),
    ("uniform", "Uniform", "online_draft-prefix-retest-20261021/uniform"),
    (
        "prefix_balanced",
        "Prefix-Balanced α=1",
        "online_draft-prefix-balanced-study-20261021/prefix_balanced_alpha1",
    ),
    (
        "position09_rejection2",
        "Position 0.9 + 拒绝固定 2",
        "online_draft-position-rejection-study-20261021/position09_rejection2",
    ),
)
COLORS = ("#334155", "#0072B2", "#009E73", "#8B5CF6", "#D55E00")
MARKERS = ("o", "s", "^", "D", "P")
STYLES = ("-", "-", "--", "-.", ":")


def timestamp(line: str) -> datetime:
    match = re.search(r"\b(\d{2}-\d{2} \d{2}:\d{2}:\d{2})\b", line)
    if match is None:
        raise ValueError(f"missing timestamp: {line}")
    return datetime.strptime("2026-" + match[1], "%Y-%m-%d %H:%M:%S")


def load_case(case: str, label: str, directory: str) -> tuple[list[dict], dict]:
    result_path = ROOT / "results" / directory / "results.jsonl"
    log_path = result_path.parent / "part1/server.log"
    records = [json.loads(line) for line in result_path.read_text().splitlines()]
    selected = [record for record in records if record["id"] == "aime2026-23"]
    assert len(selected) == 1
    record = selected[0]
    assert record.get("error") is None
    request_id = record["spec_decode_prefixes"]["1000"]["request_id"]
    assert request_id == record["spec_decode_prefixes"]["2000"]["request_id"]
    lines = log_path.read_text().splitlines()
    installs = [
        i
        for i, line in enumerate(lines)
        if f"request_id={request_id} " in line and "installed weights:" in line
    ]
    first, last = installs[0], installs[-1]
    start = max(i for i, line in enumerate(lines[:first]) if '"POST /tokenize ' in line)
    end = next(
        i
        for i in range(last + 1, len(lines))
        if '"POST /v1/chat/completions ' in lines[i]
    )
    prefix_line = next(
        i
        for i, line in enumerate(lines)
        if "SPEC_DECODE_PREFIX " in line
        and request_id in line
        and '"drafts": 2000' in line
    )
    origin = timestamp(lines[first])
    rows = []
    excluded = []
    previous = None
    for i, line in enumerate(lines):
        if "SpecDecoding metrics:" not in line:
            continue
        if start < i < end:
            pure = previous is not None and previous > start
            if not pure:
                excluded.append(i + 1)
            else:
                al = re.search(r"Mean acceptance length: ([\d.]+)", line)
                counts = re.search(r"Accepted: (\d+) tokens, Drafted: (\d+)", line)
                assert al is not None and counts is not None
                left = (timestamp(lines[previous]) - origin).total_seconds()
                right = (timestamp(line) - origin).total_seconds()
                assert right > left >= 0
                rows.append(
                    {
                        "case": case,
                        "label": label,
                        "question_id": record["id"],
                        "request_id": request_id,
                        "interval_start_seconds": left,
                        "interval_end_seconds": right,
                        "interval_mean_al": float(al[1]),
                        "accepted_tokens": int(counts[1]),
                        "draft_tokens": int(counts[2]),
                        "before_2000_snapshot": i < prefix_line,
                        "previous_log_line": previous + 1,
                        "log_line": i + 1,
                        "server_log": str(log_path.relative_to(ROOT)),
                    }
                )
        previous = i
    assert len(rows) >= 2
    metadata = {
        "case": case,
        "label": label,
        "result_path": str(result_path.relative_to(ROOT)),
        "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "server_log": str(log_path.relative_to(ROOT)),
        "server_log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
        "request_config": record["request_config"],
        "time_origin": origin.isoformat(),
        "first_install_line": first + 1,
        "tokenize_line": start + 1,
        "completion_line": end + 1,
        "round_2000_snapshot_seconds": (
            timestamp(lines[prefix_line]) - origin
        ).total_seconds(),
        "round_2000_snapshot_line": prefix_line + 1,
        "excluded_mixed_interval_lines": excluded,
        "points": len(rows),
        "points_before_2000_snapshot": sum(r["before_2000_snapshot"] for r in rows),
    }
    return rows, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/online_draft-aime23-al-time-20261021",
    )
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rows, metadata = [], []
    for case, label, directory in CASES:
        case_rows, source = load_case(case, label, directory)
        rows.extend(case_rows)
        metadata.append(source)
    assert all(m["request_config"] == metadata[0]["request_config"] for m in metadata)
    with (output / "interval_al.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "sources.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )

    plt.switch_backend("Agg")
    available = {font.name for font in font_manager.fontManager.ttflist}
    fonts = ["Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback"]
    font = next((name for name in fonts if name in available), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": font,
            "font.size": 12,
            "axes.unicode_minus": False,
            "svg.fonttype": "path",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 8), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.975, top=0.70, bottom=0.24, wspace=0.13)
    fig.text(
        0.075, 0.94, "AIME2026 [23]：区间平均 AL 随时间变化", fontsize=23, weight="bold"
    )
    fig.text(
        0.075,
        0.89,
        "Qwen3-8B + EAGLE3 · D=7 · temperature=0.6 · seed=0 · 每组一次已有运行",
        fontsize=12,
        color="#475569",
    )
    for index, (case, label, _) in enumerate(CASES):
        own = [row for row in rows if row["case"] == case]
        for ax, prefix_only in zip(axes, (True, False), strict=True):
            visible = [r for r in own if r["before_2000_snapshot"] or not prefix_only]
            ax.plot(
                [r["interval_end_seconds"] for r in visible],
                [r["interval_mean_al"] for r in visible],
                color=COLORS[index],
                marker=MARKERS[index],
                linestyle=STYLES[index],
                linewidth=2.4,
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.7,
                label=label,
            )
    titles = ("第 2000 轮快照之前的完整区间", "整段生成过程的完整区间")
    for ax, title, xmax in zip(axes, titles, (31, 52), strict=True):
        ax.set_title(title, loc="left", fontsize=14, pad=18)
        ax.set_xlim(0, xmax)
        ax.set_ylim(1.98, 2.85)
        ax.set_yticks([2.0, 2.2, 2.4, 2.6, 2.8])
        ax.set_xticks(list(range(0, xmax, 10)))
        ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#CBD5E1")
        ax.tick_params(length=0, pad=8)
    axes[0].set_ylabel("区间平均接收长度 AL（含 bonus token）", labelpad=14)
    axes[1].axvspan(29, 30, color="#94A3B8", alpha=0.18, zorder=0)
    axes[1].text(
        29.5,
        2.81,
        "各组第 2000 轮\n约 29–30 秒",
        ha="center",
        fontsize=10,
        color="#64748B",
    )
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.07, 0.85),
        ncol=3,
        frameon=False,
        fontsize=12,
    )
    fig.supxlabel(
        "距该题首次权重安装的时间（秒）；点位于统计区间结束时刻", y=0.16, fontsize=13
    )
    fig.text(
        0.075,
        0.095,
        "每个点为日志中约 10 秒区间的平均值（原日志保留两位小数），"
        "不是累计 AL 或逐轮 AL。",
        fontsize=11,
        color="#475569",
    )
    fig.text(
        0.075,
        0.055,
        "已排除跨题区间；左图还排除跨第 2000 轮快照的区间。"
        "时间零点是日志代理，精度为 1 秒。",
        fontsize=11,
        color="#475569",
    )
    for extension in ("png", "svg", "pdf"):
        fig.savefig(
            output / f"aime23_al_vs_time.{extension}", dpi=200, facecolor="white"
        )
    plt.close(fig)
    print(output)
    print(
        json.dumps(
            [
                {k: m[k] for k in ("case", "points", "points_before_2000_snapshot")}
                for m in metadata
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
