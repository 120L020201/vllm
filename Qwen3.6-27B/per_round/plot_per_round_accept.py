# SPDX-License-Identifier: Apache-2.0
"""Plot per-round accepted length and report median/variance.

Usage:
  # single run (one method):
  python plot_per_round_accept.py run.json

  # compare multiple runs on the same axes (label:path or just path):
  python plot_per_round_accept.py mtp:per_round_accept_aime2026.json \
      eagle3:per_round_accept_eagle3_aime2026.json --out compare.png
"""
import argparse
import json
import os
import statistics

import matplotlib

matplotlib.use("Agg")  # headless: write PNG, no display needed
import matplotlib.pyplot as plt  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "inputs",
        nargs="+",
        help="JSON file(s). Optionally prefix with a label, e.g. mtp:path.json",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output PNG path. Defaults to the first input's name + .png "
        "(single run) or per_round_accept_compare.png (multiple runs).",
    )
    p.add_argument(
        "--stats-out",
        default=None,
        help="Where to dump the per-(run,problem) stats JSON.",
    )
    return p.parse_args()


def load_runs(inputs):
    """Return list of (label, data) where data is the parsed JSON list."""
    runs = []
    for spec in inputs:
        if ":" in spec and not os.path.exists(spec):
            label, path = spec.split(":", 1)
        else:
            label, path = os.path.splitext(os.path.basename(spec))[0], spec
        runs.append((label, json.load(open(path))))
    return runs


def compute_stats(rounds):
    median = statistics.median(rounds) if rounds else 0.0
    variance = statistics.pvariance(rounds) if len(rounds) > 1 else 0.0
    stdev = statistics.pstdev(rounds) if len(rounds) > 1 else 0.0
    mean = statistics.fmean(rounds) if rounds else 0.0
    mx = max(rounds) if rounds else 0
    return median, variance, stdev, mean, mx


def main():
    args = parse_args()
    runs = load_runs(args.inputs)
    multi = len(runs) > 1

    # Group problems by problem_idx so the same problem lines up across runs.
    # One subplot per problem; within a subplot, one line per run.
    all_idx = sorted({rec["problem_idx"] for _, data in runs for rec in data})
    n = len(all_idx)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=False)
    if n == 1:
        axes = [axes]
    ax_by_idx = dict(zip(all_idx, axes))

    colors = ["steelblue", "darkorange", "seagreen", "crimson"]

    header = f"{'run':>10} {'prob':>5} {'rounds':>7} {'max':>4} {'mean':>7} " \
             f"{'median':>7} {'variance':>9} {'stdev':>7}"
    print(header)
    print("-" * len(header))

    stats_rows = []
    for ri, (label, data) in enumerate(runs):
        color = colors[ri % len(colors)]
        for rec in data:
            idx = rec["problem_idx"]
            rounds = rec["per_round_accept"]
            median, variance, stdev, mean, mx = compute_stats(rounds)

            print(f"{label:>10} {idx:>5} {len(rounds):>7} {mx:>4} {mean:>7.3f} "
                  f"{median:>7.1f} {variance:>9.3f} {stdev:>7.3f}")
            stats_rows.append(
                {
                    "run": label,
                    "problem_idx": idx,
                    "num_rounds": len(rounds),
                    "max": mx,
                    "mean": round(mean, 3),
                    "median": median,
                    "variance": round(variance, 3),
                    "stdev": round(stdev, 3),
                }
            )

            ax = ax_by_idx[idx]
            lbl = label if multi else None
            ax.plot(range(len(rounds)), rounds, linewidth=0.5, color=color,
                    alpha=0.8, label=lbl)
            ax.axhline(median, color=color, linestyle="--", linewidth=1)

    for idx in all_idx:
        ax = ax_by_idx[idx]
        ax.set_title(f"Problem {idx}")
        ax.set_ylabel("accepted len")
        ax.set_ylim(-1, 33)
        ax.grid(True, alpha=0.3)
        if multi:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("round index")

    title = "Per-round accepted length (num_spec_tokens=32) — AIME2026 first 5"
    if multi:
        title += "  [" + " vs ".join(lbl for lbl, _ in runs) + "]"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.99])

    if args.out:
        out_png = args.out
    elif multi:
        out_png = "per_round_compare_accept_aime2026.png"
    else:
        out_png = os.path.splitext(args.inputs[0].split(":", 1)[-1])[0] + ".png"
    fig.savefig(out_png, dpi=120)
    print(f"\nwrote {out_png}")

    stats_out = args.stats_out or (
        os.path.splitext(out_png)[0] + "_stats.json"
    )
    with open(stats_out, "w") as f:
        json.dump(stats_rows, f, ensure_ascii=False, indent=2)
    print(f"wrote {stats_out}")


if __name__ == "__main__":
    main()
