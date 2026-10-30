# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Free the CPU and GPU before an online_draft run without touching SSH sessions.

Only the current user's leftover experiment processes are stopped: vLLM
servers and EngineCore workers, run.sh, benchmark.py, server_guard.py,
methods.run, run_study.py, and pytest, including their child processes.
SSH, VS Code remote servers, terminals, tmux/screen, coding agents and every
other user's processes are never signalled; heavy ones are only reported.
After cleanup the script waits until the machine is idle, so a run does not
start while someone else is using the CPU.

Usage (dry run lists what would happen; --apply acts):
    .venv/bin/python experiments/quiet_cpu.py
    .venv/bin/python experiments/quiet_cpu.py --apply && <run command>
"""

import argparse
import contextlib
import getpass
import os
import subprocess
import sys
import time

import psutil

EXPERIMENT_MARKERS = (
    "vllm.entrypoints",
    "vllm serve",
    "VLLM::",
    "experiments/run.sh",
    "experiments/benchmark.py",
    "experiments/server_guard.py",
    "methods.run",
    "methods/run.py",
    "run_study.py",
    "pytest",
)
PROTECTED_MARKERS = (
    "sshd",
    ".vscode-server",
    "vscode",
    "tmux",
    "screen",
    "claude",
    "codex",
    "systemd",
)


def describe(proc: psutil.Process) -> str:
    try:
        return " ".join(proc.cmdline()) or proc.name()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return "?"


def is_protected(proc: psutil.Process, keep: set[int]) -> bool:
    if proc.pid in keep:
        return True
    text = describe(proc)
    return any(marker in text for marker in PROTECTED_MARKERS)


def leftover_experiments(user: str, keep: set[int]) -> list[psutil.Process]:
    roots = []
    for proc in psutil.process_iter(["username"]):
        try:
            if proc.info["username"] != user or is_protected(proc, keep):
                continue
            title = proc.name() + " " + describe(proc)
            if any(marker in title for marker in EXPERIMENT_MARKERS):
                roots.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    found: dict[int, psutil.Process] = {}
    for root in roots:
        for proc in [root, *root.children(recursive=True)]:
            try:
                if proc.username() == user and not is_protected(proc, keep):
                    found[proc.pid] = proc
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    return list(found.values())


def alive(processes: list[psutil.Process]) -> list[psutil.Process]:
    running = []
    for proc in processes:
        try:
            if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                running.append(proc)
        except psutil.NoSuchProcess:
            continue
    return running


def stop(processes: list[psutil.Process], grace: float) -> list[psutil.Process]:
    # Polling instead of psutil.wait_procs: its pidfd wait fails with EINVAL
    # for processes that are not our children on this kernel.
    for signal_name, action in (("SIGTERM", "terminate"), ("SIGKILL", "kill")):
        for proc in alive(processes):
            with contextlib.suppress(psutil.NoSuchProcess):
                getattr(proc, action)()
        deadline = time.monotonic() + grace
        while alive(processes) and time.monotonic() < deadline:
            time.sleep(0.2)
        if not alive(processes):
            return []
        survivors = len(alive(processes))
        print(f"[quiet_cpu] {survivors} process(es) survived {signal_name}")
    return alive(processes)


def gpu_processes() -> list[tuple[int, str]]:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            rows.append((int(parts[0]), parts[1]))
    return rows


def top_consumers(sample: float, limit: int = 8) -> list[tuple[float, str, int, str]]:
    procs = list(psutil.process_iter(["username"]))
    for proc in procs:
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            proc.cpu_percent(None)
    time.sleep(sample)
    rows = []
    for proc in procs:
        try:
            usage = proc.cpu_percent(None)
            if usage >= 5:
                rows.append(
                    (
                        usage,
                        proc.info["username"] or "?",
                        proc.pid,
                        describe(proc)[:110],
                    )
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return sorted(rows, reverse=True)[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--apply", action="store_true", help="stop leftovers (default: dry run)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="idle when whole-machine CPU use is below this percent (default 5)",
    )
    parser.add_argument(
        "--samples", type=int, default=3, help="consecutive idle 2 s samples needed"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=900,
        help="max seconds to wait for idle; 0 skips",
    )
    parser.add_argument(
        "--grace", type=float, default=15, help="seconds between SIGTERM and SIGKILL"
    )
    args = parser.parse_args()

    user = getpass.getuser()
    me = psutil.Process(os.getpid())
    keep = {me.pid, *(p.pid for p in me.parents())}

    leftovers = leftover_experiments(user, keep)
    print(f"[quiet_cpu] {len(leftovers)} leftover experiment process(es) of {user}")
    for proc in leftovers:
        print(f"  pid {proc.pid}: {describe(proc)[:140]}")
    if leftovers and args.apply:
        survivors = stop(leftovers, args.grace)
        if survivors:
            for proc in survivors:
                print(f"  could not stop pid {proc.pid}: {describe(proc)[:140]}")
            return 1
        print("[quiet_cpu] stopped them")
    elif leftovers:
        print("[quiet_cpu] dry run: rerun with --apply to stop them")

    busy_gpu = []
    for pid, memory in gpu_processes():
        try:
            owner = psutil.Process(pid).username()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            owner = "?"
        busy_gpu.append((pid, owner, memory))
    for pid, owner, memory in busy_gpu:
        print(f"[quiet_cpu] GPU still in use: pid {pid} ({owner}) {memory} MiB")

    if args.timeout <= 0:
        return 0 if not busy_gpu or not args.apply else 1
    deadline = time.monotonic() + args.timeout
    idle = 0
    while time.monotonic() < deadline:
        usage = psutil.cpu_percent(interval=2)
        idle = idle + 1 if usage < args.threshold else 0
        if idle >= args.samples and not (args.apply and gpu_processes()):
            print(f"[quiet_cpu] machine idle: CPU {usage:.1f}% < {args.threshold}%")
            return 0
        if idle == 0:
            print(
                f"[quiet_cpu] waiting: whole-machine CPU {usage:.1f}% "
                f"(threshold {args.threshold}%)"
            )
    print("[quiet_cpu] still busy after timeout; top CPU users (not touched):")
    for usage, owner, pid, command in top_consumers(2):
        print(f"  {usage:6.1f}%  {owner:10s} pid {pid}: {command}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
