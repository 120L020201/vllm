# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Atomic worker statistics shared with the benchmark process."""

import json
import os
import tempfile
import threading
from pathlib import Path


class WorkerStats:
    def __init__(self) -> None:
        raw_path = os.environ.get("OSD_STATS_FILE")
        self.path = Path(raw_path) if raw_path else None
        self.updates = 0
        self.pending = 0
        self.active_requests = 0
        self._lock = threading.Lock()
        self._write_unlocked()

    def begin_request(self) -> None:
        with self._lock:
            self.active_requests = 1
            self._write_unlocked()

    def enqueue(self) -> None:
        with self._lock:
            self.pending += 1
            self._write_unlocked()

    def complete(self, updates: int) -> None:
        if updates < 0:
            raise ValueError("updates must be nonnegative")
        with self._lock:
            self.pending = max(0, self.pending - 1)
            self.updates += updates
            self._write_unlocked()

    def add_updates(self, updates: int) -> None:
        if updates < 0:
            raise ValueError("updates must be nonnegative")
        with self._lock:
            self.updates += updates
            self._write_unlocked()

    def end_request(self) -> None:
        with self._lock:
            self.active_requests = 0
            self._write_unlocked()

    def write(self) -> None:
        with self._lock:
            self._write_unlocked()

    def _write_unlocked(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "active_requests": self.active_requests,
            "updates": self.updates,
            "pending": self.pending,
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.path.parent,
            prefix=f".{self.path.name}-",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(payload, output, sort_keys=True)
            output.write("\n")
        temporary.replace(self.path)
