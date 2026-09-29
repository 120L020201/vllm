# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Atomic worker statistics shared with the benchmark process."""

import json
import os
import tempfile
from pathlib import Path


class WorkerStats:
    def __init__(self) -> None:
        raw_path = os.environ.get("OSD_STATS_FILE")
        self.path = Path(raw_path) if raw_path else None
        self.updates = 0
        self.pending = 0
        self.write()

    def enqueue(self) -> None:
        self.pending += 1
        self.write()

    def complete(self, updates: int) -> None:
        if updates < 0:
            raise ValueError("updates must be nonnegative")
        self.pending = max(0, self.pending - 1)
        self.updates += updates
        self.write()

    def write(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"updates": self.updates, "pending": self.pending}
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
