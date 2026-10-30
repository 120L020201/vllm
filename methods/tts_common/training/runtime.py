# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def profile_range(name: str, enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    with torch.profiler.record_function(name):
        yield
