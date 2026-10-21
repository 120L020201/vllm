# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import torch


@cache
def _parallel_runtime():
    if (
        not torch.backends.openmp.is_available()
        or not torch.backends.mkl.is_available()
    ):
        raise ValueError("per-worker CPU threads require an OpenMP/MKL PyTorch build")
    library = ctypes.CDLL(str(Path(torch.__file__).parent / "lib/libtorch_cpu.so"))
    library.omp_set_num_threads.argtypes = [ctypes.c_int]
    library.omp_set_num_threads.restype = None
    library.MKL_Set_Num_Threads_Local.argtypes = [ctypes.c_int]
    library.MKL_Set_Num_Threads_Local.restype = ctypes.c_int
    return library


@dataclass(frozen=True)
class CPUWorkerConfig:
    """Set thread-local parallelism without resizing inference's thread pool."""

    threads: int | None = None
    cpus: tuple[int, ...] = ()

    def __post_init__(self):
        if self.threads is not None:
            if self.threads < 1:
                raise ValueError("CPU worker threads must be positive")
            _parallel_runtime()
        if self.cpus and not set(self.cpus).issubset(os.sched_getaffinity(0)):
            raise ValueError("CPU worker affinity must use available CPUs")

    def apply(self):
        if self.cpus:
            os.sched_setaffinity(0, self.cpus)
        if self.threads is not None:
            library = _parallel_runtime()
            library.omp_set_num_threads(self.threads)
            library.MKL_Set_Num_Threads_Local(self.threads)
