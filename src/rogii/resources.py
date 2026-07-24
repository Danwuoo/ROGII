"""Small deterministic CPU resource configuration helper."""

from __future__ import annotations

import os
from typing import Any


_THREAD_LIMITER: Any | None = None


def configure_cpu(*, workers: int) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    names = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    for name in names:
        os.environ[name] = str(workers)

    global _THREAD_LIMITER
    from threadpoolctl import threadpool_info, threadpool_limits

    _THREAD_LIMITER = threadpool_limits(limits=workers)
    return {
        "cpu_workers": workers,
        "thread_environment": {name: os.environ[name] for name in names},
        "threadpools": threadpool_info(),
    }
