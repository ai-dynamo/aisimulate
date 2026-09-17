# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Admit and supervise bounded waves while retaining completed evaluations."""

from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .resources import ResourceLimitError
from .supervision import checkpoint, close_pool, mark_execution_ready, terminate_pool


@dataclass(frozen=True)
class InterruptedEvaluation:
    reason: str
    resource_limited: bool
    metadata: dict[str, Any]


def _initialize(factory, initializer, ready_directory):
    initializer(factory)
    Path(ready_directory, str(os.getpid())).touch()


def evaluate_waves(specs, *, factory, initializer, evaluate, workers: int, timeout: float | None):
    """Yield indexed results. Retry interrupted work at most twice, at fewer workers.

    One admitted wave owns the workers at a time. Workers are reaped before
    admission of the next wave, so retained runtime allocations cannot overlap
    a replacement's reservation. Completed results are yielded before recovery.
    """
    pending = list(range(len(specs)))
    attempts = dict.fromkeys(pending, 0)
    capacity = max(1, workers)
    while pending:
        wave = pending[:capacity]
        while True:
            try:
                plan = factory.admit_wave([specs[index] for index in wave])
                checkpoint("wave_admitted", plan)
                break
            except ResourceLimitError as exc:
                if len(wave) > 1:
                    wave = wave[: max(1, len(wave) // 2)]
                    continue
                index = wave[0]
                pending.remove(index)
                yield index, InterruptedEvaluation(str(exc), True, exc.plan)
                wave = []
                break
        if not wave:
            continue
        pending = pending[len(wave) :]
        for index in wave:
            attempts[index] += 1
        ready = tempfile.TemporaryDirectory(prefix="aisimulate-wave-")
        pool = ProcessPoolExecutor(
            max_workers=len(wave),
            mp_context=mp.get_context("spawn"),
            initializer=_initialize,
            initargs=(factory, initializer, ready.name),
        )
        active = {}
        interrupted = None
        try:
            active = {pool.submit(evaluate, specs[index]): index for index in wave}
            started = time.monotonic()
            initialized = False
            initialization_timeout = getattr(getattr(factory, "policy", None), "initialization_timeout_seconds", 60.0)
            deadline = started + timeout if timeout else None
            while active:
                if not initialized:
                    initialized = len(list(Path(ready.name).iterdir())) == len(wave)
                    if initialized:
                        mark_execution_ready()
                    elif time.monotonic() - started >= initialization_timeout:
                        interrupted = InterruptedEvaluation("worker initialization timed out", False, {})
                        break
                done, _ = wait(active, timeout=0.05, return_when=FIRST_COMPLETED)
                for future in done:
                    index = active[future]
                    try:
                        result = future.result()
                    except ResourceLimitError as exc:
                        interrupted = InterruptedEvaluation(str(exc), True, exc.plan)
                        continue
                    del active[future]
                    yield index, result
                if interrupted:
                    break
                if not active:
                    break
                pressure = factory.live_pressure()
                if pressure:
                    interrupted = InterruptedEvaluation("live memory pressure", True, pressure)
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    interrupted = InterruptedEvaluation(f"exceed runtime: replay > {timeout:g}s", False, {})
                    break
            if interrupted:
                unfinished = [active[future] for future in active]
                terminate_pool(pool)
                pool = None
                capacity = max(1, len(wave) // 2)
                retry = []
                for index in unfinished:
                    if interrupted.resource_limited and attempts[index] < 3:
                        retry.append(index)
                    else:
                        yield index, interrupted
                checkpoint(
                    "wave_interrupted",
                    {"reason": interrupted.reason, "retry_candidates": retry, "next_parallelism": capacity},
                )
                pending = retry + pending
            else:
                close_pool(pool)
                pool = None
        finally:
            if pool is not None:
                terminate_pool(pool)
            ready.cleanup()
