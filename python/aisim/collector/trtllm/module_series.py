# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure several points of one TRT-LLM attention-module key in one task.

A module key is (model, heads, precision, batch); a point is one
(seq_len, prefix_len) of it. A series measures several points of one key:

- The layer comes once from the collector's per-worker layer cache.
- Every point still builds its own KV cache manager, metadata and inputs and
  tears them down, exactly like a one-point case. The manager's capacity is a
  kernel input, so one manager sized for the largest point would change what the
  smaller points measure:
  - the attention metadata reads ``max_seq_len`` from the manager
    (``TrtllmAttentionMetadata.max_seq_len``, attention/backends/trtllm.py);
  - DeepSeek-V4 sizes its compressed-index buffers from that value
    (attention/backends/sparse/deepseek_v4/metadata.py, ``max_compressed_indices``);
  - an MSA decode step selects blocks over the manager's worst-case block
    extent, which picks the selection kernel (see
    collect_msa_module.run_msa_module_series).
- The dry run only checks the forward and warms lazily initialised paths before
  timing, so it runs once per seq_len.
- Points run largest first, so a capacity failure shows on the first point.

A point that fails for a non-fatal reason (an OOM, a kernel limit) is recorded,
and the remaining points still run. The failure is raised after the series so
collect.py records the case. A sticky CUDA failure aborts the series at once:
nothing measured on that context afterwards can be trusted.
"""

from __future__ import annotations

import gc
import sys
import traceback
from collections.abc import Callable, Iterable, Sequence

KERNEL_LIMIT_PREFIX = "FIXME(kernel-limit)"


def ordered_points(points: Iterable[Sequence[int]]) -> list[tuple[int, int]]:
    """The distinct (seq_len, prefix_len) points of a series, largest total length first."""
    distinct: set[tuple[int, int]] = set()
    for point in points:
        if isinstance(point, (str, bytes)) or len(point) != 2:
            raise TypeError(f"a series point is (seq_len, prefix_len), got {point!r}")
        seq_len, prefix_len = (int(value) for value in point)
        if seq_len < 1 or prefix_len < 0:
            raise ValueError(f"a series point needs seq_len >= 1 and prefix_len >= 0, got {point!r}")
        distinct.add((seq_len, prefix_len))
    if not distinct:
        raise ValueError("a module series needs at least one point")
    return sorted(distinct, key=lambda point: (point[0] + point[1], point[0]), reverse=True)


def is_fatal_cuda_error(exc: BaseException) -> bool:
    """A sticky CUDA failure: nothing measured on this context afterwards can be trusted."""
    torch = sys.modules.get("torch")
    accelerator_error = getattr(torch, "AcceleratorError", None)
    if isinstance(accelerator_error, type) and isinstance(exc, accelerator_error):
        return True
    text = str(exc).lower()
    markers = ("cuda error", "illegal memory access", "cudaerrorillegaladdress", "device-side assert")
    return any(marker in text for marker in markers)


def run_points(
    points: Iterable[Sequence[int]],
    run_point: Callable[[int, int, bool], float],
    *,
    label: str,
) -> list[float]:
    """Call ``run_point(seq_len, prefix_len, dry_run)`` for every point, largest first.

    ``dry_run`` is True for the first point of each seq_len that has not yet
    run through. Returns the latencies of the measured points, in run order.
    """
    ordered = ordered_points(points)
    dry_run_done: set[int] = set()
    latencies: list[float] = []
    failures: list[tuple[tuple[int, int], BaseException]] = []
    for seq_len, prefix_len in ordered:
        try:
            latencies.append(run_point(seq_len, prefix_len, seq_len not in dry_run_done))
        except Exception as exc:
            if len(ordered) == 1 or is_fatal_cuda_error(exc):
                raise
            # The traceback holds the point's frames (KV cache, metadata, activations):
            # clear them, or that memory stays pinned and the smaller points fail too.
            traceback.clear_frames(exc.__traceback__)
            failures.append(((seq_len, prefix_len), exc))
            print(
                f"  [{label}] s={seq_len} prefix={prefix_len}: point failed "
                f"({type(exc).__name__}: {str(exc)[:200]}); the series continues"
            )
            gc.collect()
            continue
        dry_run_done.add(seq_len)
    if len(ordered) > 1:
        print(f"  [{label}] {len(latencies)}/{len(ordered)} points measured")
    if failures:
        if len(failures) == 1:
            raise failures[0][1]
        summary = "; ".join(f"s={s} prefix={p}: {type(exc).__name__}: {str(exc)[:200]}" for (s, p), exc in failures)
        # Keep the kernel-limit classification when every point failed on a kernel limit.
        prefix = (
            f"{KERNEL_LIMIT_PREFIX}: " if all(str(exc).startswith(KERNEL_LIMIT_PREFIX) for _, exc in failures) else ""
        )
        raise RuntimeError(f"{prefix}{len(failures)}/{len(ordered)} point(s) of the module series failed: {summary}")
    return latencies
