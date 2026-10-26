# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel-only timing of the GLM attention collectors: trace attribution."""

import pytest
from collector.glm53flash_attention_runtime import (
    REPETITION_RANGE,
    TimingAttributionError,
    attribute_repetitions,
)

pytestmark = pytest.mark.unit


def _range(index, ts, dur):
    return {"ph": "X", "cat": "user_annotation", "name": f"{REPETITION_RANGE}{index}", "ts": ts, "dur": dur}


def _launch(correlation, ts, name="cudaLaunchKernel", cat="cuda_runtime"):
    return {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": 2.0, "args": {"correlation": correlation}}


def _gpu(correlation, ts, dur, cat="kernel", name="k", device=0):
    args = {"device": device, "stream": 7}
    if correlation is not None:
        args["correlation"] = correlation
    return {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur, "args": args}


def _graph_and_breaks(offset):
    """One repetition: a graph launch (3 node kernels) and an eager break whose
    kernels run after a 40 us host gap, plus a memset; ends with a sync."""
    return [
        _range(offset // 1000, offset, 400.0),
        _launch(offset + 1, offset + 5, name="cudaGraphLaunch"),
        _gpu(offset + 1, offset + 20, 10.0, name="gemm"),
        _gpu(offset + 1, offset + 31, 5.0, name="norm"),
        _gpu(offset + 1, offset + 36, 4.0, name="quant"),
        _launch(offset + 2, offset + 70, name="cuLaunchKernel", cat="cuda_driver"),
        _gpu(offset + 2, offset + 80, 20.0, name="mqa_logits"),
        _launch(offset + 3, offset + 90, name="cudaMemsetAsync"),
        _gpu(offset + 3, offset + 100, 1.0, cat="gpu_memset", name="Memset"),
        # A second-stream kernel overlapping mqa_logits: counted once by the union.
        _launch(offset + 4, offset + 75),
        _gpu(offset + 4, offset + 90, 15.0, name="side"),
    ]


def test_busy_union_excludes_host_gaps_and_counts_overlap_once():
    trace = _graph_and_breaks(0) + _graph_and_breaks(1000)
    # gemm 20..30, norm+quant 31..40, then 80..105 (logits, side stream and
    # memset overlap) -> 10 + 9 + 25 us busy; host gaps never count.
    assert attribute_repetitions(trace, 2, device=0) == pytest.approx([44.0, 44.0])


def test_other_devices_and_activity_outside_repetitions_are_ignored():
    trace = _graph_and_breaks(0)
    trace.append(_gpu(99, 10.0, 5.0, device=1))
    trace += [_launch(500, 2000.0), _gpu(500, 2010.0, 50.0)]
    assert attribute_repetitions(trace, 1, device=0) == pytest.approx([44.0])


def test_kernels_without_launch_records_are_attributed_by_time():
    trace = _graph_and_breaks(0) + [_gpu(None, 200.0, 10.0, name="untracked")]
    assert attribute_repetitions(trace, 1, device=0) == pytest.approx([54.0])


def test_contradictions_and_foreign_work_fail():
    # Launched in repetition 0 but executed inside repetition 1's range.
    trace = _graph_and_breaks(0) + _graph_and_breaks(1000)
    trace.append(_gpu(1, 1200.0, 3.0))
    with pytest.raises(TimingAttributionError, match="launched in repetition 0"):
        attribute_repetitions(trace, 2, device=0)
    # Launched outside every repetition, executed inside one.
    trace = _graph_and_breaks(0) + [_launch(77, 900.0), _gpu(77, 300.0, 3.0)]
    with pytest.raises(TimingAttributionError, match="outside every repetition"):
        attribute_repetitions(trace, 1, device=0)
    # Missing or empty repetitions.
    with pytest.raises(TimingAttributionError, match="ranges"):
        attribute_repetitions(_graph_and_breaks(0), 2, device=0)
    with pytest.raises(TimingAttributionError, match="no attributed GPU kernel"):
        attribute_repetitions([_range(0, 0.0, 10.0)], 1, device=0)
    overlapping = _graph_and_breaks(0) + [_range(1, 100.0, 50.0)]
    with pytest.raises(TimingAttributionError, match="overlap"):
        attribute_repetitions(overlapping, 2, device=0)
