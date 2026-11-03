# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from audit_forward import project_forwards


def test_forward_gpu_span_follows_correlation_beyond_cpu_scope_and_keeps_stream_gaps():
    name = "step[VERIFY bs=1 g=1 g_sq=4 g_sk=8192 g_sqsk=32768]"
    events = [
        {"ph": "X", "cat": "user_annotation", "name": name, "pid": 9, "tid": 10,
         "ts": 100, "dur": 10, "args": {"External id": 1}},
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaGraphLaunch", "pid": 9, "tid": 10,
         "ts": 105, "dur": 2, "args": {"correlation": 3}},
        {"ph": "X", "cat": "kernel", "name": "kernel_a", "ts": 200, "dur": 20,
         "args": {"correlation": 3, "device": 0, "stream": 7}},
        {"ph": "X", "cat": "kernel", "name": "kernel_b", "ts": 230, "dur": 10,
         "args": {"correlation": 3, "device": 0, "stream": 8}},
        {"ph": "X", "cat": "gpu_user_annotation", "name": name, "pid": 0, "tid": 7,
         "ts": 200, "dur": 20, "args": {"External id": 1}},
        {"ph": "X", "cat": "gpu_user_annotation", "name": name, "pid": 0, "tid": 8,
         "ts": 230, "dur": 10, "args": {"External id": 1}},
    ]
    result = project_forwards({"traceEvents": events}, 0)
    row = result["forwards"][0]
    assert row["qualified"]
    assert row["cpu_span_ms"] == 0.01 and row["gpu_span_ms"] == 0.04
    assert row["streams"] == [7, 8] and row["shape"]["g_sqsk"] == 32768
    assert row["correlated_kernel_rows"] == 2
    events[-2]["ts"] = 201
    bad = project_forwards({"traceEvents": events}, 0)["forwards"][0]
    assert not bad["qualified"] and "does not enclose" in bad["reasons"][0]


def test_forward_missing_gpu_evidence_is_unavailable_and_wrong_rank_is_rejected():
    name = "step[DRAFT bs=1]"
    document = {"traceEvents": [
        {"ph": "X", "cat": "user_annotation", "name": name, "pid": 1, "tid": 2,
         "ts": 1, "dur": 10, "args": {"External id": 9}},
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaGraphLaunch", "pid": 1, "tid": 2,
         "ts": 2, "dur": 1, "args": {"correlation": 5}},
        {"ph": "X", "cat": "kernel", "name": "kernel", "ts": 5, "dur": 2,
         "args": {"correlation": 5, "device": 1}},
    ]}
    result = project_forwards(document, 0)
    assert result["stage_counts"] == {"DRAFT": 1} and result["qualified_forward_count"] == 0
    assert "gpu_span_ms" not in result["forwards"][0]
    assert len(result["forwards"][0]["reasons"]) == 2


def test_parent_native_draft_phase_includes_child_forward_gpu_annotations():
    child = "step[DRAFT bs=1]"
    events = [
        {"ph": "X", "cat": "user_annotation", "name": "draft_extend", "pid": 1, "tid": 2,
         "ts": 0, "dur": 100, "args": {"External id": 1}},
        {"ph": "X", "cat": "user_annotation", "name": child, "pid": 1, "tid": 2,
         "ts": 10, "dur": 20, "args": {"External id": 2}},
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaGraphLaunch", "pid": 1, "tid": 2,
         "ts": 11, "dur": 1, "args": {"correlation": 3}},
        {"ph": "X", "cat": "kernel", "name": "draft_kernel", "ts": 200, "dur": 20,
         "args": {"correlation": 3, "device": 0}},
        {"ph": "X", "cat": "gpu_user_annotation", "name": child, "pid": 0, "tid": 7,
         "ts": 200, "dur": 20, "args": {"External id": 2}},
    ]
    result = project_forwards({"traceEvents": events}, 0)
    assert result["stage_counts"] == {"draft_extend": 1, "DRAFT": 1}
    assert result["qualified_forward_count"] == 2
    assert [row["gpu_span_ms"] for row in result["forwards"]] == [0.02, 0.02]
