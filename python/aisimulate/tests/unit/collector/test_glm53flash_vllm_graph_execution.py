# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY trace fixtures; neither native GPU observations nor qualification."""

import copy
import hashlib
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from collector.glm53flash_graph_nodes import (
    VLLM_EXECUTION_RANGE,
    VLLM_LOGITS_RANGE,
    bind_replay_kernels,
    bind_vllm_execution_activity,
)

pytestmark = pytest.mark.unit


def trace():
    def call(name, correlation, ts):
        return {
            "cat": "cuda_runtime",
            "name": name,
            "ts": ts,
            "dur": 1,
            "pid": 1,
            "tid": 2,
            "args": {"correlation": correlation},
        }

    def kernel(name, correlation, ts, duration, graph=0, node=0):
        return {
            "cat": "kernel",
            "name": name,
            "ts": ts,
            "dur": duration,
            "args": {
                "correlation": correlation,
                "stream": 1,
                "graph id": graph,
                "graph node id": node,
                "grid": [1, 1, 1],
                "block": [32, 1, 1],
                "shared memory": 0,
            },
        }

    events = [
        {"ph": "X", "name": VLLM_EXECUTION_RANGE, "ts": 0, "dur": 50, "pid": 1, "tid": 2},
        {"ph": "X", "name": VLLM_LOGITS_RANGE, "ts": 20, "dur": 10, "pid": 1, "tid": 2},
        call("cudaMemsetAsync", 1, 2),
        call("cudaGraphLaunch", 2, 8),
        call("cudaLaunchKernel", 3, 22),
        {
            "cat": "gpu_memset",
            "name": "metadata",
            "ts": 3,
            "dur": 5,
            "args": {"bytes": 128, "stream": 1, "correlation": 1},
        },
        kernel("model", 2, 10, 14, 7, 70),
        # Deliberate GPU interval overlap: CPU ownership must not follow time/order.
        kernel("logits", 3, 21, 6),
    ]
    registry = {"graph_id": 7, "nodes": [{"node_id": 70, "node_type": 0, "name": "attention_0"}]}
    return bind_replay_kernels(registry, events, correlation=2), events


def test_external_logits_and_setup_are_distinct_source_owned_measured_units():
    binding, events = trace()
    original = copy.deepcopy(events)
    result = bind_vllm_execution_activity(binding, events)
    units = {row["operation"]: row["active_union_us"] for row in result["operation_activity_unions"]}
    assert units == {"attention_0": 14, "native_graph_setup": 5, "logits": 6}
    assert result["approximate_additive_operation_union_us"] == 25
    assert result["activity_union_us"] == 22  # Cross-unit overlap is preserved.
    assert result["execution_range"]["dur"] == 50  # No CPU-gap/residual allocation.
    assert result["outside_graph_setup"][0]["activity"] == "gpu_memset"
    assert result["outside_graph_operations"][0]["launch_correlation"] == 3
    assert events == original


@pytest.mark.parametrize("boundary", [20, 30])
def test_zero_duration_launch_at_logits_boundary_cannot_be_reclassified_as_setup(boundary):
    binding, events = trace()
    events[4].update(ts=boundary, dur=0)
    with pytest.raises(ValueError, match="ambiguous ownership"):
        bind_vllm_execution_activity(binding, events)


def test_zero_duration_launch_inside_logits_retains_source_ownership():
    binding, events = trace()
    events[4]["dur"] = 0
    result = bind_vllm_execution_activity(binding, events)
    assert result["outside_graph_operations"][0]["operation"] == "logits"
    assert result["approximate_additive_operation_union_us"] == 25


@pytest.mark.parametrize(
    "defect",
    [
        "missing_logits_range",
        "duplicate_logits_range",
        "wrong_thread",
        "outside_outer",
        "graph_in_logits",
        "straddling",
        "missing_logits_activity",
        "missing_setup_activity",
        "logits_double_owned",
        "unknown_graph",
    ],
)
def test_incomplete_or_ambiguous_external_operation_evidence_is_rejected(defect):
    binding, events = trace()
    if defect == "missing_logits_range":
        events.pop(1)
    elif defect == "duplicate_logits_range":
        events.append(dict(events[1]))
    elif defect == "wrong_thread":
        events[1]["tid"] = 4
    elif defect == "outside_outer":
        events[1]["ts"] = 51
    elif defect == "graph_in_logits":
        events[3]["ts"] = 21
    elif defect == "straddling":
        events[4]["ts"] = 19.5
    elif defect == "missing_logits_activity":
        events.pop()
    elif defect == "missing_setup_activity":
        events.pop(5)
    elif defect == "logits_double_owned":
        binding["activities"][0]["operation"] = "logits"
    elif defect == "unknown_graph":
        events[-1]["args"]["graph id"] = 99
    with pytest.raises(ValueError):
        bind_vllm_execution_activity(binding, events)


@pytest.mark.parametrize("runtime_mode", ["FULL", "PIECEWISE"])
def test_actual_execution_observer_requires_later_completed_native_samples(monkeypatch, tmp_path, runtime_mode):
    from collector import glm53flash_vllm_graph_ops as graph

    timeline = []
    _, events = trace()

    class Range:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            timeline.append(("enter", self.name))

        def __exit__(self, *args):
            timeline.append(("exit", self.name))

    class Profiler:
        def start(self):
            timeline.append("profile_start")

        def stop(self):
            timeline.append("profile_stop")

        def export_chrome_trace(self, path):
            Path(path).write_text(json.dumps({"traceEvents": events}))

    class LogitsProcessor:
        use_all_gather, logits_as_input = True, False
        head_dtype = soft_cap = None
        scale, org_vocab_size = 1.0, 154880

        def forward(self, *args, **kwargs):
            timeline.append("native_logits")
            return "original_result"

    torch = SimpleNamespace(
        bfloat16="TEST_ONLY_bf16",
        profiler=SimpleNamespace(
            profile=lambda **kwargs: Profiler(),
            record_function=Range,
            ProfilerActivity=SimpleNamespace(CPU="CPU", CUDA="CUDA"),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.layers.logits_processor", SimpleNamespace(LogitsProcessor=LogitsProcessor)
    )
    monkeypatch.setattr(
        graph, "LOGITS_SOURCE_PIN", hashlib.sha256(Path(inspect.getfile(LogitsProcessor)).read_bytes()).hexdigest()
    )
    logits = LogitsProcessor()
    model = SimpleNamespace(modules=lambda: [SimpleNamespace(lm_head=object(), logits_processor=logits)])
    native_graph, descriptor = object(), object()
    registry = {
        "graph_id": 7,
        "nodes": [{"node_id": 70, "node_type": 0, "name": "attention_0"}],
        "capture_scope": "vllm_hidden_states",
        "uncaptured_operations": ["logits"],
    }
    live_piecewise = None
    if runtime_mode == "PIECEWISE":
        from collector import glm53flash_vllm_piecewise as pw

        from .test_glm53flash_vllm_piecewise_activity import fixture

        registry, events = fixture()
        native_capture = object()

        def validate_replay(actual):
            assert actual is native_capture

        live_piecewise = SimpleNamespace(
            profiling=False, bound_capture=registry, capture=native_capture, validate_replay=validate_replay
        )
        monkeypatch.setattr(pw, "piecewise_capture_for_descriptor", lambda *args: live_piecewise)
    path = tmp_path / "TEST_ONLY-capture.json"
    path.write_text(json.dumps(registry))
    if live_piecewise is not None:
        live_piecewise.capture_artifact = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manager = SimpleNamespace(
        graphs={descriptor: native_graph},
        _aisim_glm53_node_captures={
            descriptor: {
                "graph": native_graph,
                "registry": registry,
                "artifact": {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
            }
        },
    )
    observer = graph.NativeVllmGraphExecution(
        SimpleNamespace(model=model, cudagraph_manager=manager),
        tmp_path,
        0,
        include_piecewise=runtime_mode == "PIECEWISE",
    )
    record = {
        "stage": "measure",
        "runtime_mode": runtime_mode,
        "run_id": "TEST_ONLY_run",
        "request_set": "TEST_ONLY_set",
        "tp_rank": 0,
        "invocation": 1,
        "forward_id": "rank-0/forward-1",
        "phase": "generation",
        "benchmark_id": 1,
        "repetition": 5,
        "sampling_role": "measurement",
        "dataset_role": "calibration",
        "corpus_sha256": "a" * 64,
        "request_ids": ["r"],
        "requests": [{}],
    }
    observer.begin(record, descriptor)
    if live_piecewise is not None:
        assert live_piecewise.profiling is True
    observer.start_range()
    assert logits.forward(object(), object()) == "original_result"
    observer.end_logits()
    if live_piecewise is not None:
        assert live_piecewise.profiling is False
    assert timeline == [
        "profile_start",
        ("enter", VLLM_EXECUTION_RANGE),
        ("enter", VLLM_LOGITS_RANGE),
        "native_logits",
        ("exit", VLLM_LOGITS_RANGE),
        ("exit", VLLM_EXECUTION_RANGE),
        "profile_stop",
    ]
    with pytest.raises(RuntimeError, match="completed same-request"):
        observer.finish(record)
    record.update(gpu_completed=True, native_graph_replay_completed=True)
    record["requests"][0]["sampled_token_id"] = 9
    if live_piecewise is not None:
        with pytest.raises(RuntimeError, match="completed native entry replay"):
            observer.finish(record)
        record["native_piecewise_replay_completed"] = True
    result = observer.finish(record)
    assert result["capture_registry_file"] == path.name and "capture_registry" not in result
    assert result["measurement_method"] == (
        "native_cupti_piecewise_graphs_eager_and_external_logits"
        if live_piecewise is not None
        else "native_cupti_graph_nodes_and_external_logits"
    )
    assert any(row["operation"] == "logits" for row in result["replay_nodes"]["outside_graph_operations"])
    assert observer.active is None
    original = json.loads((tmp_path / result["replay_nodes"]["trace_file"]).read_text())
    assert original["aisim_native_forward"]["invocation"] == 1
    assert original["aisim_native_execution"]["failed"] is False
    # Reusing one rank/forward file cannot supply a second measurement.
    observer.begin(record, descriptor)
    observer.start_range()
    logits.forward(object(), object())
    with pytest.raises(RuntimeError, match="overwrite"):
        observer.end_logits()
    observer.abort(RuntimeError("TEST_ONLY failed duplicate forward"))
    failed_record = dict(record, invocation=2, forward_id="rank-0/forward-2", gpu_completed=False)
    observer.begin(failed_record, descriptor)
    observer.start_range()
    observer.abort(RuntimeError("TEST_ONLY native execution failed before logits"))
    assert observer.active is None
    if live_piecewise is not None:
        assert live_piecewise.profiling is False
    failed_trace = json.loads((tmp_path / "failed-graph-profile-rank-0-forward-2.json").read_text())
    assert failed_trace["aisim_native_forward"]["invocation"] == 2
    assert failed_trace["aisim_native_execution"]["failed"] is True
    assert not (tmp_path / "graph-profile-rank-0-forward-2.json").exists()
    assert json.loads((tmp_path / result["replay_nodes"]["trace_file"]).read_text()) == original


def test_default_full_execution_observer_rejects_piecewise_before_capture_or_profiling():
    from collector.glm53flash_vllm_graph_ops import NativeVllmGraphExecution

    observer = NativeVllmGraphExecution.__new__(NativeVllmGraphExecution)
    observer.active, observer.include_piecewise = None, False
    with pytest.raises(RuntimeError, match="fresh target"):
        observer.begin({"stage": "measure", "runtime_mode": "PIECEWISE"}, object())
