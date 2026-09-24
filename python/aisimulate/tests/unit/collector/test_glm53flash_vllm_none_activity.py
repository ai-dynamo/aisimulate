# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY trace ownership; no GPU execution or accepted latency fixtures."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from collector import glm53flash_vllm_none_activity as none

pytestmark = pytest.mark.unit


def trace_fixture():
    def scope(name, start, end):
        return {"name": name, "cat": "user_annotation", "ph": "X", "ts": start, "dur": end - start, "pid": 1, "tid": 2}

    events = [
        scope(none.NONE_EXECUTION_RANGE, 0, 100),
        scope(none.NONE_SETUP_RANGES["prepared_inputs_to_raw_model_entry"], 1, 10),
        scope(none.NONE_MODEL_RANGE, 12, 70),
        scope(none.OPERATION_RANGE_PREFIX + "embedding", 15, 25),
        scope(none.OPERATION_RANGE_PREFIX + "ffn_0", 30, 60),
        scope(none.OPERATION_RANGE_PREFIX + "allreduce_0", 40, 45),
        scope(none.NONE_SETUP_RANGES["raw_model_return_to_logits_entry"], 72, 80),
        scope(none.OPERATION_RANGE_PREFIX + "logits", 82, 95),
    ]
    for correlation, start in enumerate([3, 18, 32, 41, 50, 83], 1):
        memory = correlation == 1
        events.append(
            {
                **scope("cudaMemsetAsync" if memory else "cudaLaunchKernel", start, start + 1),
                "cat": "cuda_runtime",
                "args": {"correlation": correlation},
            }
        )
        events.append(
            {
                "cat": "gpu_memset" if memory else "kernel",
                "name": "TEST_ONLY_memset" if memory else f"TEST_ONLY_kernel_{correlation}",
                "ph": "X",
                "ts": start + 2,
                "dur": 1,
                "args": {
                    "correlation": correlation,
                    "stream": 7,
                    **({"bytes": 16} if memory else {"grid": [1, 1, 1], "block": [32, 1, 1], "shared memory": 0}),
                },
            }
        )
    events.append({**scope("cudaEventRecord", 75, 76), "cat": "cuda_runtime", "args": {"correlation": 7}})
    names = ["embedding", "ffn_0", "allreduce_0", "logits"]
    calls = [
        {
            "operation": name,
            "source": "TEST_ONLY.Native." + name,
            "completed": True,
            "included_sources": [],
            "excluded_collective_sources": ["TEST_ONLY.Native.allreduce_0"] if name == "ffn_0" else [],
            "parent_operation": "ffn_0" if name == "allreduce_0" else None,
        }
        for name in names
    ]
    return events, calls, names


def test_none_ownership_uses_actual_source_nested_calls_and_correlations():
    events, calls, names = trace_fixture()
    result = none.bind_none_execution(events, calls, names)
    assert result["graph"] is None
    assert result["native_calls"] == calls
    assert result["dispatch_interpolation_admitted"] is False
    assert result["formal_admission"] is False
    assert result["whole_forward_accuracy"] == "NOT_EVALUATED"
    assert {row["launch_correlation"]: row["operation"] for row in result["activities"]} == {
        1: "native_graph_setup",
        2: "embedding",
        3: "ffn_0",
        4: "allreduce_0",
        5: "ffn_0",
        6: "logits",
    }
    assert result["setup_activity_indices"]["raw_model_return_to_logits_entry"] == []
    assert result["timing_method"] == "separate_native_cuda_event_intervals_not_profiler_activity_time"
    # The same-named GPU annotations do not create additional CPU scopes.
    duplicate = dict(events[0], cat="gpu_user_annotation", pid=8, tid=9)
    assert none.bind_none_execution([*events, duplicate], calls, names) == result


@pytest.mark.parametrize(
    "defect",
    [
        "missing_operation",
        "duplicate_call",
        "missing_scope",
        "duplicate_scope",
        "foreign_scope",
        "wrong_category",
        "wrong_thread",
        "incomplete_call",
        "missing_parent",
        "wrong_source",
        "extra_exclusion",
        "overlap",
        "scope_order",
        "graph_launch",
        "graph_activity",
        "missing_activity",
        "orphan_activity",
        "unknown_api",
        "duplicate_correlation",
        "crossing_call",
        "boundary_call",
        "unowned_launch",
        "missing_gpu_work",
    ],
)
def test_none_trace_rejects_incomplete_or_ambiguous_native_evidence(defect):
    events, calls, names = trace_fixture()
    apis = [row for row in events if row["cat"] == "cuda_runtime"]
    gpu = [row for row in events if row["cat"] in ("kernel", "gpu_memset")]
    if defect == "missing_operation":
        calls.pop()
    elif defect == "duplicate_call":
        calls[-1] = copy.deepcopy(calls[0])
    elif defect == "missing_scope":
        events.pop(3)
    elif defect == "duplicate_scope":
        events.append(copy.deepcopy(events[3]))
    elif defect == "foreign_scope":
        events.append(dict(events[3], name=none.OPERATION_RANGE_PREFIX + "unknown"))
    elif defect == "wrong_category":
        events[3]["cat"] = "gpu_user_annotation"
    elif defect == "wrong_thread":
        events[3]["tid"] = 99
    elif defect == "incomplete_call":
        calls[1]["completed"] = False
    elif defect == "missing_parent":
        calls[2]["parent_operation"] = None
    elif defect == "wrong_source":
        calls[2]["source"] = "TEST_ONLY.other_source"
    elif defect == "extra_exclusion":
        calls[1]["excluded_collective_sources"].append("TEST_ONLY.other_source")
    elif defect == "overlap":
        events[3].update(ts=20, dur=20)
    elif defect == "scope_order":
        events[6].update(ts=65, dur=15)
    elif defect == "graph_launch":
        apis[0]["name"] = "cudaGraphLaunch"
    elif defect == "graph_activity":
        gpu[0]["args"]["graph id"] = 4
    elif defect == "missing_activity":
        events.remove(gpu[0])
    elif defect == "orphan_activity":
        gpu[0]["args"]["correlation"] = 999
    elif defect == "unknown_api":
        apis[-1]["name"] = "cudaUnknownDeviceWork"
    elif defect == "duplicate_correlation":
        apis[-1]["args"]["correlation"] = 1
    elif defect == "crossing_call":
        apis[3].update(ts=39, dur=3)
    elif defect == "boundary_call":
        apis[3].update(ts=40, dur=0)
    elif defect == "unowned_launch":
        apis[0].update(ts=11, dur=0.5)
    elif defect == "missing_gpu_work":
        events = [row for row in events if row not in gpu and row not in apis[:-1]]
    with pytest.raises(ValueError):
        none.bind_none_execution(events, calls, names)


def test_unknown_model_control_and_empty_operation_preserve_completed_source_proof():
    events, calls, names = trace_fixture()
    # A source-called unit may have no device-work API. Its measured event
    # duration must come from the independent event row, not a zero from here.
    events = [row for row in events if row.get("args", {}).get("correlation") != 2]
    result = none.bind_none_execution(events, calls, names)
    assert result["operation_activity_indices"]["embedding"] == []
    assert result["native_calls"][0]["completed"] is True
    calls[0]["completed"] = False
    with pytest.raises(ValueError, match="completed source"):
        none.bind_none_execution(events, calls, names)


def execution_fixture(tmp_path):
    from collector.glm53flash_observer import NativeOperationObserver

    events, calls, names = trace_fixture()
    lifecycle = []

    class Scope:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            lifecycle.append(("enter", self.name))

        def __exit__(self, *args):
            lifecycle.append(("exit", self.name))

    class Profiler:
        def start(self):
            lifecycle.append(("profiler", "start"))

        def stop(self):
            lifecycle.append(("profiler", "stop"))

        def export_chrome_trace(self, path):
            lifecycle.append(("profiler", "export"))
            Path(path).write_text(json.dumps({"traceEvents": events}))

        def events(self):
            return []

    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_current_stream_capturing=lambda: False),
        profiler=SimpleNamespace(
            profile=lambda **kwargs: Profiler(),
            record_function=Scope,
            ProfilerActivity=SimpleNamespace(CPU="TEST_ONLY_CPU", CUDA="TEST_ONLY_CUDA"),
        ),
    )
    observer = NativeOperationObserver(
        {"phases": {"context": [{"name": name} for name in names]}},
        {"backend": "vllm"},
        0,
        torch_module=torch,
    )
    helper = none.NativeNoneExecution(observer, tmp_path)
    return helper, observer, lifecycle, calls


def execution_record(sample):
    return {
        "run_id": "TEST_ONLY_RUN",
        "request_set": "TEST_ONLY_REQUESTS",
        "tp_rank": 0,
        "invocation": 100 + sample,
        "forward_id": f"TEST_ONLY_FORWARD_{100 + sample}",
        "phase": "context",
        "benchmark_id": 1,
        "repetition": sample,
        "sampling_role": "warmup" if sample < 5 else "measurement",
        "dataset_role": "calibration",
        "corpus_sha256": "a" * 64,
        "serving_none_model_sha256": "b" * 64,
        "request_ids": ["TEST_ONLY_REQUEST"],
    }


def test_none_only_profiles_fifth_warmup_and_stops_before_sampling(tmp_path):
    from collector.glm53flash_observer import NativeWorkload

    helper, observer, lifecycle, calls = execution_fixture(tmp_path)
    for sample in range(15):
        record = execution_record(sample)
        observer.begin(
            NativeWorkload("context", 1, 128, 0, "full_prefill", ("TEST_ONLY_REQUEST",), (), sample, 100 + sample)
        )
        helper.begin(record)
        observer.events = [
            {
                "name": row["operation"],
                "source": row["source"],
                "collectives": [(None, None, source) for source in row["excluded_collective_sources"]],
                "parent_operation": row["parent_operation"],
            }
            for row in calls
        ]
        helper.boundary("model")
        helper.boundary("before_logits")
        helper.boundary("logits")
        helper.end_logits()
        assert observer.profiler is None
        lifecycle.append(("native", "sampling"))
        helper.complete(record)
        assert record["profiled"] is (sample == 4)
        assert ("native_none_profile" in record) is (sample == 4)
        observer.events.clear()
        observer.workload = None
    assert lifecycle.count(("profiler", "start")) == lifecycle.count(("profiler", "stop")) == 1
    assert lifecycle.index(("profiler", "stop")) < lifecycle.index(("profiler", "export"))
    saved = list(tmp_path.glob("none-profile-*.json"))
    assert len(saved) == 1 and saved[0].name == "none-profile-rank-0-forward-104.json"
    trace = json.loads(saved[0].read_text())
    assert trace["aisim_native_forward"]["invocation"] == 104
    assert trace["aisim_native_none"]["native_calls"] == calls
    assert trace["aisim_native_none"]["failed"] is False


@pytest.mark.parametrize(
    "defect", ["profiled_measurement", "missing_warmup_profile", "scope_order", "incomplete", "abort"]
)
def test_none_lifecycle_rejects_or_preserves_failed_original_trace(tmp_path, defect):
    from collector.glm53flash_observer import NativeWorkload

    helper, observer, lifecycle, _ = execution_fixture(tmp_path)
    sample = 5 if defect == "profiled_measurement" else 4
    record = execution_record(sample)
    observer.begin(
        NativeWorkload("context", 1, 128, 0, "full_prefill", ("TEST_ONLY_REQUEST",), (), sample, 100 + sample)
    )
    if defect == "profiled_measurement":
        observer.profiler = object()
    elif defect == "missing_warmup_profile":
        observer.profiler = None
    if defect in ("profiled_measurement", "missing_warmup_profile"):
        with pytest.raises(RuntimeError, match="fifth excluded warmup"):
            helper.begin(record)
        return
    helper.begin(record)
    if defect == "scope_order":
        with pytest.raises(RuntimeError, match="boundary"):
            helper.boundary("logits")
    elif defect == "incomplete":
        with pytest.raises(RuntimeError, match="completed observation"):
            helper.complete(record)
    else:
        helper.abort(RuntimeError("TEST_ONLY_NATIVE_FAILURE"))
        assert helper.active is None and observer.profiler is None
        trace = json.loads(next(tmp_path.glob("failed-none-profile-*.json")).read_text())
        assert trace["aisim_native_none"]["failed"] is True
        assert trace["aisim_native_none"]["native_calls"] == []
        assert "native_none_profile" not in record
        assert lifecycle[-2:] == [("profiler", "stop"), ("profiler", "export")]
