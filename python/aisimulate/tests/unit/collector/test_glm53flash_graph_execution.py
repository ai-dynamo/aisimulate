# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source-bound execution-scope regressions; CPU doubles are never GPU evidence."""

import copy
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest
from collector import glm53flash_sglang_graph_ops as sglang_graph
from collector.glm53flash_graph_nodes import EXECUTION_RANGE, bind_execution_activity, bind_replay_kernels

pytestmark = pytest.mark.unit


def fixture_events():
    events = [
        {"cat": "user_annotation", "ph": "X", "name": EXECUTION_RANGE, "ts": 0, "dur": 20, "pid": 1, "tid": 2},
        {
            "cat": "cuda_runtime",
            "name": "cudaMemsetAsync",
            "ts": 1,
            "dur": 1,
            "pid": 1,
            "tid": 2,
            "args": {"correlation": 8},
        },
        {
            "cat": "cuda_runtime",
            "name": "cudaGraphLaunch",
            "ts": 4,
            "dur": 1,
            "pid": 1,
            "tid": 2,
            "args": {"correlation": 9},
        },
        {
            "cat": "gpu_memset",
            "name": "metadata memset",
            "ts": 2,
            "dur": 4,
            "args": {"bytes": 1024, "stream": 1, "correlation": 8, "graph id": 0, "graph node id": 0},
        },
        {
            "cat": "kernel",
            "name": "native_model",
            "ts": 5,
            "dur": 3,
            "args": {
                "stream": 1,
                "grid": [1, 1, 1],
                "block": [32, 1, 1],
                "shared memory": 0,
                "correlation": 9,
                "graph id": 4,
                "graph node id": 11,
            },
        },
    ]
    registry = {"graph_id": 4, "nodes": [{"node_id": 11, "node_type": 0, "name": "attention_0"}]}
    return bind_replay_kernels(registry, events, correlation=9), events


def test_native_preparation_is_measured_as_setup_without_allocating_cpu_gaps():
    binding, events = fixture_events()
    result = bind_execution_activity(binding, events)
    assert result["graph"] is binding
    assert result["outside_graph_setup"][0]["native_launch"] == "cudaMemsetAsync"
    assert result["outside_graph_setup"][0]["launch_correlation"] == 8
    units = {row["operation"]: row for row in result["operation_activity_unions"]}
    assert units["native_graph_setup"]["active_union_us"] == 4
    assert units["attention_0"]["active_union_us"] == 3
    assert result["activity_union_us"] == 6
    assert result["approximate_additive_operation_union_us"] == 7
    assert result["formal_admission"] is False
    assert result["execution_range"]["dur"] == 20  # Remaining time is not fitted into either unit.


@pytest.mark.parametrize(
    "defect", ["thread", "outside", "missing_range", "extra_graph", "unknown_graph", "bytes", "nan"]
)
def test_native_preparation_cannot_borrow_unscoped_or_unregistered_activity(defect):
    binding, events = fixture_events()
    events = copy.deepcopy(events)
    if defect == "thread":
        events[1]["tid"] = 3
    elif defect == "outside":
        events[1]["ts"] = 21
    elif defect == "missing_range":
        events.pop(0)
    elif defect == "extra_graph":
        events[1]["name"] = "cudaGraphLaunch"
    elif defect == "unknown_graph":
        events[3]["args"]["graph id"] = 5
    elif defect == "bytes":
        events[3]["args"]["bytes"] = True
    elif defect == "nan":
        events[3]["dur"] = float("nan")
    with pytest.raises(ValueError):
        bind_execution_activity(binding, events)


def test_holdout_gpu_events_enclose_native_preparation_and_model_but_not_sampling(monkeypatch, tmp_path):
    timeline = []
    stream = object()

    class Event:
        def __init__(self, enable_timing):
            assert enable_timing

        def record(self, actual):
            assert actual is stream
            self.position = len(timeline)
            timeline.append(self)

    @dataclass(frozen=True)
    class Shape:
        size: int

    class FullCudaGraphBackend:
        def __init__(self, runner):
            self._cuda_graph_runner = runner
            self._reuse_output_buffer = False
            self._graphs = {}

        def capture_one(self, shape, forward):
            self._graphs[shape] = object()
            forward()

        def replay(self, shape, batch):
            timeline.append("model including logits")
            return "native logits"

    class DecodeCudaGraphRunner:
        def __init__(self):
            self.enable_torch_compile = False
            self.model_runner = SimpleNamespace(ps=SimpleNamespace(tp_rank=0))
            self._metadata_glue = None
            self.backend = FullCudaGraphBackend(self)

        def execute(self, batch):
            timeline.append("native metadata and copies")
            return self.backend.replay(Shape(4), batch)

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            Event=Event,
            current_stream=lambda: stream,
            is_current_stream_capturing=lambda: False,
        )
    )
    modules = {
        "torch": fake_torch,
        "sglang": SimpleNamespace(__file__=str(tmp_path / "__init__.py")),
        "sglang.srt.model_executor.runner.decode_cuda_graph_runner": {"DecodeCudaGraphRunner": DecodeCudaGraphRunner},
        "sglang.srt.model_executor.runner_backend.full_cuda_graph_backend": {
            "FullCudaGraphBackend": FullCudaGraphBackend
        },
    }
    for name, value in modules.items():
        if isinstance(value, dict):
            module = ModuleType(name)
            module.__dict__.update(value)
            value = module
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(sglang_graph, "SOURCE_PINS", {})

    # The production function also writes the source SHA into the receipt.
    class Pins(dict):
        def __getitem__(self, key):
            return "TEST_ONLY"

    monkeypatch.setattr(sglang_graph, "SOURCE_PINS", Pins())
    from collector import glm53flash_graph_policy

    def test_only_snapshot(runner, output):
        assert output == tmp_path
        timeline.append("policy snapshot")
        return {"file": "TEST_ONLY", "sha256": "a" * 64}

    monkeypatch.setattr(glm53flash_graph_policy, "persist_snapshot", test_only_snapshot)
    sglang_graph.install(None, {"TEST_ONLY": True}, tmp_path, holdout=True)
    runner = DecodeCudaGraphRunner()
    runner.backend.capture_one(Shape(4), lambda: None)
    assert runner.execute(object()) == "native logits"  # Native warmup: no events.
    timeline.clear()
    record = {"stage": "measure", "phase": "generation", "query_lengths": [1, 1, 1], "batch_size": 3, "invocation": 1}
    runner.model_runner._aisim_glm53_graph_forward = record
    assert runner.execute(object()) == "native logits"
    timeline.append("sampling and token readback")
    pending = runner.model_runner._aisim_glm53_graph_pending[1]
    assert timeline == [
        "policy snapshot",
        pending["start"],
        "native metadata and copies",
        "model including logits",
        pending["end"],
        "sampling and token readback",
    ]
    assert pending["native_shape_key"] == {"size": 4}
    assert pending["ops_instrumented"] is False and pending["profiled"] is False
    # An enabled secondary metadata graph remains explicit unsupported scope.
    record["invocation"] = 2
    runner._metadata_glue = SimpleNamespace(disabled=False)
    with pytest.raises(RuntimeError, match="separate capture-node"):
        runner.execute(object())


@pytest.mark.parametrize("name", ["cudaMemsetAsync", "cudaMemcpyAsync", "cudaLaunchKernel", "cuLaunchKernelEx"])
def test_device_work_call_cannot_disappear_from_gpu_activity(name):
    binding, events = fixture_events()
    events = [row for row in events if row.get("cat") != "gpu_memset"]
    for row in events:
        if row.get("cat") == "cuda_runtime" and row.get("name") == "cudaMemsetAsync":
            row["name"] = name
    with pytest.raises(ValueError, match="device-work call lacks"):
        bind_execution_activity(binding, events)


def test_unknown_cuda_dispatch_is_not_a_zero_cost_setup():
    binding, events = fixture_events()
    events.append(
        {
            "cat": "cuda_runtime",
            "name": "cudaFutureUnknownDispatch",
            "ts": 7,
            "dur": 1,
            "pid": 1,
            "tid": 2,
            "args": {"correlation": 17},
        }
    )
    with pytest.raises(ValueError, match="unknown native CUDA dispatch"):
        bind_execution_activity(binding, events)
