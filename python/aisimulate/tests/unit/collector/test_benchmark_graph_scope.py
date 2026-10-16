# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check native-call ordering for the actual helper, without importing CUDA."""

import ast
import logging
import sys
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
COLLECTOR = Path(__file__).resolve().parents[3] / "collector"
SOURCE = COLLECTOR / "helper.py"


def trace_benchmark(monkeypatch, *, source=SOURCE, failure=None, explicit=None, eager=False, fallback=False):
    """Execute a helper source snapshot; retain graph references like a traceback."""
    events, graphs = [], []
    state = {"stream": "caller", "replays": 0, "fault_fired": False}
    error = RuntimeError(f"{failure} failed")

    def fault(stage):
        if failure == stage and not state["fault_fired"]:
            state["fault_fired"] = True
            events.append(f"raise:{stage}")
            raise error

    class Graph:
        def __init__(self):
            events.append("graph:create")
            fault("graph_create")
            self.resets = 0
            graphs.append(self)

        def replay(self):
            state["replays"] += 1
            events.append("graph:replay")
            fault("replay")
            if state["replays"] == 3:
                fault("timed_replay")

        def reset(self):
            assert state["stream"] == "caller"
            events.append("graph:reset")
            fault("reset")
            self.resets += 1

    @contextmanager
    def capture(graph):
        events.append("capture:begin")
        state["stream"] = "capture"
        fault("capture_begin")
        yield
        events.append("capture:end")
        fault("capture_end")
        state["stream"] = "caller"

    def current_stream():
        events.append("stream:get")
        fault("current_stream")
        return state["stream"]

    def set_stream(stream):
        events.append(f"stream:set:{stream}")
        state["stream"] = stream

    def kernel():
        events.append("kernel")
        fault("initial_warmup")
        if state["stream"] == "capture":
            fault("capture_body")

    class Event:
        def __init__(self, **kwargs):
            events.append("event:create")

        def record(self):
            events.append("event:record")

        def elapsed_time(self, end):
            events.append("event:elapsed")
            return 12.0

    cuda = SimpleNamespace(
        is_available=lambda: True,
        CUDAGraph=Graph,
        graph=capture,
        current_stream=current_stream,
        set_stream=set_stream,
        synchronize=lambda: events.append("synchronize"),
        empty_cache=lambda: events.append("empty_cache"),
        Event=Event,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "benchmark_with_power")
    namespace = {"contextmanager": contextmanager, "get_device_module": lambda: cuda, "logging": logging}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    kwargs = {
        "device": SimpleNamespace(index=0),
        "kernel_func": kernel,
        "num_warmups": 2,
        "num_runs": 3,
        "measure_power": False,
        "power_min_duration": 1.0,
        "use_cuda_graph": not eager,
        "allow_graph_fail": fallback,
    }
    if explicit is not None:
        kwargs["explicit_graph_cleanup"] = explicit
    result, caught = None, None
    try:
        with namespace["benchmark_with_power"](**kwargs) as result:
            events.append("caller:body")
            fault("caller")
    except RuntimeError as exc:
        caught = {"type": type(exc).__name__, "message": str(exc), "original_error": exc is error}
    return {
        "events": events,
        "result": result,
        "error": caught,
        "stream": state["stream"],
        "resets": [graph.resets for graph in graphs],
    }


@pytest.mark.parametrize("explicit", [False, True])
def test_success_preserves_timing_body_and_scopes_cleanup(monkeypatch, explicit):
    observed = trace_benchmark(monkeypatch, explicit=explicit)
    assert observed["error"] is None
    assert observed["result"]["latency_ms"] == 4.0
    assert observed["result"]["used_cuda_graph"] is True
    assert observed["events"].count("kernel") == 3  # two eager warmups + capture
    assert observed["events"].count("graph:replay") == 5  # two warmups + three runs
    assert observed["resets"] == [int(explicit)]
    assert observed["events"][-3:] == (
        ["caller:body", "graph:reset", "empty_cache"] if explicit else ["caller:body", "synchronize", "empty_cache"]
    )
    assert ("stream:set:caller" in observed["events"]) is explicit


@pytest.mark.parametrize("failure", ["capture_begin", "capture_body", "capture_end"])
@pytest.mark.parametrize("explicit", [False, True])
def test_capture_failure_lifetime_is_opt_in(monkeypatch, failure, explicit):
    observed = trace_benchmark(monkeypatch, failure=failure, explicit=explicit)
    assert observed["error"]["original_error"] is True
    assert observed["resets"] == [int(explicit)]
    assert ("empty_cache" in observed["events"]) is explicit
    assert observed["stream"] == ("caller" if explicit else "capture")
    assert "graph:replay" not in observed["events"]


@pytest.mark.parametrize("failure", ["capture_begin", "capture_body", "capture_end"])
@pytest.mark.parametrize("explicit", [False, True])
def test_existing_opted_fallback_keeps_its_measurement_policy(monkeypatch, failure, explicit):
    observed = trace_benchmark(monkeypatch, failure=failure, explicit=explicit, fallback=True)
    assert observed["error"] is None
    assert observed["result"]["used_cuda_graph"] is False
    assert observed["resets"] == [int(explicit)]
    assert "graph:replay" not in observed["events"]
    assert observed["events"].index("empty_cache") < observed["events"].index("event:record")


@pytest.mark.parametrize("failure", ["caller", "replay", "timed_replay"])
@pytest.mark.parametrize("explicit", [False, True])
def test_cleanup_after_capture_keeps_caller_and_replay_errors(monkeypatch, failure, explicit):
    observed = trace_benchmark(monkeypatch, failure=failure, explicit=explicit)
    assert observed["error"]["original_error"] is True
    assert observed["resets"] == [int(explicit)]
    assert observed["events"][-2:] == (["graph:reset", "empty_cache"] if explicit else ["synchronize", "empty_cache"])


def test_default_is_identical_to_explicit_false(monkeypatch):
    assert trace_benchmark(monkeypatch) == trace_benchmark(monkeypatch, explicit=False)


def test_eager_does_not_touch_graph_lifetime_in_either_mode(monkeypatch):
    default = trace_benchmark(monkeypatch, eager=True)
    opted = trace_benchmark(monkeypatch, eager=True, explicit=True)
    assert default == opted
    assert default["resets"] == []
    assert default["events"].count("kernel") == 7


def test_only_vllm_mla_module_binds_explicit_graph_cleanup():
    path = COLLECTOR / "vllm/collect_mla_module.py"
    tree = ast.parse(path.read_text())
    binding = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "benchmark_with_power" for target in node.targets)
    )
    calls = []

    def shared_benchmark(**kwargs):
        calls.append(kwargs)
        return "native-context"

    namespace = {"partial": partial, "_benchmark_with_power": shared_benchmark}
    exec(compile(ast.Module(body=[binding], type_ignores=[]), str(path), "exec"), namespace)
    kernel = object()
    assert namespace["benchmark_with_power"](kernel_func=kernel, num_runs=30) == "native-context"
    assert calls == [{"explicit_graph_cleanup": True, "kernel_func": kernel, "num_runs": 30}]
    assert namespace["_benchmark_with_power"] is shared_benchmark
