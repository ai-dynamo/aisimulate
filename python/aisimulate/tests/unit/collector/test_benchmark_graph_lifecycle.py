# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual benchmark context without CUDA, retaining graph references."""

import ast
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
SOURCE = Path(__file__).resolve().parents[3] / "collector/helper.py"


def _benchmark(monkeypatch, failure=None):
    events = []
    graphs = []  # Keep Python objects alive, as a capture traceback can do.
    error = RuntimeError(f"{failure} failed")
    streams = {"current": "caller"}

    def set_stream(stream):
        streams["current"] = stream

    class Graph:
        def __init__(self):
            self.resets = 0
            graphs.append(self)

        def replay(self):
            events.append("replay")
            if failure == "replay":
                raise error

        def reset(self):
            assert streams["current"] == "caller"
            self.resets += 1
            events.append("reset")

    @contextmanager
    def capture(graph):
        events.append("capture")
        streams["current"] = "capture"
        if failure == "capture":
            raise error
        yield
        streams["current"] = "caller"

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            events.append("event")

        def elapsed_time(self, end):
            return 12.0

    cuda = SimpleNamespace(
        is_available=lambda: True,
        CUDAGraph=Graph,
        graph=capture,
        current_stream=lambda: streams["current"],
        set_stream=set_stream,
        synchronize=lambda: events.append("synchronize"),
        empty_cache=lambda: events.append("empty_cache"),
        Event=Event,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "benchmark_with_power")
    namespace = {"contextmanager": contextmanager, "get_device_module": lambda: cuda, "logging": logging}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)

    def run(**kwargs):
        return namespace["benchmark_with_power"](
            device=SimpleNamespace(index=0),
            kernel_func=lambda: events.append("kernel"),
            measure_power=False,
            power_min_duration=1.0,
            num_warmups=2,
            num_runs=3,
            **kwargs,
        )

    return run, graphs, events, error


@pytest.mark.parametrize("failure", [None, "capture", "replay", "caller"])
def test_graph_is_reset_even_when_python_references_survive(monkeypatch, failure):
    benchmark, graphs, events, error = _benchmark(monkeypatch, failure)

    def execute():
        with benchmark(explicit_graph_cleanup=True) as result:
            assert result["used_cuda_graph"] is True
            assert result["latency_ms"] == 4.0
            if failure == "caller":
                raise error

    if failure:
        with pytest.raises(RuntimeError) as caught:
            execute()
        assert caught.value is error
    else:
        execute()
    assert len(graphs) == 1
    assert graphs[0].resets == 1
    assert events[-2:] == ["reset", "empty_cache"]
    if failure == "capture":
        assert "replay" not in events
        assert events.count("kernel") == 2  # Only the pre-capture warmup; no eager retry.


def test_explicit_capture_fallback_resets_graph_before_eager_timing(monkeypatch):
    benchmark, graphs, events, _ = _benchmark(monkeypatch, "capture")
    with benchmark(allow_graph_fail=True, explicit_graph_cleanup=True) as result:
        assert result["used_cuda_graph"] is False
    assert graphs[0].resets == 1
    assert events.index("reset") < events.index("event")
    assert "replay" not in events


def test_explicit_eager_timing_does_not_create_graph(monkeypatch):
    benchmark, graphs, _, _ = _benchmark(monkeypatch)
    with benchmark(use_cuda_graph=False) as result:
        assert result["used_cuda_graph"] is False
    assert not graphs
