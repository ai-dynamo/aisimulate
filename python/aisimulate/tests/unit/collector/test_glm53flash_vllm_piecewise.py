# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY native API doubles: no real CUDA capture or measurement evidence."""

import contextlib
import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from collector import glm53flash_vllm_piecewise as pw

pytestmark = pytest.mark.unit


@pytest.fixture
def native(monkeypatch):
    timeline, scopes, graphs = [], [], []

    class Graph:
        def __init__(self):
            self.number = len(graphs) + 1
            self.nodes = {}
            graphs.append(self)

        def replay(self):
            timeline.append(("graph", self.number))

        def raw_cuda_graph_exec(self):
            return 9000 + self.number

    class Capture:
        """Original test double for native API state, never a framework fixture."""

        def __init__(self):
            self.segments = []
            self._current_graph = None
            self._capturing = False
            self.num_graphs = self.num_eager_breaks = 0

        def _begin_segment(self):
            self._current_graph = Graph()
            self._capturing = True

        def _end_segment(self):
            if self._capturing:
                self.segments.append(self._current_graph.replay)
                self.num_graphs += 1
                self._current_graph = None
                self._capturing = False

        def add_eager(self, fn):
            self._end_segment()
            value = fn()
            self.segments.append(fn)
            self.num_eager_breaks += 1
            self._begin_segment()
            return value

        def replay(self):
            for fn in self.segments:
                fn()

    capture = Capture()

    def snapshot():
        assert capture._capturing
        graph = capture._current_graph
        return {"capture_id": 100 + graph.number, "graph_id": graph.number, "nodes": dict(graph.nodes), "edges": []}

    @contextlib.contextmanager
    def record_function(name):
        scopes.append(("enter", name))
        yield
        scopes.append(("exit", name))

    registry = pw.PiecewiseCaptureRegistry(snapshot)
    current = SimpleNamespace(registry=registry)
    monkeypatch.setattr(pw, "BREAKABLE_SOURCE_PIN", hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    original_replay = Capture.replay
    pw.install_segment_observation(Capture, lambda: current, record_function)
    assert Capture.replay is original_replay

    def eager():
        timeline.append(("eager", 99))
        return 99

    return capture, registry, timeline, scopes, eager, current


def capture_split_operation(native):
    capture, registry, timeline, scopes, eager, _ = native
    capture._begin_segment()
    token = registry.enter("attention_3", "TEST_ONLY_attention_source")
    capture._current_graph.nodes[11] = {"node_type": 0}
    assert capture.add_eager(eager) == 99
    capture._current_graph.nodes[21] = {"node_type": 0}
    registry.leave(token)
    capture._end_segment()
    return capture, registry


def test_one_operation_spans_native_graphs_and_eager_without_replacing_replay(native):
    capture, registry = capture_split_operation(native)
    _, _, timeline, scopes, _, _ = native
    result = registry.finish(capture)
    assert [row["kind"] for row in result["segments"]] == ["graph", "eager", "graph"]
    assert len(result["calls"]) == 1 and result["calls"][0]["completed"]
    assert result["segments"][0]["nodes"] == [{"node_id": 11, "node_type": 0, "name": "attention_3", "call_index": 0}]
    assert result["segments"][2]["nodes"] == [{"node_id": 21, "node_type": 0, "name": "attention_3", "call_index": 0}]
    assert result["segments"][1]["name"] == "attention_3"
    assert result["segments"][1]["call_index"] == 0
    assert timeline == [("eager", 99)] and scopes == []  # Native initialization only.
    registry.validate_replay(capture)
    capture.replay()
    assert timeline[-3:] == [("graph", 1), ("eager", 99), ("graph", 2)] and scopes == []
    registry.profiling = True
    capture.replay()
    assert scopes == [("enter", pw.EAGER_RANGE_PREFIX + "0"), ("exit", pw.EAGER_RANGE_PREFIX + "0")]
    assert timeline[-3:] == [("graph", 1), ("eager", 99), ("graph", 2)]
    assert result["formal_admission"] is False


@pytest.mark.parametrize("defect", ["duplicate", "missing", "reordered", "unknown", "counter"])
def test_mutated_native_segment_list_cannot_be_admitted(native, defect):
    capture, registry = capture_split_operation(native)
    if defect == "duplicate":
        capture.segments[2] = capture.segments[0]
    elif defect == "missing":
        capture.segments.pop()
    elif defect == "reordered":
        capture.segments.reverse()
    elif defect == "unknown":
        capture.segments[1] = lambda: None
    else:
        capture.num_eager_breaks = 0
    with pytest.raises(RuntimeError, match="piecewise"):
        registry.finish(capture)


def test_replay_cannot_replace_even_equivalent_bound_native_method(native):
    capture, registry = capture_split_operation(native)
    registry.finish(capture)
    capture.segments[0] = capture.segments[0].__self__.replay
    with pytest.raises(RuntimeError, match="initialized callable list"):
        registry.validate_replay(capture)


@pytest.mark.parametrize(
    "defect", [None, "wrong_tokens", "wrong_mode", "changed_entry", "changed_callable", "changed_shape", "disabled"]
)
def test_v2_selection_resolves_original_initialized_entry_only(native, defect):
    @dataclasses.dataclass(frozen=True)
    class Shape:
        num_tokens: int = 8
        num_reqs: object = None
        uniform: bool = False
        has_lora: bool = False
        num_active_loras: int = 0

    @dataclasses.dataclass
    class Selection:
        cg_mode: object
        num_tokens: int = 8
        num_reqs: object = None
        uniform_token_count: object = None
        max_query_len: object = None
        num_active_loras: int = 0
        num_ubatches: int = 1

    capture, registry = capture_split_operation(native)
    registry.finish(capture)
    key = Shape()
    entry = SimpleNamespace(capture=capture)
    wrapper = SimpleNamespace(
        entries={key: entry},
        _aisim_piecewise_ownership={key: {"entry": entry, "capture": capture, "registry": registry}},
    )
    manager = SimpleNamespace(use_breakable_cg=True, breakable_cg_runner=wrapper)
    registry.bound_capture = {"native_shape_key": dataclasses.asdict(key)}
    selection = Selection(SimpleNamespace(name="PIECEWISE"))
    if defect == "wrong_tokens":
        selection.num_tokens = 9
    elif defect == "wrong_mode":
        selection.cg_mode.name = "FULL"
    elif defect == "changed_entry":
        wrapper.entries[key] = SimpleNamespace(capture=capture)
    elif defect == "changed_callable":
        capture.segments.reverse()
    elif defect == "changed_shape":
        registry.bound_capture["native_shape_key"]["num_tokens"] = 9
    elif defect == "disabled":
        manager.use_breakable_cg = False
    if defect:
        with pytest.raises(RuntimeError):
            pw.piecewise_capture_for_descriptor(manager, selection)
    else:
        assert pw.piecewise_capture_for_descriptor(manager, selection) is registry


def test_unfinished_operation_or_graph_rejected(native):
    capture, registry, *_ = native
    capture._begin_segment()
    token = registry.enter("attention_3", "TEST_ONLY")
    with pytest.raises(RuntimeError, match="did not close"):
        registry.finish(capture)
    capture._end_segment()
    with pytest.raises(RuntimeError, match="did not close"):
        registry.finish(capture)
    registry.leave(token)
    assert len(registry.finish(capture)["calls"]) == 1


def test_unobserved_capture_executes_original_native_methods(native):
    capture, registry, timeline, scopes, eager, current = native
    current.registry = None
    capture._begin_segment()
    assert capture.add_eager(eager) == 99
    capture._end_segment()
    capture.replay()
    assert timeline == [("eager", 99), ("graph", 1), ("eager", 99), ("graph", 2)]
    assert registry.graphs == [] and registry.eager == [] and scopes == []


def test_pinned_native_capture_source_and_single_install_are_required(native, monkeypatch):
    capture, *_ = native
    monkeypatch.setattr(pw, "BREAKABLE_SOURCE_PIN", "0" * 64)
    # A separately named subclass still carries the existing installation flag.
    with pytest.raises(RuntimeError, match="already installed"):
        pw.install_segment_observation(type(capture), lambda: None, contextlib.nullcontext)
    del type(capture)._aisim_piecewise_segments
    with pytest.raises(RuntimeError, match="reviewed native source"):
        pw.install_segment_observation(type(capture), lambda: None, contextlib.nullcontext)


def test_native_graph_identity_or_node_type_change_rejected(native):
    capture, registry, *_ = native
    capture._begin_segment()
    token = registry.enter("attention_3", "TEST_ONLY")
    capture._current_graph.nodes[11] = {"node_type": 0}
    registry._flush()
    capture._current_graph.nodes[11] = {"node_type": 2}
    with pytest.raises(RuntimeError, match="source type"):
        registry.leave(token)


def test_original_wrapper_capture_binds_exact_initialized_entry(native):
    capture, _, timeline, scopes, eager, _ = native
    cls = type(capture)
    for method in ("_begin_segment", "_end_segment", "add_eager"):
        setattr(cls, method, getattr(cls, method).__wrapped__)
    del cls._aisim_piecewise_segments

    @dataclasses.dataclass(frozen=True)
    class Descriptor:
        num_tokens: int
        num_reqs: int | None = None
        uniform: bool = False
        has_lora: bool = False
        num_active_loras: int = 0

    def snapshot(stream):
        assert stream == 123 and capture._capturing
        graph = capture._current_graph
        return {"capture_id": 100 + graph.number, "graph_id": graph.number, "nodes": dict(graph.nodes), "edges": []}

    @contextlib.contextmanager
    def record_function(name):
        scopes.append(name)
        yield

    torch = SimpleNamespace(
        cuda=SimpleNamespace(current_stream=lambda: SimpleNamespace(cuda_stream=123)),
        profiler=SimpleNamespace(record_function=record_function),
    )
    manifest = {"phases": {phase: [{"name": "attention_3"}, {"name": "logits"}] for phase in ("context", "generation")}}
    observer = pw.NativePiecewiseGraphObserver(
        manifest, {"TEST_ONLY": True}, 0, SimpleNamespace(snapshot=snapshot, libraries={}), torch_module=torch
    )

    class Model:
        def forward(self, value):
            capture._current_graph.nodes[11] = {"node_type": 0}
            assert capture.add_eager(eager) == 99
            capture._current_graph.nodes[21] = {"node_type": 0}
            return value

    model = Model()
    observer.wrap(model, "forward", "attention_3")

    class Wrapper:
        def __init__(self):
            self.entries = {}

        def _capture(self, entry, args, kwargs):
            capture._begin_segment()
            value = model.forward(*args, **kwargs)
            capture._end_segment()
            entry.capture = capture
            entry.output = value
            return value

    saved = []
    wrapper, descriptor = Wrapper(), Descriptor(8)
    entry = SimpleNamespace(batch_descriptor=descriptor, capture=None, output=None)
    wrapper.entries[descriptor] = entry
    pw.install_piecewise_capture(
        Wrapper,
        cls,
        lambda actual: observer if actual is wrapper else None,
        lambda *args: saved.append(args),
        torch_module=torch,
    )
    original_value = object()
    assert wrapper._capture(entry, (original_value,), {}) is original_value
    assert saved[0][0] is wrapper and saved[0][1] is entry
    registry, receipt = saved[0][2:]
    assert receipt["physical_padded_tokens"] == 8
    assert receipt["uncaptured_operations"] == ["logits"]
    assert len(receipt["calls"]) == 1 and len(receipt["segments"]) == 3
    assert observer.registry is None
    assert pw.captured_piecewise_registry(wrapper, descriptor) is registry
    registry.profiling = True
    entry.capture.replay()
    assert timeline[-3:] == [("graph", 1), ("eager", 99), ("graph", 2)]
    assert scopes == [pw.EAGER_RANGE_PREFIX + "0"]
    entry.capture = object()
    with pytest.raises(RuntimeError, match="exact native initialized entry"):
        pw.captured_piecewise_registry(wrapper, descriptor)


@pytest.mark.parametrize("missing_callback", [False, True])
def test_shared_original_callbacks_bind_each_native_segment_once(native, tmp_path, missing_callback):
    capture, registry = capture_split_operation(native)
    source = registry.finish(capture)
    source["native_api_libraries"] = {}
    source_path = tmp_path / "TEST_ONLY-source.json"
    source_path.write_text(json.dumps(source))
    rows = []
    for graph in registry.graphs:
        original = graph["graph"]
        number = original.number
        node = next(iter(graph["nodes"]))
        rows.extend(
            [
                {
                    "kind": "graph_exec_created",
                    "graph_id": number,
                    "graph_exec_id": 500 + number,
                    "raw_fields": {"graph": 8000 + number, "graphExec": original.raw_cuda_graph_exec()},
                },
                {
                    "kind": "node_cloned",
                    "original_node_id": node,
                    "node_id": node + 100,
                    "node_type": 0,
                    "raw_fields": {
                        "nodeType": 0,
                        "node": node + 3000,
                        "originalNode": node + 2000,
                        "graph": original.raw_cuda_graph_exec(),
                        "originalGraph": 8000 + number,
                    },
                },
            ]
        )
    if missing_callback:
        rows.pop()

    class Callbacks:
        subscription_closed = True

        def receipt(self, graph):
            return {
                "callbacks": self.rows,
                "callback_errors": [],
                "actual_graph_exec_id": 500 + graph.number,
                "graph_mutations": False,
                "callback_subscription_closed": True,
            }

    callbacks = Callbacks()
    callbacks.errors = []
    callbacks.rows = rows
    item = {
        "registry": registry,
        "source": source,
        "entry": SimpleNamespace(capture=capture),
        "source_receipt": {"file": source_path.name, "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()},
    }
    if missing_callback:
        with pytest.raises(ValueError, match="complete unique node"):
            pw.bind_piecewise_instantiations([item], callbacks, SimpleNamespace(), tmp_path, "TEST_ONLY")
        assert not hasattr(registry, "bound_capture")
        assert json.loads((tmp_path / "TEST_ONLY-piecewise-callbacks.json").read_text())["callbacks"] == rows
        return
    pw.bind_piecewise_instantiations([item], callbacks, SimpleNamespace(), tmp_path, "TEST_ONLY")
    bound = registry.bound_capture
    graphs = [row for row in bound["segments"] if row["kind"] == "graph"]
    assert [row["graph_id"] for row in graphs] == [501, 502]
    assert [row["nodes"][0]["node_id"] for row in graphs] == [111, 121]
    assert [row["nodes"][0]["capture_node_id"] for row in graphs] == [11, 21]
    assert all(row["nodes"][0]["name"] == "attention_3" for row in graphs)
    assert graphs[0]["shared_callback_receipt"] == graphs[1]["shared_callback_receipt"]
    shared = graphs[0]["shared_callback_receipt"]
    assert hashlib.sha256((tmp_path / shared["file"]).read_bytes()).hexdigest() == shared["sha256"]
    assert json.loads((tmp_path / shared["file"]).read_text())["callbacks"] == rows
    assert len(list(tmp_path.glob("*-callbacks.json"))) == 1
    assert json.loads((tmp_path / registry.capture_artifact["file"]).read_text()) == bound
