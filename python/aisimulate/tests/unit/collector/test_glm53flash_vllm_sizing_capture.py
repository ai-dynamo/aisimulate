# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY lifecycle doubles; no native model, CUDA or GPU qualification."""

import copy
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from collector import glm53flash_vllm_graph_ops as graph
from collector import glm53flash_vllm_graph_policy as policy
from collector import glm53flash_vllm_piecewise as piecewise
from collector.glm53flash_contract import BACKENDS

pytestmark = pytest.mark.unit


def full_snapshot():
    full = [policy._expected("FULL", n) for n in (1, 2, 4)]
    return {
        "backend": "vllm",
        "backend_version": BACKENDS["vllm"][0],
        "backend_revision": BACKENDS["vllm"][1],
        "source_pins": dict(policy.SOURCE_PINS),
        "native_flags": dict.fromkeys(policy.FLAGS, False),
        "capture_sizes": [1, 2, 4],
        "max_num_reqs": 4,
        "max_capture_tokens": 4,
        "decode_query_len": 1,
        "graphs_captured": True,
        "lora_capture_cases": [0],
        "dp_size": 1,
        "tp_size": 4,
        "tp_rank": 0,
        "resolved_mode": "FULL_DECODE_ONLY",
        "use_breakable_cg": False,
        "capture_descriptors": {"FULL": list(reversed(full))},
        "full_graphs": full,
        "candidates": [
            {"num_tokens": n, "num_active_loras": 0, "descriptors": [full[i]]}
            for n, i in ((0, 0), (1, 0), (2, 1), (3, 2), (4, 2))
        ],
        "piecewise_entries": [],
    }


@pytest.fixture
def native(monkeypatch, tmp_path):
    calls, lookups, snapshots = [], [], []
    original_piecewise_install = piecewise.install_piecewise_capture
    output = tmp_path / "output"
    package = tmp_path / "vllm"
    pins = {}
    for name in policy.SOURCE_PINS | graph.SOURCE_PINS:
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("TEST_ONLY native source pin: " + name)
        pins[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(graph, "SOURCE_PINS", pins)
    monkeypatch.setattr(policy, "SOURCE_PINS", pins)

    class Base:
        def capture(self, factory, *args, **kwargs):
            calls.append(("base", self, factory, args, kwargs))
            if self._capture_mem_samples is not None:
                self._capture_mem_samples.extend([101, 7])
            self._graphs_captured = True

    class Manager(Base):
        def __init__(self, sizing=False):
            self._max_full_descs_to_capture = 2 if sizing else None
            self._capture_mem_samples = [] if sizing else None
            self._graphs_captured = False
            self.ubatch_runner = None
            self.graphs = {}
            self.snapshot = full_snapshot()
            if sizing:
                self.snapshot["full_graphs"] = self.snapshot["full_graphs"][1:]
            self.error = None
            self.returned = object()

        def capture(self, model, *args, **kwargs):
            calls.append(("model", self, model, args, kwargs))
            if lookups:
                # An installed PIECEWISE wrapper sees no observer during sizing.
                lookups[-1](SimpleNamespace(unwrap=lambda: model))
            if self.error:
                raise self.error
            if hasattr(self, "during_capture"):
                self.during_capture()
            super().capture(lambda *_args: None)
            return self.returned

    class Noop:
        pass

    offloader = Noop()
    modules = {
        "torch": SimpleNamespace(),
        "vllm": SimpleNamespace(__file__=str(package / "__init__.py")),
        "vllm.distributed": SimpleNamespace(get_tensor_model_parallel_rank=lambda: 0),
        "vllm.model_executor.offloader.base": SimpleNamespace(NoopOffloader=Noop, get_offloader=lambda: offloader),
        "vllm.v1.worker.gpu.cudagraph_utils": SimpleNamespace(
            ModelCudaGraphManager=Manager, CudaGraphManager=Base, has_compiled_submodule=lambda model: False
        ),
        "vllm.compilation.breakable_cudagraph": SimpleNamespace(
            BreakableCUDAGraphCapture=type("Capture", (), {}), BreakableCUDAGraphWrapper=type("Wrapper", (), {})
        ),
    }
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)

    def persist(manager, model, rank, target):
        snapshots.append(manager)
        return policy.validate_snapshot(manager.snapshot)

    def install_hooks(model, observer, backend):
        calls.append(("operation_hooks", model))
        return []

    class Observer:
        def __init__(self, manifest, provenance, rank, api):
            calls.append(("observer",))
            self.api = api
            self.registry = None

    class Callbacks:
        def __init__(self, *args):
            calls.append(("callbacks",))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def lookup_install(wrapper, capture, getter, save, **kwargs):
        def lookup(value):
            result = getter(value)
            calls.append(("piecewise_lookup", result))
            return result

        lookups.append(lookup)

    monkeypatch.setattr(graph, "persist_snapshot", persist)
    monkeypatch.setattr(graph, "install_native_hooks", install_hooks)

    def graph_api(*, capture_memset_parameters, memset_contract):
        assert capture_memset_parameters is True
        assert memset_contract == "cuda13_live_source_memset_reported_kind_v2"
        calls.append(("api",))

    monkeypatch.setattr(graph, "NativeGraphAPI", graph_api)
    monkeypatch.setattr(graph, "NativeGraphOperationObserver", Observer)
    monkeypatch.setattr(graph, "CloneCallbacks", Callbacks)
    monkeypatch.setattr(graph, "bind_captured_graphs", lambda manager, pending: calls.append(("bind",)))
    monkeypatch.setattr(piecewise, "install_piecewise_capture", lookup_install)
    return SimpleNamespace(
        Manager=Manager,
        output=output,
        calls=calls,
        snapshots=snapshots,
        original_piecewise_install=original_piecewise_install,
    )


def install(native, mode, include_piecewise=False):
    if mode == "holdout":
        graph.install_holdout_capture(native.output, include_piecewise=include_piecewise)
    else:
        graph.install({}, {}, native.output, include_piecewise=include_piecewise)


@pytest.mark.parametrize("mode", ["holdout", "calibration"])
@pytest.mark.parametrize("include_piecewise", [False, True])
def test_sizing_native_capture_runs_once_without_serving_observation(native, mode, include_piecewise):
    install(native, mode, include_piecewise)
    manager, model, argument = native.Manager(sizing=True), object(), object()
    samples = manager._capture_mem_samples
    assert manager.capture(model, argument, native_flag=7) is manager.returned
    model_calls = [x for x in native.calls if x[0] == "model"]
    assert model_calls == [("model", manager, model, (argument,), {"native_flag": 7})]
    assert len([x for x in native.calls if x[0] == "base"]) == 1
    assert manager._capture_mem_samples is samples and samples == [101, 7]
    assert manager._graphs_captured is True
    assert not native.snapshots
    assert not any(x[0] in {"api", "observer", "operation_hooks", "callbacks", "bind"} for x in native.calls)
    assert all(x[1] is None for x in native.calls if x[0] == "piecewise_lookup")
    assert not any(name.startswith("_aisim") for name in vars(manager))
    assert not list(native.output.glob("*.json"))


@pytest.mark.parametrize("mode", ["holdout", "calibration"])
def test_sizing_exception_propagates_same_object_without_snapshot(native, mode):
    install(native, mode)
    manager = native.Manager(sizing=True)
    original = RuntimeError("TEST_ONLY native failure")
    manager.error = original
    with pytest.raises(RuntimeError) as caught:
        manager.capture(object())
    assert caught.value is original
    assert len([x for x in native.calls if x[0] == "model"]) == 1
    assert not native.snapshots
    assert not any(name.startswith("_aisim") for name in vars(manager))


@pytest.mark.parametrize("mode", ["holdout", "calibration"])
@pytest.mark.parametrize("incomplete", [False, True])
def test_real_manager_after_native_sizing_replacement_keeps_strict_full_validation(native, mode, incomplete):
    install(native, mode)
    model = object()
    disposable = native.Manager(sizing=True)
    disposable.capture(model)
    manager = native.Manager()  # Native teardown/reinitialization creates a new manager.
    if incomplete:
        manager.snapshot["full_graphs"].pop(0)
        with pytest.raises(ValueError, match="initialized captures"):
            manager.capture(model)
    else:
        assert manager.capture(model) is manager.returned
        captured = getattr(manager, "_aisim_glm53_holdout_capture" if mode == "holdout" else "_aisim_glm53_ops_capture")
        assert captured[0] is model and captured[1] == manager.snapshot
    assert native.snapshots == [manager]
    assert len([x for x in native.calls if x[0] == "model" and x[1] is manager]) == 1
    if mode == "calibration":
        assert len([x for x in native.calls if x[0] == "operation_hooks"]) == 1
        assert len([x for x in native.calls if x[0] == "callbacks"]) == 1


@pytest.mark.parametrize("mode", ["holdout", "calibration"])
@pytest.mark.parametrize("defect", ["missing", "mixed", "bool", "wrong_limit", "nonlist", "used_samples", "captured"])
def test_ambiguous_native_sizing_markers_cannot_bypass_serving_validation(native, mode, defect):
    install(native, mode)
    manager = native.Manager(sizing=True)
    if defect == "missing":
        del manager._max_full_descs_to_capture
    elif defect == "mixed":
        manager._max_full_descs_to_capture = None
    elif defect == "bool":
        manager._max_full_descs_to_capture = True
    elif defect == "wrong_limit":
        manager._max_full_descs_to_capture = 3
    elif defect == "nonlist":
        manager._capture_mem_samples = ()
    elif defect == "used_samples":
        manager._capture_mem_samples = [1]
    else:
        manager._graphs_captured = True
    with pytest.raises(RuntimeError, match="sizing"):
        manager.capture(object())
    assert not native.calls and not native.snapshots


def test_sizing_does_not_reuse_calibration_observer_from_real_manager(native):
    install(native, "calibration")
    model = object()
    native.Manager().capture(model)
    before = copy.copy(native.calls)
    with pytest.raises(RuntimeError, match="reuse a serving operation observer"):
        native.Manager(sizing=True).capture(model)
    assert native.calls == before


def test_installed_piecewise_and_segment_wrappers_delegate_sizing_without_observer(native, monkeypatch):
    timeline = []
    model, entry, token, result = object(), object(), object(), object()

    class Capture:
        def _begin_segment(self):
            timeline.append("begin")
            self._capturing = True

        def _end_segment(self):
            timeline.append("end")
            self._capturing = False

        def add_eager(self, fn):
            timeline.append("eager")
            return fn()

        def replay(self):
            raise AssertionError("sizing must not replay")

    class Wrapper:
        def unwrap(self):
            return model

        def _capture(self, actual_entry, actual_token, *, mode):
            assert actual_entry is entry and actual_token is token and mode == "TEST_ONLY"
            timeline.append("wrapper")
            capture = Capture()
            capture._begin_segment()
            assert capture.add_eager(lambda: result) is result
            capture._end_segment()
            return result

    def no_profiler(*args):
        raise AssertionError("sizing must not create a profiler scope")

    classes = sys.modules["vllm.compilation.breakable_cudagraph"]
    classes.BreakableCUDAGraphCapture, classes.BreakableCUDAGraphWrapper = Capture, Wrapper
    sys.modules["torch"].profiler = SimpleNamespace(record_function=no_profiler)
    monkeypatch.setattr(piecewise, "BREAKABLE_SOURCE_PIN", hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    monkeypatch.setattr(piecewise, "install_piecewise_capture", native.original_piecewise_install)
    original_replay = Capture.replay
    install(native, "calibration", include_piecewise=True)
    manager, wrapper = native.Manager(sizing=True), Wrapper()

    def inner():
        assert wrapper._capture(entry, token, mode="TEST_ONLY") is result

    manager.during_capture = inner
    assert manager.capture(model) is manager.returned
    assert timeline == ["wrapper", "begin", "eager", "end"]
    assert Capture.replay is original_replay
    assert not hasattr(wrapper, "_aisim_piecewise_ownership")
    assert not native.snapshots
    assert [x[0] for x in native.calls] == ["model", "base"]
    assert not list(native.output.glob("*.json"))
