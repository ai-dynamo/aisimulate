# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY purpose routing: native imports and selected installers run once."""

import hashlib
import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def bootstrap(monkeypatch):
    source = Path(__file__).parents[3] / "collector/fpm_forward/runtime/glm53flash/sitecustomize.py"
    monkeypatch.setenv("DYN_FPM_GLM53FLASH_REAL_KV", "1")
    before = list(sys.meta_path)
    spec = importlib.util.spec_from_file_location("TEST_ONLY_combined_bootstrap", source)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.meta_path[:] = before


@pytest.mark.parametrize("purpose", [None, "fpm", "ops", "ops_holdout", "ops_graph", "ops_graph_holdout"])
@pytest.mark.parametrize("name", ["gpu_worker", "gpu_model_runner", "gpu.model_runner"])
def test_purpose_selects_one_native_worker_installer(bootstrap, monkeypatch, tmp_path, purpose, name):
    if purpose is None:
        monkeypatch.delenv("AISIM_GLM53_PURPOSE", raising=False)
    else:
        monkeypatch.setenv("AISIM_GLM53_PURPOSE", purpose)
    fullname = "vllm.v1.worker." + name
    source = tmp_path / "TEST_ONLY_native.py"
    source.write_text("# TEST_ONLY native import\n")
    events = []

    class NativeLoader:
        def create_module(self, _spec):
            return None

        def exec_module(self, module):
            events.append("native")
            module.native_value = object()

    original = NativeLoader()
    native_spec = importlib.machinery.ModuleSpec(fullname, original, origin=str(source))
    monkeypatch.setattr(importlib.machinery.PathFinder, "find_spec", lambda *a, **k: native_spec)
    monkeypatch.setattr(
        bootstrap,
        "_source_pins",
        lambda: {fullname.replace(".", "/") + ".py": hashlib.sha256(source.read_bytes()).hexdigest()},
    )
    monkeypatch.setitem(
        sys.modules, "glm53flash_worker_hardware", SimpleNamespace(install=lambda module: events.append("fpm"))
    )
    monkeypatch.setitem(
        sys.modules,
        "collector.glm53flash_vllm_runtime",
        SimpleNamespace(
            install=lambda: events.append("ops_v1"),
            install_v2=lambda: events.append("ops_v2"),
            install_worker_lifecycle=lambda: events.append("ops_worker"),
        ),
    )
    selected = bootstrap._SchedulerFinder().find_spec(fullname)
    if purpose in (None, "fpm") and name != "gpu_worker":
        assert selected is None and events == []
        return
    assert selected.loader.create_module(selected) is None
    module = SimpleNamespace(__name__=fullname, __spec__=selected, __file__=str(source))
    selected.loader.exec_module(module)
    expected = (
        "fpm"
        if purpose in (None, "fpm")
        else {"gpu_worker": "ops_worker", "gpu_model_runner": "ops_v1", "gpu.model_runner": "ops_v2"}[name]
    )
    assert events == ["native", expected]
    assert module.native_value is not None


@pytest.mark.parametrize("purpose", ["fpm", "ops_graph"])
def test_wrong_native_source_never_installs_either_observer(bootstrap, monkeypatch, tmp_path, purpose):
    monkeypatch.setenv("AISIM_GLM53_PURPOSE", purpose)
    source = tmp_path / "TEST_ONLY_native.py"
    source.write_text("# TEST_ONLY wrong native bytes\n")
    events = []
    original = SimpleNamespace(exec_module=lambda module: events.append("native"))
    fullname = "vllm.v1.worker.gpu_worker"
    module = SimpleNamespace(__name__=fullname, __spec__=SimpleNamespace(origin=str(source)), __file__=str(source))
    monkeypatch.setattr(bootstrap, "_source_pins", lambda: {fullname.replace(".", "/") + ".py": "0" * 64})
    loader = bootstrap._SchedulerLoader(original) if purpose == "fpm" else bootstrap._WorkerLoader(original)
    with pytest.raises(RuntimeError, match="source"):
        loader.exec_module(module)
    # Preserve each original branch's source-check/native-import order.
    assert events == ([] if purpose == "fpm" else ["native"])


def test_helper_process_imports_do_not_activate_native_loaders(bootstrap, monkeypatch):
    monkeypatch.setenv("AISIM_GLM53_PURPOSE", "fpm")
    for name in ("subprocess", "torch", "vllm", "triton", "nvrtc", "flashinfer"):
        assert bootstrap._SchedulerFinder().find_spec(name) is None


@pytest.mark.parametrize("phase,count", [("prefill", 1), ("decode", 2)])
def test_fpm_sglang_submission_preserves_original_call_and_inputs(tmp_path, phase, count):
    from collector.fpm_forward.sglang_driver import generate_native_request

    inputs, request_ids, result = [[7, 8]], ["TEST_ONLY-original-request"], object()
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return result

    assert (
        generate_native_request(
            SimpleNamespace(generate=generate),
            root=tmp_path,
            benchmark_id=1,
            repetition=5,
            mode=phase,
            request_ids=request_ids,
            inputs=inputs,
        )
        is result
    )
    assert len(calls) == 1
    assert calls[0]["input_ids"] is inputs and calls[0]["rid"] is request_ids
    assert calls[0]["sampling_params"] == {"temperature": 0, "max_new_tokens": count, "ignore_eos": True}
    assert not list(tmp_path.iterdir())
