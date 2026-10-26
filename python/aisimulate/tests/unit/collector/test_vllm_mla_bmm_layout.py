# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise BMM storage and timing contracts on CPU without importing vLLM."""

import ast
import hashlib
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from collector.case_generator import get_mla_bmm_case_specs
from collector.framework_manifest import require_collector_runtime
from collector.version_resolver import _check_compat, resolve_module
from collector.vllm.registry import REGISTRY

pytestmark = pytest.mark.unit
COLLECTOR = Path(__file__).resolve().parents[3] / "collector" / "vllm"


class _Tensor:
    def __init__(self, array):
        self.array = array

    def transpose(self, first, second):
        return _Tensor(self.array.swapaxes(first, second))


@pytest.fixture
def bmm_modules(monkeypatch):
    calls = {"bmm": [], "benchmark": [], "published": [], "capture_error": None}
    rng = np.random.default_rng(0)

    def tensor(shape, *, device, dtype):
        assert device == "cuda:0" and dtype == "bfloat16"
        # CPU float32 checks geometry and storage, not CUDA BF16 numerics.
        return _Tensor(rng.standard_normal(shape).astype(np.float32))

    def bmm(left, right, *, out):
        calls["bmm"].append((left.array, right.array, out.array))
        np.matmul(left.array, right.array, out=out.array)

    @contextmanager
    def benchmark(**kwargs):
        calls["benchmark"].append(kwargs)
        if calls["capture_error"] is not None:
            raise calls["capture_error"]
        kwargs["kernel_func"]()
        yield {"latency_ms": 0.012, "power_stats": None}

    source = COLLECTOR / "collect_mla_bmm_v1.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name != "_log_row"]
    old = ModuleType("collector.vllm.collect_mla_bmm_v1")
    old.__dict__.update(
        torch=SimpleNamespace(
            cuda=SimpleNamespace(set_device=lambda device: None),
            set_default_device=lambda device: None,
            bfloat16="bfloat16",
            randn=tensor,
            empty=tensor,
            bmm=bmm,
        ),
        benchmark_with_power=benchmark,
        _log_row=lambda **kwargs: calls["published"].append(kwargs),
        get_mla_bmm_case_specs=get_mla_bmm_case_specs,
        _SUPPORTED_DTYPES={"bfloat16"},
    )
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), old.__dict__)
    monkeypatch.setitem(sys.modules, old.__name__, old)
    new = ModuleType("collector.vllm.collect_mla_bmm_v2")
    source = COLLECTOR / "collect_mla_bmm_v2.py"
    exec(compile(source.read_text(), str(source), "exec"), new.__dict__)
    return old, new, calls


@pytest.mark.parametrize(
    "route,function,expected_shape,expected_strides,op_name",
    [
        ("old", "run_mla_gen_pre", (2, 3, 512), (3 * 512, 512, 1), "mla_gen_pre"),
        ("new", "run_mla_gen_pre", (2, 3, 512), (512, 2 * 512, 1), "mla_gen_pre"),
        ("old", "run_mla_gen_post", (2, 3, 128), (128, 2 * 128, 1), "mla_gen_post"),
    ],
)
def test_bmm_output_storage_and_timing(bmm_modules, route, function, expected_shape, expected_strides, op_name):
    old, new, calls = bmm_modules
    module = new if route == "new" else old
    getattr(module, function)(3, 2, "bfloat16", 2, 10, perf_filename="mla_bmm_perf.txt")
    assert len(calls["bmm"]) == 1
    left, right, out = calls["bmm"][0]
    assert out.shape == expected_shape
    assert tuple(stride // out.itemsize for stride in out.strides) == expected_strides
    np.testing.assert_allclose(out, np.einsum("nbk,nkl->nbl", left, right), rtol=1e-4, atol=1e-4)
    timing = calls["benchmark"][0]
    assert (timing["num_warmups"], timing["num_runs"], timing["repeat_n"]) == (2, 10, 1)
    assert timing.get("use_cuda_graph", True) is True
    assert timing.get("allow_graph_fail", False) is False
    assert len(calls["published"]) == 1
    assert calls["published"][0]["op_name"] == op_name
    assert calls["published"][0]["results"]["latency_ms"] == 0.012


def test_pre_bmm_capture_failure_does_not_publish(bmm_modules):
    _, new, calls = bmm_modules
    calls["capture_error"] = RuntimeError("capture failed")
    with pytest.raises(RuntimeError, match="capture failed"):
        new.run_mla_gen_pre(3, 2, "bfloat16", 2, 10, perf_filename="mla_bmm_perf.txt")
    assert calls["published"] == []


def test_pre_bmm_route_is_exact_and_keeps_full_case_grid(bmm_modules):
    old, new, _ = bmm_modules
    pre = next(entry for entry in REGISTRY if entry.op == "mla_bmm_gen_pre")
    post = next(entry for entry in REGISTRY if entry.op == "mla_bmm_gen_post")
    for version in ("0.25.0", "0.27.0"):
        assert resolve_module(pre, version) == old.__name__
    assert resolve_module(pre, "0.31.0") == new.__name__
    assert resolve_module(post, "0.31.0") == old.__name__
    assert _check_compat(new.__compat__, "0.31.0+cu130")
    for version in ("0.30.0", "0.31.1", "0.32.0"):
        assert not _check_compat(new.__compat__, version)
    cases = new.get_mla_gen_pre_test_cases()
    assert len(cases) == 300
    assert {case[2] for case in cases} == {"bfloat16"}
    assert {(case[0], case[1]) for case in cases} == {
        (case.num_tokens, case.num_heads)
        for case in get_mla_bmm_case_specs("vllm", "mla_bmm_gen_pre")
        if case.dtype == "bfloat16"
    }


def test_b200_campaign_runtime_is_opt_in(monkeypatch):
    with pytest.raises(RuntimeError, match="requires exactly 0.27.0"):
        require_collector_runtime("vllm", "0.31.0", requested_ops={"mla_bmm_gen_pre"})
    manifest = COLLECTOR.parent / "campaigns" / "vllm031-b200-runtime.json"
    monkeypatch.setenv("AISIM_COLLECTOR_RUNTIME_MANIFEST", str(manifest))
    monkeypatch.setenv("AISIM_COLLECTOR_RUNTIME_MANIFEST_SHA256", hashlib.sha256(manifest.read_bytes()).hexdigest())
    runtime = require_collector_runtime("vllm", "0.31.0", requested_ops={"mla_bmm_gen_pre"})
    assert runtime.source_commit == "db9527a46873454610df6dbedf79a36d6bf1a7f6"
    assert runtime.image().endswith("@sha256:3fb023cc987cc7c2bca7ca14c1e0aff5ae06bd9431bf07773731ce7e8a682460")
