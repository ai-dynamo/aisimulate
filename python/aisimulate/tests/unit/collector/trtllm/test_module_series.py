# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free tests for the TRT-LLM attention-module series runners.

module_series is framework-free and is tested directly. The MLA/DSA and MSA
collectors import tensorrt_llm at module scope, so their series entry points are
AST-extracted and run against a fake per-point function. The DeepSeek-V4
collector imports the framework lazily and loads with a torch stub."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from collector.trtllm import module_series
from collector.trtllm.module_series import is_fatal_cuda_error, ordered_points, run_points

pytestmark = pytest.mark.unit
COLLECTOR = Path(__file__).resolve().parents[4] / "collector" / "trtllm"


def test_points_are_distinct_and_run_largest_total_length_first():
    points = [(128, 0), (64, 4096), (128, 0), (1024, 2048), (64, 1024), (2048, 1024)]
    # (64, 4096) has the largest total; equal totals put the longer seq_len first
    assert ordered_points(points) == [(64, 4096), (2048, 1024), (1024, 2048), (64, 1024), (128, 0)]


@pytest.mark.parametrize("points", [[], [(0, 0)], [(16, -1)]])
def test_points_need_a_positive_seq_len_and_a_prefix(points):
    with pytest.raises(ValueError):
        ordered_points(points)


@pytest.mark.parametrize("point", [(16,), (16, 0, 1), "16"])
def test_a_point_is_a_pair(point):
    with pytest.raises(TypeError):
        ordered_points([point])


class _Recorder:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail or {}

    def __call__(self, seq_len, prefix_len, dry_run):
        self.calls.append((seq_len, prefix_len, dry_run))
        exc = self.fail.get((seq_len, prefix_len))
        if exc is not None:
            raise exc
        return float(seq_len + prefix_len)


def test_the_dry_run_happens_once_per_seq_len():
    run = _Recorder()
    latencies = run_points([(64, 0), (64, 2048), (128, 0), (64, 128)], run, label="t")
    assert run.calls == [(64, 2048, True), (64, 128, False), (128, 0, True), (64, 0, False)]
    assert latencies == [2112.0, 192.0, 128.0, 64.0]


def test_a_failed_point_does_not_count_as_the_seq_len_s_dry_run():
    run = _Recorder(fail={(64, 2048): MemoryError("out of memory")})
    with pytest.raises(MemoryError):
        run_points([(64, 2048), (64, 128)], run, label="t")
    assert run.calls == [(64, 2048, True), (64, 128, True)]


def test_one_failed_point_is_raised_after_the_others_ran():
    oom = MemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
    run = _Recorder(fail={(1024, 0): oom})
    with pytest.raises(MemoryError) as caught:
        run_points([(1024, 0), (512, 0), (256, 0)], run, label="t")
    assert caught.value is oom
    assert [call[:2] for call in run.calls] == [(1024, 0), (512, 0), (256, 0)]


def test_several_failed_points_are_summarised():
    run = _Recorder(fail={(1024, 0): MemoryError("oom a"), (512, 0): ValueError("bad shape")})
    with pytest.raises(RuntimeError) as caught:
        run_points([(1024, 0), (512, 0), (256, 0)], run, label="t")
    message = str(caught.value)
    assert message.startswith("2/3 point(s) of the module series failed")
    assert "s=1024 prefix=0: MemoryError: oom a" in message and "s=512 prefix=0: ValueError: bad shape" in message


def test_kernel_limit_failures_keep_their_classification():
    limit = "FIXME(kernel-limit): no kernel for this shape"
    run = _Recorder(fail={(1024, 0): RuntimeError(limit), (512, 0): ValueError(limit)})
    with pytest.raises(RuntimeError) as caught:
        run_points([(1024, 0), (512, 0), (256, 0)], run, label="t")
    assert str(caught.value).startswith("FIXME(kernel-limit): 2/3 point(s)")

    mixed = _Recorder(fail={(1024, 0): RuntimeError(limit), (512, 0): MemoryError("oom")})
    with pytest.raises(RuntimeError) as caught:
        run_points([(1024, 0), (512, 0), (256, 0)], mixed, label="t")
    assert str(caught.value).startswith("2/3 point(s)")


def test_a_sticky_cuda_error_aborts_the_series():
    run = _Recorder(fail={(1024, 0): RuntimeError("CUDA error: an illegal memory access was encountered")})
    with pytest.raises(RuntimeError, match="illegal memory access"):
        run_points([(1024, 0), (512, 0), (256, 0)], run, label="t")
    assert [call[:2] for call in run.calls] == [(1024, 0)]


def test_a_one_point_series_behaves_like_a_one_point_case():
    run = _Recorder(fail={(256, 0): MemoryError("oom")})
    with pytest.raises(MemoryError):
        run_points([(256, 0)], run, label="t")
    assert run.calls == [(256, 0, True)]


def test_fatal_cuda_errors_are_recognised_by_type_and_message(monkeypatch):
    class AcceleratorError(RuntimeError):
        pass

    torch_stub = ModuleType("torch")
    torch_stub.AcceleratorError = AcceleratorError
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    assert is_fatal_cuda_error(AcceleratorError("anything"))
    assert is_fatal_cuda_error(RuntimeError("device-side assert triggered"))
    assert not is_fatal_cuda_error(MemoryError("CUDA out of memory"))
    assert not is_fatal_cuda_error(ValueError("FIXME(kernel-limit): no kernel"))


# ---------------------------------------------------------------------------
# Collector wiring
# ---------------------------------------------------------------------------


def _function_params(tree: ast.Module, name: str) -> list[str]:
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return [arg.arg for arg in fn.args.args]


def _load_series(filename: str, names: list[str], namespace: dict):
    tree = ast.parse((COLLECTOR / filename).read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert sorted(n.name for n in selected) == sorted(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), filename, "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    ("filename", "single", "series"),
    [
        ("collect_mla_module.py", "run_mla_module_worker", "run_mla_module_series_worker"),
        ("collect_msa_module.py", "run_msa_module_worker", "run_msa_module_series_worker"),
        ("collect_dsv4_attn.py", "run_dsv4_attn_worker", "run_dsv4_attn_series_worker"),
    ],
)
def test_series_workers_take_the_single_worker_fields_with_points_first(filename, single, series):
    tree = ast.parse((COLLECTOR / filename).read_text())
    single_params = _function_params(tree, single)
    series_params = _function_params(tree, series)
    expected = ["points"] + [p for p in single_params if p not in ("seq_len", "prefix_len")]
    assert series_params == expected


@pytest.mark.parametrize(
    ("filename", "point_fn"),
    [
        ("collect_mla_module.py", "run_mla_module"),
        ("collect_msa_module.py", "run_msa_module"),
        ("collect_dsv4_attn.py", "run_dsv4_attn"),
    ],
)
def test_a_one_point_case_still_dry_runs(filename, point_fn):
    tree = ast.parse((COLLECTOR / filename).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == point_fn)
    defaults = dict(zip([a.arg for a in fn.args.kwonlyargs], fn.args.kw_defaults, strict=True))
    assert isinstance(defaults["dry_run"], ast.Constant) and defaults["dry_run"].value is True


def test_mla_series_worker_runs_each_point_on_the_one_point_path():
    calls = []
    namespace = _load_series(
        "collect_mla_module.py",
        ["run_mla_module_series", "run_mla_module_series_worker"],
        {
            "run_points": run_points,
            "get_sm_version": lambda: 90,
            "run_mla_module": lambda **kwargs: calls.append(kwargs) or 1.0,
        },
    )
    namespace["run_mla_module_series_worker"](
        [(64, 0), (64, 4096), (2048, 1024)],
        4,
        64,
        "fp8",
        "bfloat16",
        "fp8_block",
        "deepseek-ai/DeepSeek-V3.2",
        "dsa",
        perf_filename="/out/dsa_context_module_perf.txt",
    )
    assert [(c["seq_len"], c["prefix_len"], c["dry_run"]) for c in calls] == [
        (64, 4096, True),
        (2048, 1024, True),
        (64, 0, False),
    ]
    assert {c["model_path"] for c in calls} == {"deepseek-ai/DeepSeek-V3.2"}
    assert {(c["batch_size"], c["num_heads"], c["gemm_type"], c["attn_type"]) for c in calls} == {
        (4, 64, "fp8_block", "dsa")
    }


def test_msa_series_worker_runs_each_point_on_the_one_point_path():
    calls = []
    namespace = _load_series(
        "collect_msa_module.py",
        ["run_msa_module_series", "run_msa_module_series_worker"],
        {"run_points": run_points, "run_msa_module": lambda **kwargs: calls.append(kwargs) or 1.0},
    )
    namespace["run_msa_module_series_worker"](
        [(8192, 0), (16384, 0)],
        8,
        64,
        "bfloat16",
        "bfloat16",
        "bfloat16",
        "MiniMaxAI/MiniMax-M3",
        perf_filename="/out/msa_generation_module_perf.txt",
    )
    assert [(c["seq_len"], c["prefix_len"], c["dry_run"]) for c in calls] == [(16384, 0, True), (8192, 0, True)]


def _load_dsv4(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    torch_stub = ModuleType("torch")
    torch_stub.cuda = SimpleNamespace(empty_cache=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    spec = importlib.util.spec_from_file_location("trtllm_dsv4_series_target", COLLECTOR / "collect_dsv4_attn.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dsv4_series_worker_runs_each_point_on_the_one_point_path(monkeypatch):
    module = _load_dsv4(monkeypatch)
    calls = []
    monkeypatch.setattr(module, "run_dsv4_attn", lambda **kwargs: calls.append(kwargs) or 1.0)
    monkeypatch.setattr(module.sys, "argv", ["collect.py"])
    module.run_dsv4_attn_series_worker(
        [(128, 0), (128, 65536), (512, 4096)],
        8,
        2,
        "fp8",
        "bfloat16",
        "fp8_block",
        "sgl-project/DeepSeek-V4-Flash-FP8",
        "hca",
        perf_filename="/out/dsv4_hca_context_module_perf.txt",
    )
    assert [(c["seq_len"], c["prefix_len"], c["dry_run"]) for c in calls] == [
        (128, 65536, True),
        (512, 4096, True),
        (128, 0, False),
    ]
    assert {(c["mode"], c["tp_size"], c["attn_kind"], c["warming_up"], c["test_ite"]) for c in calls} == {
        ("context", 2, "hca", 10, 6)
    }


def test_the_series_helper_stays_framework_free():
    tree = ast.parse(Path(module_series.__file__).read_text())
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (node.names if isinstance(node, ast.Import) else [SimpleNamespace(name=node.module or "")])
    }
    assert not imported & {"torch", "tensorrt_llm"}
