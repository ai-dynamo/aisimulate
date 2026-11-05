# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the real GEMM timing/publication block without a GPU or vLLM install.

Only layer construction is omitted: the suffix beginning at ``kernel_func``
executes unchanged against a recording benchmark context. This checks behavior,
not a source substring, and catches both graph-off requests and publication of
an eager result even when a benchmark helper silently changes its behavior.
"""

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
SOURCE = Path(__file__).resolve().parents[3] / "collector" / "vllm" / "collect_gemm.py"


def _run_timing_block(gemm_type, result, *, capture_error=None):
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_gemm")
    start = next(i for i, n in enumerate(run.body) if isinstance(n, ast.FunctionDef) and n.name == "kernel_func")
    calls = {"benchmark": [], "forward": [], "published": []}
    x = object()

    @contextmanager
    def benchmark_with_power(**kwargs):
        calls["benchmark"].append(kwargs)
        # Fail at the boundary if a dtype-specific eager exception is restored.
        assert kwargs["use_cuda_graph"] is True
        assert kwargs["allow_graph_fail"] is False
        if capture_error is not None:
            raise capture_error
        kwargs["kernel_func"]()
        yield result

    namespace = {
        "torch": SimpleNamespace(device=lambda d: d, cuda=SimpleNamespace(get_device_name=lambda d: "test GPU")),
        "benchmark_with_power": benchmark_with_power,
        "log_perf": lambda **kwargs: calls["published"].append(kwargs),
        "device": "cuda:0",
        "gemm_type": gemm_type,
        "m": 1,
        "n": 768,
        "k": 7168,
        "x": x,
        "op_list": [SimpleNamespace(forward=lambda arg: calls["forward"].append(arg))],
        "outside_loop_count": 1,
        "vllm_version": "test-version",
        "kernel_source": "test-selected-kernel",
        "perf_filename": "test-perf-file",
    }
    code = compile(ast.Module(body=run.body[start:], type_ignores=[]), str(SOURCE), "exec")
    return lambda: exec(code, namespace), calls, x


@pytest.mark.parametrize("gemm_type", ["bfloat16", "fp8", "fp8_block", "nvfp4"])
def test_gemm_publishes_only_graph_timing(gemm_type):
    run, calls, x = _run_timing_block(gemm_type, {"used_cuda_graph": True, "latency_ms": 0.012, "power_stats": None})
    run()
    assert calls["forward"] == [x]
    assert len(calls["benchmark"]) == 1
    benchmark = calls["benchmark"][0]
    assert (benchmark["num_warmups"], benchmark["num_runs"], benchmark["repeat_n"]) == (3, 6, 1)
    assert len(calls["published"]) == 1
    published = calls["published"][0]
    assert published["kernel_source"] == "test-selected-kernel"
    assert published["item_list"][0]["gemm_dtype"] == gemm_type
    assert published["item_list"][0]["latency"] == 0.012


@pytest.mark.parametrize("graph_flag", [False, None, 0, 1, "false", "true", "missing"])
def test_eager_or_unproven_result_is_not_published(graph_flag):
    result = {"latency_ms": 0.15, "power_stats": None}
    if graph_flag != "missing":
        result["used_cuda_graph"] = graph_flag
    run, calls, _ = _run_timing_block("fp8_block", result)
    with pytest.raises(RuntimeError, match="refusing to publish eager timing"):
        run()
    assert calls["published"] == []


def test_capture_failure_propagates_without_publication():
    failure = RuntimeError("capture failed")
    run, calls, _ = _run_timing_block("fp8_block", {}, capture_error=failure)
    with pytest.raises(RuntimeError, match="capture failed") as exc:
        run()
    assert exc.value is failure
    assert calls["published"] == []
