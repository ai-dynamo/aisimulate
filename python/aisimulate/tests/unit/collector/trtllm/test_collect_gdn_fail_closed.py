# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The trtllm GDN collector publishes CUDA-graph-timed rows only.

Until 2026-10-11 every benchmark_with_power call in collector/trtllm/collect_gdn.py
passed allow_graph_fail=True: a failed capture logged one WARNING, timed the
Python launch loop eagerly and shipped that floor as the kernel latency with no
trace in the row. Capture failures now raise, the failed cell is counted, and
the case ends as a classified strict-completeness failure (the sglang GDN
collector's contract)."""

import ast
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE = REPO_ROOT / "collector" / "trtllm" / "collect_gdn.py"


def _tree():
    return ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))


def _function(name):
    return next(node for node in _tree().body if isinstance(node, ast.FunctionDef) and node.name == name)


def test_trtllm_gdn_benchmarks_never_fall_back_to_eager_timing():
    calls = [
        node
        for node in ast.walk(_tree())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "benchmark_with_power"
    ]
    assert len(calls) == 9
    for call in calls:
        keyword = next((kw for kw in call.keywords if kw.arg == "allow_graph_fail"), None)
        assert keyword is None or not (isinstance(keyword.value, ast.Constant) and keyword.value.value is True)
        assert all(kw.arg != "use_cuda_graph" for kw in call.keywords)


@pytest.mark.parametrize("function_name", ["run_gdn_context_benchmark", "run_gdn_generation_benchmark"])
def test_trtllm_gdn_phase_runners_end_in_a_strict_completeness_raise(function_name):
    function = _function(function_name)
    # A per-cell `continue` is only acceptable because the function re-raises at the end.
    raises = [
        node
        for node in function.body
        if isinstance(node, ast.If) and any(isinstance(statement, ast.Raise) for statement in node.body)
    ]
    assert raises, f"{function_name} swallows cell failures without a final raise"
    assert "strict completeness" in ast.unparse(raises[-1])


class _Tensor:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def reshape(self, *shape):
        return _Tensor(shape)

    def float(self):
        return self


def _fake_torch():
    def factory(*shape, **_kwargs):
        return _Tensor(shape)

    return SimpleNamespace(
        bfloat16="bf16",
        float32="fp32",
        int64="i64",
        device=lambda name: name,
        cuda=SimpleNamespace(
            set_device=lambda _d: None,
            synchronize=lambda: None,
            empty_cache=lambda: None,
            get_device_name=lambda _d: "fake-gpu",
        ),
        set_default_device=lambda _d: None,
        randn=factory,
        zeros=factory,
        arange=lambda *a, **k: _Tensor((1,)),
        sigmoid=lambda t: t,
        nn=SimpleNamespace(functional=SimpleNamespace(logsigmoid=lambda t: t)),
    )


def _load_context_runner(benchmark_with_power, log_perf):
    namespace = {
        "torch": _fake_torch(),
        "gc": SimpleNamespace(collect=lambda: None),
        "aic_debug": 0,
        "aic_cached_inputs": 1,
        "benchmark_with_power": benchmark_with_power,
        "log_perf": log_perf,
        "causal_conv1d_fn": lambda *a, **k: None,
        "chunk_gated_delta_rule": lambda *a, **k: None,
        "_CHUNK_KERNEL_SOURCE": "chunk_gated_delta_rule_flashinfer",
        "_make_input_pool": None,
    }
    module = ast.Module(body=[_function("run_gdn_context_benchmark")], type_ignores=[])
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["run_gdn_context_benchmark"]


def test_trtllm_gdn_context_capture_failure_fails_the_case_but_keeps_the_other_cells():
    calls = {"bench": 0}
    logged = []

    @contextmanager
    def benchmark_with_power(**_kwargs):
        calls["bench"] += 1
        if calls["bench"] == 3:  # second (batch, seq) cell: the conv capture fails
            raise RuntimeError("CUDA graph capture failed: operation not permitted when stream is capturing")
        yield {"latency_ms": 0.01, "power_stats": None, "used_cuda_graph": True}

    def log_perf(**kwargs):
        logged.append(kwargs["kernel_source"])
        return True

    run = _load_context_runner(benchmark_with_power, log_perf)
    expected = r"strict completeness: ok=1 error=1 skip=0.*seq_len=64: RuntimeError"
    with pytest.raises(RuntimeError, match=expected) as info:
        run(
            d_model=2048,
            d_conv=4,
            num_k_heads=16,
            head_k_dim=128,
            num_v_heads=32,
            head_v_dim=128,
            batch_size_list=[1],
            seq_len_list=[32, 64],
            model_name="Qwen/Qwen3.5-test",
            perf_filename="gdn_perf.txt",
            trtllm_version="1.3.0rc29",
            device="cuda:0",
        )
    assert "CUDA graph capture failed" in str(info.value)
    # the first cell shipped both of its rows; the failed cell shipped none
    assert logged == ["causal_conv1d_fn", "chunk_gated_delta_rule_flashinfer"]


def test_trtllm_gdn_context_all_cells_captured_returns_quietly():
    @contextmanager
    def benchmark_with_power(**_kwargs):
        yield {"latency_ms": 0.01, "power_stats": None, "used_cuda_graph": True}

    run = _load_context_runner(benchmark_with_power, lambda **_k: True)
    with nullcontext():
        run(
            d_model=2048,
            d_conv=4,
            num_k_heads=16,
            head_k_dim=128,
            num_v_heads=32,
            head_v_dim=128,
            batch_size_list=[1, 2],
            seq_len_list=[32],
            model_name="Qwen/Qwen3.5-test",
            perf_filename="gdn_perf.txt",
            trtllm_version="1.3.0rc29",
        )
