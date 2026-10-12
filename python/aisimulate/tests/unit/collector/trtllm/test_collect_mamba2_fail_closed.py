# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The trtllm Mamba2 collector publishes CUDA-graph-timed rows only and never
drops a kernel lane silently.

Until 2026-10-12 every benchmark_with_power call in collector/trtllm/collect_mamba2.py
passed allow_graph_fail=True, and each (batch, seq) cell sat in a bare
`except Exception: continue`. The shipped rc20 mamba2 tables of all eight
systems and the rc29 b200 package contain only the causal_conv1d lanes: the
mamba_chunk_scan_combined / selective_state_update lanes vanished with zero
failure records. Capture failures now raise, every failed cell is counted, and
the case ends as a classified strict-completeness failure (the GDN contract)."""

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE = REPO_ROOT / "collector" / "trtllm" / "collect_mamba2.py"


def _tree():
    return ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))


def _function(name):
    return next(node for node in _tree().body if isinstance(node, ast.FunctionDef) and node.name == name)


def test_trtllm_mamba2_benchmarks_never_fall_back_to_eager_timing():
    calls = [
        node
        for node in ast.walk(_tree())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "benchmark_with_power"
    ]
    assert len(calls) == 8
    for call in calls:
        assert all(keyword.arg not in ("allow_graph_fail", "use_cuda_graph") for keyword in call.keywords)


@pytest.mark.parametrize("function_name", ["run_mamba2_context_benchmark", "run_mamba2_generation_benchmark"])
def test_trtllm_mamba2_phase_runners_end_in_a_strict_completeness_raise(function_name):
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
    """Shape-only stand-in; arithmetic and method calls return another stand-in."""

    def __init__(self, shape=()):
        self.shape = tuple(shape)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *args, **kwargs: _Tensor(self.shape)

    def __neg__(self):
        return _Tensor(self.shape)

    def __add__(self, other):
        return _Tensor(self.shape)

    __radd__ = __add__
    __sub__ = __add__
    __rsub__ = __add__
    __mul__ = __add__
    __rmul__ = __add__


def _factory(*shape, **_kwargs):
    return _Tensor(shape)


def _fake_torch():
    return SimpleNamespace(
        bfloat16="bf16",
        float32="fp32",
        device=lambda name: name,
        set_default_device=lambda _device: None,
        cuda=SimpleNamespace(
            set_device=lambda _device: None,
            synchronize=lambda: None,
            empty_cache=lambda: None,
            get_device_name=lambda _device: "fake-gpu",
        ),
        randn=_factory,
        rand=_factory,
        zeros=_factory,
        empty_like=lambda tensor: _Tensor(tensor.shape),
    )


def _load_runner(name, benchmark_with_power, log_perf):
    namespace = {
        "torch": _fake_torch(),
        "gc": SimpleNamespace(collect=lambda: None),
        "aic_debug": 0,
        "aic_cached_inputs": 1,
        "benchmark_with_power": benchmark_with_power,
        "log_perf": log_perf,
        "repeat": lambda tensor, _pattern, **_axes: tensor,
        "tensorrt_llm": SimpleNamespace(__version__="1.3.0rc29"),
        "causal_conv1d_fn": lambda *args, **kwargs: None,
        "causal_conv1d_update": lambda *args, **kwargs: None,
        "mamba_chunk_scan_combined": lambda *args, **kwargs: None,
        "selective_state_update": lambda *args, **kwargs: None,
        "_token_major_prefill_conv": lambda: False,
        "_prefill_conv_token_major_inputs": None,
        "_run_prefill_conv_token_major": None,
        "_make_input_pool": None,
    }
    module = ast.Module(body=[_function(name)], type_ignores=[])
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace[name]


def _failing_on(call_index):
    """A fake benchmark_with_power whose ``call_index``-th capture fails (0 = never)."""
    calls = {"bench": 0}
    logged = []

    @contextmanager
    def benchmark_with_power(**_kwargs):
        calls["bench"] += 1
        if calls["bench"] == call_index:
            raise RuntimeError("CUDA graph capture failed: operation not permitted when stream is capturing")
        yield {"latency_ms": 0.01, "power_stats": None, "used_cuda_graph": True}

    def log_perf(**kwargs):
        logged.append(kwargs["kernel_source"])
        return True

    return benchmark_with_power, log_perf, logged


_LAYER = {
    "d_model": 4096,
    "d_state": 128,
    "d_conv": 4,
    "nheads": 128,
    "head_dim": 64,
    "n_groups": 8,
    "chunk_size": 256,
    "model_name": "nvidia/Nemotron-test",
    "perf_filename": "mamba2_perf.txt",
    "device": "cuda:0",
}


def test_trtllm_mamba2_context_scan_capture_failure_fails_the_case_instead_of_dropping_the_lane():
    # The shipped-data failure mode: a cell's conv row ships, then its scan lane raises.
    benchmark_with_power, log_perf, logged = _failing_on(2)
    run = _load_runner("run_mamba2_context_benchmark", benchmark_with_power, log_perf)
    expected = r"strict completeness: ok=1 error=1 skip=0.*seq_len=32: RuntimeError"
    with pytest.raises(RuntimeError, match=expected) as info:
        run(batch_size_list=[1], seq_len_list=[32, 64], **_LAYER)
    assert "CUDA graph capture failed" in str(info.value)
    # cell (1, 32) shipped its conv row and lost the scan; cell (1, 64) shipped both
    assert logged == ["causal_conv1d_fn", "causal_conv1d_fn", "mamba_chunk_scan_combined"]


def test_trtllm_mamba2_context_all_cells_captured_returns_quietly():
    benchmark_with_power, log_perf, logged = _failing_on(0)
    run = _load_runner("run_mamba2_context_benchmark", benchmark_with_power, log_perf)
    assert run(batch_size_list=[1, 2], seq_len_list=[32], **_LAYER) is None
    assert logged == ["causal_conv1d_fn", "mamba_chunk_scan_combined"] * 2


def test_trtllm_mamba2_generation_state_update_capture_failure_fails_the_case():
    benchmark_with_power, log_perf, logged = _failing_on(4)
    run = _load_runner("run_mamba2_generation_benchmark", benchmark_with_power, log_perf)
    expected = r"strict completeness: ok=1 error=1 skip=0.*batch_size=2: RuntimeError"
    with pytest.raises(RuntimeError, match=expected):
        run(batch_size_list=[1, 2], **_LAYER)
    assert logged == ["causal_conv1d_update", "selective_state_update", "causal_conv1d_update"]


def test_trtllm_mamba2_generation_all_cells_captured_returns_quietly():
    benchmark_with_power, log_perf, logged = _failing_on(0)
    run = _load_runner("run_mamba2_generation_benchmark", benchmark_with_power, log_perf)
    assert run(batch_size_list=[1], **_LAYER) is None
    assert logged == ["causal_conv1d_update", "selective_state_update"]
