# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real quantization callback without importing CUDA packages."""

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from collector.version_resolver import _check_compat

pytestmark = pytest.mark.unit
SOURCE = Path(__file__).resolve().parents[3] / "collector/vllm/collect_computescale.py"


def _callback(dynamic_ms, static_ms, *, failure=None):
    tree = ast.parse(SOURCE.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_computescale")
    calls, published = [], []
    tensor, scale = object(), object()

    def quantize(value, **kwargs):
        assert value is tensor
        calls.append(kwargs)
        if failure is not None:
            raise failure

    durations = iter([dynamic_ms, static_ms])

    @contextmanager
    def benchmark_with_power(**kwargs):
        assert kwargs["repeat_n"] == 1
        kwargs["kernel_func"]()
        yield {"latency_ms": next(durations), "power_stats": None}

    namespace = {
        "torch": SimpleNamespace(
            device=lambda x: x,
            cuda=SimpleNamespace(set_device=lambda x: None, get_device_name=lambda x: "test GPU"),
            set_default_device=lambda x: None,
            randn=lambda *a, **k: tensor,
            tensor=lambda *a, **k: scale,
            bfloat16="bf16",
            float32="fp32",
        ),
        "setup_distributed": lambda x: None,
        "GroupShape": SimpleNamespace(PER_TENSOR=SimpleNamespace(row=-1, col=-1)),
        "ops": SimpleNamespace(scaled_fp8_quant=quantize),
        "benchmark_with_power": benchmark_with_power,
        "log_perf": lambda **kwargs: published.append(kwargs),
        "vllm_version": "0.31.0",
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return (
        lambda: namespace["run_computescale"](17, 4096, perf_filename="computescale_perf.txt"),
        calls,
        published,
        scale,
    )


@pytest.mark.parametrize("dynamic_ms,static_ms,overhead", [(15.0, 5.0, 2.0), (5.0, 15.0, 0.0)])
def test_quantization_modes_timing_normalization_and_clipped_difference(dynamic_ms, static_ms, overhead):
    run, calls, published, scale = _callback(dynamic_ms, static_ms)
    run()
    assert calls[:5] == [{"scale": None, "use_per_token_if_dynamic": True}] * 5
    assert calls[5:] == [{"scale": scale, "group_shape": (-1, -1)}] * 5
    assert [p["op_name"] for p in published] == ["compute_scale", "scale_matrix"]
    assert [p["perf_filename"] for p in published] == ["computescale_perf.txt", "scale_matrix_perf.txt"]
    assert [p["item_list"][0]["latency"] for p in published] == [overhead, static_ms / 5]


def test_quantization_failure_is_not_published_as_success():
    failure = RuntimeError("native quantization failed")
    run, _, published, _ = _callback(1, 1, failure=failure)
    with pytest.raises(RuntimeError, match="native quantization failed"):
        run()
    assert published == []


def test_compatibility_admits_target_but_keeps_unaudited_releases_closed():
    tree = ast.parse(SOURCE.read_text())
    compat = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__compat__" for t in n.targets)
    )
    for version in ["0.24.0", "0.25.0", "0.31.0"]:
        assert _check_compat(compat, version)
    for version in ["0.25.1", "0.26.0", "0.27.0", "0.27.1", "0.28.0", "0.29.0", "0.30.0", "0.31.1"]:
        assert not _check_compat(compat, version)
