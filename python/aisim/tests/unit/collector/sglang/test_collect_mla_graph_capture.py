# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""sglang mla_context / mla_generation are timed under CUDA-graph replay only.

Between 2026-10-05 and 2026-10-11 the trtllm_mla PREFILL lane was timed eagerly
because flashinfer 0.6.18 refused to capture trtllm_ragged_attention_deepseek
without CPU length mirrors; the eager rows floored at the Python launch loop
(B200 0.13 ms, GB200 0.29 ms for every isl <= 1024 cell). flashinfer 0.7.2
(sglang 0.5.23's pin) assumes every row active by default and captures the
call sglang makes (prefill.py:6115, :6472-6491 @v0.7.2), so the collector runs
sglang's own TRTLLMMLABackend and still refuses to publish any eager result."""

import ast
import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE = REPO_ROOT / "collector" / "sglang" / "collect_mla.py"


def _tree():
    return ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))


def _load(names, namespace):
    selected = [
        node for node in _tree().body if isinstance(node, ast.FunctionDef | ast.ClassDef) and node.name in names
    ]
    assert len(selected) == len(names)
    loaded = dict(namespace)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), loaded)
    return loaded


def test_benchmark_layer_refuses_eager_timing():
    benchmark_kwargs = {}

    @contextlib.contextmanager
    def benchmark_with_power(**kwargs):
        benchmark_kwargs.update(kwargs)
        yield {"latency_ms": 0.13, "power_stats": None, "used_cuda_graph": False}

    benchmark_layer = _load(
        ["benchmark_layer"],
        {
            "benchmark_with_power": benchmark_with_power,
            "forward_context": lambda _ctx: contextlib.nullcontext(),
            "ForwardContext": lambda **_kw: None,
        },
    )["benchmark_layer"]

    with pytest.raises(RuntimeError, match="refusing to publish eager timing"):
        benchmark_layer(lambda *a, **k: None, object(), SimpleNamespace(), SimpleNamespace(device="cuda:0"), None, None)

    # graph timing is the helper default; the lane must not opt out of it
    assert "use_cuda_graph" not in benchmark_kwargs
    assert benchmark_kwargs.get("allow_graph_fail") is not True


def test_benchmark_layer_times_the_layer_call_inside_the_backend_context():
    calls = []
    contexts = []

    @contextlib.contextmanager
    def benchmark_with_power(*, kernel_func, **_kwargs):
        kernel_func()
        yield {"latency_ms": 1.0, "power_stats": None, "used_cuda_graph": True}

    @contextlib.contextmanager
    def forward_context(ctx):
        contexts.append(ctx)
        yield

    benchmark_layer = _load(
        ["benchmark_layer"],
        {
            "benchmark_with_power": benchmark_with_power,
            "forward_context": forward_context,
            "ForwardContext": lambda **kw: kw,
        },
    )["benchmark_layer"]
    backend, batch = object(), object()
    q = SimpleNamespace(device="cuda:0")

    latency, _ = benchmark_layer(lambda *a, **k: calls.append((a, k)), backend, batch, q, "k", "v", save_kv_cache=False)

    assert latency == 1.0
    assert contexts == [{"attn_backend": backend}]
    assert calls == [((q, "k", "v", batch), {"save_kv_cache": False})]


def test_run_mla_has_no_eager_prefill_switch():
    source = SOURCE.read_text(encoding="utf-8")
    assert "use_cuda_graph=False" not in source
    run_mla = next(node for node in _tree().body if isinstance(node, ast.FunctionDef) and node.name == "run_mla")
    constructed = {
        ast.unparse(node.func)
        for node in ast.walk(run_mla)
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("TRTLLMMLABackend")
    }
    assert constructed == {"TRTLLMMLABackend"}
    # No collector subclass rewires how sglang calls its kernels.
    assert not [
        node.name
        for node in _tree().body
        if isinstance(node, ast.ClassDef) and any(ast.unparse(base) == "TRTLLMMLABackend" for base in node.bases)
    ]
