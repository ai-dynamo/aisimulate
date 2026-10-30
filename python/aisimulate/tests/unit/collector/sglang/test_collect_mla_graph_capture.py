# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""sglang mla_context on SM100 is timed under CUDA-graph replay again.

Between 2026-10-05 and 2026-10-11 the trtllm_mla PREFILL lane was timed eagerly
because flashinfer 0.6.18 refuses to capture trtllm_ragged_attention_deepseek
without the CPU length mirrors sglang 0.5.21 does not pass; the eager rows
floored at the Python launch loop (B200 0.13 ms, GB200 0.29 ms for every
isl <= 1024 cell). The collector now hands flashinfer the mirrors from
sglang's own extend_seq_lens_cpu and refuses to publish any eager result."""

import ast
import contextlib
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE = REPO_ROOT / "collector" / "sglang" / "collect_mla.py"
FORWARD_MODE = SimpleNamespace(EXTEND="extend", DECODE="decode")


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


def _fake_torch():
    return SimpleNamespace(
        int32="int32",
        tensor=lambda data, dtype=None, device=None: SimpleNamespace(values=tuple(data), dtype=dtype, device=device),
    )


def _install_fake_flashinfer(monkeypatch, *, accepts_cpu_lens):
    calls = []

    if accepts_cpu_lens:

        def trtllm_ragged_attention_deepseek(
            *, query, cum_seq_lens_q, cum_seq_lens_kv, q_seq_lens_cpu=None, kv_seq_lens_cpu=None
        ):
            calls.append({"q_seq_lens_cpu": q_seq_lens_cpu, "kv_seq_lens_cpu": kv_seq_lens_cpu})
            return "out"

    else:

        def trtllm_ragged_attention_deepseek(*, query, cum_seq_lens_q, cum_seq_lens_kv):
            calls.append({})
            return "out"

    prefill = SimpleNamespace(trtllm_ragged_attention_deepseek=trtllm_ragged_attention_deepseek)
    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(prefill=prefill))
    monkeypatch.setitem(sys.modules, "flashinfer.prefill", prefill)
    return prefill, calls, trtllm_ragged_attention_deepseek


class _ServingBackend:
    """Stand-in for sglang 0.5.21's TRTLLMMLABackend: metadata init + the kernel hook."""

    def __init__(self):
        self.metadata_batches = []

    def init_forward_metadata(self, forward_batch):
        self.metadata_batches.append(forward_batch)

    def _run_prefill_kernel(self, *, q, cum_seq_lens_q, cum_seq_lens_kv, **_rest):
        # sglang calls flashinfer through the package attribute at call time
        # (trtllm_mla_backend.py:1121 @v0.5.21), without the CPU mirrors.
        import flashinfer

        return flashinfer.prefill.trtllm_ragged_attention_deepseek(
            query=q, cum_seq_lens_q=cum_seq_lens_q, cum_seq_lens_kv=cum_seq_lens_kv
        )


def _backend():
    loaded = _load(
        ["_extend_seq_lens_cpu", "_ragged_prefill_with_cpu_seq_lens", "_GraphCapturableTRTLLMMLABackend"],
        {
            "torch": _fake_torch(),
            "ForwardMode": FORWARD_MODE,
            "TRTLLMMLABackend": _ServingBackend,
            "contextlib": contextlib,
            "inspect": inspect,
        },
    )
    return loaded["_GraphCapturableTRTLLMMLABackend"]()


def _extend_batch(lens):
    return SimpleNamespace(forward_mode=FORWARD_MODE.EXTEND, extend_seq_lens_cpu=list(lens))


def test_no_prefix_prefill_call_carries_the_cpu_length_mirrors(monkeypatch):
    prefill, calls, original = _install_fake_flashinfer(monkeypatch, accepts_cpu_lens=True)
    backend = _backend()
    backend.init_forward_metadata(_extend_batch([512, 512]))
    cum = object()

    assert backend._run_prefill_kernel(q="q", cum_seq_lens_q=cum, cum_seq_lens_kv=cum) == "out"

    (call,) = calls
    assert call["q_seq_lens_cpu"].values == (512, 512)
    assert call["q_seq_lens_cpu"].dtype == "int32"
    assert call["q_seq_lens_cpu"].device == "cpu"
    assert call["kv_seq_lens_cpu"] is call["q_seq_lens_cpu"]
    # the patch is scoped to the call: serving's function is back afterwards
    assert prefill.trtllm_ragged_attention_deepseek is original


def test_chunked_prefix_call_goes_through_sglang_unchanged(monkeypatch):
    _, calls, _ = _install_fake_flashinfer(monkeypatch, accepts_cpu_lens=True)
    backend = _backend()
    backend.init_forward_metadata(_extend_batch([512]))

    backend._run_prefill_kernel(q="q", cum_seq_lens_q=object(), cum_seq_lens_kv=object())

    assert calls == [{"q_seq_lens_cpu": None, "kv_seq_lens_cpu": None}]


def test_decode_metadata_leaves_the_prefill_hook_inert(monkeypatch):
    _, calls, _ = _install_fake_flashinfer(monkeypatch, accepts_cpu_lens=True)
    backend = _backend()
    backend.init_forward_metadata(SimpleNamespace(forward_mode=FORWARD_MODE.DECODE, seq_lens_cpu=[9]))
    cum = object()

    backend._run_prefill_kernel(q="q", cum_seq_lens_q=cum, cum_seq_lens_kv=cum)

    assert calls == [{"q_seq_lens_cpu": None, "kv_seq_lens_cpu": None}]


def test_flashinfer_without_the_parameters_is_called_as_is(monkeypatch):
    prefill, calls, original = _install_fake_flashinfer(monkeypatch, accepts_cpu_lens=False)
    backend = _backend()
    backend.init_forward_metadata(_extend_batch([512]))
    cum = object()

    assert backend._run_prefill_kernel(q="q", cum_seq_lens_q=cum, cum_seq_lens_kv=cum) == "out"
    assert calls == [{}]
    assert prefill.trtllm_ragged_attention_deepseek is original


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
    forward_batch = SimpleNamespace(attn_backend=None)

    with pytest.raises(RuntimeError, match="refusing to publish eager timing"):
        benchmark_layer(lambda *a, **k: None, forward_batch, SimpleNamespace(device="cuda:0"), None, None, None, None)

    # graph timing is the helper default; the lane must not opt out of it
    assert "use_cuda_graph" not in benchmark_kwargs
    assert benchmark_kwargs.get("allow_graph_fail") is not True


def test_run_mla_has_no_eager_prefill_switch():
    source = SOURCE.read_text(encoding="utf-8")
    assert "_eager_prefill" not in source
    assert "use_cuda_graph=False" not in source
    run_mla = next(node for node in _tree().body if isinstance(node, ast.FunctionDef) and node.name == "run_mla")
    constructed = {
        ast.unparse(node.func)
        for node in ast.walk(run_mla)
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("TRTLLMMLABackend")
    }
    assert constructed == {"_GraphCapturableTRTLLMMLABackend"}
