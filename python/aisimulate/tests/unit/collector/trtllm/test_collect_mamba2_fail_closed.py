# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The trtllm Mamba2 collector publishes CUDA-graph-timed rows only, never
drops a kernel lane silently, and runs the SSM kernels the way serving does.

Until 2026-10-12 every benchmark_with_power call in collector/trtllm/collect_mamba2.py
passed allow_graph_fail=True, each (batch, seq) cell sat in a bare
`except Exception: continue`, and neither SSM kernel received the caller-owned
`out` buffer TRT-LLM requires (ssd_chunk_scan.py:667, selective_state_update.py
`out.dim()` @1.3.0rc20/rc29). The shipped rc20 mamba2 tables of all eight
systems and the rc29 b200 package contain only the causal_conv1d lanes: the
mamba_chunk_scan_combined / selective_state_update lanes vanished with zero
failure records. Capture failures now raise, every failed cell is counted, the
case ends as a classified strict-completeness failure (the GDN contract), both
SSM calls carry `out=`, and the decode state update is the kernel serving's
Mamba2Mixer gate selects (FlashInfer or TRT-LLM's Triton), labelled as such."""

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


def _callee(node):
    return ast.unparse(node.func)


def test_trtllm_mamba2_benchmarks_never_fall_back_to_eager_timing():
    calls = [
        node for node in ast.walk(_tree()) if isinstance(node, ast.Call) and _callee(node) == "benchmark_with_power"
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


def test_every_ssm_kernel_call_hands_over_the_caller_owned_out_buffer():
    # TRT-LLM's chunk scan asserts on `out` and its state update calls
    # `out.dim()`; the missing keyword is what emptied the SSM lanes.
    scan_calls = [
        node
        for node in ast.walk(_function("run_mamba2_context_benchmark"))
        if isinstance(node, ast.Call) and _callee(node) == "mamba_chunk_scan_combined"
    ]
    update_calls = [
        node
        for node in ast.walk(_function("run_mamba2_generation_benchmark"))
        if isinstance(node, ast.Call) and _callee(node) == "ssu_func"
    ]
    assert len(scan_calls) == 4  # cached + pooled, warm-up call + timed closure
    assert len(update_calls) == 4
    for call in scan_calls + update_calls:
        assert any(keyword.arg == "out" for keyword in call.keywords), ast.unparse(call)
    for call in update_calls:
        assert any(keyword.arg == "state_batch_indices" for keyword in call.keywords), ast.unparse(call)


def test_generation_runner_never_calls_the_triton_state_update_directly():
    # The decode kernel comes from the serving gate, never from a hard-coded name.
    direct = [
        node
        for node in ast.walk(_function("run_mamba2_generation_benchmark"))
        if isinstance(node, ast.Call) and _callee(node) == "selective_state_update"
    ]
    assert direct == []
    for literal in ("selective_state_update", "selective_state_update_flashinfer", "selective_state_update_triton"):
        assert all(
            not (isinstance(node, ast.Constant) and node.value == literal)
            for node in ast.walk(_function("run_mamba2_generation_benchmark"))
        ), f"{literal!r} is hard-coded in the generation runner"


def _load_gate():
    tree = _tree()
    body = [
        node
        for node in tree.body
        if (isinstance(node, ast.Assign) and ast.unparse(node.targets[0]).startswith("_FLASHINFER_SSU_"))
        or (isinstance(node, ast.FunctionDef) and node.name == "_serving_selects_flashinfer_ssu")
    ]
    namespace = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["_serving_selects_flashinfer_ssu"]


@pytest.mark.parametrize(
    ("nheads", "n_groups", "head_dim", "d_state", "flashinfer"),
    [
        (64, 8, 64, 128, True),  # Nemotron-3 Nano / generic 4K: ratio 8
        (256, 8, 64, 256, True),  # Nemotron-H 56B: ratio 32, d_state 256
        (16, 4, 64, 64, True),  # generic 1K: ratio 4 (rc29 widened the list; rc20 had [1, 8, 16])
        (64, 8, 80, 128, False),  # Nemotron-v2-Nano head_dim 80: no FlashInfer kernel
        (48, 16, 64, 128, False),  # ratio 3
        (64, 8, 64, 32, False),  # d_state 32
        (64, 0, 64, 128, False),  # no groups
    ],
)
def test_decode_kernel_gate_mirrors_the_rc29_mixer(nheads, n_groups, head_dim, d_state, flashinfer):
    assert _load_gate()(nheads, n_groups, head_dim, d_state) is flashinfer


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
        int32="i32",
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
        arange=lambda n, **_kwargs: _Tensor((n,)),
        empty_like=lambda tensor: _Tensor(tensor.shape),
    )


def _fake_input_pool(shapes, count, _dtype, _device):
    return {name: [_Tensor(shape) for _ in range(count)] for name, shape in shapes.items()}


def _load_runner(
    name, benchmark_with_power, log_perf, *, cached_inputs=1, ssu_label="selective_state_update_flashinfer"
):
    ssu_calls = []

    def ssu(*args, **kwargs):
        ssu_calls.append(kwargs)

    namespace = {
        "torch": _fake_torch(),
        "gc": SimpleNamespace(collect=lambda: None),
        "aic_debug": 0,
        "aic_cached_inputs": cached_inputs,
        "benchmark_with_power": benchmark_with_power,
        "log_perf": log_perf,
        "repeat": lambda tensor, _pattern, **_axes: tensor,
        "tensorrt_llm": SimpleNamespace(__version__="1.3.0rc29"),
        "causal_conv1d_fn": lambda *args, **kwargs: None,
        "causal_conv1d_update": lambda *args, **kwargs: None,
        "mamba_chunk_scan_combined": lambda *args, **kwargs: None,
        "selective_state_update": lambda *args, **kwargs: None,
        "_decode_state_update_kernel": lambda *geometry: (ssu, ssu_label),
        "_decode_dt": lambda batch_size, nheads, head_dim, _device: _Tensor((batch_size, nheads, head_dim)),
        "_token_major_prefill_conv": lambda: False,
        "_prefill_conv_token_major_inputs": None,
        "_run_prefill_conv_token_major": None,
        "_make_input_pool": _fake_input_pool,
    }
    module = ast.Module(body=[_function(name)], type_ignores=[])
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace[name], ssu_calls


def _failing_on(call_index):
    """A fake benchmark_with_power whose ``call_index``-th capture fails (0 = never)."""
    calls = {"bench": 0}
    logged = []

    @contextmanager
    def benchmark_with_power(**kwargs):
        calls["bench"] += 1
        if calls["bench"] == call_index:
            raise RuntimeError("CUDA graph capture failed: operation not permitted when stream is capturing")
        kwargs["kernel_func"]()  # the real helper runs the closure for its warmups and the capture
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
    run, _ = _load_runner("run_mamba2_context_benchmark", benchmark_with_power, log_perf)
    expected = r"strict completeness: ok=1 error=1 skip=0.*seq_len=32: RuntimeError"
    with pytest.raises(RuntimeError, match=expected) as info:
        run(batch_size_list=[1], seq_len_list=[32, 64], **_LAYER)
    assert "CUDA graph capture failed" in str(info.value)
    # cell (1, 32) shipped its conv row and lost the scan; cell (1, 64) shipped both
    assert logged == ["causal_conv1d_fn", "causal_conv1d_fn", "mamba_chunk_scan_combined"]


@pytest.mark.parametrize("cached_inputs", [1, 0])
def test_trtllm_mamba2_context_all_cells_captured_returns_quietly(cached_inputs):
    benchmark_with_power, log_perf, logged = _failing_on(0)
    run, _ = _load_runner("run_mamba2_context_benchmark", benchmark_with_power, log_perf, cached_inputs=cached_inputs)
    assert run(batch_size_list=[1, 2], seq_len_list=[32], **_LAYER) is None
    assert logged == ["causal_conv1d_fn", "mamba_chunk_scan_combined"] * 2


def test_trtllm_mamba2_generation_state_update_capture_failure_fails_the_case():
    benchmark_with_power, log_perf, logged = _failing_on(4)
    run, _ = _load_runner("run_mamba2_generation_benchmark", benchmark_with_power, log_perf)
    expected = r"strict completeness: ok=1 error=1 skip=0.*batch_size=2: RuntimeError"
    with pytest.raises(RuntimeError, match=expected):
        run(batch_size_list=[1, 2], **_LAYER)
    assert logged == ["causal_conv1d_update", "selective_state_update_flashinfer", "causal_conv1d_update"]


@pytest.mark.parametrize("cached_inputs", [1, 0])
@pytest.mark.parametrize("ssu_label", ["selective_state_update_flashinfer", "selective_state_update_triton"])
def test_trtllm_mamba2_generation_labels_the_rows_with_the_kernel_serving_selected(cached_inputs, ssu_label):
    benchmark_with_power, log_perf, logged = _failing_on(0)
    run, ssu_calls = _load_runner(
        "run_mamba2_generation_benchmark",
        benchmark_with_power,
        log_perf,
        cached_inputs=cached_inputs,
        ssu_label=ssu_label,
    )
    assert run(batch_size_list=[1], **_LAYER) is None
    assert logged == ["causal_conv1d_update", ssu_label]
    # warm-up call + the timed closure; every call carries the serving-shaped extras
    assert len(ssu_calls) == 2
    for kwargs in ssu_calls:
        assert kwargs["out"].shape == (1, 128, 64)
        assert kwargs["state_batch_indices"].shape == (1,)
        assert kwargs["dt_softplus"] is True and kwargs["z"] is None


def test_generation_runner_builds_dt_only_through_the_serving_layout_helper():
    # Both input modes take dt from _decode_dt; no dense (batch, nheads, head_dim) dt is made.
    runner = _function("run_mamba2_generation_benchmark")
    helper_calls = [node for node in ast.walk(runner) if isinstance(node, ast.Call) and _callee(node) == "_decode_dt"]
    assert len(helper_calls) == 2
    dt_assignments = [
        node
        for node in ast.walk(runner)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "dt" for t in node.targets)
    ]
    assert dt_assignments, "cached-input mode no longer binds dt"
    for assignment in dt_assignments:
        assert isinstance(assignment.value, ast.Call) and _callee(assignment.value) == "_decode_dt"


def test_decode_dt_is_an_fp32_per_head_tensor_broadcast_over_head_dim():
    # FlashInfer's binding requires dt.stride(1) == 1 and dt.stride(2) == 0
    # (csrc/selective_state_update.cu:187-188 @v0.6.18); serving gets there with
    # fp32 (batch, nheads) + einops.repeat "b h -> b h p" (mamba2_mixer.py:764 @rc29).
    seen = {}

    def randn(*shape, dtype=None, device=None):
        seen["randn"] = (shape, dtype)
        return _Tensor(shape)

    def repeat(tensor, pattern, **axes):
        seen["repeat"] = (tensor.shape, pattern, axes)
        return _Tensor(tensor.shape + (axes["p"],))

    namespace = {"torch": SimpleNamespace(randn=randn, float32="fp32", Tensor=object), "repeat": repeat}
    module = ast.Module(body=[_function("_decode_dt")], type_ignores=[])
    exec(compile(module, str(SOURCE), "exec"), namespace)
    dt = namespace["_decode_dt"](4, 128, 64, "cuda:0")
    assert seen["randn"] == ((4, 128), "fp32")
    assert seen["repeat"] == ((4, 128), "b h -> b h p", {"p": 64})
    assert dt.shape == (4, 128, 64)
