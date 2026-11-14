# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SGLang GEMM and compute_scale collectors call serving's own entry points.

``collect_gemm.py`` and ``collect_computescale.py`` import sglang and torch at
module scope and run only on a GPU node, so these tests AST-extract the real
function bodies and run them against recorders, the technique
``tests/unit/collector/test_collect_gemm_fp8block_prepack.py`` uses. Each
sandbox defines only the names the serving path may call; any other name
raises ``NameError``.
"""

import ast
import contextlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
GEMM_PATH = REPO_ROOT / "collector" / "sglang" / "collect_gemm.py"
COMPUTESCALE_PATH = REPO_ROOT / "collector" / "sglang" / "collect_computescale.py"

BFLOAT16, FLOAT32, FLOAT8 = "bfloat16", "float32", "float8_e4m3fn"


def _exec(nodes, path: Path, namespace: dict) -> dict:
    loaded = dict(namespace)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), loaded)
    return loaded


def _top_level(path: Path, *names: str, namespace: dict) -> dict:
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    return _exec(nodes, path, namespace)


def _create_gemm(namespace: dict):
    """``create_gemm`` is nested in ``run_gemm``; its free variables become globals."""
    tree = ast.parse(GEMM_PATH.read_text(), filename=str(GEMM_PATH))
    run_gemm = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_gemm")
    create_gemm = next(
        node for node in ast.walk(run_gemm) if isinstance(node, ast.FunctionDef) and node.name == "create_gemm"
    )
    return _exec([create_gemm], GEMM_PATH, namespace)["create_gemm"]


class _Tensor:
    """Shape/dtype stand-in; values are never read."""

    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype

    def clamp(self, min=None, max=None):  # noqa: A002 - mirrors torch.Tensor.clamp
        return self

    def to(self, dtype):
        return _Tensor(self.shape, dtype)

    def t(self):
        return _Tensor(self.shape[::-1], self.dtype)

    def view(self, dtype):
        return _Tensor(self.shape, dtype)

    def __sub__(self, _other):
        return self

    def __mul__(self, _other):
        return self

    __rmul__ = __mul__

    def __rtruediv__(self, _other):
        return self


def _shape(shape):
    return tuple(shape[0]) if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else tuple(shape)


def _fake_torch():
    return SimpleNamespace(
        bfloat16=BFLOAT16,
        float32=FLOAT32,
        float8_e4m3fn=FLOAT8,
        finfo=lambda _dtype: SimpleNamespace(min=-448.0, max=448.0),
        rand=lambda *shape, device=None: _Tensor(_shape(shape), FLOAT32),
        randn=lambda *shape, device=None, dtype=None: _Tensor(_shape(shape), dtype),
        empty=lambda *shape, device=None, dtype=None: _Tensor(_shape(shape), dtype),
        tensor=lambda values, device=None, dtype=None: _Tensor((len(values),), dtype),
        nn=SimpleNamespace(Module=_Module, Parameter=_Parameter),
    )


class _Parameter:
    def __init__(self, data, requires_grad=True):
        self.data = data
        self.shape = data.shape
        self.dtype = data.dtype
        self.requires_grad = requires_grad


class _Module:
    def register_parameter(self, name, param):
        setattr(self, name, param)


# --- fp8: apply_fp8_linear and the GEMM it ran -------------------------------


def test_fp8_gemm_op_is_serving_apply_fp8_linear_with_a_per_channel_weight():
    calls = []

    def apply_fp8_linear(**kwargs):
        calls.append(kwargs)
        return "out"

    gemm_op = _create_gemm(
        {
            "torch": _fake_torch(),
            "apply_fp8_linear": apply_fp8_linear,
            "gemm_type": "fp8",
            "M": 16,
            "N": 256,
            "K": 512,
            "device": "cpu",
        }
    )()

    assert gemm_op() == "out"
    (call,) = calls
    assert call["input"].shape == (16, 512) and call["input"].dtype == BFLOAT16
    # Fp8LinearMethod stores the [N, K] FP8 weight as its [K, N] view and an [N, 1] scale.
    assert call["weight"].shape == (512, 256) and call["weight"].dtype == FLOAT8
    assert call["weight_scale"].shape == (256, 1) and call["weight_scale"].dtype == FLOAT32
    assert call["input_scale"] is None
    assert call["bias"] is None
    assert call["cutlass_fp8_supported"] is True
    assert call["use_per_token_if_dynamic"] is False


class _Trace:
    def __init__(self, backends):
        self.backends = backends
        self.enabled = False
        self.records = []
        self.events = []

    def namespace(self):
        def clear():
            self.events.append("clear")
            self.records.clear()

        def enable():
            self.events.append("enable")
            self.enabled = True

        def disable():
            self.events.append("disable")
            self.enabled = False

        return {
            "clear_fused_op_trace": clear,
            "enable_fused_op_trace": enable,
            "disable_fused_op_trace": disable,
            "get_fused_op_trace": lambda: list(self.records),
        }

    def gemm_op(self):
        assert self.enabled
        self.records.append(SimpleNamespace(op="quantization.other", backend="jit"))
        self.records.extend(SimpleNamespace(op="gemm.fp8_scaled_mm", backend=b) for b in self.backends)


@pytest.mark.parametrize(
    ("backends", "label"),
    [
        (["aot"], "sglang_sgl_kernel_fp8_scaled_mm"),
        (["torch"], "sglang_torch_scaled_mm"),
        ([], "sglang_triton_scaled_mm"),
    ],
)
def test_fp8_label_names_the_backend_the_fused_op_trace_recorded(backends, label):
    trace = _Trace(backends)
    probe = _top_level(GEMM_PATH, "_fp8_linear_kernel_source", namespace=trace.namespace())["_fp8_linear_kernel_source"]

    assert probe(trace.gemm_op) == label
    assert trace.enabled is False
    assert trace.records == []
    assert trace.events[-2:] == ["disable", "clear"]


def test_fp8_label_refuses_a_mixed_backend_trace():
    trace = _Trace(["aot", "torch"])
    probe = _top_level(GEMM_PATH, "_fp8_linear_kernel_source", namespace=trace.namespace())["_fp8_linear_kernel_source"]

    with pytest.raises(RuntimeError, match="unexpected backends"):
        probe(trace.gemm_op)
    assert trace.enabled is False


def test_fp8_label_probe_turns_tracing_off_when_the_call_raises():
    trace = _Trace([])

    def failing_op():
        raise ValueError("kernel failed")

    probe = _top_level(GEMM_PATH, "_fp8_linear_kernel_source", namespace=trace.namespace())["_fp8_linear_kernel_source"]
    with pytest.raises(ValueError, match="kernel failed"):
        probe(failing_op)
    assert trace.enabled is False


# --- nvfp4: serving's activation quant, then the SM120 KDA kernel or mm_fp4 ---


def _round_up(x, m):
    return -(-x // m) * m


def _nvfp4_padding(calls: list) -> dict:
    """serving's modelopt_quant padding helpers, by shape: N and K to 32 (K packed 2 per byte)."""

    def pad_nvfp4_weight(w):
        rows, col_bytes = w.shape
        padded = _Tensor((_round_up(rows, 32), _round_up(col_bytes * 2, 32) // 2), w.dtype)
        calls.append(("pad_weight", w.shape, padded.shape))
        return padded, padded.shape[1] - col_bytes

    def pad_nvfp4_activation_for_cutlass(x, cols):
        calls.append(("pad_activation", cols))
        return _Tensor((x.shape[0], x.shape[1] + cols), x.dtype)

    def slice_nvfp4_output(out, n):
        calls.append(("slice", n))
        return out

    return {
        "pad_nvfp4_weight": pad_nvfp4_weight,
        "pad_nvfp4_activation_for_cutlass": pad_nvfp4_activation_for_cutlass,
        "slice_nvfp4_output": slice_nvfp4_output,
    }


def _nvfp4_gemm_op(kda_result, ran: set, calls: list, fp4_backend="cutlass", n=256, k=512):
    def flashinfer_fp4_quantize(x, global_scale, is_sf_swizzled_layout):
        calls.append(("flashinfer_fp4_quantize", x.shape))
        rows, k = x.shape
        return _Tensor((rows, k // 2), "uint8"), _Tensor((rows, k // 16), "uint8")

    def sglang_fp4_quantize(x, global_scale):
        calls.append(("sglang_fp4_quantize", x.shape))
        rows, k = x.shape
        return _Tensor((rows, k // 2), "uint8"), _Tensor((rows, k // 16), "uint8")

    def try_qwen3x_nvfp4_gemm(a_fp4, b_fp4, a_sf, b_sf, alpha, dtype, out_features):
        calls.append(("kda", b_fp4.shape, b_sf.shape, b_sf.dtype, out_features))
        return kda_result

    def flashinfer_mm_fp4(a_fp4, b_fp4, a_sf, b_sf, alpha, dtype, *, backend):
        calls.append(("mm_fp4", backend, a_fp4.shape, b_fp4.shape))
        return "mm-out"

    return _create_gemm(
        {
            **_nvfp4_padding(calls),
            "torch": _fake_torch(),
            "flashinfer_fp4_quantize": flashinfer_fp4_quantize,
            "sglang_fp4_quantize": sglang_fp4_quantize,
            "try_qwen3x_nvfp4_gemm": try_qwen3x_nvfp4_gemm,
            "flashinfer_mm_fp4": flashinfer_mm_fp4,
            "nvfp4_ran": ran,
            "fp4_backend": fp4_backend,
            "gemm_type": "nvfp4",
            "M": 4,
            "N": n,
            "K": k,
            "device": "cpu",
        }
    )()


def test_nvfp4_gemm_op_quantizes_activations_with_serving_fp4_quantize_then_runs_mm_fp4():
    ran, calls = set(), []
    gemm_op = _nvfp4_gemm_op(None, ran, calls, fp4_backend="cute-dsl")
    # Setup quantizes the weight once with FlashInfer's own quantizer, then pads it
    # as process_weights_after_loading does (aligned here: no padding).
    assert calls == [("flashinfer_fp4_quantize", (256, 512)), ("pad_weight", (256, 256), (256, 256))]
    calls.clear()

    assert gemm_op() == "mm-out"

    assert calls[0] == ("sglang_fp4_quantize", (4, 512))
    assert calls[1] == ("pad_activation", 0)
    # fp4_gemm tries the KDA kernel first with the [K/2, N] weight and its
    # float8_e4m3fn [K/16, N] scale view, as ModelOptFp4LinearMethod.apply passes them.
    assert calls[2] == ("kda", (256, 256), (32, 256), FLOAT8, 256)
    assert calls[3] == ("mm_fp4", "cute-dsl", (4, 256), (256, 256))
    assert calls[4] == ("slice", 256)
    assert ran == {"cute-dsl"}


def test_nvfp4_gemm_op_pads_a_misaligned_shape_as_serving_does():
    # n=48, k=48: serving pads the packed weight to N=64, K=64 (32 bytes), pads the
    # activation by the same 8 bytes, runs the GEMM at N=64 and slices back to 48.
    ran, calls = set(), []
    gemm_op = _nvfp4_gemm_op(None, ran, calls, n=48, k=48)
    assert calls[-1] == ("pad_weight", (48, 24), (64, 32))
    calls.clear()

    gemm_op()

    assert calls[1] == ("pad_activation", 8)
    assert calls[2][0] == "kda" and calls[2][1] == (32, 64) and calls[2][-1] == 64
    assert calls[3] == ("mm_fp4", "cutlass", (4, 32), (32, 64))
    assert calls[4] == ("slice", 48)


def test_nvfp4_gemm_op_returns_the_kda_output_when_serving_would():
    ran, calls = set(), []
    gemm_op = _nvfp4_gemm_op("kda-out", ran, calls)
    calls.clear()

    assert gemm_op() == "kda-out"
    assert [c[0] for c in calls] == ["sglang_fp4_quantize", "pad_activation", "kda", "slice"]
    assert ran == {"kda"}


# --- compute_scale: serving's dynamic and static quantization ----------------


def _benchmark_recorder():
    from contextlib import contextmanager

    @contextmanager
    def benchmark_with_power(device, kernel_func, repeat_n):
        kernel_func()
        yield {"latency_ms": 5.0, "power_stats": None}

    return benchmark_with_power


def test_computescale_dynamic_quant_is_serving_sglang_per_token_quant_fp8():
    calls = []
    bench = _top_level(
        COMPUTESCALE_PATH,
        "_bench_dynamic",
        namespace={
            "_OUTSIDE_LOOP_COUNT": 5,
            "benchmark_with_power": _benchmark_recorder(),
            "sglang_per_token_quant_fp8": lambda x: calls.append(x),
        },
    )["_bench_dynamic"]

    latency, _ = bench("cuda:0", "x")

    assert calls == ["x"] * 5
    assert latency == 1.0


@pytest.mark.parametrize(("sm", "repeat_scale"), [(89, True), (90, False), (100, False), (103, False), (120, False)])
def test_computescale_static_quant_is_serving_static_quant_fp8(sm, repeat_scale):
    calls = []
    bench = _top_level(
        COMPUTESCALE_PATH,
        "_bench_static",
        namespace={
            "_OUTSIDE_LOOP_COUNT": 5,
            "benchmark_with_power": _benchmark_recorder(),
            "get_sm_version": lambda: sm,
            "static_quant_fp8": lambda x, scale, repeat_scale: calls.append((x, scale.shape, repeat_scale)),
            "torch": _fake_torch(),
        },
    )["_bench_static"]

    bench("cuda:0", "x")

    assert calls == [("x", (1,), repeat_scale)] * 5


# --- bf16: UnquantizedLinearMethod.apply and the branch it took ---------------


def test_bf16_gemm_op_is_serving_unquantized_linear_method_apply():
    calls = []

    class UnquantizedLinearMethod:
        def apply(self, layer, x, bias=None):
            calls.append((layer, x, bias))
            return "out"

    gemm_op = _create_gemm(
        {
            "torch": _fake_torch(),
            "unquant": SimpleNamespace(UnquantizedLinearMethod=UnquantizedLinearMethod),
            "gemm_type": "bfloat16",
            "M": 8,
            "N": 256,
            "K": 512,
            "device": "cpu",
        }
    )()

    assert gemm_op() == "out"
    ((layer, x, bias),) = calls
    assert x.shape == (8, 512) and x.dtype == BFLOAT16
    # The frozen [N, K] Parameter UnquantizedLinearMethod.create_weights builds.
    assert layer.weight.shape == (256, 512) and layer.weight.dtype == BFLOAT16
    assert layer.weight.requires_grad is False
    assert bias is None


def _bf16_label(m, n, k, *, backend="cutedsl", splitk=True, warp=(), direct=(), tgv=()):
    fake_unquant = SimpleNamespace(
        get_bf16_gemm_backend=lambda: SimpleNamespace(
            is_cutedsl=lambda: backend == "cutedsl", is_gemv=lambda: backend == "gemv"
        ),
        _enable_bf16_splitk_gemm=splitk,
        use_bf16_splitk_gemm=lambda m, n, k: (m, n, k) in {*warp, *direct, (1, 256, 8192)},
        _BF16_WARP_SPLITK_TUNED_TACTICS=dict.fromkeys(warp, None),
        _prefer_direct=lambda m, n, k: (m, n, k) in direct,
        _use_cutedsl_bf16_gemm=(lambda m, n, k: (m, n, k) in tgv) if backend == "cutedsl" else None,
    )
    return _top_level(GEMM_PATH, "_bf16_kernel_source", namespace={"unquant": fake_unquant})["_bf16_kernel_source"](
        m, n, k
    )


def test_bf16_label_follows_the_dispatch_order_serving_takes():
    assert _bf16_label(4, 4608, 8192, warp=[(4, 4608, 8192)]) == "sglang_flashinfer_bf16_warp_splitk"
    assert _bf16_label(2, 512, 2560, direct=[(2, 512, 2560)]) == "sglang_flashinfer_bf16_direct"
    assert _bf16_label(1, 256, 8192) == "sglang_flashinfer_bf16_splitk"
    # The split-K table wins over TGV for the same shape.
    assert _bf16_label(1, 256, 8192, tgv=[(1, 256, 8192)]) == "sglang_flashinfer_bf16_splitk"
    assert _bf16_label(16, 4096, 4096, tgv=[(16, 4096, 4096)]) == "sglang_cutedsl_tgv_bf16"
    assert _bf16_label(16, 4096, 4096) == "sglang_torch_linear"
    # With split-K disabled the tuned table is not consulted.
    assert _bf16_label(1, 256, 8192, splitk=False) == "sglang_torch_linear"


def test_bf16_label_is_cublas_off_the_cutedsl_backend_and_refuses_gemv():
    assert _bf16_label(1, 256, 8192, backend="auto", tgv=[(1, 256, 8192)]) == "sglang_torch_linear"
    with pytest.raises(RuntimeError, match="gemv"):
        _bf16_label(1, 256, 8192, backend="gemv")


# --- run_gemm: backend config, autotune scope and the logged label -----------


def _run_gemm_namespace(events: list, logged: list, **overrides):
    tree = ast.parse(GEMM_PATH.read_text(), filename=str(GEMM_PATH))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        or (
            isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and t.id.startswith("_") for t in node.targets)
        )
    ]
    torch = _fake_torch()
    torch.cuda = SimpleNamespace(
        set_device=lambda device: None,
        get_device_properties=lambda device: SimpleNamespace(total_memory=10**9),
        nvtx=SimpleNamespace(range_push=lambda tag: None, range_pop=lambda: None),
        get_device_name=lambda device: "fake",
        empty_cache=lambda: None,
    )

    @contextlib.contextmanager
    def benchmark_with_power(device, kernel_func, num_warmups, num_runs, repeat_n):
        events.append("benchmark")
        kernel_func()
        yield {"latency_ms": 1.0, "power_stats": None}

    namespace = _exec(nodes, GEMM_PATH, {"torch": torch, "contextlib": contextlib})
    namespace.update(
        {
            "_ensure_serving_gemm_config": lambda: events.append("serving_config"),
            "get_sm_version": lambda: 100,
            "_dist_version": lambda name: "0.5.23",
            "benchmark_with_power": benchmark_with_power,
            "log_perf": lambda **kwargs: logged.append(kwargs) or True,
        }
    )
    namespace.update(overrides)
    return namespace


def _fake_tuning_bucket(m, max_tuned_tokens=None):
    """helper.flashinfer_tuning_bucket for the token counts these tests use: FlashInfer's
    hybrid ceil (48 -> 64, 600 -> 768), None past the tuned list."""
    bucket = {1: 1, 48: 64, 64: 64, 600: 768}[m]
    return bucket if max_tuned_tokens is None or bucket <= max_tuned_tokens else None


def _nvfp4_overrides(events: list, *, autotune_enabled=True, decode_max_tokens=512):
    v2 = {"active": False}

    @contextlib.contextmanager
    def flashinfer_autotune_v2(**kwargs):
        events.append(("autotune_v2", kwargs))
        v2["active"] = True
        try:
            yield
        finally:
            v2["active"] = False

    @contextlib.contextmanager
    def flashinfer_autotuned(num_tokens, tune_fn, *, label=""):
        events.append(("autotuned", num_tokens, label))
        if autotune_enabled:
            tune_fn()
        yield autotune_enabled

    def flashinfer_mm_fp4(a, b, a_sf, b_sf, alpha, dtype, *, backend):
        events.append(("mm_fp4", backend, v2["active"]))
        return "mm-out"

    runner = SimpleNamespace(
        is_flashinfer_cutedsl=lambda: True,
        is_flashinfer_cutlass=lambda: False,
        get_flashinfer_backend=lambda: "cute-dsl",
        value="flashinfer_cutedsl",
    )

    def quantize(x, scale, **_kwargs):
        return _Tensor((x.shape[0], x.shape[1] // 2), "uint8"), _Tensor(x.shape, "uint8")

    return {
        **_nvfp4_padding([]),
        "HAS_FLASHINFER_FP4": True,
        "sglang_fp4_quantize": quantize,
        "flashinfer_fp4_quantize": quantize,
        "try_qwen3x_nvfp4_gemm": lambda *args: None,
        "flashinfer_mm_fp4": flashinfer_mm_fp4,
        "flashinfer_autotune_v2": flashinfer_autotune_v2,
        "flashinfer_autotuned": flashinfer_autotuned,
        "flashinfer_tuning_bucket": _fake_tuning_bucket,
        "_decode_autotune_max_tokens": lambda: decode_max_tokens,
        "get_fp4_gemm_runner_backend": lambda: runner,
    }


def test_nvfp4_is_timed_on_the_tactic_tuned_inside_serving_autotune_scope():
    events, logged = [], []
    namespace = _run_gemm_namespace(events, logged, **_nvfp4_overrides(events))

    namespace["run_gemm"]("nvfp4", 64, 256, 512, perf_filename="gemm_perf.txt")

    assert events[0] == "serving_config"
    assert events[1] == ("autotuned", 64, "gemm nvfp4 n=256 k=512")
    # The tuning pass runs mm_fp4 inside autotune_v2(mode="tune", persistent_cache=False),
    # as SGLang's warm-up does, at the bucket mm_fp4 maps this token count to (64 is one).
    assert events[2] == ("autotune_v2", {"mode": "tune", "persistent_cache": False, "tuning_buckets": (64,)})
    assert ("mm_fp4", "cute-dsl", True) in events[3 : events.index("benchmark")]
    # The timed calls run after the tuning pass, outside the v2 context.
    assert events[events.index("benchmark") + 1 :] and all(
        event == ("mm_fp4", "cute-dsl", False) for event in events[events.index("benchmark") + 1 :]
    )
    (row,) = logged
    assert row["kernel_source"] == "sglang_flashinfer_cutedsl_nvfp4"


def test_nvfp4_tunes_at_the_bucket_serving_maps_m_to():
    events, logged = [], []
    namespace = _run_gemm_namespace(events, logged, **_nvfp4_overrides(events))

    namespace["run_gemm"]("nvfp4", 48, 256, 512, perf_filename="gemm_perf.txt")

    # mm_fp4 maps m=48 up to the 64 bucket; the warm-up profiled 64, so 48 runs 64's tactic.
    assert events[1] == ("autotuned", 64, "gemm nvfp4 n=256 k=512")
    assert events[2] == ("autotune_v2", {"mode": "tune", "persistent_cache": False, "tuning_buckets": (64,)})
    assert logged[0]["item_list"][0]["m"] == 48


def test_nvfp4_above_the_decode_warmup_runs_untuned_like_serving():
    events, logged = [], []
    namespace = _run_gemm_namespace(events, logged, **_nvfp4_overrides(events, decode_max_tokens=512))

    namespace["run_gemm"]("nvfp4", 600, 256, 512, perf_filename="gemm_perf.txt")

    # 600 maps to the 768 bucket, past the decode warm-up's 512: no tuned entry in serving.
    assert not any(isinstance(event, tuple) and event[0] in ("autotuned", "autotune_v2") for event in events)
    assert ("mm_fp4", "cute-dsl", False) in events
    assert logged[0]["kernel_source"] == "sglang_flashinfer_cutedsl_nvfp4"


@pytest.mark.parametrize(
    ("gpu_mem_mib", "tp_size", "expected"),
    [
        (None, 1, (4096, 160)),
        (16 * 1024, 1, (2048, 8)),
        (32 * 1024, 1, (4096, 48)),
        (32 * 1024, 4, (4096, 160)),
        (48 * 1024, 1, (4096, 32)),
        (48 * 1024, 8, (4096, 160)),
        (80 * 1024, 1, (8192, 256)),
        (97_887, 1, (8192, 256)),  # RTX PRO 6000 Blackwell
        (97_887, 4, (8192, 512)),
        (144_000, 8, (8192, 512)),  # H200
        (183_359, 1, (16384, 512)),  # B200
        (288 * 1024, 2, (16384, 512)),  # B300
    ],
)
def test_serving_memory_defaults_follow_sglang_memory_table(monkeypatch, gpu_mem_mib, tp_size, expected):
    from collector.sglang import runtime_limits

    common = types.ModuleType("sglang.srt.utils.common")
    common.get_device_memory_capacity = lambda: gpu_mem_mib
    for name in ("sglang", "sglang.srt", "sglang.srt.utils"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "sglang.srt.utils.common", common)
    assert runtime_limits.serving_memory_defaults(tp_size) == expected
    assert runtime_limits.decode_autotune_max_tokens(tp_size) == expected[1]


def test_gemm_decode_ceiling_is_the_tp1_warmup_batch():
    namespace = _run_gemm_namespace([], [])
    calls = []
    namespace["decode_autotune_max_tokens"] = lambda tp_size: calls.append(tp_size) or 256
    assert namespace["_decode_autotune_max_tokens"]() == 256
    assert calls == [1]


def test_nvfp4_kda_path_is_labelled_kda():
    events, logged = [], []
    overrides = _nvfp4_overrides(events)
    overrides["try_qwen3x_nvfp4_gemm"] = lambda *args: "kda-out"
    namespace = _run_gemm_namespace(events, logged, **overrides)

    namespace["run_gemm"]("nvfp4", 1, 24576, 4096, perf_filename="gemm_perf.txt")

    assert not any(isinstance(event, tuple) and event[0] == "mm_fp4" for event in events)
    assert logged[0]["kernel_source"] == "sglang_kda_qwen3x_nvfp4"


def test_nvfp4_refuses_a_runner_that_is_not_flashinfer_fp4():
    events, logged = [], []
    overrides = _nvfp4_overrides(events)
    overrides["get_fp4_gemm_runner_backend"] = lambda: SimpleNamespace(
        is_flashinfer_cutedsl=lambda: False, is_flashinfer_cutlass=lambda: False, value="marlin"
    )
    namespace = _run_gemm_namespace(events, logged, **overrides)

    with pytest.raises(ValueError, match="marlin"):
        namespace["run_gemm"]("nvfp4", 1, 256, 512, perf_filename="gemm_perf.txt")
    assert logged == []


def test_bf16_runs_serving_dispatch_without_autotune_and_logs_its_label():
    events, logged = [], []

    class UnquantizedLinearMethod:
        def apply(self, layer, x, bias=None):
            events.append("apply")

    def flashinfer_autotuned(*_args, **_kwargs):
        raise AssertionError("only FlashInfer-tuned NVFP4 cases enter the autotune pass")

    namespace = _run_gemm_namespace(
        events,
        logged,
        unquant=SimpleNamespace(UnquantizedLinearMethod=UnquantizedLinearMethod),
        flashinfer_autotuned=flashinfer_autotuned,
        _bf16_kernel_source=lambda m, n, k: f"label-{m}-{n}-{k}",
    )

    namespace["run_gemm"]("bfloat16", 16, 256, 512, perf_filename="gemm_perf.txt")

    assert events[0] == "serving_config"
    assert "apply" in events
    assert logged[0]["kernel_source"] == "label-16-256-512"


def test_fp8_block_dispatches_like_fp8_linear_method_and_logs_its_label():
    events, logged = [], []
    dispatch_calls = []

    def block_linear(**kwargs):
        events.append("block_linear")

    def dispatch_w8a8_block_fp8_linear(**kwargs):
        dispatch_calls.append(kwargs)
        return block_linear

    namespace = _run_gemm_namespace(
        events,
        logged,
        dispatch_w8a8_block_fp8_linear=dispatch_w8a8_block_fp8_linear,
        requant_block_scale_ue8m0_for_deepgemm=lambda *args, **kwargs: False,
        deepgemm_w8a8_block_fp8_linear_with_fallback=block_linear,
        _fp8_block_kernel_source=lambda linear, n, k: f"{linear.__name__}-{n}-{k}",
    )

    namespace["run_gemm"]("fp8_block", 16, 256, 512, perf_filename="gemm_perf.txt")

    # A checkpoint without ue8m0 scales: Fp8LinearMethod passes act_scale_ue8m0=False.
    assert dispatch_calls == [{"weight_block_size": [128, 128], "act_scale_ue8m0": False}]
    assert events.index("serving_config") < events.index("block_linear")
    assert logged[0]["kernel_source"] == "block_linear-256-512"
