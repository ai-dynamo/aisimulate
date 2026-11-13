# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

MODULE_PATH = Path(__file__).resolve().parents[4] / "collector" / "sglang" / "collect_mhc_module.py"


def _flag(value):
    return SimpleNamespace(get=lambda: value)


def _install_sglang(monkeypatch, *, tilelang_pre=True, deepgemm=True, tilelang_post=True, flashinfer=False):
    """Stub the SGLang modules the collector imports lazily; returns the call log of HcRead/HcUpdate."""
    torch = ModuleType("torch")
    torch.OutOfMemoryError = type("OutOfMemoryError", (Exception,), {})
    torch.cuda = SimpleNamespace(
        OutOfMemoryError=type("CudaOutOfMemoryError", (Exception,), {}),
        empty_cache=lambda: None,
        synchronize=lambda: None,
        get_device_name=lambda _device: "test-gpu",
    )
    torch.no_grad = lambda: __import__("contextlib").nullcontext()
    torch.distributed = SimpleNamespace(is_initialized=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch)

    calls: list[tuple] = []

    class HcReadCall:
        def __init__(self, layer, attention):
            self.layer, self.attention = layer, attention

        def read(self, residual, norm):
            calls.append(("read", self.attention, norm))
            return f"x_{'attn' if self.attention else 'ffn'}", f"state_{'attn' if self.attention else 'ffn'}"

        def update_and_read(self, update, value, residual, norm):
            calls.append(("update_and_read", self.attention, update.attention, value, residual, norm))

    class HcRead:
        def __init__(self, layer, attention):
            self.layer, self.attention = layer, attention

        def bind(self, batch, call):
            assert batch is None and call is None
            return HcReadCall(self.layer, self.attention)

    class HcUpdate:
        def __init__(self, layer, attention):
            self.layer, self.attention = layer, attention

        def update(self, output, state):
            calls.append(("update", self.attention, output, state))

    boundary = ModuleType("sglang.srt.models.deepseek_v4_boundary")
    boundary.HcRead, boundary.HcUpdate = HcRead, HcUpdate
    models = ModuleType("sglang.srt.models")
    models.deepseek_v4_boundary = boundary
    deepseek_v4 = ModuleType("sglang.srt.models.deepseek_v4")
    deepseek_v4._HC_PRENORM_DEEPGEMM_MIN_TOKENS = 1024
    environ = ModuleType("sglang.srt.environ")
    environ.envs = SimpleNamespace(
        SGLANG_OPT_USE_TILELANG_MHC_PRE=_flag(tilelang_pre),
        SGLANG_OPT_DEEPGEMM_HC_PRENORM=_flag(deepgemm),
        SGLANG_OPT_USE_TILELANG_MHC_POST=_flag(tilelang_post),
        SGLANG_OPT_USE_FLASHINFER_MHC=_flag(flashinfer),
    )
    mhc_kernels = ModuleType("sglang.kernels.ops.layernorm.mhc")
    mhc_kernels._use_deep_gemm_hc_prenorm = lambda: deepgemm
    parallel_state = ModuleType("sglang.srt.distributed.parallel_state")
    parallel_state.destroy_model_parallel = lambda: None
    parallel_state.destroy_distributed_environment = lambda: None
    expert_location = ModuleType("sglang.srt.eplb.expert_location")
    expert_location._global_expert_location_metadata = None
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.distributed",
        "sglang.srt.eplb",
        "sglang.kernels",
        "sglang.kernels.ops",
        "sglang.kernels.ops.layernorm",
    ):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    for module in (boundary, models, deepseek_v4, environ, mhc_kernels, parallel_state, expert_location):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return calls


def _load(monkeypatch, name: str):
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _layer(**overrides):
    fields = dict(
        hc_mult=4,
        hc_sinkhorn_iters=20,
        config=SimpleNamespace(hidden_size=4096),
        input_layernorm="input_norm",
        post_attention_layernorm="post_attention_norm",
        boundary_pre_from_prev=False,
        boundary_mhc_fusions=True,
        use_fused_mhc_post_pre=True,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_mhc_sweep_fails_closed_after_an_inner_error(monkeypatch):
    _install_sglang(monkeypatch)
    module = _load(monkeypatch, "mhc_fail_closed_target")

    runner = SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(layers=[_layer()])))
    monkeypatch.setattr(module, "_load_one_layer_runner", lambda *_args, **_kwargs: runner)
    monkeypatch.setattr(module, "get_version", lambda _package: "0.5.23")
    monkeypatch.setattr(module, "_observe_norm_fusion", lambda _layer, _device: True)
    monkeypatch.setattr(module, "_make_residual", lambda _layer, num_tokens, _device: num_tokens)
    monkeypatch.setattr(module, "_make_kernel", lambda _layer, _op, residual: residual)
    monkeypatch.setattr(module, "_log_result", lambda **_kwargs: None)

    def benchmark(*, kernel_func, **_kwargs):
        if kernel_func == 2:
            raise RuntimeError("injected inner failure")
        return {"latency_ms": 1.0, "num_runs_executed": 3}

    monkeypatch.setattr(module, "_benchmark_mhc_kernel", benchmark)

    with pytest.raises(RuntimeError, match=r"mHC sweep failed: ok=1 error=1 skip=0 total=2"):
        module.run_mhc_module(ops=["pre"], num_tokens_cases=[1, 2], num_iterations=3)


def test_ops_run_on_serving_reads_and_updates(monkeypatch):
    calls = _install_sglang(monkeypatch)
    module = _load(monkeypatch, "mhc_serving_objects_target")
    layer = _layer()

    module._make_kernel(layer, "pre", "residual")()
    assert calls == [("read", True, "input_norm"), ("read", False, "post_attention_norm")]

    calls.clear()
    kernel = module._make_kernel(layer, "post", "residual")
    calls.clear()  # the reads that build the post inputs run once, outside the timed kernel
    kernel()
    assert calls == [("update", True, "x_attn", "state_attn"), ("update", False, "x_ffn", "state_ffn")]

    calls.clear()
    kernel = module._make_kernel(layer, "fused_post_pre", "residual")
    calls.clear()
    kernel()
    # Attention reads after the previous layer's FFN post, the FFN after this layer's attention post.
    assert calls == [
        ("update_and_read", True, False, "x_ffn", "state_ffn", "input_norm"),
        ("update_and_read", False, True, "x_attn", "state_attn", "post_attention_norm"),
    ]


@pytest.mark.parametrize(
    ("flags", "op", "num_tokens", "norm_fused", "expected"),
    [
        ({}, "pre", 64, True, "sglang.hc_pre[tilelang,prenorm_gemm=deepgemm,norm=fused]"),
        ({}, "post", 64, True, "sglang.hc_post[tilelang]"),
        ({}, "fused_post_pre", 32, True, "sglang.mhc_fused_post_pre[fused_post_gemm,norm=fused]"),
        ({}, "fused_post_pre", 33, True, "sglang.mhc_fused_post_pre[post+prenorm_gemm=deepgemm,norm=fused]"),
        # SM120 serving: the model hook turns TileLang pre and the DeepGEMM prenorm off.
        ({"tilelang_pre": False, "deepgemm": False}, "pre", 4096, False, "sglang.hc_pre[torch,norm=separate]"),
        (
            {"tilelang_pre": False, "deepgemm": False},
            "fused_post_pre",
            4096,
            False,
            "sglang.mhc_fused_post_pre[post+prenorm_gemm=tilelang,norm=fused]",
        ),
        ({"tilelang_pre": False}, "pre", 1023, False, "sglang.hc_pre[torch,norm=separate]"),
        ({"tilelang_pre": False}, "pre", 1024, False, "sglang.hc_pre[torch,prenorm_gemm=deepgemm,norm=separate]"),
        ({"tilelang_post": False}, "post", 64, True, "sglang.hc_post[torch]"),
    ],
)
def test_kernel_source_follows_serving_dispatch(monkeypatch, flags, op, num_tokens, norm_fused, expected):
    _install_sglang(monkeypatch, **flags)
    module = _load(monkeypatch, "mhc_kernel_source_target")
    assert module._kernel_source(_layer(), op, num_tokens, norm_fused) == expected


def test_unfused_seam_is_labelled_as_post_then_read(monkeypatch):
    _install_sglang(monkeypatch)
    module = _load(monkeypatch, "mhc_unfused_seam_target")
    label = module._kernel_source(_layer(use_fused_mhc_post_pre=False), "fused_post_pre", 64, True)
    assert label == "sglang.hc_post[tilelang]+sglang.hc_pre[tilelang,prenorm_gemm=deepgemm,norm=fused]"


@pytest.mark.parametrize(
    ("layer_overrides", "flashinfer", "match"),
    [
        ({"boundary_pre_from_prev": True}, False, "lagged"),
        ({}, True, "SGLANG_OPT_USE_FLASHINFER_MHC"),
    ],
)
def test_paths_serving_does_not_take_by_default_raise(monkeypatch, layer_overrides, flashinfer, match):
    _install_sglang(monkeypatch, flashinfer=flashinfer)
    module = _load(monkeypatch, "mhc_refusal_target")
    with pytest.raises(RuntimeError, match=match):
        module._refuse_unaudited_paths(_layer(**layer_overrides))


def test_worker_refuses_architectures_without_an_audited_route(monkeypatch):
    _install_sglang(monkeypatch)
    module = _load(monkeypatch, "mhc_worker_target")
    monkeypatch.setattr(module, "get_version", lambda _package: "0.5.23")
    monkeypatch.setattr(module, "run_mhc_module", lambda **_kwargs: pytest.fail("must not run"))
    with pytest.raises(module.MhcRuntimeNotAuditedError):
        module.run_mhc_module_worker(
            "pre", "zai-org/GLM-5.3-Flash", "Glm5NextForConditionalGeneration", perf_filename="/tmp/x_perf.txt"
        )
    module._require_audited_runtime(module.DSV4_ARCHITECTURE, "0.5.23")
    with pytest.raises(module.MhcRuntimeNotAuditedError):
        module._require_audited_runtime(module.DSV4_ARCHITECTURE, "0.5.21")
