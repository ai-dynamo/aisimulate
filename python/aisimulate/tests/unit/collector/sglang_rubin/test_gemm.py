# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from collector import case_generator
from collector.sglang_rubin import collect_gemm, collect_moe

pytestmark = pytest.mark.unit


def test_collectors_import_without_torch_or_sglang():
    package_root = Path(collect_gemm.__file__).resolve().parents[2]
    code = (
        f"import sys; sys.path.insert(0, {str(package_root)!r}); "
        "from collector.sglang_rubin import collect_gemm, collect_moe; "
        "assert not {'torch', 'sglang'} & sys.modules.keys()"
    )
    subprocess.run([sys.executable, "-I", "-S", "-c", code], check=True)


@pytest.mark.parametrize("model", ["", "zai-org/GLM-5.2-FP8", "nvidia/GLM-5.3-NVFP4"])
def test_case_generation_requires_explicit_pilot_checkpoint(monkeypatch, model):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model)
    for getter in (collect_gemm.get_gemm_test_cases, collect_moe.get_moe_test_cases):
        with pytest.raises(ValueError, match="COLLECTOR_MODEL_PATH"):
            getter()


def test_bf16_cases_preserve_declared_physical_shapes(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", collect_gemm.PILOT_MODEL)
    monkeypatch.delenv("AIC_COLLECT_GEMM_TYPES", raising=False)
    expected = [(1, 256, 6144), (32, 6144, 512)]

    def specs(*, backend):
        assert backend == "sglang"
        return [case_generator.GemmCommonTestCase(*shape) for shape in expected]

    monkeypatch.setattr(case_generator, "get_gemm_case_specs", specs)
    assert collect_gemm.get_gemm_test_cases() == [["bfloat16", *shape] for shape in expected]
    monkeypatch.setenv("AIC_COLLECT_GEMM_TYPES", "bfloat16,nvfp4")
    with pytest.raises(ValueError, match="supports only bfloat16"):
        collect_gemm.get_gemm_test_cases()


def test_gemm_rejects_nonpilot_quantization_before_loading_framework():
    with pytest.raises(ValueError, match="supports only bfloat16"):
        collect_gemm.run_gemm("nvfp4", 1, 6144, 2048, perf_filename="unused")


def test_bf16_provenance_tracks_framework_shape_dispatch():
    backend = SimpleNamespace(value="cutedsl", is_cutedsl=lambda: True)
    unquant = SimpleNamespace(
        get_bf16_gemm_backend=lambda: backend,
        _use_cutedsl_bf16_gemm=lambda m, n, k: m == 1,
    )
    assert collect_gemm._kernel_source(unquant, 1, 6144, 2048) == "sglang_cutedsl_bf16_gemm"
    assert collect_gemm._kernel_source(unquant, 32, 6144, 2048) == "sglang_torch_linear"
    backend.value = "auto"
    backend.is_cutedsl = lambda: False
    with pytest.raises(RuntimeError, match="Unresolved"):
        collect_gemm._kernel_source(unquant, 1, 6144, 2048)


def test_runtime_preflight_failure_is_not_ignored(monkeypatch):
    monkeypatch.setattr(collect_gemm.runtime, "collect_inventory", lambda: {"observed": {}})
    with pytest.raises(RuntimeError, match="Linux/aarch64"):
        collect_gemm._require_runtime()


def test_runtime_requires_exact_installed_distribution_not_only_build_label(monkeypatch):
    inventory = {"observed": {"package_versions": {"sglang": {"version": collect_gemm.SGLANG_DISTRIBUTION_VERSION}}}}
    monkeypatch.setattr(collect_gemm.runtime, "collect_inventory", lambda: inventory)
    monkeypatch.setattr(collect_gemm.runtime, "validate_runtime", lambda _: [])
    collect_gemm._require_runtime()
    inventory["observed"]["package_versions"]["sglang"]["version"] = "0.5.18+02c5a855"
    with pytest.raises(RuntimeError, match="Expected SGLang distribution"):
        collect_gemm._require_runtime()


@pytest.fixture
def gemm_runtime(monkeypatch):
    """Exercise run_gemm without importing the GPU-only distribution.

    These fakes test collector integration, not native dispatch correctness.
    The source-pinned native caller audit separately executes MoEGate's AST.
    """
    events = []
    rows = []
    measured = []

    class Tensor:
        def __init__(self, shape, dtype="bfloat16"):
            self.shape = shape
            self.dtype = dtype

        def normal_(self):
            return self

        def zero_(self):
            return self

    class Module:
        def __call__(self, *args):
            return self.forward(*args)

        def requires_grad_(self, enabled):
            assert enabled is False
            return self

        def eval(self):
            return self

    class LinearMethod:
        def create_weights(self, layer, k, partitions, full_k, n, dtype):
            assert partitions == [n] and full_k == k and dtype == "bfloat16"
            layer.weight = Tensor((n, k))

        def process_weights_after_loading(self, layer):
            events.append("process_linear_weights")

        def apply(self, layer, x, bias):
            assert bias is None
            events.append("linear")
            return Tensor((x.shape[0], layer.weight.shape[0]))

    def router_jit(x, weight, out_dtype="float32"):
        events.append("router_jit")
        return Tensor((x.shape[0], weight.shape[0]), out_dtype)

    def router_cublas(x, weight):
        events.append("router_cublas")
        return Tensor((x.shape[0], weight.shape[0]), "float32")

    class Gate(Module):
        def __init__(self, config, quant_config):
            assert config.hidden_size == 6144 and config.n_routed_experts == 256
            assert config.architectures == ["GlmMoeDsaForCausalLM"]
            assert quant_config == {"native_quant_config": True}
            self.weight = Tensor((config.n_routed_experts, config.hidden_size))
            self.e_score_correction_bias = Tensor((config.n_routed_experts,), "float32")

        def forward(self, x):
            if x.shape[0] <= 16:
                return router_jit(x, self.weight)
            return router_cublas(x, self.weight)

    def quant_from_config(config):
        assert config["quant_algo"] == "NVFP4"
        return {"native_quant_config": True}

    @contextmanager
    def benchmark(**kwargs):
        assert sys.getprofile() is None
        assert {key: kwargs[key] for key in ("num_warmups", "num_runs", "repeat_n")} == {
            "num_warmups": 3,
            "num_runs": 6,
            "repeat_n": 1,
        }
        measured.extend(kwargs["kernel_func"]() for _ in range(2))
        yield {"latency_ms": 0.1, "power_stats": {}}

    def log_perf(**kwargs):
        rows.append(kwargs)
        return True

    backend = SimpleNamespace(value="torch", is_cutedsl=lambda: False)
    context = SimpleNamespace(server_args=object(), override_server_args=lambda **_: nullcontext())
    modules = {
        "torch": SimpleNamespace(
            bfloat16="bfloat16",
            float32="float32",
            cuda=SimpleNamespace(
                set_device=lambda _: None,
                empty_cache=lambda: events.append("empty_cache"),
                get_device_name=lambda _: "Rubin",
            ),
            nn=SimpleNamespace(Module=Module),
            device=lambda _: nullcontext(),
            no_grad=nullcontext,
            randn=lambda m, k, **_: Tensor((m, k)),
        ),
        "sglang.srt.layers.quantization": SimpleNamespace(
            unquant=SimpleNamespace(
                initialize_bf16_gemm_config=lambda _: None,
                UnquantizedLinearMethod=LinearMethod,
                get_bf16_gemm_backend=lambda: backend,
            )
        ),
        "sglang.srt.runtime_context": SimpleNamespace(
            get_context=lambda: context,
            get_exec=lambda: SimpleNamespace(deterministic=SimpleNamespace(enable_deterministic_inference=False)),
        ),
        "sglang.srt.model_loader.utils": SimpleNamespace(set_default_torch_dtype=lambda _: nullcontext()),
        "sglang.srt.models.deepseek_v2": SimpleNamespace(MoEGate=Gate),
        "sglang.srt.layers.quantization.modelopt_quant": SimpleNamespace(
            ModelOptFp4Config=SimpleNamespace(from_config=quant_from_config)
        ),
        "sglang.kernels.ops.gemm.dsv3_router_gemm": SimpleNamespace(dsv3_router_gemm=router_jit),
        "sglang.kernels.ops.attention.dsv4.gemm": SimpleNamespace(_linear_bf16_fp32_cublas=router_cublas),
        "collector.helper": SimpleNamespace(benchmark_with_power=benchmark, log_perf=log_perf),
    }
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", collect_gemm.PILOT_MODEL)
    monkeypatch.setattr(collect_gemm, "_require_runtime", lambda: None)
    monkeypatch.setattr(collect_gemm.importlib.metadata, "version", lambda _: collect_gemm.SGLANG_DISTRIBUTION_VERSION)
    return SimpleNamespace(events=events, rows=rows, measured=measured, modules=modules, Tensor=Tensor, Gate=Gate)


@pytest.mark.parametrize(
    "m,leaf",
    [
        (1, "dsv3_router_gemm"),
        (16, "dsv3_router_gemm"),
        (17, "linear_bf16_fp32_cublas"),
        (1024, "linear_bf16_fp32_cublas"),
    ],
)
def test_router_gemm_measures_native_gate_and_preserves_fp32_output(gemm_runtime, m, leaf):
    collect_gemm.run_gemm("bfloat16", m, 256, 6144, perf_filename="gemm.txt")
    assert gemm_runtime.rows[0]["kernel_source"] == f"sglang_{leaf}"
    assert [output.dtype for output in gemm_runtime.measured] == ["float32", "float32"]
    assert "linear" not in gemm_runtime.events
    assert gemm_runtime.rows[0]["item_list"] == [
        {"gemm_dtype": "bfloat16", "m": m, "n": 256, "k": 6144, "latency": 0.1}
    ]


@pytest.mark.parametrize(
    "n,k", [(1024, 6144), (6144, 512), (6144, 6144), (6144, 3072), (51200, 6144), (65536, 6144), (256, 2048)]
)
def test_standard_projections_keep_the_existing_bf16_path(gemm_runtime, n, k):
    collect_gemm.run_gemm("bfloat16", 32, n, k, perf_filename="gemm.txt")
    assert gemm_runtime.rows[0]["kernel_source"] == "sglang_torch_linear"
    assert [output.dtype for output in gemm_runtime.measured] == ["bfloat16", "bfloat16"]
    assert gemm_runtime.events == ["process_linear_weights", "linear", "linear", "empty_cache"]


def test_router_rejects_changed_checkpoint_metadata(gemm_runtime, monkeypatch, tmp_path):
    config = tmp_path / "config.json"
    config.write_text('{"hidden_size": 6144, "n_routed_experts": 256}')
    monkeypatch.setattr(collect_gemm, "_MODEL_CONFIG", config)
    with pytest.raises(RuntimeError, match="frozen.*metadata"):
        collect_gemm.run_gemm("bfloat16", 1, 256, 6144, perf_filename="gemm.txt")
    assert not gemm_runtime.rows and not gemm_runtime.measured


def test_router_rejects_deterministic_inference(gemm_runtime):
    context = gemm_runtime.modules["sglang.srt.runtime_context"]
    context.get_exec = lambda: SimpleNamespace(deterministic=SimpleNamespace(enable_deterministic_inference=True))
    with pytest.raises(RuntimeError, match="deterministic inference disabled"):
        collect_gemm.run_gemm("bfloat16", 1, 256, 6144, perf_filename="gemm.txt")
    assert not gemm_runtime.rows and not gemm_runtime.measured


@pytest.mark.parametrize("fault", ["no_leaf", "two_leaves", "dtype", "shape", "exception"])
def test_router_dispatch_observation_fails_closed_and_cleans_up(gemm_runtime, monkeypatch, fault):
    original = gemm_runtime.Gate.forward

    def forward(self, x):
        if fault == "no_leaf":
            return gemm_runtime.Tensor((x.shape[0], 256), "float32")
        output = original(self, x)
        if fault == "two_leaves":
            original(self, x)
        elif fault == "dtype":
            output.dtype = "bfloat16"
        elif fault == "shape":
            output.shape = (x.shape[0], 128)
        elif fault == "exception":
            raise RuntimeError("native failure")
        return output

    monkeypatch.setattr(gemm_runtime.Gate, "forward", forward)
    with pytest.raises(RuntimeError, match="native failure|Unexpected Rubin router execution"):
        collect_gemm.run_gemm("bfloat16", 1, 256, 6144, perf_filename="gemm.txt")
    assert sys.getprofile() is None
    assert not gemm_runtime.rows and not gemm_runtime.measured
    assert gemm_runtime.events[-1] == "empty_cache"


def test_router_does_not_replace_an_existing_profiler(gemm_runtime):
    def existing(*_):
        pass

    sys.setprofile(existing)
    try:
        with pytest.raises(RuntimeError, match="no existing Python profiler"):
            collect_gemm.run_gemm("bfloat16", 1, 256, 6144, perf_filename="gemm.txt")
        assert sys.getprofile() is existing
    finally:
        sys.setprofile(None)
    assert not gemm_runtime.rows and not gemm_runtime.measured


def test_direct_gemm_execution_rejects_an_unrelated_model(gemm_runtime, monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "zai-org/GLM-5.2-FP8")
    with pytest.raises(ValueError, match="COLLECTOR_MODEL_PATH"):
        collect_gemm.run_gemm("bfloat16", 1, 256, 6144, perf_filename="gemm.txt")
    assert not gemm_runtime.rows and not gemm_runtime.events
