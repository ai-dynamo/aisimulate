# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Precision dispatch and native serialized-weight population contracts."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit
SOURCE = Path(__file__).resolve().parents[3] / "collector/vllm/collect_mla_module.py"


def _load_function(name, namespace=None):
    node = next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {} if namespace is None else namespace
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


@pytest.fixture
def torch(monkeypatch):
    with monkeypatch.context() as context:
        if isinstance(sys.modules.get("torch"), MagicMock):
            context.delitem(sys.modules, "torch")
        yield pytest.importorskip("torch")


@pytest.mark.parametrize("mode,block", [("fp8", None), ("fp8_block", [128, 128])])
def test_fp8_modes_construct_distinct_native_configs(monkeypatch, mode, block):
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.fp8",
        SimpleNamespace(Fp8Config=lambda **kwargs: kwargs),
    )
    config = _load_function("_create_gemm_quant_config")(mode)
    assert config == {
        "is_checkpoint_fp8_serialized": True,
        "activation_scheme": "dynamic",
        "weight_block_size": block,
    }


def test_nvfp4_config_preserves_native_modelopt_without_projection_exclusions(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.modelopt",
        SimpleNamespace(ModelOptNvFp4Config=lambda **kwargs: kwargs),
    )
    assert _load_function("_create_gemm_quant_config")("nvfp4") == {
        "is_checkpoint_nvfp4_serialized": True,
        "kv_cache_quant_algo": None,
        "exclude_modules": [],
    }


def _nvfp4_fixture(monkeypatch, torch, *, invalid_scales=False):
    class NativeMethod:
        quant_config = SimpleNamespace(group_size=16)

    calls = []

    def native_quantize(values, inverse_scale, *, is_sf_swizzled_layout):
        # This tests the native producer boundary, not a reimplementation of
        # NVFP4. Actual packing/layout/kernel correctness requires the GPU canary.
        calls.append((values.clone(), inverse_scale.clone()))
        assert values.dtype == torch.bfloat16
        assert values.shape[0] * values.shape[1] <= 1048576
        assert torch.isfinite(values).all()
        assert (values < 0).any() and (values > 0).any()
        assert is_sf_swizzled_layout is False
        packed = torch.full((values.shape[0], values.shape[1] // 2), 0x12, dtype=torch.uint8)
        scales = torch.full(
            (values.shape[0], values.shape[1] // 16),
            float("nan") if invalid_scales else 2.0,
            dtype=torch.float32,
        ).to(torch.float8_e4m3fn)
        return packed, scales

    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(_custom_ops=SimpleNamespace(scaled_fp4_quant=native_quantize))
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.modelopt",
        SimpleNamespace(ModelOptNvFp4LinearMethod=NativeMethod),
    )
    module = torch.nn.Module()
    layer = module.projection = torch.nn.Module()
    layer.quant_method = NativeMethod()
    layer.weight = torch.nn.Parameter(torch.zeros(1025, 512, dtype=torch.uint8), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(torch.zeros(1025, 64, dtype=torch.float8_e4m3fn), requires_grad=False)
    layer.weight_scale_2 = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
    layer.input_scale = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
    module.other_weight = torch.nn.Parameter(torch.full((2, 2), 17.0), requires_grad=False)
    module.register_buffer("rotary", torch.tensor([0.3, 0.7]), persistent=False)
    return module, calls


def test_nvfp4_populates_native_weight_and_scale_formats_without_rng_or_buffer_changes(monkeypatch, torch):
    module, calls = _nvfp4_fixture(monkeypatch, torch)
    initialize = _load_function("_initialize_nvfp4_parameters", {"torch": torch})
    rng = torch.get_rng_state()
    initialize(module)
    assert torch.equal(torch.get_rng_state(), rng)
    assert [v.shape[0] for v, _ in calls] == [1024, 1]
    layer = module.projection
    assert torch.all(layer.weight == 0x12)
    assert torch.all(layer.weight_scale.float() == 2.0)
    # ModelOpt stores dequantizing global scales, not the inverse required by
    # the producer API. Fused logical partitions must share the same scale.
    assert torch.allclose(layer.weight_scale_2 * calls[0][1], torch.ones(2))
    assert torch.all(layer.input_scale > 0)
    assert torch.unique(layer.weight_scale_2).numel() == 1
    assert torch.unique(layer.input_scale).numel() == 1
    assert torch.equal(module.rotary, torch.tensor([0.3, 0.7]))
    assert torch.all(module.other_weight == 17.0)
    first_values = calls[0][0]
    calls.clear()
    initialize(module)
    assert torch.equal(first_values, calls[0][0])


def test_nvfp4_rejects_invalid_native_scale_output(monkeypatch, torch):
    module, _ = _nvfp4_fixture(monkeypatch, torch, invalid_scales=True)
    with pytest.raises(ValueError, match="invalid block scales"):
        _load_function("_initialize_nvfp4_parameters", {"torch": torch})(module)


def test_nvfp4_cannot_be_claimed_without_a_native_nvfp4_projection(monkeypatch, torch):
    module, _ = _nvfp4_fixture(monkeypatch, torch)
    module.projection.quant_method = object()
    with pytest.raises(RuntimeError, match="no native ModelOpt NVFP4 linear"):
        _load_function("_initialize_nvfp4_parameters", {"torch": torch})(module)
