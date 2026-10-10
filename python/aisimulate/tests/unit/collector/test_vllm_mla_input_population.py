# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical input preparation without importing vLLM or requiring CUDA."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit
SOURCE = Path(__file__).resolve().parents[3] / "collector/vllm/collect_mla_module.py"


@pytest.fixture
def torch(monkeypatch):
    # Resolve optional tensors at execution time so CPU-only CI can still
    # inventory every test. Other collector suites may retain a torch mock.
    with monkeypatch.context() as context:
        if isinstance(sys.modules.get("torch"), MagicMock):
            context.delitem(sys.modules, "torch")
        yield pytest.importorskip("torch")


def _load_function(name, torch=None):
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {} if torch is None else {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("meta_weight", [False, True])
def test_materialization_preserves_nonpersistent_rotary_and_scale_buffers(meta_weight, torch):
    module = torch.nn.Module()
    module.projection = torch.nn.Linear(4, 4, device="meta" if meta_weight else "cpu")
    module.rotary = torch.nn.Module()
    expected_rope = torch.tensor([[1.0, 0.0], [0.54, 0.84]])
    module.rotary.register_buffer("cos_sin_cache", expected_rope.clone(), persistent=False)
    module.register_buffer("scale", torch.tensor(0.125))

    move = _load_function("_move_module_preserving_buffers", torch)
    moved = move(module, "cpu")
    assert moved is module
    assert not moved.projection.weight.is_meta
    assert torch.equal(moved.rotary.cos_sin_cache, expected_rope)
    assert moved.scale.item() == 0.125
    assert "rotary.cos_sin_cache" not in moved.state_dict()


def test_materialization_rejects_meta_runtime_buffer_without_a_constructor_value(torch):
    module = torch.nn.Linear(4, 4, device="meta")
    module.register_buffer("cos_sin_cache", torch.empty(4, device="meta"), persistent=False)
    move = _load_function("_move_module_preserving_buffers", torch)
    with pytest.raises(RuntimeError, match="uninitialized meta buffer"):
        move(module, "cpu")


class _Rotary:
    """Stand-in for vLLM's RotaryEmbeddingBase: a non-persistent cos/sin table built by a
    device-free recipe in __init__ (rotary_embedding/base.py:60-63 @ v0.30.0)."""

    def __new__(cls, torch, scale, dtype):
        class Rotary(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = scale
                self.register_buffer("cos_sin_cache", self._compute_cos_sin_cache().to(dtype), persistent=False)

            def _compute_cos_sin_cache(self):
                return torch.arange(6, dtype=torch.float32).reshape(3, 2) * self.scale

        return Rotary()


def _meta_attention_module(torch):
    # Serving-parity construction instantiates the model under torch.device("meta"): the
    # two rotary tables (rotary_emb, indexer_rope_emb) and the KV-quant scale buffers that
    # MLAAttention registers (attention.py:129-135,184 @ v0.30.0) are all meta.
    with torch.device("meta"):
        module = torch.nn.Module()
        module.projection = torch.nn.Linear(4, 4)
        module.rotary_emb = _Rotary(torch, 1.0, torch.bfloat16)
        module.indexer_rope_emb = _Rotary(torch, 0.5, torch.float32)
        module.mla_attn = torch.nn.Module()
        for scale in ("_k_scale", "_v_scale", "_q_scale", "_prob_scale"):
            module.mla_attn.register_buffer(scale, torch.tensor(1.0, dtype=torch.float32))
        module.mla_attn._k_scale_cpu = torch.tensor(1.0, dtype=torch.float32)
        module.mla_attn._v_scale_cpu = torch.tensor(1.0, dtype=torch.float32)
    return module


def test_materialization_rebuilds_meta_rotary_tables_and_quant_scales_from_the_owner_recipe(torch):
    module = _meta_attention_module(torch)
    assert module.rotary_emb.cos_sin_cache.is_meta and module.mla_attn._k_scale.is_meta

    move = _load_function("_move_module_preserving_buffers", torch)
    moved = move(module, "cpu")

    recipe = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    assert moved is module
    assert not moved.projection.weight.is_meta
    assert moved.rotary_emb.cos_sin_cache.dtype == torch.bfloat16
    assert torch.equal(moved.rotary_emb.cos_sin_cache, recipe.to(torch.bfloat16))
    assert moved.indexer_rope_emb.cos_sin_cache.dtype == torch.float32
    assert torch.equal(moved.indexer_rope_emb.cos_sin_cache, recipe * 0.5)
    assert "rotary_emb.cos_sin_cache" not in moved.state_dict()
    for scale in ("_k_scale", "_v_scale", "_q_scale", "_prob_scale"):
        tensor = getattr(moved.mla_attn, scale)
        assert not tensor.is_meta and tensor.dtype == torch.float32 and tensor.item() == 1.0
        assert f"mla_attn.{scale}" in moved.state_dict()
    assert moved.mla_attn._k_scale_cpu.device.type == "cpu" and moved.mla_attn._k_scale_cpu.item() == 1.0
    assert moved.mla_attn._v_scale_cpu.item() == 1.0


def test_materialization_handles_a_rotary_shared_between_attention_and_indexer(torch):
    # get_rope() caches instances per argument tuple, so both names can hold one module.
    with torch.device("meta"):
        module = torch.nn.Module()
        module.projection = torch.nn.Linear(4, 4)
        shared = _Rotary(torch, 2.0, torch.bfloat16)
        module.rotary_emb = shared
        module.indexer_rope_emb = shared
    move = _load_function("_move_module_preserving_buffers", torch)
    moved = move(module, "cpu")
    recipe = (torch.arange(6, dtype=torch.float32).reshape(3, 2) * 2.0).to(torch.bfloat16)
    assert moved.rotary_emb is moved.indexer_rope_emb
    assert torch.equal(moved.indexer_rope_emb.cos_sin_cache, recipe)


def test_materialization_refuses_a_meta_buffer_whose_owner_has_no_recipe(torch):
    module = _meta_attention_module(torch)
    with torch.device("meta"):
        module.register_buffer("block_table_cache", torch.zeros(4), persistent=False)
    move = _load_function("_move_module_preserving_buffers", torch)
    with pytest.raises(RuntimeError, match="uninitialized meta buffer.*block_table_cache"):
        move(module, "cpu")


@pytest.mark.parametrize("dtype_name", ["float8_e4m3fn", "float8_e5m2"])
def test_synthetic_fp8_weights_produce_nonzero_queries_without_changing_buffers_or_rng(dtype_name, torch):
    def make_module():
        module = torch.nn.Module()
        module.weight = torch.nn.Parameter(torch.empty(128, 256, dtype=getattr(torch, dtype_name)), requires_grad=False)
        module.weight_scale_inv = torch.nn.Parameter(torch.empty(1, dtype=torch.float32), requires_grad=False)
        module.dense_weight = torch.nn.Parameter(torch.empty(4, dtype=torch.bfloat16), requires_grad=False)
        module.packed_weight = torch.nn.Parameter(torch.empty(8, dtype=torch.uint8), requires_grad=False)
        module.register_buffer("cos_sin_cache", torch.tensor([0.5, 0.75]), persistent=False)
        return module

    initialize = _load_function("_initialize_synthetic_parameters", torch)
    first, second = make_module(), make_module()
    rng_before = torch.get_rng_state()
    initialize(first)
    initialize(second)
    assert torch.equal(torch.get_rng_state(), rng_before)
    weights = first.weight.float()
    assert torch.equal(weights, second.weight.float())
    assert torch.isfinite(weights).all()
    assert (weights < 0).any() and (weights > 0).any()
    queries = torch.ones(3, 256) @ weights.T
    assert torch.isfinite(queries).all()
    assert torch.count_nonzero(queries) > 0
    assert queries[0].unique().numel() > 64
    assert first.weight_scale_inv.item() == 0.5
    assert torch.equal(first.cos_sin_cache, torch.tensor([0.5, 0.75]))
    assert torch.equal(first.dense_weight, torch.full((4,), 0.01, dtype=torch.bfloat16))
    assert torch.count_nonzero(first.packed_weight) == 0


def test_nvfp4_fp8_scale_storage_is_not_treated_as_a_projection_weight(torch):
    module = torch.nn.Module()
    module.projection = torch.nn.Module()
    module.projection.weight = torch.nn.Parameter(torch.empty(16, 64, dtype=torch.uint8), requires_grad=False)
    module.projection.weight_scale = torch.nn.Parameter(
        torch.empty(16, 8, dtype=torch.float8_e4m3fn), requires_grad=False
    )
    module.projection.weight_scale_2 = torch.nn.Parameter(torch.empty(1, dtype=torch.float32), requires_grad=False)
    initialize = _load_function("_initialize_synthetic_parameters", torch)
    initialize(module)
    # Packed weights are populated by the native NVFP4 initializer after
    # this generic pass; scales must not receive floating projection values.
    assert torch.count_nonzero(module.projection.weight) == 0
    assert torch.count_nonzero(module.projection.weight_scale.float()) == 0
    assert module.projection.weight_scale_2.item() == 0.5


def test_indexer_history_uses_native_quantization_with_bounded_diverse_keys(monkeypatch, torch):
    calls = []

    def native_insert(keys, cache, slots, block_size, scale_format):
        calls.append((keys.clone(), slots.clone(), block_size, scale_format))
        assert keys.device.type == "cpu"
        assert keys.dtype == torch.bfloat16
        assert torch.isfinite(keys).all()
        assert keys.shape[0] <= 8192
        assert keys.flatten()[:4096].unique().numel() > 100

    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(_custom_ops=SimpleNamespace(indexer_k_quant_and_cache=native_insert))
    )
    populate = _load_function("_populate_indexer_kv_cache", torch)
    indexer = SimpleNamespace(head_dim=256, quant_block_size=128, scale_fmt="ue8m0")
    cache = torch.empty(300, 64, 264, dtype=torch.uint8)
    block_table = torch.stack([torch.arange(260) + 3, torch.arange(260) + 5, torch.arange(260) + 7])
    metadata = SimpleNamespace(block_table_tensor=block_table)
    rng_before = torch.get_rng_state()
    populate(cache, metadata, [16385, 2, 0], indexer)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert [keys.shape[0] for keys, *_ in calls] == [8192, 8192, 1, 2]
    assert all((block, fmt) == (128, "ue8m0") for _, _, block, fmt in calls)
    first_request_slots = torch.cat([entry[1] for entry in calls[:3]])
    assert torch.equal(first_request_slots, torch.arange(16385) + 3 * 64)
    assert torch.equal(calls[-1][1], torch.tensor([5 * 64, 5 * 64 + 1]))
    first_keys = calls[0][0]
    calls.clear()
    populate(cache, metadata, [16385, 2, 0], indexer)
    assert torch.equal(first_keys, calls[0][0])


def test_indexer_history_rejects_an_incompatible_packed_cache(monkeypatch, torch):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(_custom_ops=SimpleNamespace()))
    populate = _load_function("_populate_indexer_kv_cache", torch)
    with pytest.raises(ValueError, match="FP8 values and FP32 scales"):
        populate(
            torch.empty(1, 64, 128, dtype=torch.uint8),
            SimpleNamespace(block_table_tensor=torch.zeros(1, 1, dtype=torch.int32)),
            [1],
            SimpleNamespace(head_dim=128, quant_block_size=128, scale_fmt="ue8m0"),
        )


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
    initialize = _load_function("_initialize_nvfp4_parameters", torch)
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
        _load_function("_initialize_nvfp4_parameters", torch)(module)


def test_nvfp4_cannot_be_claimed_without_a_native_nvfp4_projection(monkeypatch, torch):
    module, _ = _nvfp4_fixture(monkeypatch, torch)
    module.projection.quant_method = object()
    with pytest.raises(RuntimeError, match="no native ModelOpt NVFP4 linear"):
        _load_function("_initialize_nvfp4_parameters", torch)(module)
