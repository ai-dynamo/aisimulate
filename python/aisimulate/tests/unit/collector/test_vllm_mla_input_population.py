# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical input preparation without importing vLLM or requiring CUDA."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
SOURCE = Path(__file__).resolve().parents[3] / "collector/vllm/collect_mla_module.py"


def _load_function(name, torch):
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("meta_weight", [False, True])
def test_materialization_preserves_nonpersistent_rotary_and_scale_buffers(meta_weight):
    torch = pytest.importorskip("torch")
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


def test_materialization_rejects_meta_runtime_buffer_without_a_constructor_value():
    torch = pytest.importorskip("torch")
    module = torch.nn.Linear(4, 4, device="meta")
    module.register_buffer("cos_sin_cache", torch.empty(4, device="meta"), persistent=False)
    move = _load_function("_move_module_preserving_buffers", torch)
    with pytest.raises(RuntimeError, match="uninitialized meta buffer"):
        move(module, "cpu")


def test_indexer_history_uses_native_quantization_with_bounded_diverse_keys(monkeypatch):
    torch = pytest.importorskip("torch")
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


def test_indexer_history_rejects_an_incompatible_packed_cache(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(_custom_ops=SimpleNamespace()))
    populate = _load_function("_populate_indexer_kv_cache", torch)
    with pytest.raises(ValueError, match="FP8 values and FP32 scales"):
        populate(
            torch.empty(1, 64, 128, dtype=torch.uint8),
            SimpleNamespace(block_table_tensor=torch.zeros(1, 1, dtype=torch.int32)),
            [1],
            SimpleNamespace(head_dim=128, quant_block_size=128, scale_fmt="ue8m0"),
        )
