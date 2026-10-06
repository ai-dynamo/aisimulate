# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The synthetic history populates actual request slots via native writers."""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
SOURCE = Path(__file__).resolve().parents[4] / "collector/sglang/collect_mla_module.py"


def load_initializer(torch):
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_initialize_dsa_history")
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns[node.name]


@pytest.mark.parametrize("length", [1, 63, 64, 65, 8193])
def test_native_writers_cover_each_history_slot_and_no_new_token(monkeypatch, length):
    torch = pytest.importorskip("torch")
    quant_module = ModuleType("sglang.srt.layers.attention.dsa.triton_kernel")
    quant_module.act_quant = object()
    monkeypatch.setitem(sys.modules, quant_module.__name__, quant_module)
    initialize = load_initializer(torch)
    batch = SimpleNamespace(req_pool_indices=torch.tensor([2, 0]), batch_size=2)
    table = torch.arange(3 * (length + 1)).reshape(3, length + 1)
    expected = table[[2, 0], :length].flatten()
    mla_calls, index_calls = [], []
    layer = object()

    def mla_writer(actual_layer, slots, latent, rope):
        assert actual_layer is layer
        assert latent.shape == (len(slots), 1, 512)
        assert rope.shape == (len(slots), 1, 64)
        assert latent.dtype == rope.dtype == torch.bfloat16
        assert torch.isfinite(latent).all() and latent.float().std() > 0
        assert torch.isfinite(rope).all() and rope.float().std() > 0
        mla_calls.append((slots.clone(), latent.clone(), rope.clone()))

    def index_writer(actual_batch, layer_id, key, *, act_quant, out_cache_loc):
        assert actual_batch is batch and layer_id == 1
        assert act_quant is quant_module.act_quant
        assert key.shape == (len(out_cache_loc), 128)
        assert key.dtype == torch.bfloat16 and torch.isfinite(key).all()
        assert key.float().std() > 0
        index_calls.append((out_cache_loc.clone(), key.clone()))

    pool = SimpleNamespace(index_head_dim=128, set_mla_kv_buffer=mla_writer)
    runner = SimpleNamespace(token_to_kv_pool=pool, req_to_token_pool=SimpleNamespace(req_to_token=table))
    attention = SimpleNamespace(
        attn_mqa=layer,
        indexer=SimpleNamespace(_store_index_k_cache=index_writer),
        layer_id=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )
    rng_before = torch.random.get_rng_state().clone()
    initialize(runner, attention, batch, length)
    assert torch.equal(torch.cat([c[0] for c in mla_calls]), expected)
    assert torch.equal(torch.cat([c[0] for c in index_calls]), expected)
    assert max(len(c[0]) for c in index_calls) <= 8192
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    first = index_calls[0][1].clone()
    mla_calls.clear()
    index_calls.clear()
    initialize(runner, attention, batch, length)
    assert torch.equal(index_calls[0][1], first)


def test_empty_history_needs_no_native_import_or_pool():
    initialize = load_initializer(None)
    initialize(None, None, None, 0)
    with pytest.raises(ValueError, match="nonnegative"):
        initialize(None, None, None, -1)
