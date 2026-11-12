# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SGLang DeepSeek-V4 CSA module times its top-k on written history and usable index scores."""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.unit

ATTN_SOURCE = Path(__file__).resolve().parents[4] / "collector" / "sglang" / "collect_dsv4_attn.py"


def _load(source_path: Path, *names: str, namespace: dict | None = None) -> dict:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(selected) == len(names)
    loaded = dict(namespace or {})
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), loaded)
    return loaded


class _Tensor:
    """A numpy stand-in for the few tensor calls the history writer makes."""

    def __init__(self, values, device="cuda:0"):
        self.values = np.asarray(values)
        self.device = device

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index = tuple(item.values if isinstance(item, _Tensor) else item for item in index)
        return _Tensor(self.values[index], self.device)

    def __floordiv__(self, divisor):
        return _Tensor(self.values // divisor, self.device)

    def long(self):
        return self

    def flatten(self):
        return _Tensor(self.values.reshape(-1), self.device)

    def contiguous(self):
        return self

    def numel(self):
        return int(self.values.size)


def _fake_torch():
    drawn = []

    class Generator:
        def __init__(self, device):
            self.device = device

        def manual_seed(self, seed):
            self.seed = seed

    def randn(shape, *, generator, device, dtype):
        assert generator.device == device and dtype == "bfloat16"
        drawn.append(tuple(shape))
        return _Tensor(np.random.default_rng(generator.seed).standard_normal(shape), device)

    return SimpleNamespace(Generator=Generator, randn=randn, bfloat16="bfloat16"), drawn


def _history_fixture(history_rows=1 << 16, *, use_fp4_indexer=False):
    torch, drawn = _fake_torch()
    write = _load(ATTN_SOURCE, "_write_index_history", namespace={"torch": torch, "_HISTORY_ROWS": history_rows})[
        "_write_index_history"
    ]
    stores = []
    # Request row r holds the slots 1000*r + position: a page-aligned run per request.
    table = _Tensor(np.arange(4)[:, None] * 1000 + np.arange(64)[None, :])
    runner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=table),
        token_to_kv_pool=SimpleNamespace(
            set_index_k_fused=lambda *, layer_id, loc, cache_k: stores.append((layer_id, loc.values, cache_k.values))
        ),
    )
    attention = SimpleNamespace(indexer=SimpleNamespace(head_dim=128, use_fp4_indexer=use_fp4_indexer), layer_id=5)
    batch = SimpleNamespace(req_pool_indices=_Tensor([2, 0]))
    return write, runner, attention, batch, stores, drawn


def test_history_keys_go_to_the_slots_serving_writes_them_at():
    """Key j of a request sits at (slot of token 4j+3) // 4, the c4_out_loc serving derives
    (metadata_kernel._init_compressed_attn_metadata_kernel at v0.5.23); a trailing partial
    group has no key yet."""
    write, runner, attention, batch, stores, drawn = _history_fixture()
    write(runner, attention, batch, 11)
    ((layer_id, slots, keys),) = stores
    assert layer_id == 5
    # Request rows 2 and 0, tokens 3 and 7 of each.
    assert slots.tolist() == [2003 // 4, 2007 // 4, 3 // 4, 7 // 4]
    assert keys.shape == (4, 128) and np.isfinite(keys).all() and keys.std() > 0.5
    assert drawn == [(4, 128)]
    # Seeded: a second point writes the same keys.
    stores.clear()
    write(runner, attention, batch, 11)
    assert np.array_equal(stores[0][2], keys)


def test_a_long_history_repeats_one_block_of_keys():
    write, runner, attention, batch, stores, drawn = _history_fixture(history_rows=3)
    write(runner, attention, batch, 16)  # 4 keys per request, 8 slots
    assert drawn == [(3, 128)]  # sampled once, bounded by the block
    assert [len(slots) for _layer, slots, _keys in stores] == [3, 3, 2]
    assert np.concatenate([slots for _layer, slots, _keys in stores]).tolist() == [
        *(s // 4 for s in (2003, 2007, 2011, 2015)),
        *(s // 4 for s in (3, 7, 11, 15)),
    ]
    block = stores[0][2]
    assert np.array_equal(stores[1][2], block) and np.array_equal(stores[2][2], block[:2])


def test_no_history_and_modules_without_an_indexer_write_nothing():
    write, runner, attention, batch, stores, drawn = _history_fixture()
    write(runner, attention, batch, 0)
    write(runner, attention, batch, 3)  # no complete group of four
    write(None, SimpleNamespace(indexer=None), None, 4096)  # HCA
    write(None, SimpleNamespace(), None, 4096)
    assert not stores and not drawn


def test_the_fp4_indexer_layout_is_refused_not_overwritten():
    write, runner, attention, batch, stores, _drawn = _history_fixture(use_fp4_indexer=True)
    with pytest.raises(RuntimeError, match="FP4 indexer is enabled"):
        write(runner, attention, batch, 64)
    assert not stores


class _Weight:
    def __init__(self):
        self.filled = []
        self.data = SimpleNamespace(fill_=self.filled.append)


def _magnitude_fixture(monkeypatch, load_format):
    initialized = []
    weight_utils = ModuleType("sglang.srt.model_loader.weight_utils")
    weight_utils.initialize_dummy_weights = lambda module, **kwargs: initialized.append((module, kwargs))
    monkeypatch.setitem(sys.modules, weight_utils.__name__, weight_utils)
    give = _load(ATTN_SOURCE, "_give_index_scores_magnitude")["_give_index_scores_magnitude"]
    q_norm, key_norm, kv_norm = _Weight(), _Weight(), _Weight()
    indexer = SimpleNamespace(
        compressor=SimpleNamespace(norm=SimpleNamespace(weight=key_norm)),
        weights_proj=object(),
        wq_b=object(),
    )
    attention = SimpleNamespace(
        indexer=indexer, q_norm=SimpleNamespace(weight=q_norm), kv_norm=SimpleNamespace(weight=kv_norm)
    )
    runner = SimpleNamespace(server_args=SimpleNamespace(load_format=load_format))
    return give, runner, attention, (q_norm, key_norm, kv_norm), initialized


def test_dummy_weights_get_unit_gains_and_larger_head_weights_only(monkeypatch):
    """The index scores are q_norm's output through wq_b, times the compressor-norm keys,
    times weights_proj: the two gains and weights_proj change, wq_b (FP8 under fp8_block)
    and the attention's own key norm do not."""
    give, runner, attention, (q_norm, key_norm, kv_norm), initialized = _magnitude_fixture(monkeypatch, "dummy")
    give(runner, attention)
    assert q_norm.filled == [1.0] and key_norm.filled == [1.0] and kv_norm.filled == []
    assert initialized == [(attention.indexer.weights_proj, {"low": -1.0, "high": 1.0})]


def test_a_loaded_checkpoint_and_a_module_without_an_indexer_keep_their_weights(monkeypatch):
    give, runner, attention, weights, initialized = _magnitude_fixture(monkeypatch, "auto")
    give(runner, attention)
    give(None, SimpleNamespace(indexer=None))  # HCA: the runner is not even read
    assert [w.filled for w in weights] == [[], [], []] and not initialized


def _calls(function, name):
    return [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == name
    ]


def test_the_module_is_prepared_before_timing_and_no_environment_switch_is_left():
    source = ATTN_SOURCE.read_text(encoding="utf-8")
    assert "AIC_DSV4_PROPER_INIT" not in source
    tree = ast.parse(source, filename=str(ATTN_SOURCE))
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_dsv4_mla_module")
    (magnitude,) = _calls(run, "_give_index_scores_magnitude")
    (build,) = _calls(run, "_build_forward_batch")
    (history,) = _calls(run, "_write_index_history")
    (bench,) = _calls(run, "_bench_cuda_events")
    # Once per runner, then per point: history after the slots are bound, before timing.
    assert magnitude < build < history < bench
    environment_reads = [
        node
        for node in ast.walk(run)
        if isinstance(node, ast.Attribute) and node.attr == "environ" and getattr(node.value, "id", None) == "os"
    ]
    assert not environment_reads
