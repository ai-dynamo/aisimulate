# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 dummy cuts: source layers travel with a consumer, layer
pointers are renumbered, engram hash tables shrink to a memory-only size.

The generic depth cut produced a 2-layer V4.1 dummy holding only sliding-window
layers with every source list left verbatim (2026-09-27): no compressor,
indexer, compressed-KV or candidate path was ever exercised, and the stale-ref
post-check missed it because the keys end in ``_layer_ids`` not ``layers``.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
COMPONENTS = Path(__file__).resolve().parents[4] / "collector" / "opharness" / "components"


def _load():
    spec = importlib.util.spec_from_file_location("dummies_t", COMPONENTS / "dummies.py")
    mod = importlib.util.module_from_spec(spec); sys.modules["dummies_t"] = mod; spec.loader.exec_module(mod)
    return mod


def _v41_config():
    """Shape of deepseek-ai/DeepSeek-V4.1-Flash text_config (structure fields only)."""
    return {
        "architectures": ["DeepseekV41ForCausalLM"],
        "text_config": {
            "num_hidden_layers": 40,
            "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],  # 40 backbone + 3 MTP
            "kv_source_layer_ids": [2, 8, 14, 20],
            "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
            "candidate_source_layer_id": 20,
            "candidate_topk_blocks": 2048,
            "candidate_block_size": 8,
            "engram_layer_ids": [1, 14],
            "engram_num_embeddings": [384006168, 384016682],
            "engram_vocab_size": 16000000,
            "engram_max_ngram_size": 4,
            "engram_n_heads": 8,
            "engram_head_dim": 256,
            "engram_compressed_vocab_size": 99092,
            "num_nextn_predict_layers": 3,
            "dspark_target_layer_ids": [37, 38, 39],
            "dspark_block_size": 5,
        },
        "vision_config": {"num_hidden_layers": 32},
    }


def test_engram_rows_reproduce_shipped_table_sizes():
    d = _load()
    assert d.engram_table_rows(16000000, 2, 4, 8) == [384006168, 384016682]


def test_rep_cut_holds_every_source_with_a_consumer():
    d = _load()
    variants = d.variants_dsv41(_v41_config())
    assert [v["name"] for v in variants] == ["rep", "rep_min"]
    # swa, engram swa, c2 source + consumer, candidate/c1 source + consumer, masked indexer
    assert variants[0]["sel"] == [0, 1, 2, 3, 20, 21, 24]
    assert variants[1]["sel"] == [1, 2, 3, 20, 21]


def test_apply_renumbers_pointers_and_shrinks_engram():
    d = _load()
    cfg = _v41_config()
    var = d.variants_dsv41(cfg)[0]
    edits: list[str] = []
    d.apply_dsv41(cfg, var, edits)
    tc = cfg["text_config"]
    assert tc["num_hidden_layers"] == 7
    assert tc["compress_ratios"] == [0, 0, 2, 2, 1, 1, 1]
    assert tc["kv_source_layer_ids"] == [2, 4]
    assert tc["index_source_layer_ids"] == [2, 4, 6]
    assert tc["candidate_source_layer_id"] == 4
    assert tc["engram_layer_ids"] == [1]
    assert tc["engram_vocab_size"] == d.ENGRAM_DUMMY_VOCAB_SIZE
    # rows follow the frameworks' prime-sum rule for the shrunken vocab
    assert tc["engram_num_embeddings"] == d.engram_table_rows(d.ENGRAM_DUMMY_VOCAB_SIZE, 1, 4, 8)
    assert tc["engram_num_embeddings"][0] < 3_000_000
    assert tc["num_nextn_predict_layers"] == 0
    assert not [k for k in tc if k.startswith("dspark_")]
    assert d._check_no_stale_layer_refs(cfg, 7) == []
    # vision tower untouched (separate depth axis)
    assert cfg["vision_config"]["num_hidden_layers"] == 32


def test_shrink_refuses_when_shipped_sizes_break_the_rule():
    d = _load()
    cfg = _v41_config()
    cfg["text_config"]["engram_num_embeddings"] = [384006168, 1]
    with pytest.raises(SystemExit):
        d.apply_dsv41(cfg, d.variants_dsv41(cfg)[0], [])


def test_cut_without_a_source_fails_loudly():
    d = _load()
    cfg = _v41_config()
    with pytest.raises(SystemExit):
        d.apply_dsv41(cfg, {"name": "bad", "sel": [0, 3]}, [])  # ratio-2 layer 3 without source 2


def test_stale_check_flags_layer_id_keys():
    """Regression: the generic cut left these verbatim and the check was silent."""
    d = _load()
    cfg = {"text_config": {"num_hidden_layers": 2, "kv_source_layer_ids": [2, 8, 14, 20],
                           "candidate_source_layer_id": 20, "engram_layer_ids": [1, 14]}}
    stale = d._check_no_stale_layer_refs(cfg, 2)
    assert any("kv_source_layer_ids" in s for s in stale)
    assert any("candidate_source_layer_id" in s for s in stale)
    assert any("engram_layer_ids" in s for s in stale)
    assert d._check_no_stale_layer_refs({"candidate_source_layer_id": -1}, 2) == []  # -1 = none


def test_generic_axis_from_layer_id_index_list():
    """Inkling: local_layer_ids (55 of 66) is a layer-kind partition; the
    representative cut must hold one local and one global layer and remap."""
    d = _load()
    n = 66
    local = [i for i in range(n) if i % 6 != 5]
    cfg = {"text_config": {"num_hidden_layers": n, "local_layer_ids": local}}
    variants = d.variants_generic(cfg)
    assert variants[0]["name"] == "all_kinds" and variants[0]["sel"] == [0, 5]
    edits: list[str] = []
    d.apply_generic(cfg, variants[0], edits)
    assert cfg["text_config"]["local_layer_ids"] == [0]
    assert d._check_no_stale_layer_refs(cfg, 2) == []


def test_stale_check_uses_nested_sub_config_depth():
    """Inkling mtp_config.local_layer_ids indexes the 8 MTP layers, not the backbone."""
    d = _load()
    cfg = {"text_config": {"num_hidden_layers": 2, "local_layer_ids": [0]},
           "mtp_config": {"num_nextn_predict_layers": 8, "local_layer_ids": [0, 2, 4, 5, 6, 7]},
           "vision_config": {"num_hidden_layers": 32, "layer_types": ["a"] * 32}}
    assert d._check_no_stale_layer_refs(cfg, 2) == []
    cfg["mtp_config"]["local_layer_ids"].append(9)
    assert d._check_no_stale_layer_refs(cfg, 2) != []
