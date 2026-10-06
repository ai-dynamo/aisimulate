# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""dsv411: the parallel DeepSeek-V4.1 decomposition selected by the Rust-owned
``dsv41_family`` switch. Coexists with ``DEEPSEEKV41`` (legacy)."""

from __future__ import annotations

import json

import pytest

from aisimulate_core.sdk import config as sdk_config
from aisimulate_core.sdk.deepseek_v41 import MODEL_PATH
from aisimulate_core.sdk.engine import EngineHandle, compile_engine
from aisimulate_core.sdk.models import get_model

pytestmark = pytest.mark.unit

_SYSTEM = "gb300"
_VERSION = {"sglang": "0.5.14", "vllm": "0.24.0"}


def _model(backend: str, family: str = "dsv411", tp: int = 4):
    cfg = sdk_config.ModelConfig(tp_size=tp, moe_tp_size=tp, moe_ep_size=1, system=_SYSTEM)
    cfg.dsv41_family = family
    return get_model(MODEL_PATH, cfg, backend)


def _specs(model, context: bool) -> list[dict]:
    ops = model.context_ops if context else model.generation_ops
    return [json.loads(op._spec_json()) for op in ops]


def _stage_children(spec: dict) -> list[dict]:
    return spec["Dsv411Stage"]["children"]


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
def test_switch_selects_the_dsv411_family_and_keeps_legacy_default(backend):
    legacy = _model(backend, "legacy")
    ops_family = _model(backend)
    assert legacy.model_family == "DEEPSEEKV41"
    assert ops_family.model_family == "DEEPSEEKV411"
    assert type(legacy).__name__ == "DeepSeekV41Model"
    assert type(ops_family).__name__ == "DeepSeekV411Model"


def test_stage_children_follow_layer_roles_and_backend_facts():
    model = _model("sglang")
    d = model.extra_params
    stages = [s for s in _specs(model, True) if "Dsv411Stage" in s]
    assert len(stages) == d.num_hidden_layers
    for layer, stage in enumerate(stages):
        kinds = [next(iter(child)) for child in _stage_children(stage)]
        role = d.layer_role(layer)
        assert ("Dsv411Indexer" in kinds) == (role in ("full", "reindex")), (layer, role, kinds)
        assert kinds.count("Dsv411AttentionCore") == 1
        assert kinds.count("Dsv411Mhc") == 1
        assert kinds.count("Dsv411SharedLinear") == 2
        assert ("Dsv411Engram" in kinds) == (layer in d.engram_layer_ids)
        core = next(c["Dsv411AttentionCore"] for c in _stage_children(stage) if "Dsv411AttentionCore" in c)
        assert core["role"] == role
        assert core["name"] == "context_attention"
        assert core["kv_layout"] == {"window_entry_bytes": 584.0, "main_entry_bytes": 584.0, "index_entry_bytes": 68.0}
        assert core["fmha_quant_mode"] == "fp8"
    # No overlap coefficient anywhere: stages are plain sequential sums.
    assert all(set(s["Dsv411Stage"]) == {"name", "is_context", "children"} for s in stages)


def test_indexer_carries_per_backend_scoring_facts():
    def indexers(model):
        return [
            c["Dsv411Indexer"]
            for s in _specs(model, False)
            if "Dsv411Stage" in s
            for c in _stage_children(s)
            if "Dsv411Indexer" in c
        ]

    sglang = indexers(_model("sglang"))
    vllm = indexers(_model("vllm"))
    assert len(sglang) == len(vllm) > 0
    assert {i["index_entry_bytes"] for i in sglang} == {68.0}
    assert {i["scoring_quant_mode"] for i in sglang} == {"bfloat16"}
    assert {i["skip_within_topk"] for i in sglang} == {False}
    assert {i["index_entry_bytes"] for i in vllm} == {132.0}
    assert {i["scoring_quant_mode"] for i in vllm} == {"fp8"}
    assert {i["skip_within_topk"] for i in vllm} == {True}
    assert all(i["name"] == "generation_indexer" for i in sglang)
    source = [i for i in vllm if i["is_candidate_source"]]
    assert len(source) == 1 and source[0]["candidate_limit"] == 0


def test_engram_sharding_is_a_backend_fact():
    def engrams(model):
        return [
            c["Dsv411Engram"]
            for s in _specs(model, True)
            if "Dsv411Stage" in s
            for c in _stage_children(s)
            if "Dsv411Engram" in c
        ]

    assert {e["sharding"] for e in engrams(_model("sglang"))} == {"row"}
    assert {e["sharding"] for e in engrams(_model("vllm"))} == {"head"}


def test_kv_memory_uses_backend_index_entry_bytes():
    sglang = _model("sglang")
    vllm = _model("vllm")
    legacy = _model("sglang", "legacy")
    seq = 65536
    assert vllm.get_kvcache_bytes_per_sequence(seq) > sglang.get_kvcache_bytes_per_sequence(seq)
    # The 584-byte compressed row is shared with the legacy sglang contract.
    assert sglang.get_kvcache_bytes_per_sequence(seq) == pytest.approx(legacy.get_kvcache_bytes_per_sequence(seq))
    assert sglang.get_resident_weights_bytes() == pytest.approx(legacy.get_resident_weights_bytes(), rel=0.05)


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
def test_compile_engine_sol_prediction_is_finite_and_monotone(backend):
    handle = EngineHandle.compile(
        MODEL_PATH,
        _SYSTEM,
        backend,
        backend_version=_VERSION[backend],
        tp_size=4,
        moe_tp_size=4,
        moe_ep_size=1,
        dsv41_family="dsv411",
        database_mode="SOL",
    )
    short = handle.predict_prefill_latency(1, 1024, 0)
    long = handle.predict_prefill_latency(1, 8192, 0)
    assert 0 < short < long
    decode_small = handle.predict_decode_latency(1, 4096)
    decode_big = handle.predict_decode_latency(64, 65536)
    assert 0 < decode_small < decode_big


def test_switch_rejections():
    with pytest.raises(ValueError, match="dsv41_family"):
        compile_engine(MODEL_PATH, _SYSTEM, "sglang", dsv41_family="ops")
    with pytest.raises(ValueError, match="sglang' or 'vllm"):
        compile_engine(MODEL_PATH, _SYSTEM, "trtllm", dsv41_family="dsv411")
    with pytest.raises(ValueError, match="decoder_replay"):
        compile_engine(MODEL_PATH, _SYSTEM, "sglang", dsv41_family="dsv411", decoder_replay=True)
    cfg = sdk_config.ModelConfig(tp_size=4, moe_tp_size=4, moe_ep_size=1, system=_SYSTEM, forward_model="fpm")
    cfg.dsv41_family = "dsv411"
    with pytest.raises(NotImplementedError, match="op-level family"):
        get_model(MODEL_PATH, cfg, "sglang")


def test_legacy_engine_config_dict_carries_the_switch():
    from aisimulate_core.sdk.rust_engine_step import ForwardPassPerfModelConfig

    legacy = {
        "schema_version": 1,
        "model_name": MODEL_PATH,
        "system_name": _SYSTEM,
        "backend": "vllm",
        "backend_version": _VERSION["vllm"],
        "tp_size": 4,
        "pp_size": 1,
        "moe_tp_size": 4,
        "moe_ep_size": 1,
        "attention_dp_size": 1,
        "database_mode": "SOL",
        "dsv41_family": "dsv411",
    }
    config = ForwardPassPerfModelConfig.from_legacy_engine_config(legacy, "aggregated")
    assert config.dsv41_family == "dsv411"
