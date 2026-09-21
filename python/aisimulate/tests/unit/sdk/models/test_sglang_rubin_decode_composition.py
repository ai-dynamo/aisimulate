# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
# SPDX-License-Identifier: Apache-2.0
# Modified analytical composition fixtures based on SGLang:
# https://gitlab-master.nvidia.com/dl/sglang/sglang/-/tree/02c5a855aceb968c310e6fbc6632270e26edc84b/python/sglang/srt

"""Source-backed outer decode operations for the established Rubin regime."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import aisimulate_core as core
from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel, common, engine
from aisimulate_core.sdk.config_builders import build_model_config
from aisimulate_core.sdk.models import get_model
from aisimulate_core.sdk.speculation import SpeculationConfig

pytestmark = pytest.mark.unit
MODEL = "nvidia/GLM-5.2-NVFP4"
SYSTEM = "vr200_hecate"
VERSION = "0.5.18+nvinternal.rubin.0.8full.66997102"
SYSTEMS = str(Path(core.__file__).parent / "systems")
ADDED = {"generation_embedding_ar", "generation_routed_shared_add", "generation_final_add_norm"}
PROFILES = [None, "observed_glm52_nvfp4_decode_1ab2c747975e_v1", "observed_glm52_nvfp4_decode_composite_v2"]
MODEL_KWARGS = dict(
    tp_size=4,
    pp_size=1,
    attention_dp_size=1,
    moe_tp_size=4,
    moe_ep_size=1,
    gemm_quant_mode="bfloat16",
    moe_quant_mode="nvfp4",
    kvcache_quant_mode="fp8",
    fmha_quant_mode="bfloat16",
    comm_quant_mode="half",
)


def model_for(model_path=MODEL, backend="sglang", **changes):
    return get_model(model_path, build_model_config(**(MODEL_KWARGS | changes)), backend)


def spec_for(model, **changes):
    kwargs = dict(
        model_path=model.model_path,
        system=SYSTEM,
        backend="sglang",
        backend_version=VERSION,
        systems_path=SYSTEMS,
        kv_block_size=None,
        nextn=0,
        shared_layer=False,
        strict_provenance=True,
    )
    return json.loads(engine.build_engine_spec_json(model, **(kwargs | changes)))


def named(ops):
    return {fields["name"]: (tag, fields) for op in ops for tag, fields in op.items()}


@pytest.mark.parametrize("profile", PROFILES)
def test_exact_runtime_adds_three_source_backed_outer_operations(profile):
    model = model_for(decode_workload_distribution=profile)
    context = json.loads(engine.build_ops_json(model.context_ops))
    generation = json.loads(engine.build_ops_json(model.generation_ops))
    spec = spec_for(model)
    leaves = named(spec["generation_ops"])
    # vocab_parallel_embedding.py:566-579; deepseek_v2.py:1009-1030,
    # 2906-2910. Deferred finalize is disabled by the existing pilot runtime
    # contract; source defaults alone do not establish this composition.
    assert leaves.keys() >= ADDED
    assert leaves["generation_embedding_ar"] == (
        "CustomAllReduce",
        dict(name="generation_embedding_ar", scale_factor=1.0, hidden_size=6144, tp_size=4, quant="half", seq_split=1),
    )
    # BF16 h+h -> h: two reads and one write. Terminal residual RMSNorm
    # retains the existing analytical two-input/two-output convention.
    for name, scale, byte_count in (
        ("generation_routed_shared_add", 75, 36864),
        ("generation_final_add_norm", 1, 49152),
    ):
        assert leaves[name] == (
            "Elementwise",
            dict(name=name, scale_factor=scale, bytes_per_token=byte_count, scale_num_tokens=1, seq_split=1),
        )
    order = list(leaves)
    assert order[order.index("generation_embedding") + 1] == "generation_embedding_ar"
    assert order[order.index("generation_moe_overlap") + 1] == "generation_routed_shared_add"
    assert order[order.index("generation_logits_gemm") - 1] == "generation_final_add_norm"
    # No overlap movement, altered old leaf, or context edit is permitted.
    assert [op for op in spec["generation_ops"] if next(iter(op.values()))["name"] not in ADDED] == generation
    assert spec["context_ops"] == context
    assert spec_for(model) == spec
    assert json.loads(engine.build_ops_json(model.generation_ops)) == generation
    assert json.loads(engine.build_ops_json(model.context_ops)) == context


@pytest.mark.parametrize(
    "changes",
    [
        {"system": "b200_sxm"},
        {"backend_version": "0.5.18+another-build"},
        {"backend": "vllm"},
        {"model_path": "nvidia/GLM-5.1-NVFP4"},
    ],
)
def test_other_resolved_identities_preserve_cached_graph(changes):
    model = model_for()
    original = json.loads(engine.build_ops_json(model.generation_ops))
    qualified = spec_for(model)
    # A loaded database's literal version takes precedence over the request.
    # This isolates composition from unlisted-version admission, which has
    # independent coverage in test_engine_spec_version_literal.py.
    database = SimpleNamespace(version=changes.get("backend_version", VERSION))
    other = spec_for(model, **changes, database=database)
    assert other["generation_ops"] == original
    assert other["context_ops"] == qualified["context_ops"]
    assert spec_for(model) == qualified


def test_hook_uses_resolved_version_and_keeps_model_cache_unmodified():
    model = model_for()
    expected = spec_for(model)
    assert spec_for(model, backend_version="current") == expected
    assert spec_for(model, backend_version=None) == expected
    # A current alias must not qualify a database bound to another literal.
    other = spec_for(model, backend_version="current", database=SimpleNamespace(version="other-build"))
    assert other["generation_ops"] == json.loads(engine.build_ops_json(model.generation_ops))
    assert named(spec_for(model)["generation_ops"]).keys() >= ADDED


@pytest.mark.parametrize(
    "changes",
    [
        {"tp_size": 2, "moe_tp_size": 2},
        {"moe_tp_size": 2, "moe_ep_size": 2},
        {"pp_size": 2},
        {"tp_size": 2, "attention_dp_size": 2},
        {"tp_size": 1, "cp_size": 4},
        {"gemm_quant_mode": common.GEMMQuantMode.nvfp4},
        {"moe_quant_mode": common.MoEQuantMode.fp8},
        {"fmha_quant_mode": common.FMHAQuantMode.fp8},
        {"kvcache_quant_mode": common.KVCacheQuantMode.bfloat16},
        {"comm_quant_mode": common.CommQuantMode.fp8},
        {"nextn": 1},
        {"speculation": SpeculationConfig("mtp", {"depth": 1})},
        {"overwrite_num_layers": 4},
        {"overwrite_num_layers": 78},
        {"decoder_replay": True},
        {"enable_eplb": True},
        {"wideep_num_slots": 256},
        {"attention_backend": common.AttentionBackend.fa3},
        {"moe_backend": common.MoEBackend.megamoe},
        {"forward_model": "fpm"},
    ],
)
def test_other_model_configurations_keep_their_existing_operations(changes):
    cfg = replace(build_model_config(**MODEL_KWARGS), **changes)
    model = get_model(MODEL, cfg, "sglang")
    original = {
        phase: json.loads(engine.build_ops_json(getattr(model, phase))) for phase in ("context_ops", "generation_ops")
    }
    spec = spec_for(model)
    for phase, ops in original.items():
        assert spec[phase] == ops


@pytest.mark.parametrize("model_path,backend", [("nvidia/GLM-5.1-NVFP4", "sglang"), (MODEL, "vllm")])
def test_other_model_and_backend_graphs_are_unchanged(model_path, backend):
    model = model_for(model_path, backend)
    original = json.loads(engine.build_ops_json(model.generation_ops))
    assert spec_for(model, backend=backend, database=SimpleNamespace(version=VERSION))["generation_ops"] == original


def test_prefill_only_profile_retains_both_original_operation_lists():
    model = model_for(prefill_graph_profile="sglang_glm52_nvfp4_vr200_tp4_graph_v1")
    spec = spec_for(model)
    for phase in ("context_ops", "generation_ops"):
        assert spec[phase] == json.loads(engine.build_ops_json(getattr(model, phase)))


def canonical_config(profile):
    return {
        "model": MODEL,
        "system": SYSTEM,
        "backend": "sglang",
        "backend_version": VERSION,
        "worker_type": "decode",
        "tp": 4,
        "pp": 1,
        "attention_dp": 1,
        **{key: value for key, value in MODEL_KWARGS.items() if key not in ("tp_size", "pp_size", "attention_dp_size")},
        "estimation_mode": "op_level",
        "fallback_policy": "deny",
        "database_mode": "SILICON",
        "enable_shared_layer": False,
        "strict_provenance": True,
        "systems_paths": [SYSTEMS],
        "estimator_config": {
            "op_level": {"decode_workload_distribution": profile} if profile else {},
            "correction": {"enabled": False},
        },
    }


@pytest.mark.parametrize("profile", PROFILES)
def test_canonical_config_and_binary_reload_retain_inventory_and_values(profile, monkeypatch):
    captured = []
    original = engine.build_engine_spec_json

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(engine, "build_engine_spec_json", capture)
    model = RustForwardPassPerfModel.best_available(canonical_config(profile))
    spec_json = captured[-1]
    spec = json.loads(spec_json)
    assert named(spec["generation_ops"]).keys() >= ADDED
    saved = json.loads(json.dumps(model.diagnostics()["provenance"]["config"]))
    reloaded = RustForwardPassPerfModel.best_available(ForwardPassPerfModelConfig(**saved))
    assert json.loads(captured[-1]) == spec
    assert reloaded.diagnostics()["provenance"]["config"] == saved
    blob = bytes(core.engine_spec_bincode_from_json(spec_json))
    compiled = engine.EngineHandle(blob, systems_path=SYSTEMS)
    twice = engine.EngineHandle(compiled.spec_bytes, systems_path=SYSTEMS)
    for batch in (1, 3):
        call = dict(batch_size=batch, input_tokens=1024, output_tokens=2, prefill=False)
        latency = model.static_phase_latency(**call)
        assert reloaded.static_phase_latency(**call) == latency
        assert compiled.predict_decode_latency(batch, 1024, 2) == latency
        assert twice.predict_decode_latency(batch, 1024, 2) == latency
        diagnostics = model.static_phase_diagnostics(batch_size=batch, context_length=1024, prefill=False)
        assert diagnostics == reloaded.static_phase_diagnostics(batch_size=batch, context_length=1024, prefill=False)
        assert len(diagnostics) == len({op["name"] for op in diagnostics})
        # Independent hand-derived oracle in Rust runtime.rs:
        # glm52_rubin_decode_outer_terms_match_source_inventory_oracle.
        ar, add, norm = {
            1: (0.005369920134544372, 0.12416196776269622, 0.0016567196801166505),
            3: (0.005253988674708775, 0.12471401723901163, 0.0016665338930289244),
        }[batch]
        expected = dict(
            zip(
                ("generation_embedding_ar", "generation_routed_shared_add", "generation_final_add_norm"),
                (ar, add, norm),
                strict=True,
            )
        )
        for op in diagnostics:
            if op["name"] in expected:
                assert op["latency_ms"] == pytest.approx(expected[op["name"]], rel=1e-12)
                assert op["source"] == ("silicon" if op["name"] == "generation_embedding_ar" else "empirical")
        # Source-audited pre-change B1/B3 K1024 totals at 30a7377f.
        # Only the three independently priced outer leaves may change them.
        if profile is None:
            baseline = {1: 5.650986119005678, 3: 6.41587280664244}[batch]
            assert latency == pytest.approx(baseline + ar + add + norm, rel=1e-12)
