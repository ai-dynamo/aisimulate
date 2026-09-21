# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical API admission and the independently frozen seven-shape oracle."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

import aisimulate_core
from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel
from aisimulate_core.sdk.errors import DecodeMoeProfileError, PrefillGraphProfileError

pytestmark = pytest.mark.unit
PROFILE = "sglang_glm52_nvfp4_vr200_tp4_graph_v1"
PROFILE_ID = "829a83e1629ba546dd4bd90e75a2e2496b7fb24ddc8b60dfbf076ba02312cbce"
VERSION = "0.5.18+nvinternal.rubin.0.8full.66997102"
# Independent graph-prediction-v5 comparison, SHA256
# 5e12de81a7b80628efb8513d76d268369b48d1b62424864edf7aff7604326930.
# Columns: public (batch,total ISL,prefix), operator prediction, native mean ms.
CASES = [
    ((1, 1024, 0), 34.90489051212317, 36.83244806925456),
    ((2, 1024, 0), 48.38665649293463, 50.99899724324544),
    ((1, 2048, 1024), 37.859508044217485, 39.544115193684895),
    ((1, 8192, 0), 196.09922835363085, 202.3783935546875),
    ((2, 8192, 0), 364.91345761325437, 374.7311482747396),
    ((1, 16384, 0), 378.53291416754246, 387.1900960286458),
    ((1, 32768, 16384), 442.6694767692677, 440.9783203125),
]


def graph_config():
    return {
        "model": "nvidia/GLM-5.2-NVFP4",
        "system": "vr200_hecate",
        "backend": "sglang",
        "backend_version": VERSION,
        "worker_type": "prefill",
        "tp": 4,
        "pp": 1,
        "attention_dp": 1,
        "moe_tp_size": 4,
        "moe_ep_size": 1,
        "gemm_quant_mode": "bfloat16",
        "moe_quant_mode": "nvfp4",
        "fmha_quant_mode": "bfloat16",
        "kvcache_quant_mode": "fp8",
        "comm_quant_mode": "half",
        "estimation_mode": "op_level",
        "fallback_policy": "deny",
        "database_mode": "SILICON",
        "enable_shared_layer": False,
        "strict_provenance": True,
        "systems_paths": [str(Path(aisimulate_core.__file__).parent / "systems")],
        "estimator_config": {
            "op_level": {"prefill_graph_profile": PROFILE},
            "correction": {"enabled": False},
        },
    }


@pytest.fixture(scope="module")
def graph_model():
    return RustForwardPassPerfModel.best_available(graph_config())


@pytest.mark.parametrize("call,expected,native", CASES)
def test_canonical_direct_prefill_matches_frozen_operator_and_native_evidence(graph_model, call, expected, native):
    result = graph_model.predict_prefill_latency(*call)
    assert result == pytest.approx(expected, rel=1e-12, abs=1e-10)
    assert abs(result / native - 1.0) <= 0.15
    assert graph_model.predict_prefill_latency(*call) == result


def test_canonical_saved_config_retains_profile_identity_and_reloads(graph_model):
    provenance = graph_model.diagnostics()["provenance"]
    saved = provenance["config"]
    assert saved["estimator_config"]["op_level"] == {
        "prefill_graph_profile": PROFILE,
        "prefill_graph_profile_id": PROFILE_ID,
    }
    assert saved["systems_paths"] == graph_config()["systems_paths"]
    assert provenance["selection_failures"] == []
    assert provenance["selected_estimation_mode"] == "op_level"
    assert saved["estimator_config"]["correction"]["enabled"] is False
    normalized = json.loads(aisimulate_core.RustForwardPassPerfModel.normalize_config(json.dumps(saved)))
    assert normalized == saved
    reloaded = RustForwardPassPerfModel.best_available(ForwardPassPerfModelConfig(**saved))
    assert reloaded.predict_prefill_latency(1, 2048, 1024) == graph_model.predict_prefill_latency(1, 2048, 1024)
    changed = copy.deepcopy(saved)
    changed["estimator_config"]["op_level"]["prefill_graph_profile_id"] = "0" * 64
    with pytest.raises(PrefillGraphProfileError, match="identity"):
        RustForwardPassPerfModel.best_available(changed)


def test_canonical_profile_owns_verified_tables_despite_default_cache_and_later_source_changes(tmp_path):
    config = graph_config()
    source = Path(config["systems_paths"][0])
    shutil.copy2(source / "vr200_hecate.yaml", tmp_path / "vr200_hecate.yaml")
    shutil.copytree(source / "data/vr200_hecate", tmp_path / "data/vr200_hecate")
    config["systems_paths"] = [str(tmp_path)]
    gemm = next(tmp_path.rglob("gemm_perf.parquet"))
    approved = gemm.read_bytes()
    table = pq.read_table(gemm)
    latency = [value * 10 for value in table["latency"].to_pylist()]
    poisoned = table.set_column(table.column_names.index("latency"), "latency", pa.array(latency))
    pq.write_table(poisoned, gemm)
    ordinary_config = copy.deepcopy(config)
    ordinary_config["estimator_config"]["op_level"] = {}
    ordinary = RustForwardPassPerfModel.best_available(ordinary_config)
    ordinary.static_phase_diagnostics(batch_size=1, context_length=1024, prefill=True)
    gemm.write_bytes(approved)
    selected = RustForwardPassPerfModel.best_available(config)
    # The reviewed bytes are bound at admission, before the first selected
    # query; an unrelated default engine cannot donate its previously cached rows.
    pq.write_table(poisoned, gemm)
    for call, expected, _ in CASES:
        assert selected.predict_prefill_latency(*call) == pytest.approx(expected, rel=1e-12, abs=1e-10)
    saved = selected.diagnostics()["provenance"]["config"]
    assert saved["systems_paths"] == [str(tmp_path)]
    assert saved["estimator_config"]["op_level"]["prefill_graph_profile_id"] == PROFILE_ID
    with pytest.raises(ValueError, match="retained input changed"):
        RustForwardPassPerfModel.best_available(saved)


@pytest.mark.parametrize(
    "key,value",
    [
        ("estimation_mode", "auto"),
        ("estimation_mode", "fpm_interpolation"),
        ("estimation_mode", "fpm_regression"),
        ("fallback_policy", "allow"),
        ("worker_type", "decode"),
        ("worker_type", "aggregated"),
        ("database_mode", "HYBRID"),
        ("database_mode", "SOL"),
        ("database_mode", "EMPIRICAL"),
    ],
)
def test_canonical_profile_rejects_unqualified_construction(key, value):
    config = graph_config()
    config[key] = value
    with pytest.raises(PrefillGraphProfileError, match="requires explicit"):
        RustForwardPassPerfModel.best_available(config)


def test_profile_requires_disabled_correction():
    config = graph_config()
    del config["estimator_config"]["correction"]
    with pytest.raises(PrefillGraphProfileError, match="disabled correction"):
        RustForwardPassPerfModel.best_available(config)


@pytest.mark.parametrize("value", [True, False, 1.0, "1", None, -1, 2**32])
def test_canonical_direct_call_and_raw_binding_preserve_exact_integer_admission(graph_model, value):
    for target in (graph_model, graph_model._inner):
        for call in [(value, 1024, 0), (1, value, 0), (1, 1024, value)]:
            with pytest.raises((PrefillGraphProfileError, OverflowError, TypeError)):
                target.predict_prefill_latency(*call)


@pytest.mark.parametrize("call", [(1, 2048, 0), (2, 1536, 512), (1, 1024, 1024), (0, 1024, 0), (2**32 - 1, 2, 0)])
def test_canonical_direct_call_rejects_same_tokens_wrong_shape_and_overflow(graph_model, call):
    with pytest.raises(PrefillGraphProfileError):
        graph_model.predict_prefill_latency(*call)


def test_profile_rejects_telemetry_tuning_and_energy_routes_including_empty_input(graph_model):
    for payload in [[], {}, [{"scheduled_requests": []}]]:
        with pytest.raises((PrefillGraphProfileError, ValueError)):
            graph_model.estimate_forward_pass_time_ms(payload)
    with pytest.raises(PrefillGraphProfileError, match="only direct"):
        graph_model.tune_with_fpms([])
    for prefill in [False, True]:
        with pytest.raises(PrefillGraphProfileError, match="only direct"):
            graph_model.static_phase_diagnostics(batch_size=1, context_length=1024, prefill=prefill)


def test_missing_profile_data_never_selects_a_fallback(tmp_path):
    config = graph_config()
    config["systems_paths"] = [str(tmp_path)]
    with pytest.raises(ValueError, match="vr200_hecate.yaml"):
        RustForwardPassPerfModel.best_available(config)


def test_pinned_pilot_runtime_is_excluded_from_the_generic_fleet_next_version(tmp_path):
    from aisimulate_core.sdk.perf_database import get_version_slots

    roots = graph_config()["systems_paths"]
    root = Path(roots[0])
    # Reconstruct the packaged version-discovery tree before this pilot: all
    # existing hardware/data, with neither VR hardware nor its override.
    (tmp_path / "data").symlink_to(root / "data", target_is_directory=True)
    original_slots = yaml.safe_load((root / "query_versions.yaml").read_text())
    original_slots["overrides"].pop("vr200_hecate")
    (tmp_path / "query_versions.yaml").write_text(yaml.safe_dump(original_slots))
    existing_systems = []
    for path in root.glob("*.yaml"):
        if path.name not in {"query_versions.yaml", "vr200_hecate.yaml"}:
            (tmp_path / path.name).symlink_to(path)
            existing_systems.append(path.stem)
    assert "gb300" in existing_systems
    for system in existing_systems:
        assert get_version_slots(system, "sglang", systems_paths=roots) == get_version_slots(
            system, "sglang", systems_paths=[str(tmp_path)]
        ), system
    assert get_version_slots("vr200_hecate", "sglang", systems_paths=roots) == {"current": VERSION}


@pytest.mark.parametrize(
    "distribution,batch,outside_profile",
    [
        ("observed_glm52_nvfp4_decode_1ab2c747975e_v1", 1, 33),
        ("observed_glm52_nvfp4_decode_composite_v2", 3, 5),
    ],
)
def test_prior_decode_profile_uses_canonical_controls_and_saved_configuration(distribution, batch, outside_profile):
    config = graph_config()
    config["worker_type"] = "decode"
    config["estimator_config"]["op_level"] = {"decode_workload_distribution": distribution}
    model = RustForwardPassPerfModel.best_available(config)
    saved = model.diagnostics()["provenance"]["config"]
    assert saved["estimator_config"]["op_level"] == {"decode_workload_distribution": distribution}
    reloaded = RustForwardPassPerfModel.best_available(saved)
    # The existing observed-MoE tests anchor the exact per-layer measurements.
    # This exercises canonical forwarding: a dropped selector would silently
    # use generic MoE rows at the unsupported logical batch below.
    operations = reloaded.static_phase_diagnostics(batch_size=batch, context_length=32768, prefill=False)
    assert any(op["name"] == "generation_moe_overlap" and op["latency_ms"] > 0 for op in operations)
    with pytest.raises(DecodeMoeProfileError):
        reloaded.static_phase_diagnostics(batch_size=outside_profile, context_length=32768, prefill=False)
