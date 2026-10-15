# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense synthetic KV needs positive cache and eligibility evidence, not a label alone."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from collector.fpm_forward.config import with_kv_warmup_defaults
from collector.fpm_forward.database import (
    aggregate_cell,
    published_dense_synthetic_cells,
    validate_formal_database_commit,
    write_formal_database,
)
from collector.fpm_forward.model_capability import ResolvedModelConfig
from collector.fpm_forward.planner import _canonical_hash
from collector.fpm_forward.repeatability import load_repeatability_source
from collector.fpm_forward.runtime_memory import saved_plan_identity

from aisimulate import main as cli
from aisimulate.support.collection_readiness import assess_readiness
from aisimulate.support.validation_workflow import _collection_source

from ..test_onboard_collection_readiness import _change_decode
from ..test_onboard_finalization import build_completed_collection
from ..test_support_serving_validation import materialized_workload  # noqa: F401
from ..test_support_validation import validation_case  # noqa: F401
from ..test_support_validation_workflow import quality_case  # noqa: F401
from .test_fpm_forward import _synthetic_decode_cell
from .test_fpm_measurement_evidence import _add_measurement_protocol

pytestmark = pytest.mark.unit

_DENSE_SKIP = {"enabled": True, "warm_eligible": False, "skip_reason": "dense_model_content_insensitive"}
_FULL = {
    "type": "FullAttentionSpec",
    "dtype": "torch.bfloat16",
    "block_size": 16,
    "num_kv_heads": 2,
    "head_size": 32,
    "sliding_window": None,
    "attention_chunk_size": None,
    "non_causal": False,
}


def _engine(plan, cell, *, version=None, groups=None):
    source = plan.capability.model_config.payload
    return {
        "model": {
            "model_type": source["model_type"],
            "architectures": source["architectures"],
            "model": plan.model_path,
        },
        "parallel": {
            "tensor_parallel_size": cell.topology.tp,
            "pipeline_parallel_size": cell.topology.pp,
            "data_parallel_size": cell.topology.dp,
            "prefill_context_parallel_size": cell.topology.cp,
            "decode_context_parallel_size": cell.topology.cp,
            "enable_expert_parallel": False,
            "data_parallel_rank": 0,
        },
        "kv_cache": {"groups": copy.deepcopy([_FULL] if groups is None else groups)},
        "versions": {"vllm": version or plan.capability.aic_database_version},
    }


def _dense_cell(tmp_path, *, groups=None):
    plan, cell, directory = _synthetic_decode_cell(
        tmp_path,
        kvwarm=_DENSE_SKIP,
        markers=("kvwarm_fake_fallback", "kvwarm_fake_fallback"),
        parallel_strategy="tp",
    )
    plan.capability.is_moe = False
    plan.capability.attention_source = "dense_attention"
    plan.capability.model_config = ResolvedModelConfig(
        {"model_type": "llama", "architectures": ["LlamaForCausalLM"]},
        source_kind="explicit",
        source_reference="test",
    )
    plan.cells = (cell,)
    path = directory / "raw/pod-0/benchmark.json"
    payload = json.loads(path.read_text())
    payload["engine"] = _engine(plan, cell, groups=groups)
    for row in payload["results"]:
        row["kv_seed_regime"] = "fake_fallback"
    path.write_text(json.dumps(payload))
    return plan, cell, directory, path


def _regimes(case):
    plan, cell, directory, _path = case
    return {row["kv_seed_regime"] for row in aggregate_cell(plan, cell, directory, expected_attempt_id="attempt")}


@pytest.mark.parametrize(
    "groups",
    [[_FULL], [_FULL, _FULL], [{"type": "UniformTypeKVCacheSpecs", "specs": [_FULL, _FULL]}]],
    ids=["full", "multiple-full", "wrapped-full"],
)
def test_dense_full_attention_preserves_native_samples_and_becomes_direct_eligible(tmp_path, groups):
    case = _dense_cell(tmp_path, groups=groups)
    plan, cell, directory, path = case
    before = path.read_bytes()
    rows = aggregate_cell(plan, cell, directory, expected_attempt_id="attempt")
    assert {row["kv_seed_regime"] for row in rows} == {"skip:dense_model_content_insensitive"}
    assert [row["latency_ms"] for row in rows] == [10.0, 20.0]
    assert path.read_bytes() == before
    assert {row["kv_seed_regime"] for row in json.loads(before)["results"]} == {"fake_fallback"}


@pytest.mark.parametrize(
    "groups",
    [
        [],
        [None],
        [{}],
        [{"type": "MLAAttentionSpec"}],
        [{"type": "SlidingWindowSpec"}],
        [{"type": "MambaSpec"}],
        [{**_FULL, "sliding_window": 512}],
        [{**_FULL, "attention_chunk_size": 128}],
        [{**_FULL, "non_causal": True}],
        [{key: value for key, value in _FULL.items() if key != "sliding_window"}],
        [{key: value for key, value in _FULL.items() if key != "attention_chunk_size"}],
        [{key: value for key, value in _FULL.items() if key != "non_causal"}],
        [_FULL, {"type": "ShortConvSpec"}],
        [{"type": "UniformTypeKVCacheSpecs"}],
        [{"type": "UniformTypeKVCacheSpecs", "specs": []}],
        [{"type": "UniformTypeKVCacheSpecs", "specs": {"layer": _FULL}}],
        [{"type": "UniformTypeKVCacheSpecs", "specs": [_FULL, {"type": "MambaSpec"}]}],
        [{**_FULL, "specs": [{"type": "ShortConvSpec"}]}],
    ],
)
def test_unknown_or_non_full_attention_cache_never_promotes_dense_skip(tmp_path, groups):
    assert _regimes(_dense_cell(tmp_path, groups=groups)) == {"fake_fallback"}


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("model", "model_type", "mamba"),
        ("model", "architectures", ["DifferentForCausalLM"]),
        ("model", "model", "another/checkpoint"),
        ("parallel", "tensor_parallel_size", 8),
        ("parallel", "tensor_parallel_size", True),
        ("parallel", "pipeline_parallel_size", 2),
        ("parallel", "prefill_context_parallel_size", 2),
        ("parallel", "decode_context_parallel_size", 2),
        ("parallel", "decode_context_parallel_size", True),
        ("parallel", "enable_expert_parallel", True),
        ("parallel", "enable_expert_parallel", 0),
        ("parallel", "data_parallel_size", 2),
        ("parallel", "data_parallel_rank", 1),
        ("versions", "vllm", "0.99.0"),
        ("kv_cache", "groups", None),
    ],
)
def test_contradictory_or_missing_initialized_identity_stays_fake_fallback(tmp_path, section, key, value):
    case = _dense_cell(tmp_path)
    payload = json.loads(case[-1].read_text())
    payload["engine"][section][key] = value
    case[-1].write_text(json.dumps(payload))
    assert _regimes(case) == {"fake_fallback"}
    del payload["engine"][section][key]
    case[-1].write_text(json.dumps(payload))
    assert _regimes(case) == {"fake_fallback"}


@pytest.mark.parametrize("missing", ["engine", "model", "parallel", "versions", "kv_cache"])
def test_historical_or_incomplete_engine_evidence_is_not_promoted(tmp_path, missing):
    case = _dense_cell(tmp_path)
    payload = json.loads(case[-1].read_text())
    (payload if missing == "engine" else payload["engine"]).pop(missing)
    case[-1].write_text(json.dumps(payload))
    assert _regimes(case) == {"fake_fallback"}


def test_failed_engine_capture_cannot_establish_dense_eligibility(tmp_path):
    case = _dense_cell(tmp_path)
    payload = json.loads(case[-1].read_text())
    payload["engine"]["capture_error"] = "cache inspection failed"
    case[-1].write_text(json.dumps(payload))
    assert _regimes(case) == {"fake_fallback"}


@pytest.mark.parametrize("location", ["root", "text_config"])
@pytest.mark.parametrize("field", ["num_local_experts", "num_experts", "n_routed_experts", "moe_num_experts"])
def test_expert_evidence_in_either_config_rejects_dense_skip(tmp_path, location, field):
    case = _dense_cell(tmp_path)
    capability = case[0].capability
    source = capability.model_config.payload
    target = source if location == "root" else source.setdefault("text_config", {})
    target[field] = 1  # Dynamo treats even one declared expert as a non-dense model.
    capability.model_config = ResolvedModelConfig(source, source_kind="explicit", source_reference="test")
    assert _regimes(case) == {"fake_fallback"}


@pytest.mark.parametrize("field,value", [("is_moe", True), ("attention_source", "mla_module")])
def test_frozen_capability_must_agree_with_dense_full_attention(tmp_path, field, value):
    case = _dense_cell(tmp_path)
    setattr(case[0].capability, field, value)
    assert _regimes(case) == {"fake_fallback"}


@pytest.mark.parametrize("field", ["is_moe", "attention_source", "model_config"])
def test_unavailable_frozen_metadata_does_not_establish_dense_eligibility(tmp_path, field):
    case = _dense_cell(tmp_path)
    setattr(case[0].capability, field, None)
    assert _regimes(case) == {"fake_fallback"}


def test_mounted_model_identity_requires_the_frozen_deployment_hash(tmp_path):
    case = _dense_cell(tmp_path / "campaign/cells")
    plan, _cell, directory, path = case
    overrides = {"K8sConfig": {"k8s_model_path_in_pvc": "checkpoint", "k8s_pvc_mount_path": "/models"}}
    plan.generator_config_sha256 = _canonical_hash(with_kv_warmup_defaults(overrides))
    generator = directory.parent.parent / "generator-overrides.json"
    generator.write_text(json.dumps(overrides))
    for name in ("fpm_env.sh", "collector-runtime-env.sh"):
        (directory / name).write_text("")
    (directory / "generator-request.json").write_text(
        json.dumps({"ServiceConfig": {"model_path": "/models/checkpoint"}})
    )
    (directory / "run.sh").write_text("engine_command=(python3 -m dynamo.vllm --model /models/checkpoint)\n")
    payload = json.loads(path.read_text())
    payload["engine"]["model"]["model"] = "/models/checkpoint"
    path.write_text(json.dumps(payload))
    assert _regimes(case) == {"skip:dense_model_content_insensitive"}
    overrides["K8sConfig"]["k8s_model_path_in_pvc"] = "another-checkpoint"
    generator.write_text(json.dumps(overrides))
    assert _regimes(case) == {"fake_fallback"}


def test_historical_saved_identity_retains_dense_evidence_without_reloading_sources():
    source = Path(__file__).parent / "fixtures/fpm_collection_plan_v10.json"
    original = source.read_bytes()
    payload = json.loads(original)
    identity = saved_plan_identity(payload)
    assert identity.sha256 == payload["sha256"]
    assert identity.generator_config_sha256 == payload["generator_config_sha256"]
    assert identity.capability.is_moe is payload["capability"]["is_moe"]
    assert identity.capability.attention_source == payload["capability"]["attention_source"]
    assert identity.capability.model_config.payload == payload["capability"]["model_config"]["payload"]
    assert source.read_bytes() == original


@pytest.mark.parametrize(
    "meta",
    [
        {**_DENSE_SKIP, "enabled": False},
        {"enabled": True, "warm_eligible": True, "skip_reason": None},
        {"enabled": True, "warm_eligible": False, "skip_reason": "dataset_unavailable"},
        {"enabled": True, "warm_eligible": False, "skip_reason": "hybrid_state_layers_unsupported"},
    ],
)
def test_full_attention_does_not_convert_failed_or_disabled_warmup(tmp_path, meta):
    case = _dense_cell(tmp_path)
    payload = json.loads(case[-1].read_text())
    payload["kvwarm"] = meta
    case[-1].write_text(json.dumps(payload))
    assert _regimes(case) == {"fake_fallback"}


def test_dense_skip_with_real_kv_injection_is_rejected(tmp_path):
    case = _dense_cell(tmp_path)
    payload = json.loads(case[-1].read_text())
    for row, group in zip(payload["results"], payload["iteration_groups"], strict=True):
        row["kv_seed_regime"] = "real_kv"
        row["point"]["sample_reasons"] = ["kvwarm_real_kv"]
        group["point"]["sample_reasons"] = ["kvwarm_real_kv"]
    case[-1].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="real KV.*warm-up"):
        _regimes(case)


def test_unstamped_dense_point_is_not_promoted_from_skip_reason_alone(tmp_path):
    case = _dense_cell(tmp_path)
    payload = json.loads(case[-1].read_text())
    for row, group in zip(payload["results"], payload["iteration_groups"], strict=True):
        row.pop("kv_seed_regime")
        row["point"]["sample_reasons"] = []
        group["point"]["sample_reasons"] = []
    case[-1].write_text(json.dumps(payload))
    assert _regimes(case) == {"fake_fallback"}


def test_existing_moe_by_construction_and_legacy_contracts_are_unchanged(tmp_path):
    plan, cell, directory = _synthetic_decode_cell(
        tmp_path,
        kvwarm={**_DENSE_SKIP, "skip_reason": "moe_tp_balanced_by_construction"},
        parallel_strategy="tp",
        markers=("kvwarm_fake_fallback",),
    )
    assert {row["kv_seed_regime"] for row in aggregate_cell(plan, cell, directory, expected_attempt_id="attempt")} == {
        "skip:moe_tp_balanced_by_construction"
    }
    path = directory / "raw/pod-0/benchmark.json"
    payload = json.loads(path.read_text())
    del payload["kvwarm"]
    path.write_text(json.dumps(payload))
    assert {row["kv_seed_regime"] for row in aggregate_cell(plan, cell, directory, expected_attempt_id="attempt")} == {
        "legacy"
    }


def test_dense_publication_keeps_legacy_classification_until_fresh_publication(tmp_path):
    import pyarrow.parquet as pq

    plan, cell, directory, _path = _dense_cell(tmp_path)
    legacy_rows = aggregate_cell(plan, cell, directory, expected_attempt_id="attempt", dense_synthetic_kv=False)
    old_table, old_meta, _ = write_formal_database(plan, legacy_rows, systems_root=tmp_path / "old")
    old_bytes = old_table.read_bytes(), old_meta.read_bytes()
    committed = validate_formal_database_commit(old_table, old_meta, plan)
    assert published_dense_synthetic_cells(committed) == frozenset()
    assert pq.read_table(old_table).to_pylist() == legacy_rows
    current_rows = aggregate_cell(plan, cell, directory, expected_attempt_id="attempt")
    with pytest.raises(ValueError, match="conflicting FPM database row"):
        write_formal_database(plan, current_rows, systems_root=tmp_path / "old")
    assert (old_table.read_bytes(), old_meta.read_bytes()) == old_bytes

    new_table, new_meta, _ = write_formal_database(plan, current_rows, systems_root=tmp_path / "new")
    committed = validate_formal_database_commit(new_table, new_meta, plan)
    assert published_dense_synthetic_cells(committed) == {cell.cell_id}
    assert pq.read_table(new_table).to_pylist() == current_rows
    assert [row["latency_ms"] for row in legacy_rows] == [row["latency_ms"] for row in current_rows]
    assert (old_table.read_bytes(), old_meta.read_bytes()) == old_bytes


@pytest.mark.parametrize("decode_cp", [None, 1, 2])
def test_formal_dense_publication_requires_observed_decode_context_parallel_size_one(tmp_path, decode_cp):
    plan, cell, directory, path = _dense_cell(tmp_path)
    # DCP2 is valid with TP2, but differs from this frozen CP1 collection.
    cell = replace(cell, topology=replace(cell.topology, tp=2))
    plan.cells = (cell,)
    payload = json.loads(path.read_text())
    parallel = payload["engine"]["parallel"]
    parallel["tensor_parallel_size"] = 2
    if decode_cp is None:
        parallel.pop("decode_context_parallel_size")
    else:
        parallel["decode_context_parallel_size"] = decode_cp
    path.write_text(json.dumps(payload))
    raw = path.read_bytes()
    rows = aggregate_cell(plan, cell, directory, expected_attempt_id="attempt")
    expected = "skip:dense_model_content_insensitive" if decode_cp == 1 else "fake_fallback"
    assert {row["kv_seed_regime"] for row in rows} == {expected}
    legacy = aggregate_cell(plan, cell, directory, expected_attempt_id="attempt", dense_synthetic_kv=False)
    assert {row["kv_seed_regime"] for row in legacy} == {"fake_fallback"}
    assert [row["latency_ms"] for row in rows] == [row["latency_ms"] for row in legacy]
    table, metadata, skipped = write_formal_database(plan, rows, systems_root=tmp_path / "published")
    assert not skipped
    committed = validate_formal_database_commit(table, metadata, plan)
    assert published_dense_synthetic_cells(committed) == ({cell.cell_id} if decode_cp == 1 else set())
    assert path.read_bytes() == raw


@pytest.mark.parametrize("change", ["removed", "other-cell", "empty", "duplicate", "unknown-policy"])
def test_new_dense_publication_requires_bound_classification_marker(tmp_path, change):
    plan, cell, directory, _path = _dense_cell(tmp_path)
    rows = aggregate_cell(plan, cell, directory, expected_attempt_id="attempt")
    table, meta, _ = write_formal_database(plan, rows, systems_root=tmp_path / "systems")
    payload = json.loads(meta.read_text())
    if change == "removed":
        payload.pop("kv_seed_classification")
    elif change == "unknown-policy":
        payload["kv_seed_classification"]["policy"] = "unknown"
    else:
        payload["kv_seed_classification"]["cells"] = {
            "other-cell": ["different-cell"],
            "empty": [],
            "duplicate": [cell.cell_id, cell.cell_id],
        }[change]
    meta.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="KV seed classification"):
        validate_formal_database_commit(table, meta, plan)
    with pytest.raises(ValueError, match="KV seed classification"):
        write_formal_database(plan, rows, systems_root=tmp_path / "systems")


def test_merged_database_does_not_upgrade_old_dense_cells(tmp_path):
    import pyarrow.parquet as pq

    plan, cell, directory, _path = _dense_cell(tmp_path)
    old = aggregate_cell(plan, cell, directory, expected_attempt_id="attempt", dense_synthetic_kv=False)
    table, meta, _ = write_formal_database(plan, old, systems_root=tmp_path / "systems")
    new = aggregate_cell(plan, cell, directory, expected_attempt_id="attempt")
    for row in new:
        row["cell_id"] = "another-cell"
    write_formal_database(plan, new, systems_root=tmp_path / "systems")
    committed = validate_formal_database_commit(table, meta, plan)
    assert published_dense_synthetic_cells(committed) == {"another-cell"}
    assert [row for row in pq.read_table(table).to_pylist() if row["cell_id"] == cell.cell_id] == old


@pytest.mark.parametrize("observed", ["full", "convolution", "unreported"])
def test_public_readiness_uses_corroborated_dense_regime(quality_case, observed):  # noqa: F811
    case = quality_case
    _change_decode(case, fake="all", kvwarm=_DENSE_SKIP)
    for cell in case["plan"].cells:
        if cell.workload_kind != "decode":
            continue
        for path in (case["campaign"] / "cells" / cell.cell_id / "raw").rglob("benchmark-dp*.json"):
            payload = json.loads(path.read_text())
            if observed != "unreported":
                groups = [_FULL] if observed == "full" else [_FULL, {"type": "ShortConvSpec"}]
                payload["engine"] = _engine(case["plan"], cell, groups=groups)
            _add_measurement_protocol(payload)
            path.write_text(json.dumps(payload))
    before = {path: path.read_bytes() for path in case["root"].rglob("*") if path.is_file()}
    report = assess_readiness(case["request"], case["root"], case["checkpoint"].parent)
    decode = next(row for row in report["selected_cells"] if row["cell"]["workload_kind"] == "decode")
    assert decode["formal_regime_counts"], decode["blockers"]
    assert decode["native_regime_counts"] == {"fake_fallback": 12}
    assert decode["direct_eligible_points"] == (12 if observed == "full" else 0)
    assert decode["status"] == ("ready" if observed == "full" else "blocked")
    assert decode["formal_regime_counts"] == {
        "skip:dense_model_content_insensitive" if observed == "full" else "fake_fallback": 12
    }
    assert "do not assess or upgrade an existing published table" in decode["classification_scope"]
    assert all(rank["verified"] for rank in decode["rank_diagnostics"])
    assert {path: path.read_bytes() for path in case["root"].rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("legacy", [True, False])
def test_quality_source_revalidates_exact_old_or_new_published_classification(quality_case, legacy):  # noqa: F811
    import pyarrow as pa
    import pyarrow.parquet as pq

    case = quality_case
    _change_decode(case, fake="all", kvwarm=_DENSE_SKIP)
    all_rows = []
    for cell in case["plan"].cells:
        directory = case["campaign"] / "cells" / cell.cell_id
        if cell.workload_kind == "decode":
            for path in (directory / "raw").rglob("benchmark-dp*.json"):
                payload = json.loads(path.read_text())
                payload["engine"] = _engine(case["plan"], cell)
                _add_measurement_protocol(payload)
                path.write_text(json.dumps(payload))
        all_rows.extend(
            aggregate_cell(
                case["plan"], cell, directory, expected_attempt_id="synthetic-source", dense_synthetic_kv=not legacy
            )
        )
    # Replace only fixture setup, before the publication under test exists.
    for path in (case["root"] / "systems/data").rglob("fpm_forward_perf.*"):
        path.unlink()
    table, meta, _ = write_formal_database(case["plan"], all_rows, systems_root=case["root"] / "systems/data")
    before = table.read_bytes(), meta.read_bytes()
    source, _campaign, _checkpoint, _evidence = _collection_source(case["request"], case["root"])
    assert source.sha256 == case["plan"].sha256
    readiness = assess_readiness(case["request"], case["root"], case["checkpoint"].parent)
    decode = next(row for row in readiness["selected_cells"] if row["cell"]["workload_kind"] == "decode")
    assert decode["direct_eligible_points"] == 12
    assert "do not assess or upgrade an existing published table" in decode["classification_scope"]
    assert (table.read_bytes(), meta.read_bytes()) == before
    published_rows = pq.read_table(table).to_pylist()
    assert {row["kv_seed_regime"] for row in published_rows if row["workload_kind"] == "decode"} == {
        "fake_fallback" if legacy else "skip:dense_model_content_insensitive"
    }
    published_rows[0]["latency_ms"] *= 2
    pq.write_table(pa.Table.from_pylist(published_rows), table)
    with pytest.raises(ValueError, match="does not match its commit record"):
        _collection_source(case["request"], case["root"])
    changed_meta = json.loads(meta.read_text())
    changed_meta["parquet_sha256"] = hashlib.sha256(table.read_bytes()).hexdigest()
    meta.write_text(json.dumps(changed_meta))
    with pytest.raises(ValueError, match="published formal rows differ"):
        _collection_source(case["request"], case["root"])


@pytest.mark.parametrize("legacy", [True, False])
def test_public_finalization_preserves_old_or_new_dense_classification(tmp_path, legacy):
    import pyarrow.parquet as pq

    _request, root = build_completed_collection(tmp_path)
    campaign = next((root / "fpm-artifacts").iterdir())
    plan = load_repeatability_source(campaign)
    rows = []
    for cell in plan.cells:
        directory = campaign / "cells" / cell.cell_id
        if cell.workload_kind == "decode":
            path = directory / "raw/pod-0/benchmark.json"
            payload = json.loads(path.read_text())
            payload["engine"] = _engine(plan, cell)
            payload["kvwarm"] = _DENSE_SKIP
            for row, group in zip(payload["results"], payload["iteration_groups"], strict=True):
                row["kv_seed_regime"] = "fake_fallback"
                row["point"]["sample_reasons"] = ["kvwarm_fake_fallback"]
                group["point"]["sample_reasons"] = ["kvwarm_fake_fallback"]
            path.write_text(json.dumps(payload))
        rows.extend(
            aggregate_cell(
                plan,
                cell,
                directory,
                expected_attempt_id=f"attempt-{cell.workload_kind}",
                dense_synthetic_kv=not legacy,
            )
        )
    for path in (root / "systems/data").rglob("fpm_forward_perf.*"):
        path.unlink()
    table, _meta, _ = write_formal_database(plan, rows, systems_root=root / "systems/data")
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    resolved = tmp_path / "resolved"
    assert (
        cli.main(
            [
                "onboard",
                "finalize",
                "--config",
                str(root / "request.yaml"),
                "--output-dir",
                str(root),
                "--resolved-output-dir",
                str(resolved),
            ]
        )
        == 0
    )
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == before
    exported = next(resolved.rglob("fpm_forward_perf.parquet"))
    assert exported.read_bytes() == table.read_bytes()
    assert {
        row["kv_seed_regime"] for row in pq.read_table(exported).to_pylist() if row["workload_kind"] == "decode"
    } == {"fake_fallback" if legacy else "skip:dense_model_content_insensitive"}
