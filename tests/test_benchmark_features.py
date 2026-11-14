# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measured feature provenance must survive publication without guessed states."""

import copy
import hashlib
import json
import tomllib
from pathlib import Path

import pytest
from test_agentx_accuracy import bundle, result

from scripts.e2e_accuracy.benchmark_features import (
    FEATURES,
    SILICON_FEATURES,
    add_agentx_receipt,
    add_branch_receipt,
    add_preemption_receipt,
    from_vllm_receipt,
    validate,
    validate_qwen_baseline,
)
from scripts.e2e_accuracy.run_agentx_accuracy import normalize_native
from scripts.pages import agentx_pages
from scripts.pages.agentx_pages import validate_bundle
from scripts.pages.prepare_e2e_accuracy_pages import artifact_key


def test_feature_identifiers_are_registered_in_the_pinned_support_matrix():
    matrix = Path(__file__).resolve().parents[1] / "pages/feature-support-matrix/features.toml"
    assert set(tomllib.loads(matrix.read_text())["benchmark_features"]["ids"]) == set(FEATURES)


def metadata():
    return from_vllm_receipt(
        "agentx-446063",
        {"workers": [{"cache_config": {"enable_prefix_caching": True}, "parallelism": {}, "scheduler": {}}]},
        "a" * 64,
        {"revision": "b" * 40, "sha256": "c" * 64},
    )


def silicon_metadata(identity="qwen-weka-l1"):
    """Synthetic complete receipt; the raw payload is retained by silicon_inputs."""
    evidence = {
        "sha256": hashlib.sha256(agentx_pages.canonical({"settings": {"synthetic": True}})).hexdigest(),
        "pointer": "/settings",
    }
    return {
        "schema": "aisimulate.benchmark.features.v1",
        "benchmark_id": identity,
        "taxonomy": None,
        "features": [
            {
                "id": key,
                "state": "disabled",
                "observed": {"state": "unknown", "count": None},
                "evidence": [dict(evidence)],
                "parameters": [{"name": "scope", "value": "not_applicable"}]
                if key in {"cache.g2", "cache.g3"}
                else [{"name": "method", "value": "off"}, {"name": "draft_length", "value": None}]
                if key == "speculation"
                else [],
            }
            for key in sorted(SILICON_FEATURES)
        ],
    }


def feature_sidecar(summary):
    record = metadata()
    return {
        "schema": "aisimulate.agentx.features.v1",
        "measurement_dataset_revision": summary["snapshot"]["dataset_revision"],
        "feature_dataset_revision": "d" * 40,
        "measurement_catalog_sha256": "e" * 64,
        "feature_catalog_sha256": "f" * 64,
        "records": [
            {
                "benchmark_id": record["benchmark_id"],
                "feature_artifact_sha256": hashlib.sha256(agentx_pages.canonical(record)).hexdigest(),
                "feature_metadata": record,
            }
        ],
    }


def test_feature_backfill_keeps_qualified_prediction_bytes_and_revisions(tmp_path):
    files = bundle()
    summary = json.loads(files["summary.json"])
    branch = summary["snapshot"]["branch"]
    key = artifact_key(branch)
    destination = tmp_path / "published"
    base = destination / "branches" / key
    for name, content in files.items():
        target = base / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    catalog = {
        "schema_version": 1,
        "branches": [{"branch": branch, "status": "evaluated", "summary_path": f"branches/{key}/summary.json"}],
    }
    (destination / "branches.json").write_bytes(agentx_pages.canonical(catalog))
    sidecar = feature_sidecar(summary)
    evidence = tmp_path / "reviewed-features.json"
    evidence.write_bytes(agentx_pages.canonical(sidecar))
    agentx_pages.attach_feature_evidence(destination, branch, evidence)
    assert {name: (base / name).read_bytes() for name in files} == files
    descriptor = json.loads((destination / "branches.json").read_bytes())["branches"][0]["feature_evidence"]
    assert descriptor["path"] == f"branches/{key}/feature-evidence.json"
    assert descriptor["sha256"] == hashlib.sha256((base / "feature-evidence.json").read_bytes()).hexdigest()
    assert agentx_pages.validate_feature_evidence(sidecar, summary) == sidecar
    catalog["branches"][0]["summary_path"] = "../outside/summary.json"
    (destination / "branches.json").write_bytes(agentx_pages.canonical(catalog))
    with pytest.raises(ValueError, match="destination"):
        agentx_pages.attach_feature_evidence(destination, branch, evidence)


@pytest.mark.parametrize(
    "problem", ["dataset", "missing", "duplicate", "identity", "checksum", "private", "extra", "record_field", "path"]
)
def test_feature_backfill_rejects_mixed_incomplete_or_private_evidence(problem):
    summary = json.loads(bundle()["summary.json"])
    value = feature_sidecar(summary)
    record = value["records"][0]
    if problem == "dataset":
        value["measurement_dataset_revision"] = "f" * 40
    elif problem == "missing":
        value["records"] = []
    elif problem == "duplicate":
        value["records"].append(copy.deepcopy(record))
    elif problem == "identity":
        record["feature_metadata"]["benchmark_id"] = "another-benchmark"
    elif problem == "checksum":
        record["feature_artifact_sha256"] = "0" * 64
    elif problem == "private":
        record["feature_metadata"]["prompt"] = "private input"
    elif problem == "extra":
        value["raw_log"] = "private log"
    elif problem == "record_field":
        record["source_path"] = "/Users/private"
    elif problem == "path":
        record["feature_metadata"]["features"][0]["evidence"] = [{"sha256": "a" * 64, "pointer": "/Users/private"}]
    with pytest.raises(ValueError):
        agentx_pages.validate_feature_evidence(value, summary)


def test_enabled_does_not_claim_exercised_and_missing_is_unknown():
    value = metadata()
    rows = {row["id"]: row for row in value["features"]}
    assert rows["cache.g1_prefix"]["state"] == "enabled"
    assert rows["cache.g1_prefix"]["observed"] == {"state": "unknown", "count": None}
    assert rows["speculation"]["state"] == "unknown"
    assert set(rows) == set(FEATURES)
    assert validate(json.loads(json.dumps(value))) == value


@pytest.mark.parametrize("bad", ["/Users/private/file", "Bearer secret", "hf_secret", {"token_ids": [1]}])
def test_private_parameters_rejected(bad):
    value = metadata()
    value["features"][0]["parameters"] = [{"name": "implementation", "value": bad}]
    with pytest.raises(ValueError):
        validate(value)


def test_incomplete_inventory_wrong_identity_and_unsupported_claims_rejected():
    value = metadata()
    with pytest.raises(ValueError, match="another benchmark"):
        validate(value, "different-benchmark")
    truncated = copy.deepcopy(value)
    truncated["features"].pop()
    with pytest.raises(ValueError, match="Incomplete"):
        validate(truncated)
    value["features"][0]["observed"] = {"state": "exercised", "count": 0}
    with pytest.raises(ValueError, match="disagrees"):
        validate(value)


def test_public_feature_record_round_trip_and_legacy_bundle():
    files = bundle(result(feature_metadata=metadata()))
    assert validate_bundle(files)["runs"][0]["feature_metadata"] == metadata()
    assert "feature_metadata" not in validate_bundle(bundle())["runs"][0]
    with pytest.raises(ValueError, match="require silicon feature metadata"):
        validate_bundle(bundle(result(id="qwen-weka-l1", category="self_collected")))


def test_complete_receipt_accepts_actual_enabled_parameters_and_evidenced_na():
    value = silicon_metadata()
    enabled = {
        "cache.sub_block_prefix_reuse": {"match_granularity_tokens": 1, "block_size": 64},
        "cache.hybrid_state": {"cache_types": ["recurrent_state"]},
        "cache.g2": {"scope": "local", "size": 4096, "implementation": "host-cache"},
        "cache.g3": {"scope": "shared", "implementation": "storage-cache"},
        "speculation": {"method": "dspark", "draft_length": 7},
    }
    for row in value["features"]:
        parameters = enabled.get(row["id"])
        row["state"] = "enabled" if parameters else "not_applicable"
        if parameters:
            row["parameters"] = [{"name": name, "value": scalar} for name, scalar in parameters.items()]
    assert validate(value, require_complete=True) == value


@pytest.mark.parametrize(
    "identity",
    [
        f"self-p{length}-l{lanes}-{cache}-r{repeat}"
        for length in (4096, 32768)
        for lanes in (1, 4, 16)
        for cache in ("cold", "warm")
        for repeat in range(3)
    ],
)
def test_only_explicit_historical_finite_runs_may_omit_complete_receipts(identity):
    from scripts.e2e_accuracy.agentx_contract import HISTORICAL_SELF_COLLECTED

    length, lanes, cache, repeat = HISTORICAL_SELF_COLLECTED[identity]
    run = result(
        id=identity,
        category="self_collected",
        model="Qwen/Qwen3.5-397B-A17B",
        backend="vllm",
        boundary="complete_finite_cohort",
        initial_input_tokens=length,
        lanes=lanes,
        cache_state=cache,
        repeat=repeat,
    )
    assert validate_bundle(bundle(run))["runs"][0]["id"] == identity
    for change in ({"id": identity + "-new"}, {"boundary": "successful_profile_span"}, {"model": "another/model"}):
        with pytest.raises(ValueError, match="require silicon feature metadata"):
            validate_bundle(bundle({**run, **change}))


def test_weka_preparation_is_not_reclassified_as_finite_cohort():
    native = {"telemetry": {"requests": [{"request_id": "primer", "phase": "preparation"}]}}
    row = normalize_native(native, [], "self_collected", "successful_profile_span")[0]
    assert row["phase"] == "preparation"


def test_saved_harness_settings_are_distinct_from_observed_usage():
    value = metadata()
    configuration = {
        "datasets": [
            {
                "dataset": "semianalysis_cc_traces_weka_062126",
                "max_context_length": 262144,
                "random_seed": 42,
                "sampling": "sequential",
                "cache_bust": {"target": "none"},
            }
        ],
        "phases": [
            {
                "timing_mode": "agentic_replay",
                "duration": 3600,
                "grace_period": 30,
                "system_idle_gap_cap_seconds": 10,
                "warmup_requests_per_lane": 10,
            }
        ],
        "endpoint": {"urls": ["http://127.0.0.1:8800"]},
    }
    requests = [
        {"conversation_id": "child", "session_id": "session", "agent_depth": 1, "cached_tokens": 10},
        {"conversation_id": "child", "session_id": "session", "agent_depth": 1, "cached_tokens": 0},
    ]
    added = add_agentx_receipt(
        value, configuration, "d" * 64, [{"snapshot": {"t_star_ms": 10}}], "e" * 64, requests, "f" * 64
    )
    rows = {row["id"]: row for row in added["features"]}
    assert rows["agentx.weka"]["state"] == "enabled"
    assert rows["agentx.cache_bust"]["state"] == "disabled"
    assert rows["routing.planner"]["state"] == "disabled"
    assert rows["events.subagent_spawn"]["observed"]["count"] == 1
    assert rows["events.cache_reuse"]["observed"]["count"] == 1
    assert rows["events.state_restore"]["observed"]["state"] == "unknown"
    with pytest.raises(ValueError, match="unresolved"):
        validate_qwen_baseline(added)


def test_arbitrary_public_parameter_names_and_values_are_not_allowed():
    value = metadata()
    for parameter in ({"name": "password", "value": "secret"}, {"name": "mode", "value": "secret"}):
        value["features"][0]["parameters"] = [parameter]
        with pytest.raises(ValueError):
            validate(value)


def test_archive_is_validated_and_keeps_its_own_snapshot(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    features = silicon_metadata()
    current = bundle(result(id="qwen-weka-l1", category="self_collected", feature_metadata=features))
    historical = bundle(
        result(
            id="self-p4096-l1-cold-r0",
            category="self_collected",
            model="Qwen/Qwen3.5-397B-A17B",
            backend="vllm",
            initial_input_tokens=4096,
            cache_state="cold",
            boundary="complete_finite_cohort",
        )
    )
    for prefix, files in ((artifacts, current), (artifacts / "archive/finite-v1", historical)):
        for name, content in files.items():
            path = prefix / artifact_key("main") / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    monkeypatch.setattr(agentx_pages, "write_assets", lambda *_: None)
    agentx_pages.package(tmp_path, tmp_path / "site", ["main"], artifacts)
    archived = json.loads(
        (tmp_path / "site/archive/finite-v1/branches" / artifact_key("main") / "summary.json").read_bytes()
    )
    assert archived["runs"][0]["id"] == "self-p4096-l1-cold-r0"
    assert archived["snapshot"] == json.loads(historical["summary.json"])["snapshot"]
    damaged = artifacts / "archive/finite-v1" / artifact_key("main") / "summary.json"
    damaged.write_text("{}")
    with pytest.raises(ValueError):
        agentx_pages.package(tmp_path, tmp_path / "rejected", ["main"], artifacts)


def test_preemption_counts_retain_observed_interval_scope_without_claiming_disabled():
    receipt = {
        "profile_start_ns": "0",
        "profile_end_ns": "3000000000",
        "intervals": [{"start_ns": "1000000000", "end_ns": "2000000000", "preemptions": 0}],
    }
    rows = {row["id"]: row for row in add_preemption_receipt(metadata(), receipt, "a" * 64)["features"]}
    assert rows["events.preemption"]["state"] == "unknown"
    assert rows["events.preemption"]["observed"] == {"state": "not_observed", "count": 0}
    parameters = {p["name"]: p["value"] for p in rows["events.preemption"]["parameters"]}
    assert parameters["observation_seconds"] == 1
    assert parameters["profile_seconds"] == 3
    receipt["intervals"][0]["preemptions"] = 2
    rows = {row["id"]: row for row in add_preemption_receipt(metadata(), receipt, "a" * 64)["features"]}
    assert rows["events.preemption"]["observed"] == {"state": "exercised", "count": 2}


def test_branch_counters_preserve_dispatch_and_join_definitions_and_unknowns():
    value = add_branch_receipt(metadata(), {"branch_stats": {"children_spawned": 5, "parents_resumed": 2}}, "a" * 64)
    rows = {row["id"]: row for row in value["features"]}
    assert rows["events.subagent_spawn"]["observed"]["count"] == 5
    assert rows["events.join"]["observed"]["count"] == 2
    assert rows["events.tool_wait"]["observed"]["count"] is None
    assert rows["events.join"]["evidence"][0]["pointer"] == "/branch_stats/parents_resumed"
    with pytest.raises(ValueError, match="branch counter"):
        add_branch_receipt(metadata(), {"branch_stats": {"children_spawned": -1}}, "a" * 64)
