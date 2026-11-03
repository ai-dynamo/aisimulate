# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measured feature provenance must survive publication without guessed states."""

import copy
import json
import tomllib
from pathlib import Path

import pytest
from test_agentx_accuracy import bundle, result

from scripts.e2e_accuracy.benchmark_features import (
    FEATURES,
    add_agentx_receipt,
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
    value["features"][0]["parameters"] = [{"name": "hidden", "value": bad}]
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
    features = metadata()
    features["benchmark_id"] = "qwen-weka-l1"
    current = bundle(result(id="qwen-weka-l1", category="self_collected", feature_metadata=features))
    historical = bundle(result(id="finite-old", category="self_collected", boundary="complete_finite_cohort"))
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
    assert archived["runs"][0]["id"] == "finite-old"
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
