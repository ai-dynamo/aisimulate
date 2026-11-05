# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Effective simulation records are complete, scoped, and separate from silicon."""

import copy
import hashlib
import json
from types import SimpleNamespace
from typing import ClassVar

import pytest
from test_agentx_accuracy import result, silicon_inputs, simulation_record

from scripts.e2e_accuracy.agentx_contract import canonical, validate_public
from scripts.e2e_accuracy.agentx_inputs import load_runs
from scripts.e2e_accuracy.simulation_features import (
    build,
    compare_speculation,
    routing_feature,
    speculation_settings,
    validate,
)


class ResolvedEngine(SimpleNamespace):
    model_fields: ClassVar = {"nextn": SimpleNamespace(metadata=[SimpleNamespace(le=5)])}


def compiled_record(*, roles=None, backend="sglang", prefix=True, unit=None, state=None, nextn=0, scheme=None):
    # Model output includes resolved defaults even when the authored config omitted them.
    roles = roles or {"aggregated": {"enable_prefix_caching": prefix, "block_size": 64}}
    workers = {
        name: {"kv_cache": {"host_offload": None, "g3_offload": None, "state_cache": None, "kvcr": None}}
        for name in roles
    }
    if unit is not None:
        roles["aggregated"]["prefix_match_unit"] = unit
    if state is not None:
        roles["aggregated"]["state_cache"] = {"bytes_per_request": state}
    engine = ResolvedEngine(nextn=nextn, nextn_accepted=1.99 if nextn else None, speculation=scheme)
    parsed = SimpleNamespace(
        engine=engine, model_dump=lambda **_: {"engine": {"model": "org/model", "workers": workers}}
    )
    deployment = SimpleNamespace(
        backend=backend,
        backend_version="1.2.3",
        **{
            field: roles.get(role)
            for role, field in (
                ("aggregated", "agg_engine_args"),
                ("prefill", "prefill_engine_args"),
                ("decode", "decode_engine_args"),
            )
        },
    )
    spec = SimpleNamespace(backend_deployment=deployment, execution_mode="offline", adapters={})
    return build(
        parsed,
        spec,
        {"telemetry": {"requests": [{"first_admission_reused_input_tokens": 64}]}},
        {"benchmark_id": "benchmark", "evaluated_revision": "a" * 40, "dataset_revision": "b" * 40},
        architecture="OtherArchitecture",
        dspark_architectures={"KimiArchitecture"},
    )


def rows(record, index=0):
    return {r["id"]: r for r in record["roles"][index]["features"]}


def test_resolved_defaults_are_complete_without_authored_cli_flags():
    record = compiled_record()
    features = rows(record)
    assert len(features) == 8
    assert features["cache.g1_prefix"]["state"] == "enabled"
    assert features["cache.g1_prefix"]["observed"] == {"state": "exercised", "count": 1}
    assert all(
        features[k]["state"] == "disabled"
        for k in ("cache.g2", "cache.g3", "cache.hybrid_state", "cache.kvcr", "speculation")
    )
    assert features["routing.kv_aware"]["state"] == "not_applicable"
    assert record["evaluated_revision"] == "a" * 40
    validate_public({"simulation_features": record})


def test_explicit_off_and_backend_derived_matching():
    assert rows(compiled_record(prefix=False))["cache.sub_block_prefix_reuse"]["state"] == "not_applicable"
    for backend in ("sglang", "vllm"):
        row = rows(compiled_record(backend=backend))["cache.sub_block_prefix_reuse"]
        assert row["state"] == "disabled"
        assert row["parameters"]["match_granularity_tokens"] == 64
    row = rows(compiled_record(backend="vllm", unit=16, state=4096))["cache.sub_block_prefix_reuse"]
    assert row["state"] == "enabled"
    assert rows(compiled_record(backend="trtllm"))["cache.sub_block_prefix_reuse"]["state"] == "failed_to_record"


def test_distinct_role_settings_and_scopes_are_not_collapsed():
    record = compiled_record(
        roles={
            "prefill": {
                "enable_prefix_caching": False,
                "block_size": 64,
                "native_host_offload": {"scope": "dp_rank_local"},
                "kvcr": {"bandwidth_gb_s": 100},
            },
            "decode": {"enable_prefix_caching": True, "block_size": 64, "g3_offload": {"scope": "cluster_shared"}},
        }
    )
    assert [r["role"] for r in record["roles"]] == ["prefill", "decode"]
    assert rows(record)["cache.g1_prefix"]["state"] == "disabled"
    assert rows(record, 1)["cache.g1_prefix"]["state"] == "enabled"
    assert rows(record)["cache.g2"]["parameters"]["scope"] == "dp_rank_local"
    assert rows(record, 1)["cache.g3"]["parameters"]["scope"] == "cluster_shared"
    assert rows(record)["cache.kvcr"]["state"] == "enabled"
    assert rows(record)["cache.kvcr"]["observed"]["state"] == "unknown"


def test_generic_and_legacy_speculation_use_evaluated_resolution_and_limits():
    legacy = ResolvedEngine(nextn=3, nextn_accepted=1.99, speculation=None)
    assert speculation_settings(legacy, "KimiArchitecture", {"KimiArchitecture"})["method"] == "dspark"
    assert speculation_settings(legacy, "OtherArchitecture", {"KimiArchitecture"})["method"] == "mtp"
    assert speculation_settings(legacy, None, None) is None
    generic = ResolvedEngine(kind="dspark", num_speculative_tokens=7, expected_accepted_tokens=2)
    legacy.speculation = generic
    settings = speculation_settings(legacy, None, None)
    assert settings["method"] == "dspark" and settings["draft_length"] == 7
    assert settings["max_draft_length"] is None  # no universal five-token assumption
    assert rows(compiled_record(nextn=3))["speculation"]["parameters"]["max_draft_length"] == 5


def silicon(method="mtp", length=4, algorithm="NEXTN"):
    return {
        "features": [
            {
                "id": "speculation",
                "parameters": [
                    {"name": "method", "value": method},
                    {"name": "draft_length", "value": length},
                    {"name": "algorithm", "value": algorithm},
                ],
            }
        ]
    }


def test_nextn_seed_count_normalization_preserves_raw_silicon_and_real_mismatches():
    record = compiled_record(nextn=3)
    measured = silicon()
    original = copy.deepcopy(measured)
    recipe = {
        "base": {
            "roles": {
                "agg": {
                    "args": {"speculative-eagle-topk": 1, "speculative-num-draft-tokens": 4, "speculative-num-steps": 3}
                }
            }
        }
    }
    comparison = compare_speculation(record, measured, {"backend": "sglang", "lanes": 1}, recipe)[
        "speculation_comparison"
    ][0]
    assert comparison["status"] == "matched"
    assert (comparison["actual_draft_length"], comparison["actual_proposals"], comparison["actual_verify_width"]) == (
        4,
        3,
        4,
    )
    assert measured == original
    mismatch = compare_speculation(compiled_record(nextn=2), measured, {"backend": "sglang", "lanes": 1}, recipe)
    assert mismatch["speculation_comparison"][0]["status"] == "mismatch"
    unsupported = compare_speculation(compiled_record(nextn=3), silicon(length=7), {"backend": "vllm", "lanes": 1})
    assert unsupported["speculation_comparison"][0]["status"] == "unsupported"
    assert unsupported["speculation_comparison"][0]["actual_draft_length"] == 7
    assert unsupported["speculation_comparison"][0]["aisim_max"] == 5


@pytest.mark.parametrize("corruption", ["identity", "revision", "missing", "duplicate", "private", "extra"])
def test_feature_records_fail_closed(corruption):
    record = compiled_record()
    if corruption == "identity":
        record["benchmark_id"] = "another"
    elif corruption == "revision":
        record["evaluated_revision"] = "main"
    elif corruption == "missing":
        record["roles"][0]["features"].pop()
    elif corruption == "duplicate":
        record["roles"].append(copy.deepcopy(record["roles"][0]))
    elif corruption == "private":
        record["roles"][0]["features"][0]["parameters"]["scope"] = "/Users/private"
    else:
        record["raw_config"] = {"token": "private"}
    with pytest.raises(ValueError):
        validate(record, "benchmark")


def test_simulation_evidence_cannot_be_an_hf_silicon_artifact(tmp_path):
    catalog = silicon_inputs(tmp_path, result())
    path = tmp_path / catalog["runs"][0]["manifest"]["path"]
    manifest = json.loads(path.read_bytes())
    manifest["artifacts"]["simulation_features"] = manifest["artifacts"]["configuration"]
    content = canonical(manifest)
    path.write_bytes(content)
    catalog["runs"][0]["manifest"].update(sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
    with pytest.raises(ValueError, match="do not belong"):
        load_runs(tmp_path, catalog)


def test_legacy_publication_missing_evidence_stays_missing():
    assert "simulation_features" not in result()
    record = simulation_record()
    assert validate(record)["configuration_sha256"] is not None


def test_dynamo_routing_requires_router_policy_and_never_uses_planner():
    adapter = SimpleNamespace(config={"router": {"policy": "kv_router"}, "planner": {"policy": "disabled"}})
    spec = SimpleNamespace(adapters={"dynamo": adapter}, execution_mode="offline")
    assert routing_feature(spec)["state"] == "enabled"
    adapter.config["router"]["policy"] = "round_robin"
    adapter.config["planner"]["policy"] = "enabled"
    assert routing_feature(spec)["state"] == "disabled"
    adapter.config.pop("router")
    assert routing_feature(spec)["state"] == "failed_to_record"
