# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance against installed matching wheels, through the actual CLI process.

Run with the Python interpreter from a fresh venv containing both built wheels:
    python -m pytest -c /dev/null -p no:cacheprovider -q \
        --confcutdir=python/aisimulate-dynamo-policy/tests \
        python/aisimulate-dynamo-policy/tests/test_cli_native.py

The trace below is self-authored. Fixed engine timing qualifies routing and
cache behavior without downloading a model or implying hardware accuracy.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

DYNAMO_REVISION = "d9eb42db1168131fdae318eef77255637e4d3495"


@pytest.fixture(scope="session")
def installed_cli() -> Path:
    """Never accidentally qualify an editable checkout or a different CLI."""
    for name in ("aisimulate", "aisimulate-dynamo-policy"):
        distribution = importlib.metadata.distribution(name)
        direct = distribution.read_text("direct_url.json")
        assert not direct or not json.loads(direct).get("dir_info", {}).get("editable"), name
        assert Path(distribution.locate_file("")).resolve().is_relative_to(Path(sys.prefix).resolve()), name
    assert importlib.metadata.version("aisimulate") == importlib.metadata.version("aisimulate-dynamo-policy")
    for name in ("aisimulate", "aisimulate_dynamo_policy"):
        spec = importlib.util.find_spec(name)
        assert spec is not None and spec.origin is not None, name
        assert Path(spec.origin).resolve().is_relative_to(Path(sys.prefix).resolve()), spec.origin
    cli = Path(sys.executable).parent / "aisimulate"
    assert cli.is_file(), f"Install the matching wheels into this interpreter's venv: {cli}"
    return cli


def _trace(path: Path) -> int:
    # Both plays deliberately reuse conversation labels and hash IDs. Their
    # local cache and group namespaces must remain independent. Each play has
    # siblings, later turns without a spawn edge, and two different parents.
    rows: list[dict] = [
        {
            "schema": "dynamo.agentic_mooncake",
            "version": 2,
            "block_size": 64,
            "hash_id_scope": "local",
            "source": {"format": "self-authored-cli-acceptance", "digest": "conversation-tree-v1"},
        }
    ]
    for play in ("play-a", "play-b"):
        for name, conversation, start, parent, relation in (
            ("root", "main", 0, None, None),
            ("left", "left", 20, "root", "spawn"),
            ("right", "right", 20, "root", "spawn"),
            ("left-next", "left", 60, "left", "sequence"),
            ("right-next", "right", 60, "right", "sequence"),
            ("left-child-a", "left-child-a", 100, "left-next", "spawn"),
            ("left-child-b", "left-child-b", 100, "left-next", "spawn"),
            ("right-child-a", "right-child-a", 140, "right-next", "spawn"),
            ("right-child-b", "right-child-b", 140, "right-next", "spawn"),
            ("left-child-a-next", "left-child-a", 180, "left-child-a", "sequence"),
            ("left-child-b-next", "left-child-b", 180, "left-child-b", "sequence"),
            ("right-child-a-next", "right-child-a", 220, "right-child-a", "sequence"),
            ("right-child-b-next", "right-child-b", 220, "right-child-b", "sequence"),
        ):
            rows.append(
                {
                    "request_id": f"{play}-{name}",
                    "play_id": play,
                    "session_id": conversation,
                    "model": "example/model",
                    "input_length": 128,
                    "output_length": 2,
                    "hash_ids": [10, 20],
                    "not_before_ms": start,
                    "recorded_api_time_ms": 2,
                    "dependencies": []
                    if parent is None
                    else [
                        {
                            "request_id": f"{play}-{parent}",
                            "relation": relation,
                            "trigger": "completion" if relation == "sequence" else "dispatch",
                            "delay_ms": 20,
                        }
                    ],
                }
            )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return len(rows) - 1


def _config(tmp_path: Path, backend: str, topology: str, affinity: str) -> dict:
    trace = tmp_path / "conversations.jsonl"
    _trace(trace)
    return {
        "traffic": {
            "source": {"type": "trace", "format": "agentic_mooncake", "paths": [str(trace)], "block_size": 64},
            "load": {"type": "trace_timestamps"},
        },
        "router": {"policy": "kv_router", "affinity": {"mode": affinity, "ttl_seconds": 3600}},
        "engine": {
            "mode": topology,
            "model": "example/model",
            "hardware": "h200_sxm",
            "backend": backend,
            "context_length": 2048,
            "workers": {
                role: {
                    "parallelism": {"replicas": 2, "tensor": 1, "pipeline": 1, "attention_data": 2},
                    "scheduler": {"max_batched_tokens": 1024, "max_sequences": 32},
                    "kv_cache": {
                        "block_size": 64,
                        "prefix_caching": True,
                        "capacity": {"type": "fixed", "blocks": 128},
                    },
                    "timing": {"type": "fixed", "prefill_ms": 1, "decode_ms": 1},
                }
                for role in (["aggregated"] if topology == "aggregated" else ["prefill", "decode"])
            },
        },
    }


def _invoke(cli: Path, tmp_path: Path, config: dict, *, name: str, stack: str | None = None):
    config_path = tmp_path / f"{name}.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / f"{name}-output"
    command = [str(cli), "predict", "--config", str(config_path), "--capture-per-request", "--format", "json"]
    command += ["--output-dir", str(output)]
    if stack is not None:
        command += ["--stack", stack]
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update(PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    process = subprocess.run(command, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=120)
    return process, output


def _run(cli: Path, tmp_path: Path, config: dict, *, name: str, stack: str | None = None) -> dict:
    process, output = _invoke(cli, tmp_path, config, name=name, stack=stack)
    assert process.returncode == 0, f"{process.stdout}\n{process.stderr}"
    report = json.loads((output / "prediction.json").read_text())
    records = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    assert records == report["per_request"]
    return report


def _expected_group(identity: dict, affinity: str) -> tuple:
    lineage = identity["lineage"]
    assert lineage["schema"] == "aisimulate.agentic.conversation-lineage.v1"
    if affinity == "session":
        return identity["play_id"], identity["conversation_id"]
    return (
        identity["play_id"],
        lineage["root_conversation_id"],
        lineage.get("parent_conversation_id") or identity["conversation_id"],
        "siblings" if lineage.get("parent_conversation_id") else "root",
    )


def _assert_native_routing(report: dict, topology: str, affinity: str) -> None:
    policy = report["dynamo_policy"]
    assert policy["dynamo_revision"] == DYNAMO_REVISION
    assert policy["native_policy"] == "dynamo.SelectionCore"
    assert policy["physical_kv_events"] > 0
    assert report["completed_requests"] > 0
    roles = {"aggregated"} if topology == "aggregated" else {"prefill", "decode"}
    decisions = {(row["request_id"], row["role"]): row for row in policy["decisions"]}
    assert set(row["role"] for row in policy["decisions"]) == roles
    grouped = defaultdict(list)
    keys = defaultdict(set)
    for record in report["per_request"]:
        assert record["terminal_status"] == "completed"
        routes = {"aggregated" if row["pool"] == "agg" else row["pool"]: row for row in record["routing_history"]}
        assert set(routes) == roles
        assert {"aggregated" if row["pool"] == "agg" else row["pool"] for row in record["admission_history"]} == roles
        expected_group = _expected_group(record["agentic"], affinity)
        for role, route in routes.items():
            decision = decisions[(record["uuid"], role)]
            assert decision["native_policy"] == "dynamo.SelectionCore"
            assert decision["worker_id"] in (0, 1)
            assert decision["dp_rank"] in (0, 1)
            assert decision["worker_id"] == route["logical_worker_id"]
            assert decision["dp_rank"] == route["dp_rank"]
            assert decision["scheduler_id"] == route["scheduler_id"]
            assert decision["overlap_blocks"] == route["selected_overlap_blocks"]
            grouped[(role, expected_group)].append((decision["worker_id"], decision["dp_rank"]))
            keys[expected_group].add(decision["group_key"])
        if topology == "disaggregated":
            # The legacy *_worker_idx fields identify rank schedulers; the
            # routing history above carries the logical worker and DP pair.
            assert record["prefill_worker_idx"] == decisions[(record["uuid"], "prefill")]["scheduler_id"]
            assert record["decode_worker_idx"] == decisions[(record["uuid"], "decode")]["scheduler_id"]
            assert record["prefill_admit_ms"] <= record["decode_admit_ms"]
        else:
            assert record["prefill_worker_idx"] is None
            assert record["decode_worker_idx"] == decisions[(record["uuid"], "aggregated")]["scheduler_id"]
    # Preparation must use the same conversation identities and native policy,
    # otherwise warmed residency may be on a different worker/DP from profile.
    phases = report.get("agentic_phases")
    if phases:
        for request in phases["requests"]:
            assert request["terminal_status"] == "completed"
            assert request["quiescent_at_ms"] <= phases["profile_start_ms"]
            expected_group = _expected_group(request["identity"], affinity)
            for role in roles:
                decision = decisions[(request["uuid"], role)]
                grouped[(role, expected_group)].append((decision["worker_id"], decision["dp_rank"]))
                keys[expected_group].add(decision["group_key"])
    # Binding is a worker/DP pair, and later turns retain their ancestral group.
    assert any(len(bindings) > 1 for bindings in grouped.values())
    assert all(len(set(bindings)) == 1 for bindings in grouped.values())
    assert all(len(group_keys) == 1 for group_keys in keys.values())
    # A different parent or a different play must not acquire the same lease.
    assert len({next(iter(group_keys)) for group_keys in keys.values()}) == len(keys)
    assert len({record["agentic"]["play_id"] for record in report["per_request"]}) >= 2
    assert any(decision["overlap_blocks"] > 0 for decision in policy["decisions"])
    assert report["first_admission_prefix_cache_reused_ratio"] > 0
    assert any(record["reused_input_tokens"] > 0 for record in report["per_request"])


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("topology", ["aggregated", "disaggregated"])
@pytest.mark.parametrize("affinity", ["session", "sibling_group"])
def test_yaml_selects_native_dynamo_with_real_affinity_and_cache(installed_cli, tmp_path, backend, topology, affinity):
    config = _config(tmp_path, backend, topology, affinity)
    report = _run(installed_cli, tmp_path, config, name="native")
    assert report["completed_requests"] == 26
    _assert_native_routing(report, topology, affinity)
    # Same workload, policy, affinity, workers and timing: disable only physical
    # prefix caching. A router's estimated overlap cannot pass this control.
    for worker in config["engine"]["workers"].values():
        worker["kv_cache"]["prefix_caching"] = False
    cold = _run(installed_cli, tmp_path, config, name="no-prefix-cache")
    assert cold["completed_requests"] == report["completed_requests"]
    assert cold["first_admission_prefix_cache_reused_ratio"] == 0
    assert all(record["reused_input_tokens"] == 0 for record in cold["per_request"])


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("topology", ["aggregated", "disaggregated"])
@pytest.mark.parametrize("affinity", ["session", "sibling_group"])
def test_snapshot_warmup_and_duration_share_the_installed_policy(installed_cli, tmp_path, backend, topology, affinity):
    config = _config(tmp_path, backend, topology, affinity)
    config["traffic"]["load"].update(
        agentic_lanes=2,
        agentic_snapshot={"seed": 42},
        agentic_warmup=True,
        agentic_profile={"duration_seconds": 0.8},
    )
    report = _run(installed_cli, tmp_path, config, name="duration")
    _assert_native_routing(report, topology, affinity)
    phases = report["agentic_phases"]
    profile = report["agentic_profile"]
    assert phases["profile_start_ms"] > 0
    assert profile["profile_start_ms"] == phases["profile_start_ms"]
    assert profile["admission_closed"]
    assert profile["admission_cutoff_ms"] - profile["profile_start_ms"] == pytest.approx(800)
    assert profile["plays_started"] > 2
    assert profile["client_completed_plays"] >= 2
    assert profile["unsettled_server_requests"] == 0
    assert not profile["cancel_drain_timed_out"]
    identities = defaultdict(set)
    for record in report["per_request"]:
        identities[record["agentic"]["play_id"]].add(record["agentic"]["cache_id"])
    assert len(identities) > 2
    assert all(len(cache_ids) == 1 for cache_ids in identities.values())
    assert len({next(iter(cache_ids)) for cache_ids in identities.values()}) == len(identities)


def test_explicit_stack_and_policy_without_affinity_remain_available(installed_cli, tmp_path):
    config = _config(tmp_path, "vllm", "aggregated", "session")
    del config["router"]["affinity"]
    report = _run(installed_cli, tmp_path, config, name="explicit", stack="dynamo-policy")
    assert report["completed_requests"] == 26
    assert report["dynamo_policy"]["native_policy"] == "dynamo.SelectionCore"
    assert all(row["group_key"] is None for row in report["dynamo_policy"]["decisions"])
    assert report["first_admission_prefix_cache_reused_ratio"] > 0


@pytest.mark.parametrize("stack", [None, "engine"])
def test_invalid_routing_never_silently_runs_default_policy(installed_cli, tmp_path, stack):
    config = _config(tmp_path, "vllm", "aggregated", "session")
    config["router"]["affinity"]["mode"] = "unsupported-affinity"
    process, output = _invoke(installed_cli, tmp_path, config, name="invalid", stack=stack)
    assert process.returncode != 0
    assert "router" in process.stderr.lower() or "affinity" in process.stderr.lower()
    assert not (output / "prediction.json").exists()


def test_without_routing_configuration_retains_core_default(installed_cli, tmp_path):
    config = _config(tmp_path, "vllm", "aggregated", "session")
    del config["router"]
    report = _run(installed_cli, tmp_path, config, name="default")
    assert report["completed_requests"] == 26
    assert "dynamo_policy" not in report
