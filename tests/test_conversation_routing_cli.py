# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from ai-dynamo/dynamo at 35ea1e7f689198333b1af0a2db8f94f545817792:
# components/src/dynamo/replay/tests/e2e/test_unified_cli_dynamo.py and its
# configs/unified_cli/predict/dynamo/13-trace-dynamo-agentic.yaml fixture.
# https://github.com/ai-dynamo/dynamo/tree/35ea1e7f689198333b1af0a2db8f94f545817792/components/src/dynamo/replay/tests/e2e
# Adapted for AISim-owned CLI execution with only the standalone policy API.

"""Conversation routing acceptance: AISim runs AgentX and calls Dynamo policy.

Set AISIM_REQUIRE_DYNAMO_ROUTER=1 for the required cross-repository acceptance
job. The job installs ai-dynamo-runtime with OfflineRouter and no ai-dynamo
runner package. Ordinary unit environments may omit this optional runtime.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.integration, pytest.mark.gpu_0, pytest.mark.timeout(240)]
_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dynamo_policy():
    required = os.environ.get("AISIM_REQUIRE_DYNAMO_ROUTER") == "1"
    try:
        core = importlib.import_module("dynamo._core")
    except ImportError as error:
        if required:
            pytest.fail(f"Dynamo policy acceptance requires ai-dynamo-runtime with OfflineRouter: {error}")
        pytest.skip("optional Dynamo policy runtime is not installed")
    if not hasattr(getattr(core, "OfflineRouter", None), "submit") or not hasattr(core, "SessionAffinity"):
        if required:
            pytest.fail(
                "installed ai-dynamo-runtime lacks native scheduler/affinity bindings; build the policy API revision"
            )
        pytest.skip("installed Dynamo runtime predates the standalone OfflineRouter API")
    if required:
        try:
            importlib.metadata.distribution("ai-dynamo")
        except importlib.metadata.PackageNotFoundError:
            pass
        else:
            pytest.fail("acceptance must run without the ai-dynamo whole-run plugin package")
    return core.OfflineRouter


def _run_cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    # Sitecustomize is inherited by supervised child interpreters as well as the
    # coordinator. Neither may delegate execution to Dynamo's replay plugin.
    guard = tmp_path / "import-guard"
    guard.mkdir(exist_ok=True)
    (guard / "sitecustomize.py").write_text(
        "import sys\n"
        "class NoDynamoRunner:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if any(fullname == name or fullname.startswith(name + '.') "
        "for name in ('dynamo.replay', 'dynamo.router', 'dynamo.planner')):\n"
        "            raise AssertionError('AISim must own simulation: ' + fullname)\n"
        "sys.meta_path.insert(0, NoDynamoRunner())\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(guard) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "aisimulate", *args],
        cwd=_REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, f"AISim CLI failed:\n{result.stdout}\n{result.stderr}"
    return result


def _conversation_prediction(tmp_path: Path, backend: str, topology: str, mode: str):
    """Self-authored two-play tree; no weights, profiles, or network access."""
    config = {
        "engine": {
            "mode": "aggregated",
            "model": str(_REPO_ROOT / "tests/e2e/configs/unified_cli/fixtures/tiny-model"),
            "hardware": "h200_sxm",
            "backend": "vllm",
            "context_length": 2048,
            "workers": {
                "aggregated": {
                    "parallelism": {"replicas": 1, "tensor": 1, "attention_data": 1},
                    "scheduler": {"max_batched_tokens": 8192, "max_sequences": 256},
                    "kv_cache": {
                        "block_size": 64,
                        "prefix_caching": True,
                        "capacity": {"type": "fixed", "blocks": 256},
                    },
                    "timing": {"type": "fixed", "prefill_ms": 1, "decode_ms": 1},
                }
            },
        }
    }
    rows = [
        {
            "schema": "dynamo.agentic_mooncake",
            "version": 2,
            "block_size": 64,
            "hash_id_scope": "local",
            "source": {
                "format": "aisimulate-policy-api-test",
                "digest": "two-branches-two-plays-v1",
            },
        }
    ]
    for play in ("first", "second"):
        for index, session, predecessor, relation in (
            (0, "owner", None, None),
            (1, "branch-a", 0, "spawn"),
            (2, "branch-b", 0, "spawn"),
            (3, "branch-a", 1, "sequence"),
            (4, "branch-b", 2, "sequence"),
        ):
            rows.append(
                {
                    "request_id": f"{play}-{index}",
                    "play_id": play,
                    "session_id": session,
                    "model": config["engine"]["model"],
                    "input_length": 128,
                    "output_length": 2,
                    "hash_ids": [101, 202],
                    "not_before_ms": index * 20,
                    "recorded_api_time_ms": 2,
                    "dependencies": []
                    if predecessor is None
                    else [
                        {
                            "request_id": f"{play}-{predecessor}",
                            "relation": relation,
                            "trigger": "dispatch" if relation == "spawn" else "completion",
                            "delay_ms": 20,
                        }
                    ],
                }
            )
    trace = tmp_path / "conversation-tree.jsonl"
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config["traffic"] = {
        "source": {
            "type": "trace",
            "format": "agentic_mooncake",
            "paths": [str(trace)],
            "block_size": 64,
        },
        "load": {"type": "trace_timestamps"},
    }
    worker = config["engine"]["workers"]["aggregated"]
    worker["parallelism"].update(replicas=2, attention_data=2)
    config["engine"].update(mode=topology, backend=backend)
    config["engine"]["workers"] = {
        role: json.loads(json.dumps(worker))
        for role in (["aggregated"] if topology == "aggregated" else ["prefill", "decode"])
    }
    config["router"] = {
        "policy": "kv_router",
        "affinity": {"mode": mode, "ttl_seconds": 3600},
    }
    return config


def _predict_conversations(tmp_path: Path, config: dict, name: str):
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config))
    output = tmp_path / name
    args = [
        "predict",
        "--config",
        str(path),
        "--output-dir",
        str(output),
        "--capture-per-request",
        "--format",
        "json",
    ]
    result = _run_cli(tmp_path, *args)
    report = json.loads((output / "prediction.json").read_text())
    assert json.loads(result.stdout)["completed_requests"] == report["completed_requests"]
    assert report["per_request"] == [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    return report


def _assert_conversation_bindings(report: dict, topology: str, mode: str, *, block_size=64):
    policies = report["routing_policy"]["roles"]
    roles = {"aggregated"} if topology == "aggregated" else {"prefill", "decode"}
    assert set(policies) == roles
    decisions = {}
    expected_requests = {row["uuid"] for row in report["per_request"]} | {
        row["uuid"] for row in report.get("agentic_phases", {}).get("requests", [])
    }
    for role, policy in policies.items():
        assert policy["native_policy"] == "dynamo.DefaultWorkerSelector"
        if role == "decode":
            # The existing decode adapter uses native load selection. Its engine
            # cache reuse is checked independently through transfer timing below.
            assert policy["physical_kv_events"] == 0
        else:
            assert policy["physical_kv_events"] > 0
        assert policy["decisions_captured"] is True
        assert policy["decision_count"] == len(policy["decisions"])
        assert policy["post_dispatch_checks"] == policy["decision_count"]
        assert policy["dispatch_aborts"] == 0
        assert {row["request_id"] for row in policy["decisions"]} == expected_requests
        for row in policy["decisions"]:
            assert row["native_policy"] == "dynamo.DefaultWorkerSelector"
            assert row["role"] == role
            key = (row["request_id"], role)
            assert key not in decisions
            decisions[key] = row
    bindings: dict[tuple, set] = {}
    group_keys: dict[tuple, set] = {}
    observations = 0
    for record in report["per_request"]:
        assert record["terminal_status"] == "completed"
        identity = record["agentic"]
        lineage = identity["lineage"]
        conversation = identity["conversation_id"]
        group = (
            (identity["play_id"], conversation)
            if mode == "session"
            else (
                identity["play_id"],
                lineage["root_conversation_id"],
                lineage.get("parent_conversation_id") or conversation,
                bool(lineage.get("parent_conversation_id")),
            )
        )
        observed_roles = set()
        for route in record["routing_history"]:
            role = "aggregated" if route["pool"] == "agg" else route["pool"]
            observed_roles.add(role)
            decision = decisions[(record["uuid"], role)]
            pair = decision["worker_id"], decision["dp_rank"]
            assert pair == (route["logical_worker_id"], route["dp_rank"])
            assert all(value in (0, 1) for value in pair)
            assert route["outcome"] == "immediate"
            assert route["selected_overlap_blocks"] == decision["overlap_blocks"]
            assert route["best_available_overlap_blocks"] == decision["best_available_overlap_blocks"]
            bindings.setdefault((group, role), set()).add(pair)
            group_keys.setdefault(group, set()).add(decision["group_key"])
            observations += 1
        assert observed_roles == roles
    assert observations > len(bindings)
    assert all(len(pairs) == 1 for pairs in bindings.values())
    assert all(len(keys) == 1 for keys in group_keys.values())
    assert len({next(iter(keys)) for keys in group_keys.values()}) == len(group_keys)
    assert any(row["binding_reused"] for row in decisions.values())
    assert report["first_admission_prefix_cache_reused_ratio"] > 0
    records = report["per_request"]
    actual_reused_tokens = sum(row["reused_input_tokens"] for row in records)
    assert actual_reused_tokens > 0
    # Per-worker tokenization runs after placement. Other requests can grow the
    # physical cache before admission, beyond the router observation at placement.
    # Without that delay, check the existing fixture's overlap bound.
    assert all(
        row["reused_input_tokens"] <= route["selected_overlap_blocks"] * block_size
        for row in records
        for route in row["routing_history"]
        if route["pool"] in ("agg", "prefill") and "tokenized_ms" not in row
    )
    assert report["first_admission_prefix_cache_reused_ratio"] == pytest.approx(
        actual_reused_tokens / sum(row["input_length"] for row in records)
    )


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("topology", ["aggregated", "disaggregated"])
@pytest.mark.parametrize("mode", ["session", "sibling_group"])
def test_conversation_cli_calls_dynamo_policy(tmp_path, backend, topology, mode, dynamo_policy):
    config = _conversation_prediction(tmp_path, backend, topology, mode)
    report = _predict_conversations(tmp_path, config, "affinity")
    assert report["completed_requests"] == 10
    _assert_conversation_bindings(report, topology, mode)
    for worker in config["engine"]["workers"].values():
        worker["kv_cache"]["prefix_caching"] = False
    cold = _predict_conversations(tmp_path, config, "cache-disabled")
    assert cold["completed_requests"] == report["completed_requests"]
    assert cold["first_admission_prefix_cache_reused_ratio"] == 0
    assert all(row["reused_input_tokens"] == 0 for row in cold["per_request"])


@pytest.mark.parametrize("topology", ["aggregated", "disaggregated"])
def test_conversation_tokenization_precedes_engine_admission(tmp_path, topology, dynamo_policy):
    config = _conversation_prediction(tmp_path, "vllm", topology, "session")
    config["engine"]["tokenization"] = {
        "workers": 1,
        "base_ms": 3.0,
        "per_input_token_us": 500.0,
    }
    report = _predict_conversations(tmp_path, config, "tokenization")
    assert report["completed_requests"] == 10
    _assert_conversation_bindings(report, topology, "session")
    for record in report["per_request"]:
        service_ms = 3.0 + 0.5 * record["input_length"]
        assert record["tokenized_ms"] - record["arrival_time_ms"] >= service_ms
        assert record["first_admit_ms"] >= record["tokenized_ms"]


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_conversation_cli_reuses_actual_decode_destination_cache(tmp_path, backend, dynamo_policy):
    config = _conversation_prediction(tmp_path, backend, "disaggregated", "session")
    trace = Path(config["traffic"]["source"]["paths"][0])
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    first = rows[1]
    repeated = {
        **first,
        "request_id": "first-repeat",
        "not_before_ms": 1000,
        "dependencies": [
            {
                "request_id": first["request_id"],
                "relation": "sequence",
                "trigger": "completion",
                "delay_ms": 1000,
            }
        ],
    }
    trace.write_text("".join(json.dumps(row) + "\n" for row in [rows[0], first, repeated]))
    config["engine"]["kv_transfer"] = {
        "bandwidth_gb_per_second": 1,
        "timing_mode": "destination_missing",
    }
    missing = _predict_conversations(tmp_path, config, "destination-missing")
    _assert_conversation_bindings(missing, "disaggregated", "session")
    config["engine"]["kv_transfer"]["timing_mode"] = "full_prompt"
    full = _predict_conversations(tmp_path, config, "full-prompt")
    config["engine"]["kv_transfer"]["timing_mode"] = "destination_missing"
    for worker in config["engine"]["workers"].values():
        worker["kv_cache"]["prefix_caching"] = False
    cold = _predict_conversations(tmp_path, config, "uncached-destination")

    def transfer_span(report):
        assert report["completed_requests"] == 2
        record = max(report["per_request"], key=lambda row: row["arrival_time_ms"])
        return record["destination_activated_ms"] - record["destination_reserved_ms"]

    # This reads actual engine handoff timestamps. A policy overlap estimate or
    # the presence of an affinity configuration cannot satisfy these assertions.
    assert transfer_span(missing) < transfer_span(full) - 1e-6
    assert transfer_span(cold) == pytest.approx(transfer_span(full), abs=1e-6)
    assert all(row["reused_input_tokens"] == 0 for row in cold["per_request"])


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("topology", ["aggregated", "disaggregated"])
@pytest.mark.parametrize("mode", ["session", "sibling_group"])
def test_conversation_duration_reuses_snapshot_warmup_pipeline(tmp_path, backend, topology, mode, dynamo_policy):
    config = _conversation_prediction(tmp_path, backend, topology, mode)
    config["traffic"]["load"].update(
        agentic_lanes=2,
        agentic_snapshot={"seed": 42},
        agentic_warmup=True,
        agentic_profile={"duration_seconds": 0.4},
    )
    report = _predict_conversations(tmp_path, config, "duration")
    _assert_conversation_bindings(report, topology, mode)
    profile = report["agentic_profile"]
    phases = report["agentic_phases"]
    assert profile["admission_closed"]
    assert profile["profile_start_ms"] == phases["profile_start_ms"]
    assert profile["admission_cutoff_ms"] - profile["profile_start_ms"] == pytest.approx(400)
    assert profile["plays_started"] > 2
    assert profile["unsettled_server_requests"] == 0
    assert not profile["cancel_drain_timed_out"]
    # Warmup requests traverse the same native policy before measured traffic.
    measured = {row["uuid"] for row in report["per_request"]}
    prepared = {row["uuid"] for row in phases["requests"]}
    assert prepared - measured
    for policy in report["routing_policy"]["roles"].values():
        decisions = {row["request_id"] for row in policy["decisions"]}
        assert measured | prepared == decisions


@pytest.mark.parametrize("topology", ["aggregated", "disaggregated"])
@pytest.mark.parametrize("mode", ["session", "sibling_group"])
def test_weka_agentx_cli_keeps_parent_and_child_lineage(tmp_path, topology, mode, dynamo_policy):
    config = _conversation_prediction(tmp_path, "vllm", topology, mode)
    config["traffic"] = {
        "source": {
            "type": "trace",
            "format": "weka",
            "block_size": 4,
            "nested_timestamp_basis": "absolute",
            "paths": [str(_REPO_ROOT / "tests/e2e/configs/unified_cli/fixtures/traces/weka-g2-restore.jsonl")],
        },
        "load": {"type": "trace_timestamps", "agentic_lanes": 1},
    }
    for worker in config["engine"]["workers"].values():
        worker["kv_cache"]["block_size"] = 4
    report = _predict_conversations(tmp_path, config, "weka")
    assert report["completed_requests"] == 4
    _assert_conversation_bindings(report, topology, mode, block_size=4)

    def source_request(suffix):
        return next(record for record in report["per_request"] if record["request_id"].endswith(suffix))

    parent = source_request(":outer:0")["agentic"]
    child = source_request(":outer:1:inner:0")["agentic"]
    resumed = source_request(":outer:2")
    independent = source_request(":outer:3")["agentic"]
    # The first and resumed inputs share hashes [1, 2, 3]. The explicit subagent
    # has hashes [11, 12, 13]; the final, unrelated [21, 22, 23] input lowers to
    # a separate worker stream, with its own conversation root.
    assert resumed["agentic"]["conversation_id"] == parent["conversation_id"]
    assert resumed["agentic"]["lineage"] == parent["lineage"]
    assert resumed["reused_input_tokens"] == 8
    assert len({identity["conversation_id"] for identity in (parent, child, independent)}) == 3
    assert len({identity["play_id"] for identity in (parent, child, independent)}) == 1
    assert child["lineage"]["parent_conversation_id"] == parent["conversation_id"]
    assert child["lineage"]["root_conversation_id"] == parent["conversation_id"]
    for root in (parent, independent):
        assert root["lineage"]["root_conversation_id"] == root["conversation_id"]
        assert "parent_conversation_id" not in root["lineage"]


@pytest.mark.parametrize("dispatch", ["dispatch_committed", "dispatch_aborted"])
def test_policy_releases_conversation_waiters_after_dispatch(dispatch, dynamo_policy):
    from aisimulate.router import _DynamoPolicy

    policy = _DynamoPolicy(
        json.dumps(
            {
                "schema_version": 1,
                "role": "aggregated",
                "dp_size": 1,
                "topology": [{"worker_id": 0, "scheduler_ids": [0]}],
                "engine": {"rank": {"block_size": 4, "num_gpu_blocks": 128, "max_num_batched_tokens": 32}},
                "placement": {"config": {"affinity": {"mode": "session", "ttl_seconds": 1}}},
                "selector_seed": 7,
                "capture_decisions": True,
            }
        )
    )

    def call(op, **fields):
        return json.loads(policy.call_json(json.dumps({"op": op, **fields})))

    def place(request_id):
        return call(
            "place",
            now_ms=0,
            input_length=4,
            session_id="conversation",
            request={"uuid": request_id, "tokens": [1, 2, 3, 4], "max_output_tokens": 2},
        )

    # Authored tokens without precomputed hashes exercise native hashing too.
    first = place("first")["result"]["placement"]
    assert first is not None
    assert place("second")["result"]["placement"] is None
    call(dispatch, request_id="first", now_ms=0)
    assert call("cancel_pending", request_id="unknown")["next_wakeup_ms"] == 0
    released = call("advance_clock", now_ms=0)
    assert [row["request_id"] for row in released["result"]] == ["second"]
    assert released["pending_count"] == 0
    evidence = call("snapshot")["evidence"]
    assert evidence["decisions"][-1]["binding_reused"] == (dispatch == "dispatch_committed")
    call("dispatch_committed", request_id="second", now_ms=0)
    call("request_terminal", request_id="first", now_ms=0)
    call("request_terminal", request_id="second", now_ms=0)
    assert not policy.active and not policy.staged
