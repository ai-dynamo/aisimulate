# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dashboard contract tests use self-authored telemetry and fixed-time replay."""

import copy
import json
from pathlib import Path

import pytest

from aisimulate.output import write_prediction_report, write_telemetry
from aisimulate.replay.dashboard import (
    build_replay_dashboard_data,
    render_replay_dashboard,
    write_replay_dashboard,
)
from aisimulate.runner import EngineReplayRunnerFactory
from aisimulate.sweeper import BackendDeploymentSpec, ReplayOutputRequirements, ReplaySpec


@pytest.fixture
def capture():
    return {
        "schema": "aisimulate.replay.telemetry.v1",
        "clock": "absolute_virtual_ms",
        "clock_origin_ms": 0,
        "units": {
            "timestamps": "ms",
            "durations": "ms",
            "traffic_duration": "s",
            "tokens": "tokens",
            "cache_blocks": "blocks",
            "cache_bytes": "bytes",
            "cache_usage": "ratio",
        },
        "sample_interval_ms": 1000,
        "profile_start_ms": 1250,
        "completion_status": "completed",
        "phase_boundaries": [
            {"phase": "preparation", "start_ms": 0, "end_ms": 1250, "completion_status": "completed"},
            {"phase": "profile", "start_ms": 1250, "end_ms": 2600, "completion_status": "completed"},
        ],
        "requests": [
            {
                "uuid": "request-1",
                "request_id": "turn-1",
                "phase": "profile",
                "arrival_time_ms": 2250,
                "terminal_time_ms": 2600,
                "first_token_ms": 2600,
                "input_length": 16,
                "output_length": 1,
                "ttft_ms": 350,
                "itl_ms": None,
                "terminal_status": "completed",
                "token_ids": [1, 2, 3],
                "agentic": {
                    "play_id": "play-1",
                    "request_id": "turn-1",
                    "conversation_id": "c1",
                    "lineage": {
                        "schema": "aisimulate.agentic.conversation-lineage.v1",
                        "root_conversation_id": "c0",
                        "parent_conversation_id": "c0",
                    },
                },
                "routing_history": [{"pool": "agg", "worker_id": 0}],
                "admission_history": [{"at_ms": 2250, "pool": "agg", "worker_id": 0}],
            }
        ],
        "samples": [
            {
                "phase": "profile",
                "kind": "final",
                "sample_ordinal": 2,
                "interval_start_ms": 2250,
                "sampled_at_ms": 2600,
                "traffic": {
                    "duration_s": 0.35,
                    "delivered_output_tokens": 1,
                    "committed_prefill_tokens": 8,
                    "arriving_requests": 1,
                    "completed_requests": 1,
                    "first_admission": {
                        "requests": 1,
                        "input_tokens": 16,
                        "hbm_reused_tokens": 8,
                        "host_reused_tokens": 0,
                        "uncached_tokens": 8,
                    },
                },
                "prefill_scheduler_metrics": [],
                "decode_scheduler_metrics": [
                    {
                        "worker_id": 0,
                        "dp_rank": 0,
                        "active_blocks": 4,
                        "inactive_blocks": 2,
                        "total_blocks": 10,
                        "active_cache_usage": 0.4,
                        "physical_cache_usage": 0.6,
                        "running_requests": 0,
                        "waiting_requests": 0,
                    }
                ],
                "prefill_interval_metrics": {},
                "decode_interval_metrics": {
                    "native_prefix_cache_lookup": {"query_count": 2, "queried_tokens": 32, "matched_tokens": 0},
                    "cache_hit_tokens": None,
                    "cache_total_tokens": None,
                    "preemptions": 0,
                },
                "router_pending_prefill_requests": 0,
                "router_pending_decode_requests": 0,
                "host_caches": [
                    {
                        "scope": "cluster_shared",
                        "role": None,
                        "capacity_blocks": 20,
                        "resident_blocks": 3,
                        "used_blocks": 5,
                        "capacity_bytes": 2000,
                        "resident_bytes": 300,
                        "used_bytes": 500,
                    }
                ],
            }
        ],
        "lifecycle": [
            {
                "phase": "profile",
                "at_ms": 2250,
                "event": "dispatch",
                "request_uuid": "request-1",
                "play_id": "play-1",
                "request_id": "turn-1",
            }
        ],
    }


def test_preserves_absolute_phases_native_cutoff_and_interval_semantics(capture):
    prediction = {"agentic_profile": {"admission_cutoff_ms": 2500}, "per_request": [{"heavy": True}]}
    original = copy.deepcopy(capture)
    data = build_replay_dashboard_data(prediction, capture)
    assert data["phases"]["profile"]["start_ms"] == 1250
    assert data["phases"]["profile"]["admission_cutoff_ms"] == 2500
    assert data["phases"]["profile"]["end_ms"] == 2600
    assert "per_request" not in data["prediction"]
    assert "samples" not in data["telemetry"]
    request = data["requests"][0]
    assert request["arrival_time_ms"] == 2250
    assert request["terminal_time_ms"] == 2600
    assert request["itl_ms"] is None
    assert "token_ids" not in request
    assert request["agentic"]["lineage"]["parent_conversation_id"] == "c0"
    sample = data["samples"][0]
    assert (sample["t"], sample["start"], sample["duration"]) == (2.6, 2.25, 0.35)
    assert sample["lookup_hits"] == 0
    assert sample["lookup_queries"] == 32
    assert sample["scheduler_hits"] is None
    assert sample["accept"] is None
    assert sample["host"] == 0
    assert sample["cache_pct"] == 60
    assert sample["workers"][0]["role"] == "agg"
    assert sample["host_pools"][0]["used_bytes"] == 500
    assert capture == original


def test_byte_capacities_and_unknown_metrics_are_preserved(capture):
    row = capture["samples"][0]
    worker = row["decode_scheduler_metrics"][0]
    worker.update(total_blocks=None, kv_cache_capacity_bytes=800, kv_cache_used_bytes=200)
    row["decode_scheduler_metrics"].append(
        {**worker, "dp_rank": 1, "kv_cache_capacity_bytes": 200, "kv_cache_used_bytes": 100}
    )
    data = build_replay_dashboard_data({}, capture)
    assert data["samples"][0]["cache_pct"] == 30
    assert data["samples"][0]["workers"][0]["total_blocks"] is None
    assert data["samples"][0]["workers"][0]["kv_cache_capacity_bytes"] == 800
    row["decode_scheduler_metrics"].append(
        {"worker_id": 1, "dp_rank": 0, "total_blocks": 4, "physical_cache_usage": 0.5}
    )
    assert build_replay_dashboard_data({}, capture)["samples"][0]["cache_pct"] is None
    row["decode_scheduler_metrics"] = []
    row.pop("router_pending_prefill_requests")
    row.pop("router_pending_decode_requests")
    sample = build_replay_dashboard_data({}, capture)["samples"][0]
    assert sample["running"] is None
    assert sample["router_pending"] is None


@pytest.mark.parametrize("missing", [None, 0])
def test_deployment_gauges_require_every_live_rank(capture, missing):
    row = capture["samples"][0]
    row["decode_scheduler_metrics"][0].update(running_requests=1, waiting_requests=1)
    row["decode_scheduler_metrics"].append(
        {**row["decode_scheduler_metrics"][0], "dp_rank": 1, "running_requests": missing, "waiting_requests": missing}
    )
    row.update(router_pending_prefill_requests=1, router_pending_decode_requests=missing)
    sample = build_replay_dashboard_data({}, capture)["samples"][0]
    expected = None if missing is None else 1
    assert sample["running"] == sample["waiting"] == sample["router_pending"] == expected


def test_aborted_preparation_and_empty_profile(capture):
    capture.update(
        completion_status="preparation_aborted", profile_start_ms=None, requests=[], samples=[], lifecycle=[]
    )
    capture["phase_boundaries"] = [{"phase": "preparation", "start_ms": 0, "end_ms": 0, "completion_status": "aborted"}]
    data = build_replay_dashboard_data({}, capture)
    assert set(data["phases"]) == {"preparation"}
    assert data["samples"] == []
    capture.update(completion_status="completed", profile_start_ms=0)
    capture["phase_boundaries"] = [{"phase": "profile", "start_ms": 0, "end_ms": 0, "completion_status": "completed"}]
    assert build_replay_dashboard_data({}, capture)["phases"]["profile"]["end_ms"] == 0


def test_nested_summary_and_preparation_failure_evidence(capture):
    prediction = {
        "completed_requests": 99,
        "summary": {"completed_requests": 1, "model": "fixture-model", "mean_ttft_ms": None},
        "agentic_phases": {
            "phase": "aborted",
            "failure_reason": "warmup request rejected",
            "failure_request_id": "request-1",
            "requests": [{"large": True}],
            "lanes": [{"large": True}],
        },
    }
    data = build_replay_dashboard_data(prediction, capture)
    assert data["prediction"]["completed_requests"] == 1
    assert data["prediction"]["mean_ttft_ms"] is None
    assert data["metadata"]["model"] == "fixture-model"
    assert data["prediction"]["agentic_phases"] == {
        "phase": "aborted",
        "failure_reason": "warmup request rejected",
        "failure_request_id": "request-1",
    }


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("schema", "unsupported.v1", "schema"),
        ("clock", "relative_ms", "clock"),
        ("sample_interval_ms", 0, "positive"),
        ("sample_interval_ms", float("nan"), "finite"),
        ("units", {"timestamps": "seconds"}, "units"),
        ("requests", {}, "list of JSON mappings"),
    ],
)
def test_rejects_unsupported_capture(capture, field, value, message):
    capture[field] = value
    with pytest.raises(ValueError, match=message):
        build_replay_dashboard_data({}, capture)


@pytest.mark.parametrize(
    "field,value",
    [
        ("delivered_output_tokens", "1"),
        ("avg_accept_length", float("inf")),
        ("duration_s", 1.0),
        ("arriving_requests", True),
    ],
)
def test_rejects_malformed_metrics(capture, field, value):
    capture["samples"][0]["traffic"][field] = value
    with pytest.raises(ValueError, match=field):
        build_replay_dashboard_data({}, capture)


def test_writer_is_standalone_safe_and_atomic(tmp_path, capture):
    dangerous = '</script><script>alert("bad")</script>\u2028__PLOTLY_JS__'
    capture["requests"][0]["request_id"] = dangerous
    path = write_replay_dashboard(tmp_path, {"completed_requests": 1}, capture)
    document = path.read_text(encoding="utf-8")
    assert path.name == "report.html"
    assert '<script>alert("bad")</script>' not in document
    assert "\\u003c/script>" in document
    assert "\\u2028__PLOTLY_JS__" in document
    assert "plotly.js v" in document
    assert "Permission is hereby granted" in document
    assert list(tmp_path.iterdir()) == [path]
    original = document
    capture["samples"][0]["traffic"]["delivered_output_tokens"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        write_replay_dashboard(tmp_path, {}, capture)
    assert path.read_text(encoding="utf-8") == original
    assert list(tmp_path.iterdir()) == [path]


def test_regeneration_uses_only_manifest_streams_and_checks_counts(tmp_path, capture):
    keep = tmp_path / "keep.txt"
    keep.write_text("keep", encoding="utf-8")
    prediction = {"agentic_profile": {"admission_cutoff_ms": 2500}, "telemetry": write_telemetry(tmp_path, capture)}
    write_prediction_report(tmp_path, prediction)
    first = write_replay_dashboard(tmp_path, prediction, capture).read_text(encoding="utf-8")
    assert render_replay_dashboard(tmp_path).read_text(encoding="utf-8") == first
    assert keep.read_text(encoding="utf-8") == "keep"
    prediction["telemetry"]["record_counts"]["requests"] += 1
    write_prediction_report(tmp_path, prediction)
    with pytest.raises(ValueError, match="record_counts.requests"):
        render_replay_dashboard(tmp_path)
    assert (tmp_path / "report.html").read_text(encoding="utf-8") == first
    prediction["telemetry"]["artifacts"]["requests"] = "../outside.jsonl"
    write_prediction_report(tmp_path, prediction)
    with pytest.raises(ValueError, match="outside the replay directory"):
        render_replay_dashboard(tmp_path)


@pytest.mark.parametrize("backend,mode", [("vllm", "agg"), ("sglang", "disagg")])
def test_native_source_fixture_all_phase_dashboard(backend, mode):
    args = {
        "engine_type": backend,
        "aic_backend": backend,
        "aic_model_path": "dashboard-fixture",
        "aic_system": "h200_sxm",
        "aic_tp_size": 2,
        "aic_attention_dp_size": 2,
        "block_size": 4,
        "num_gpu_blocks": 64,
        "enable_prefix_caching": True,
        "timing_model": {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0},
    }
    roles = (
        {"agg_engine_args": {**args, "worker_type": "aggregated"}, "num_workers": 1}
        if mode == "agg"
        else {
            "prefill_engine_args": {**args, "worker_type": "prefill"},
            "num_prefill_workers": 1,
            "decode_engine_args": {**args, "worker_type": "decode"},
            "num_decode_workers": 1,
        }
    )
    spec = ReplaySpec(
        backend_deployment=BackendDeploymentSpec(
            deployment_mode=mode, backend=backend, backend_version="test", **roles
        ),
        workload={
            "source_type": "trace",
            "load_type": "trace_timestamps",
            "trace_format": "weka",
            "trace_block_size": 4,
            "trace_path": str(Path(__file__).parent / "e2e/configs/unified_cli/fixtures/traces/weka-two-plays.jsonl"),
            "agentic_lanes": 2,
            "agentic_snapshot": {"seed": 42},
            "agentic_warmup": True,
            "agentic_profile": {"duration_seconds": 0.02, "response_grace_seconds": 0.01},
        },
        goal={"target": "throughput"},
    )
    runner = EngineReplayRunnerFactory(determinism="canonical_v1").create(0)
    try:
        report = runner.run(
            spec,
            output_requirements=ReplayOutputRequirements(
                include_raw_report=True,
                capture_telemetry=True,
                telemetry_sample_interval_ms=2.5,
            ),
        ).metadata["native_report"]
    finally:
        runner.close()
    data = build_replay_dashboard_data(report, report["telemetry"])
    assert data["phases"]["profile"]["start_ms"] > 0
    assert data["phases"]["profile"]["admission_cutoff_ms"] == report["agentic_profile"]["admission_cutoff_ms"]
    assert {sample["phase"] for sample in data["samples"]} == {"preparation", "profile"}
    assert "warmup" in {request["phase"] for request in data["requests"]}
    assert {worker["dp_rank"] for sample in data["samples"] for worker in sample["workers"]} == {0, 1}
    assert sum(sample["output_tokens"] for sample in data["samples"]) == sum(
        sample["traffic"]["delivered_output_tokens"] for sample in report["telemetry"]["samples"]
    )
    assert all(request["itl_ms"] is None for request in data["requests"] if request["output_length"] < 2)
    json.dumps(data, allow_nan=False)


def test_native_time_limit_does_not_fabricate_unfinished_request_records():
    args = {
        "worker_type": "aggregated",
        "engine_type": "vllm",
        "aic_backend": "vllm",
        "aic_model_path": "dashboard-fixture",
        "aic_system": "h200_sxm",
        "aic_tp_size": 1,
        "aic_attention_dp_size": 1,
        "block_size": 4,
        "num_gpu_blocks": 64,
        "timing_model": {"type": "fixed", "prefill_ms": 100.0, "decode_ms": 100.0},
    }
    spec = ReplaySpec(
        backend_deployment=BackendDeploymentSpec(
            deployment_mode="agg",
            backend="vllm",
            backend_version="test",
            agg_engine_args=args,
            num_workers=1,
        ),
        workload={"isl": 8, "osl": 3, "concurrency": 2, "num_request_ratio": 2, "max_sim_time_ms": 50.0},
        goal={"target": "throughput"},
    )
    runner = EngineReplayRunnerFactory(determinism="canonical_v1").create(0)
    try:
        report = runner.run(
            spec,
            output_requirements=ReplayOutputRequirements(
                include_raw_report=True,
                capture_telemetry=True,
                telemetry_sample_interval_ms=10.0,
            ),
        ).metadata["native_report"]
    finally:
        runner.close()
    data = build_replay_dashboard_data(report, report["telemetry"])
    assert data["prediction"]["num_requests"] == 2
    assert data["prediction"]["completed_requests"] == 0
    assert data["telemetry"]["completion_status"] == "time_limited"
    assert data["requests"] == []
    assert data["samples"][-1]["waiting"] == 2
