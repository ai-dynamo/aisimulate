# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from pydantic import ValidationError

from aisimulate.config.traffic import (
    TrafficPredictionConfig,
    TrafficRecommendationConfig,
)
from aisimulate.runner import EngineReplayRunnerFactory
from aisimulate.sweeper import (
    BackendDeploymentSpec,
    ReplayOutputRequirements,
    ReplaySpec,
)
from aisimulate.sweeper.config import Workload


@pytest.mark.parametrize("backend,tp,dp", [("vllm", 8, 4), ("vllm", 16, 2), ("sglang", 8, 1)])
@pytest.mark.parametrize("mode", ["agg", "disagg"])
@pytest.mark.parametrize("policy", ["repeated_prefixes", "live_trajectory_v1"])
def test_native_all_phase_telemetry_preserves_replay(backend, tp, dp, mode, policy):
    """Self-authored fixed-time workload exercises the supported serving shapes."""
    args = {
        "engine_type": backend,
        "aic_backend": backend,
        "aic_model_path": "telemetry-fixture",
        "aic_system": "h200_sxm",
        "aic_tp_size": tp,
        "aic_attention_dp_size": dp,
        "block_size": 4,
        "num_gpu_blocks": 64,
        "enable_prefix_caching": True,
        "timing_model": {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0},
    }
    if mode == "agg":
        deployment = BackendDeploymentSpec(
            deployment_mode=mode,
            backend=backend,
            backend_version="test",
            agg_engine_args={**args, "worker_type": "aggregated"},
            num_workers=1,
        )
    else:
        deployment = BackendDeploymentSpec(
            deployment_mode=mode,
            backend=backend,
            backend_version="test",
            prefill_engine_args={**args, "worker_type": "prefill"},
            decode_engine_args={**args, "worker_type": "decode"},
            num_prefill_workers=1,
            num_decode_workers=1,
        )
    trace = Path(__file__).parent / "e2e/configs/unified_cli/fixtures/traces/weka-two-plays.jsonl"
    spec = ReplaySpec(
        backend_deployment=deployment,
        workload={
            "source_type": "trace",
            "load_type": "trace_timestamps",
            "trace_path": str(trace.resolve()),
            "trace_format": "weka",
            "trace_block_size": 4,
            "agentic_lanes": 4,
            "agentic_snapshot": {"seed": 42},
            "agentic_warmup": True,
            "agentic_warmup_policy": policy,
            "agentic_profile": {
                "duration_seconds": 0.05,
                "response_grace_seconds": 0.01,
            },
        },
        goal={"target": "throughput"},
    )
    runner = EngineReplayRunnerFactory(determinism="canonical_v1").create(0)
    try:
        baseline = runner.run(
            spec,
            output_requirements=ReplayOutputRequirements(include_raw_report=True, capture_per_request=True),
        )
        captured = runner.run(
            spec,
            output_requirements=ReplayOutputRequirements(
                include_raw_report=True,
                capture_per_request=True,
                capture_telemetry=True,
                telemetry_sample_interval_ms=2.5,
            ),
        )
    finally:
        runner.close()
    # These measure simulator execution speed, not the virtual replay outcomes.
    wall_clock_metrics = {
        "wall_time_ms",
        "processed_output_tokens_per_s",
        "processed_tokens_per_s",
    }
    assert {k: v for k, v in captured.metrics.items() if k not in wall_clock_metrics} == {
        k: v for k, v in baseline.metrics.items() if k not in wall_clock_metrics
    }
    plain = baseline.metadata["native_report"]
    report = captured.metadata["native_report"]
    assert report["per_request"] == plain["per_request"]
    assert "telemetry" not in plain
    telemetry = report["telemetry"]
    assert telemetry["clock"] == "absolute_virtual_ms"
    assert telemetry["profile_start_ms"] > 0
    assert {row["phase"] for row in telemetry["samples"]} == {"preparation", "profile"}
    assert "warmup" in {row["phase"] for row in telemetry["requests"]}
    profile_rows = [row for row in telemetry["requests"] if row["phase"] == "profile"]
    assert len(profile_rows) == len(report["per_request"])
    assert len({row["agentic"]["play_id"] for row in profile_rows}) > 4
    origin = telemetry["profile_start_ms"]
    measured = {row["uuid"]: row for row in report["per_request"]}
    for row in profile_rows:
        assert row["arrival_time_ms"] == pytest.approx(measured[row["uuid"]]["arrival_time_ms"] + origin)
        assert row["terminal_time_ms"] == pytest.approx(measured[row["uuid"]]["terminal_time_ms"] + origin)
    for row in telemetry["requests"]:
        assert row["arrival_time_ms"] <= row["terminal_time_ms"]
        if row["output_length"] < 2:
            assert row["itl_ms"] is None
    gauges = [
        gauge
        for sample in telemetry["samples"]
        for role in ("prefill_scheduler_metrics", "decode_scheduler_metrics")
        for gauge in sample[role]
    ]
    assert {gauge["dp_rank"] for gauge in gauges} == set(range(dp))
    assert telemetry["lifecycle"]
    requests_by_uuid = {row["uuid"]: row for row in telemetry["requests"]}
    for event in telemetry["lifecycle"]:
        if event["event"] in {"dispatch", "causal_terminal", "quiescent"}:
            request = requests_by_uuid[event["request_uuid"]]
            assert event["phase"] == request["phase"]
            assert event["agentic"] == request["agentic"]
            assert request["arrival_time_ms"] <= event["at_ms"]
            assert event["at_ms"] <= request["terminal_time_ms"]


def traffic():
    return {
        "source": {"type": "trace", "paths": ["corpus"], "format": "weka"},
        "load": {
            "type": "trace_timestamps",
            "agentic_lanes": 2,
            "agentic_snapshot": {"seed": 42},
            "agentic_warmup": True,
            "agentic_warmup_policy": "live_trajectory_v1",
            "agentic_profile": {},
        },
    }


@pytest.mark.parametrize("schema", [TrafficPredictionConfig, TrafficRecommendationConfig])
def test_live_warmup_policy_roundtrip_and_legacy_default_omission(schema):
    source = traffic()
    parsed = schema.model_validate(source)
    assert parsed.load.agentic_warmup_policy == "live_trajectory_v1"
    assert parsed.model_dump()["load"]["agentic_warmup_policy"] == "live_trajectory_v1"
    source["load"].pop("agentic_warmup_policy")
    assert "agentic_warmup_policy" not in schema.model_validate(source).model_dump()["load"]


@pytest.mark.parametrize("missing", ["agentic_warmup", "agentic_profile"])
def test_live_warmup_requires_preparation_and_continuous_profile(missing):
    source = traffic()
    source["load"].pop(missing)
    with pytest.raises(ValidationError, match="live_trajectory_v1 requires"):
        TrafficPredictionConfig.model_validate(source)


@pytest.mark.parametrize("policy", [True, "live", {}, None])
def test_live_warmup_rejects_unknown_policy(policy):
    source = traffic()
    source["load"]["agentic_warmup_policy"] = policy
    with pytest.raises(ValidationError):
        TrafficPredictionConfig.model_validate(source)


def test_live_warmup_workload_serialization_keeps_the_policy_explicit():
    data = {
        "source_type": "trace",
        "load_type": "trace_timestamps",
        "trace_path": "corpus",
        "trace_format": "weka",
        "agentic_lanes": 2,
        "agentic_snapshot": {"seed": 42},
        "agentic_warmup": True,
        "agentic_profile": {},
    }
    assert "agentic_warmup_policy" not in Workload.model_validate(data).model_dump()
    data["agentic_warmup_policy"] = "live_trajectory_v1"
    assert Workload.model_validate(data).model_dump()["agentic_warmup_policy"] == "live_trajectory_v1"


@pytest.mark.parametrize("stack,online", [("legacy", False), ("engine", True)])
def test_supervisor_rejects_unsupported_live_warmup_before_overwrite(tmp_path, monkeypatch, capsys, stack, online):
    from aisimulate import cli_args, supervision

    prior = tmp_path / "prediction.json"
    prior.write_bytes(b"previous valid results\n")
    raw = {
        "engine": {
            "model": "example/model",
            "hardware": "h200_sxm",
            "workers": {"aggregated": {}},
        },
        "traffic": traffic(),
    }
    monkeypatch.setattr(cli_args, "_load_mapping", lambda _: raw)
    monkeypatch.setattr(supervision, "_installed_names", lambda _: ["legacy"])
    calls = []

    def forbidden_run(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("unsupported policy reached execution")

    monkeypatch.setattr(supervision, "run_process", forbidden_run)
    arguments = [
        "predict",
        "--config",
        "unused.yaml",
        "--stack",
        stack,
        "--output-dir",
        str(tmp_path),
        "--overwrite",
    ]
    if online:
        arguments.append("--online")
    with pytest.raises(SystemExit) as exc:
        supervision.main(arguments)
    assert exc.value.code == 2
    assert "live_trajectory_v1 CLI requires offline --stack engine" in capsys.readouterr().err
    assert not calls
    assert prior.read_bytes() == b"previous valid results\n"


def test_supervisor_default_recommendation_without_traffic_keeps_existing_path(tmp_path, monkeypatch):
    from aisimulate import cli_args, supervision

    raw = {
        "engine": {
            "model": "example/model",
            "hardware": "h200_sxm",
            "mode": "aggregated",
            "workers": {"aggregated": {}},
        },
        "optimization": {
            "target": "throughput_per_gpu",
            "constraints": {"max_candidate_gpus": 1},
        },
    }
    monkeypatch.setattr(cli_args, "_load_mapping", lambda _: raw)

    class ReachedExecution(Exception):
        pass

    def observe_run(*args, **kwargs):
        raise ReachedExecution

    monkeypatch.setattr(supervision, "run_process", observe_run)
    output = tmp_path / "output"
    with pytest.raises(ReachedExecution):
        supervision.main(
            [
                "recommend",
                "--config",
                "unused.yaml",
                "--stack",
                "engine",
                "--output-dir",
                str(output),
            ]
        )
    assert output.is_dir()
