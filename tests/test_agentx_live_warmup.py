# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from aisimulate.config.traffic import TrafficPredictionConfig, TrafficRecommendationConfig
from aisimulate.sweeper.config import Workload


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
        "engine": {"model": "example/model", "hardware": "h200_sxm", "workers": {"aggregated": {}}},
        "traffic": traffic(),
    }
    monkeypatch.setattr(cli_args, "_load_mapping", lambda _: raw)
    monkeypatch.setattr(supervision, "_installed_names", lambda _: ["legacy"])
    calls = []

    def forbidden_run(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("unsupported policy reached execution")

    monkeypatch.setattr(supervision, "run_process", forbidden_run)
    arguments = ["predict", "--config", "unused.yaml", "--stack", stack, "--output-dir", str(tmp_path), "--overwrite"]
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
        "optimization": {"target": "throughput_per_gpu", "constraints": {"max_candidate_gpus": 1}},
    }
    monkeypatch.setattr(cli_args, "_load_mapping", lambda _: raw)

    class ReachedExecution(Exception):
        pass

    def observe_run(*args, **kwargs):
        raise ReachedExecution

    monkeypatch.setattr(supervision, "run_process", observe_run)
    output = tmp_path / "output"
    with pytest.raises(ReachedExecution):
        supervision.main(["recommend", "--config", "unused.yaml", "--stack", "engine", "--output-dir", str(output)])
    assert output.is_dir()
