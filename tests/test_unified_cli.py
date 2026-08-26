# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest
import yaml

import aisimulate.main as cli
from aisimulate.output import prepare_output_directory
from aisimulate.sweeper.provider import AdapterReplaySpec, AdapterSearchPlan
from aisimulate.sweeper.replay import ReplayReport, RunnerCapabilities


class _Runner:
    def __init__(self) -> None:
        self.spec = None
        self.output_requirements = None
        self.closed = False

    def run(self, spec, *, output_requirements=None):
        self.spec = spec
        self.output_requirements = output_requirements
        return ReplayReport(
            metrics={
                "completed_requests": 1.0,
                "num_ttft_samples": 1.0,
                "num_tpot_samples": 1.0,
                "num_e2e_latency_samples": 1.0,
                "output_throughput_tok_s": 8.0,
                "mean_ttft_ms": 2.0,
                "mean_tpot_ms": 1.0,
                "mean_e2e_latency_ms": 4.0,
            },
            metadata={
                "native_report": {
                    "summary": {
                        "completed_requests": 1,
                        "output_throughput_tok_s": 8.0,
                    },
                    "per_request": [{"request_id": "synthetic-0"}],
                }
            },
        )

    def close(self) -> None:
        self.closed = True


class _Factory:
    def __init__(self, runner: _Runner) -> None:
        self.runner = runner

    def capabilities(self):
        return RunnerCapabilities(
            supported_backend_topologies=(("vllm", "agg"), ("trtllm", "agg"))
        )

    def create(self, worker_id: int):
        del worker_id
        return self.runner


class _PlacementAdapter:
    name = "engine.placement"
    section = "placement"
    config_adapter_api_version = 3
    api_version = 1

    def compile_prediction(self, config, context):
        del context
        if set(config) != {"policy"} or config["policy"] != "first":
            raise ValueError("placement.policy must be 'first'")
        return AdapterReplaySpec(config={"policy": "first"})

    def compile_recommendation(self, config, context):
        del context
        if config not in ({}, {"policy": "first"}):
            raise ValueError("placement has unknown fields")
        return AdapterSearchPlan(state={"policy": "first"})

    def materialize_candidate(self, plan, selection, context):
        del selection, context
        return AdapterReplaySpec(config=dict(plan.state))


def test_predict_is_the_single_concrete_cli(tmp_path, monkeypatch, capsys) -> None:
    config_path = tmp_path / "prediction.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "engine": {
                    "model": "example/model",
                    "hardware": "h200_sxm",
                    "context_length": 4096,
                    "workers": {"aggregated": {}},
                }
            }
        )
    )
    output = tmp_path / "out"
    runner = _Runner()
    monkeypatch.setattr(cli, "resolve_runner_factory", lambda stack: _Factory(runner))

    assert (
        cli.main(
            [
                "predict",
                "--config",
                str(config_path),
                "--output-dir",
                str(output),
                "--capture-per-request",
                "--format",
                "json",
            ]
        )
        == 0
    )

    assert runner.spec.workload["concurrency"] == 10
    assert runner.spec.workload["request_count"] == 100
    assert runner.closed is True
    assert (
        json.loads((output / "prediction.json").read_text())["summary"][
            "completed_requests"
        ]
        == 1
    )
    assert (
        json.loads((output / "requests.jsonl").read_text())["request_id"]
        == "synthetic-0"
    )
    assert json.loads(capsys.readouterr().out)["completed_requests"] == 1


def test_stack_resolution_precedes_config_read(monkeypatch, capsys) -> None:
    def unavailable(_stack):
        raise cli.StackResolutionError("stack unavailable")

    monkeypatch.setattr(cli, "resolve_runner_factory", unavailable)
    try:
        cli.main(["predict", "--stack", "missing", "--config", "/does/not/exist"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("configuration error must exit")
    assert "stack unavailable" in capsys.readouterr().err


def test_set_adapter_path_is_validated_and_materialized_by_adapter(
    tmp_path, monkeypatch, capsys
) -> None:
    config_path = tmp_path / "prediction.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "engine": {
                    "model": "example/model",
                    "hardware": "h200_sxm",
                    "context_length": 4096,
                    "workers": {"aggregated": {}},
                }
            }
        )
    )
    runner = _Runner()
    adapter = _PlacementAdapter()
    monkeypatch.setattr(cli, "resolve_runner_factory", lambda stack: _Factory(runner))

    def resolve(names):
        assert list(names) == ["engine.placement"]
        return {"engine.placement": adapter}

    monkeypatch.setattr(cli, "resolve_config_adapters", resolve)

    assert (
        cli.main(
            [
                "predict",
                "--config",
                str(config_path),
                "--set",
                "placement.policy=first",
                "--output-dir",
                str(tmp_path / "out"),
                "--format",
                "json",
            ]
        )
        == 0
    )
    assert runner.spec.adapters["engine.placement"].config == {"policy": "first"}
    assert json.loads(capsys.readouterr().out)["completed_requests"] == 1


def test_engine_stack_rejects_explicit_unavailable_component(
    tmp_path, monkeypatch, capsys
) -> None:
    config_path = tmp_path / "prediction.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "engine": {
                    "model": "example/model",
                    "hardware": "h200_sxm",
                    "context_length": 4096,
                    "workers": {"aggregated": {}},
                },
                "router": {"policy": "round_robin"},
            }
        )
    )
    monkeypatch.setattr(
        cli,
        "resolve_runner_factory",
        lambda stack: _Factory(_Runner()),
    )

    def unavailable(names):
        assert list(names) == ["engine.router"]
        raise cli.ConfigAdapterResolutionError(
            "config adapter 'engine.router' is unavailable"
        )

    monkeypatch.setattr(cli, "resolve_config_adapters", unavailable)

    with pytest.raises(SystemExit, match="2"):
        cli.main(["predict", "--config", str(config_path)])
    assert "engine.router" in capsys.readouterr().err


@pytest.mark.parametrize(("sla_field", "bound"), [("ttft_ms", 800.0), ("itl_ms", 30.0)])
def test_partial_sla_recommendation_yaml_round_trips_into_predict(
    tmp_path, monkeypatch, capsys, sla_field: str, bound: float
) -> None:
    config_path = tmp_path / "recommendation.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "engine": {
                    "mode": "aggregated",
                    "model": "deepseek-ai/DeepSeek-V3",
                    "hardware": "gb200",
                    "backend": "trtllm",
                    "backend_version": "1.3.0rc20",
                    "context_length": 2048,
                    "workers": {
                        "aggregated": {
                            "parallelism": {
                                "preset": False,
                                "replicas": 1,
                                "tensor": 4,
                                "pipeline": 1,
                                "attention_data": 1,
                                "moe_tensor": 4,
                                "moe_expert": 1,
                            },
                            "scheduler": {
                                "max_batched_tokens": 8192,
                                "max_sequences": 256,
                            },
                            "kv_cache": {
                                "block_size": 64,
                                "capacity": {"type": "fixed", "blocks": 256},
                            },
                            "timing": {
                                "type": "fixed",
                                "prefill_ms": 1,
                                "decode_ms": 1,
                            },
                        }
                    },
                },
                "evaluation": {"sla": {sla_field: bound}},
                "optimization": {
                    "target": "throughput",
                    "strict_sla": True,
                    "constraints": {"max_candidate_gpus": 8},
                },
                "optimizer": {
                    "algorithm": "random",
                    "max_trials": 1,
                    "parallelism": 1,
                    "candidate_timeout_seconds": 30,
                },
                "placement": {},
            }
        )
    )
    recommendation_output = tmp_path / "recommend-output"
    adapter = _PlacementAdapter()
    runner = _Runner()
    monkeypatch.setattr(cli, "resolve_runner_factory", lambda stack: _Factory(runner))

    def resolve(names):
        assert list(names) == ["engine.placement"]
        return {"engine.placement": adapter}

    monkeypatch.setattr(cli, "resolve_config_adapters", resolve)

    assert (
        cli.main(
            [
                "recommend",
                "--config",
                str(config_path),
                "--output-dir",
                str(recommendation_output),
                "--format",
                "json",
            ]
        )
        == 0
    )
    rows = json.loads(capsys.readouterr().out)
    recommendation = recommendation_output / "recommendations" / "0001.yaml"
    assert rows[0]["config_path"] == str(recommendation)
    generated = yaml.safe_load(recommendation.read_text())
    assert "router" not in generated
    assert "planner" not in generated
    assert generated["placement"] == {"policy": "first"}
    assert generated["evaluation"]["sla"] == {sla_field: bound}

    prediction_output = tmp_path / "predict-output"
    assert (
        cli.main(
            [
                "predict",
                "--config",
                str(recommendation),
                "--output-dir",
                str(prediction_output),
                "--format",
                "json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["completed_requests"] == 1
    assert runner.spec.goal["sla"] == {sla_field: bound}


def test_overwrite_only_removes_known_outputs(tmp_path) -> None:
    root = tmp_path / "out"
    recommendations = root / "recommendations"
    recommendations.mkdir(parents=True)
    unrelated = root / "keep.txt"
    unrelated.write_text("keep")
    (root / "prediction.json").write_text("old")
    (recommendations / "0001.yaml").write_text("old")
    (recommendations / "notes.txt").write_text("keep")

    with pytest.raises(ValueError, match="not empty"):
        prepare_output_directory(root, overwrite=False)

    prepare_output_directory(root, overwrite=True)

    assert unrelated.read_text() == "keep"
    assert (recommendations / "notes.txt").read_text() == "keep"
    assert not (root / "prediction.json").exists()
    assert not (recommendations / "0001.yaml").exists()
