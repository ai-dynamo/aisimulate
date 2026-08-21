# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import aisimulate.main as cli
import pytest
import yaml
from aisimulate.output import prepare_output_directory
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
            metrics={"completed_requests": 1.0},
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
            supported_backend_topologies=(("vllm", "agg"),)
        )

    def create(self, worker_id: int):
        assert worker_id == 0
        return self.runner


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
    assert json.loads((output / "prediction.json").read_text())["summary"][
        "completed_requests"
    ] == 1
    assert json.loads((output / "requests.jsonl").read_text())["request_id"] == "synthetic-0"
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


def test_recommendation_yaml_round_trips_into_predict(tmp_path, capsys) -> None:
    config_path = tmp_path / "recommendation.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "engine": {
                    "mode": "aggregated",
                    "model": "deepseek-ai/DeepSeek-V3",
                    "hardware": "gb200",
                    "backend": "trtllm",
                    "workers": {
                        "aggregated": {
                            "parallelism": {
                                "preset": False,
                                "replicas": 1,
                                "tensor": 4,
                                "pipeline": 1,
                                "attention_data": 1,
                                "moe_tensor": 1,
                                "moe_expert": 4,
                            },
                            "scheduler": {
                                "max_batched_tokens": 8192,
                                "max_sequences": 256,
                            },
                            "kv_cache": {
                                "capacity": {"type": "fixed", "blocks": 256}
                            },
                            "timing": {
                                "type": "fixed",
                                "prefill_ms": 1,
                                "decode_ms": 1,
                            },
                        }
                    },
                },
                "optimization": {
                    "target": "throughput",
                    "constraints": {"max_candidate_gpus": 8},
                },
                "optimizer": {
                    "algorithm": "random",
                    "max_trials": 1,
                    "parallelism": 1,
                    "candidate_timeout_seconds": 30,
                },
            }
        )
    )
    recommendation_output = tmp_path / "recommend-output"

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
    assert json.loads(capsys.readouterr().out)["completed_requests"] == 100


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
