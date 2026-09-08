# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import yaml

import aisimulate.main as cli
from aisimulate.output import prepare_output_directory
from aisimulate.sweeper.config import Candidate
from aisimulate.sweeper.provider import AdapterReplaySpec, AdapterSearchPlan
from aisimulate.sweeper.replay import ReplayReport, RunnerCapabilities


class _RecommendationResult:
    def __init__(self, selected_candidates, *, failed: int = 0) -> None:
        self._candidates = {
            f"candidate-{index:06d}": candidate
            for index, candidate in enumerate(selected_candidates, start=1)
        }
        self._selected_ids = list(self._candidates)
        self._failed = failed

    @property
    def selected_candidate_ids(self):
        return list(self._selected_ids)

    @property
    def selected_candidates(self):
        return [self._candidates[candidate_id] for candidate_id in self._selected_ids]

    def with_selected_prediction_configs(self, selections):
        self._selected_ids = [candidate_id for candidate_id, _ in selections]
        for candidate_id, config in selections:
            self._candidates[candidate_id] = self._candidates[candidate_id].model_copy(
                update={"prediction_config": dict(config)}
            )
        return self

    def to_json(self) -> str:
        return json.dumps(
            {
                "schema_version": "1.0",
                "counts": {
                    "feasible": len(self._candidates),
                    "failed": self._failed,
                },
                "candidates": [
                    {
                        "candidate_id": candidate_id,
                        "prediction_config": candidate.prediction_config,
                    }
                    for candidate_id, candidate in self._candidates.items()
                ],
                "views": {"top_n": self._selected_ids, "pareto_front": []},
            },
            sort_keys=True,
        )

    def to_csv(self) -> str:
        return "schema_version\n1.0\n"


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
                "power_w": 487.5,
                "power_coverage": 0.95,
            },
            metadata={
                "power": {
                    "source": "modeled",
                    "scope": "active_forward_pass_per_gpu",
                    "power_w_unit": "W",
                    "coverage_gate": 0.9,
                    "publication_status": "available",
                },
                "native_report": {
                    "summary": {
                        "completed_requests": 1,
                        "output_throughput_tok_s": 8.0,
                        "power_w": 487.5,
                        "power_coverage": 0.95,
                    },
                    "per_request": [{"request_id": "synthetic-0"}],
                },
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
    assert runner.spec.execution_mode == "offline"
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


def test_predict_online_is_forwarded_through_replay_spec(
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
    output = tmp_path / "out"
    runner = _Runner()

    class OnlineFactory(_Factory):
        def capabilities(self):
            return RunnerCapabilities(
                supported_execution_modes=("offline", "online"),
                supported_backend_topologies=(("vllm", "agg"),),
            )

    monkeypatch.setattr(
        cli, "resolve_runner_factory", lambda stack: OnlineFactory(runner)
    )

    assert (
        cli.main(
            [
                "predict",
                "--online",
                "--config",
                str(config_path),
                "--output-dir",
                str(output),
                "--format",
                "json",
            ]
        )
        == 0
    )

    assert runner.spec.execution_mode == "online"
    assert json.loads(capsys.readouterr().out)["completed_requests"] == 1


def test_predict_online_rejects_runner_without_online_capability(
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
    monkeypatch.setattr(
        cli, "resolve_runner_factory", lambda stack: _Factory(_Runner())
    )

    with pytest.raises(SystemExit, match="2"):
        cli.main(["predict", "--online", "--config", str(config_path)])

    assert "runner does not support execution mode 'online'" in capsys.readouterr().err


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
    assert rows[0]["power_w"] == 487.5
    assert rows[0]["power_coverage"] == 0.95
    generated = yaml.safe_load(recommendation.read_text())
    assert "router" not in generated
    assert "planner" not in generated
    assert generated["placement"] == {"policy": "first"}
    assert generated["evaluation"]["sla"] == {sla_field: bound}
    result = json.loads((recommendation_output / "recommendation.json").read_text())
    assert result["schema_version"] == "1.0"
    assert result["counts"]["feasible"] == 1
    assert result["views"]["top_n"] == ["candidate-000001"]
    assert result["candidates"][0]["prediction_config"] == generated
    assert result["candidates"][0]["metrics"]["power_w"] == 487.5
    assert result["candidates"][0]["metrics"]["power_coverage"] == 0.95
    assert result["candidates"][0]["provenance"]["power"] == {
        "coverage_gate": 0.9,
        "power_coverage": 0.95,
        "power_w": 487.5,
        "power_w_unit": "W",
        "publication_status": "available",
        "scope": "active_forward_pass_per_gpu",
        "source": "modeled",
    }
    with (recommendation_output / "recommendation.csv").open() as csv_file:
        csv_rows = list(csv.DictReader(csv_file))
    assert csv_rows[0]["power_w"] == "487.5"
    assert csv_rows[0]["power_coverage"] == "0.95"
    assert csv_rows[0]["power_source"] == "modeled"

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


def test_recommendation_outputs_each_concrete_prediction_once(
    tmp_path, monkeypatch, capsys
) -> None:
    concrete = {
        "traffic": {
            "source": {"type": "synthetic", "input_tokens": 8, "output_tokens": 2},
            "load": {"type": "concurrency", "concurrency": 1},
            "stop": {"requests": 1},
        },
        "engine": {
            "mode": "aggregated",
            "model": "example/model",
            "hardware": "h200_sxm",
            "backend": "vllm",
            "context_length": 4096,
            "workers": {
                "aggregated": {
                    "parallelism": {
                        "replicas": 1,
                        "tensor": 1,
                        "pipeline": 1,
                        "attention_data": 1,
                        "moe_tensor": 1,
                        "moe_expert": 1,
                    },
                    "scheduler": {
                        "max_batched_tokens": 8192,
                        "max_sequences": 256,
                    },
                    "kv_cache": {
                        "block_size": 64,
                        "prefix_caching": True,
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
        "evaluation": {},
    }
    recommendation_input = deepcopy(concrete)
    recommendation_input["engine"]["workers"]["aggregated"]["parallelism"] = {
        "preset": False,
        **recommendation_input["engine"]["workers"]["aggregated"]["parallelism"],
    }
    config_path = tmp_path / "recommend.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                **recommendation_input,
                "optimization": {
                    "target": "throughput",
                    "constraints": {"max_candidate_gpus": 1},
                },
                "optimizer": {
                    "algorithm": "random",
                    "max_trials": 2,
                    "parallelism": 1,
                },
            }
        )
    )
    candidates = [
        Candidate(
            config={"planner_selection": selection},
            used_gpus=1,
            score=score,
            metrics={},
            prediction_config=concrete,
        )
        for selection, score in (("first", 2.0), ("second", 1.0))
    ]
    monkeypatch.setattr(
        cli, "resolve_runner_factory", lambda stack: _Factory(_Runner())
    )
    monkeypatch.setattr(
        "aisimulate.recommend.run_recommendation",
        lambda *args, **kwargs: _RecommendationResult(candidates),
    )

    output = tmp_path / "out"
    assert (
        cli.main(
            [
                "recommend",
                "--config",
                str(config_path),
                "--output-dir",
                str(output),
                "--format",
                "json",
            ]
        )
        == 0
    )

    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 1
    assert rows[0]["score"] == 2.0
    assert [path.name for path in (output / "recommendations").iterdir()] == [
        "0001.yaml"
    ]
    result = json.loads((output / "recommendation.json").read_text())
    assert result["counts"]["feasible"] == 2
    assert result["views"]["top_n"] == ["candidate-000001"]
    assert result["candidates"][0]["prediction_config"] == yaml.safe_load(
        (output / "recommendations" / "0001.yaml").read_text()
    )


def test_overwrite_only_removes_known_outputs(tmp_path) -> None:
    root = tmp_path / "out"
    recommendations = root / "recommendations"
    recommendations.mkdir(parents=True)
    unrelated = root / "keep.txt"
    unrelated.write_text("keep")
    (root / "prediction.json").write_text("old")
    (root / "recommendation.json").write_text("old")
    (root / "recommendation.csv").write_text("old")
    (recommendations / "0001.yaml").write_text("old")
    (recommendations / "notes.txt").write_text("keep")

    with pytest.raises(ValueError, match="not empty"):
        prepare_output_directory(root, overwrite=False)

    prepare_output_directory(root, overwrite=True)

    assert unrelated.read_text() == "keep"
    assert (recommendations / "notes.txt").read_text() == "keep"
    assert not (root / "prediction.json").exists()
    assert not (root / "recommendation.json").exists()
    assert not (root / "recommendation.csv").exists()
    assert not (recommendations / "0001.yaml").exists()


def test_recommendation_writes_an_empty_result_before_returning_failure(
    tmp_path, monkeypatch, capsys
) -> None:
    import aisimulate.recommend as recommendation_module

    monkeypatch.setattr(
        cli.CoreRecommendationConfig,
        "model_validate",
        staticmethod(lambda raw: object()),
    )
    monkeypatch.setattr(cli, "_resolve_section_adapters", lambda sections, stack: {})
    monkeypatch.setattr(
        recommendation_module,
        "run_recommendation",
        lambda *args, **kwargs: _RecommendationResult([], failed=1),
    )
    output = tmp_path / "empty-result"

    status = cli._recommend(
        SimpleNamespace(
            stack="engine",
            format="json",
            output_dir=str(output),
            overwrite=False,
        ),
        {},
        object(),
    )

    assert status == 1
    assert json.loads((output / "recommendation.json").read_text())["counts"] == {
        "failed": 1,
        "feasible": 0,
    }
    assert "saved full result" in capsys.readouterr().err
