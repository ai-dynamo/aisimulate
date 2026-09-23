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
            f"candidate-{index:06d}": candidate for index, candidate in enumerate(selected_candidates, start=1)
        }
        self._selected_ids = list(self._candidates)
        self._failed = failed
        self.counts = SimpleNamespace(resource_limited=0)

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
    def __init__(self, *, power_w: float | None = 487.5, power_coverage: float | None = 0.95) -> None:
        self.spec = None
        self.output_requirements = None
        self.closed = False
        self.power_w = power_w
        self.power_coverage = power_coverage

    def run(self, spec, *, output_requirements=None):
        self.spec = spec
        self.output_requirements = output_requirements
        metrics = {
            "completed_requests": 1.0,
            "num_ttft_samples": 1.0,
            "num_tpot_samples": 1.0,
            "num_e2e_latency_samples": 1.0,
            "output_throughput_tok_s": 8.0,
            "mean_ttft_ms": 2.0,
            "mean_tpot_ms": 1.0,
            "mean_e2e_latency_ms": 4.0,
            "power_coverage": self.power_coverage,
        }
        summary = {
            "completed_requests": 1,
            "output_throughput_tok_s": 8.0,
            "power_coverage": self.power_coverage,
        }
        power_is_publishable = (
            self.power_w is not None and self.power_coverage is not None and self.power_coverage >= 0.9
        )
        if power_is_publishable:
            metrics["power_w"] = self.power_w
            summary["power_w"] = self.power_w
        power_diagnostics = {
            "schema_version": "1.0",
            "scope": "active_forward_pass_per_gpu",
            "power_w_unit": "W",
            "energy_unit": "W-ms",
            "latency_unit": "ms",
            "publication_status": ("available" if power_is_publishable else "withheld"),
            "coverage_gate": 0.9,
            "power_coverage": self.power_coverage,
            "energy_wms": 121.875,
            "latency_ms": 0.25,
            "covered_latency_ms": 0.25 * (self.power_coverage or 0.0),
            "phases": [
                {
                    "name": "prefill",
                    "energy_wms": 121.875,
                    "latency_ms": 0.25,
                    "covered_latency_ms": 0.25 * (self.power_coverage or 0.0),
                    "power_coverage": self.power_coverage,
                    "publication_status": ("available" if power_is_publishable else "withheld"),
                    "source": "silicon",
                    "source_kind": "measured",
                    "operations": [
                        {
                            "name": "gemm",
                            "energy_wms": 121.875,
                            "latency_ms": 0.25,
                            "covered_latency_ms": 0.25 * (self.power_coverage or 0.0),
                            "power_coverage": self.power_coverage,
                            "energy_contribution": 1.0,
                            "source": "silicon",
                            "source_kind": "measured",
                            "status": ("available" if self.power_coverage == 1.0 else "partial"),
                        }
                    ],
                }
            ],
        }
        power_diagnostics["power_w"] = self.power_w if power_is_publishable else None
        phase = power_diagnostics["phases"][0]
        phase["power_w"] = self.power_w if power_is_publishable else None
        first = phase["operations"][0]
        second = {**first, "name": "attention"}
        for op, scale in ((first, 0.8), (second, 0.2)):
            for field in ("energy_wms", "latency_ms", "covered_latency_ms", "energy_contribution"):
                op[field] *= scale
        phase["operations"].append(second)
        if power_is_publishable:
            power_diagnostics["power_w"] = self.power_w
            power_diagnostics["phases"][0]["power_w"] = self.power_w
        if self.power_coverage is not None and self.power_coverage < 1.0:
            power_diagnostics["phases"][0]["operations"][0]["uncovered_reason"] = (
                "some accumulated operation latency lacks positive energy evidence"
            )
        if self.power_coverage is None:
            power_diagnostics = {
                "schema_version": "1.0",
                "scope": "active_forward_pass_per_gpu",
                "publication_status": "unsupported",
                "power_w": None,
                "power_coverage": None,
                "coverage_gate": 0.9,
                "phases": [],
                "unavailable_reason": "test provider unsupported",
            }
        return ReplayReport(
            metrics=metrics,
            metadata={
                "power": {
                    "source": "modeled",
                    "scope": "active_forward_pass_per_gpu",
                    "power_w_unit": "W",
                    "coverage_gate": 0.9,
                    "publication_status": ("available" if power_is_publishable else "withheld"),
                },
                "native_report": {
                    "summary": summary,
                    "power_diagnostics": power_diagnostics,
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
        return RunnerCapabilities(supported_backend_topologies=(("vllm", "agg"), ("trtllm", "agg")))

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
    assert json.loads((output / "prediction.json").read_text())["summary"]["completed_requests"] == 1
    assert json.loads((output / "requests.jsonl").read_text())["request_id"] == "synthetic-0"
    assert json.loads(capsys.readouterr().out)["completed_requests"] == 1


def test_predict_online_is_forwarded_through_replay_spec(tmp_path, monkeypatch, capsys) -> None:
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

    monkeypatch.setattr(cli, "resolve_runner_factory", lambda stack: OnlineFactory(runner))

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


def test_predict_power_diagnostics_json_preserves_complete_export(tmp_path, monkeypatch, capsys) -> None:
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
    monkeypatch.setattr(cli, "resolve_runner_factory", lambda stack: _Factory(_Runner()))

    assert (
        cli.main(
            [
                "predict",
                "--config",
                str(config_path),
                "--output-dir",
                str(output),
                "--diagnostics",
                "power",
                "--format",
                "json",
            ]
        )
        == 0
    )

    stdout = json.loads(capsys.readouterr().out)
    assert stdout["summary"]["power_w"] == 487.5
    diagnostics = stdout["power_diagnostics"]
    assert diagnostics["power_w"] == diagnostics["energy_wms"] / diagnostics["latency_ms"]
    phase = diagnostics["phases"][0]
    assert phase["power_w"] == phase["energy_wms"] / phase["latency_ms"]
    assert phase["operations"][0]["name"] == "gemm"
    saved = json.loads((output / "prediction.json").read_text())
    assert saved["power_diagnostics"] == diagnostics


def test_predict_power_diagnostics_table_applies_top_n(tmp_path, monkeypatch, capsys) -> None:
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
    monkeypatch.setattr(cli, "resolve_runner_factory", lambda stack: _Factory(_Runner()))

    assert (
        cli.main(
            [
                "predict",
                "--config",
                str(config_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--diagnostics",
                "power",
                "--diagnostics-top-n",
                "1",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    assert "active forward-pass energy diagnostics (per GPU)" in stdout
    assert "gemm" in stdout

    assert "attention" not in stdout
    assert "... 1 more in prediction.json" in stdout
    saved = json.loads((tmp_path / "out" / "prediction.json").read_text())
    assert len(saved["power_diagnostics"]["phases"][0]["operations"]) == 2


def test_predict_online_rejects_runner_without_online_capability(tmp_path, monkeypatch, capsys) -> None:
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
    monkeypatch.setattr(cli, "resolve_runner_factory", lambda stack: _Factory(_Runner()))

    with pytest.raises(SystemExit, match="2"):
        cli.main(["predict", "--online", "--config", str(config_path)])

    assert "runner does not support execution mode 'online'" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["predict", "recommend"])
def test_dry_run_is_rejected_before_execution(command, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "resolve_runner_factory", lambda _: pytest.fail("removed option must stop before replay"))

    with pytest.raises(SystemExit) as exc:
        cli.main([command, "--config", "unused.yaml", "--dry-run"])

    assert exc.value.code == 2
    assert "unrecognized arguments: --dry-run" in capsys.readouterr().err


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


@pytest.mark.filterwarnings("error")
def test_recommend_runner_incompatibility_is_cli_config_error(tmp_path, monkeypatch, capsys) -> None:
    from aisimulate.recommend import _run_recommendation

    monkeypatch.setattr("aisimulate.recommend.run_recommendation", _run_recommendation)
    config_path = tmp_path / "recommendation.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "engine": {
                    "mode": "aggregated",
                    "model": "example/model",
                    "hardware": "h200_sxm",
                    "backend": "vllm",
                    "context_length": 4096,
                    "workers": {"aggregated": {}},
                },
                "optimization": {
                    "target": "throughput",
                    "constraints": {"max_candidate_gpus": 1},
                },
                "optimizer": {"max_trials": 1, "parallelism": 1},
            }
        )
    )

    class IncompatibleFactory(_Factory):
        def capabilities(self):
            return RunnerCapabilities(supported_backend_topologies=(("trtllm", "agg"),))

    monkeypatch.setattr(
        cli,
        "resolve_runner_factory",
        lambda stack: IncompatibleFactory(_Runner()),
    )

    with pytest.raises(SystemExit, match="2"):
        cli.main(
            [
                "recommend",
                "--config",
                str(config_path),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )

    error = capsys.readouterr().err
    assert "no configured backend/topology is supported by the Replay runner" in error
    assert "deployment_mode='agg'" in error
    assert "runner-incompatible backends=['vllm']" in error
    assert not (tmp_path / "out").exists()


def test_set_adapter_path_is_validated_and_materialized_by_adapter(tmp_path, monkeypatch, capsys) -> None:
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


def test_engine_stack_rejects_explicit_unavailable_component(tmp_path, monkeypatch, capsys) -> None:
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
        raise cli.ConfigAdapterResolutionError("config adapter 'engine.router' is unavailable")

    monkeypatch.setattr(cli, "resolve_config_adapters", unavailable)

    with pytest.raises(SystemExit, match="2"):
        cli.main(["predict", "--stack", "engine", "--config", str(config_path)])
    assert "engine.router" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("router", "explicit", "selected"),
    [
        (None, None, "engine"),
        ({"policy": "kv_router"}, None, "dynamo-policy"),
        ({"policy": "kv_router", "affinity": {"mode": "sibling_group"}}, None, "dynamo-policy"),
        ({"policy": "kv_router"}, "engine", "engine"),
        ({"policy": "kv_router"}, "dynamo", "dynamo"),
    ],
)
def test_prediction_selects_stack_from_router_only_when_not_explicit(
    tmp_path, monkeypatch, router, explicit, selected
) -> None:
    raw = {"engine": {}}
    if router is not None:
        raw["router"] = router
    path = tmp_path / "routing.yaml"
    path.write_text(yaml.safe_dump(raw))
    factory = object()
    loaded = []

    def resolve(name):
        loaded.append(name)
        return factory

    def predict(args, config, actual_factory):
        assert args.stack == selected
        assert config == raw
        assert actual_factory is factory
        return 0

    monkeypatch.setattr(cli, "resolve_runner_factory", resolve)
    monkeypatch.setattr(cli, "_predict", predict)
    argv = ["predict", "--config", str(path)]
    if explicit is not None:
        argv += ["--stack", explicit]
    assert cli.main(argv) == 0
    assert loaded == [selected]


def test_prediction_router_override_selects_native_policy(tmp_path, monkeypatch) -> None:
    path = tmp_path / "routing.yaml"
    path.write_text("engine: {}\n")
    loaded = []
    monkeypatch.setattr(cli, "resolve_runner_factory", lambda name: loaded.append(name) or object())
    monkeypatch.setattr(cli, "_predict", lambda args, raw, factory: 0)
    assert cli.main(["predict", "--config", str(path), "--set", "router.policy=kv_router"]) == 0
    assert loaded == ["dynamo-policy"]


def test_prediction_router_missing_integration_is_configuration_error(tmp_path, monkeypatch, capsys) -> None:
    from aisimulate.stack import resolve_runner_factory

    path = tmp_path / "routing.yaml"
    path.write_text("engine: {}\nrouter: {policy: kv_router}\n")
    monkeypatch.setattr(cli, "resolve_runner_factory", lambda name: resolve_runner_factory(name, entry_points=[]))
    with pytest.raises(SystemExit, match="2"):
        cli.main(["predict", "--config", str(path)])
    error = capsys.readouterr().err
    assert "does not provide native Dynamo routing" in error
    assert "pip install" in error
    assert "will not fall back" in error


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
    assert result["schema_version"] == "1.1"
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

    runner.power_coverage = 0.9
    prediction_output = tmp_path / "predict-output"
    assert (
        cli.main(
            [
                "predict",
                "--config",
                str(recommendation),
                "--output-dir",
                str(prediction_output),
                "--diagnostics",
                "power",
                "--format",
                "json",
            ]
        )
        == 0
    )
    prediction_stdout = json.loads(capsys.readouterr().out)
    prediction_summary = prediction_stdout["summary"]
    assert prediction_summary["completed_requests"] == 1
    assert prediction_summary["power_w"] == 487.5
    assert prediction_summary["power_coverage"] == 0.9
    prediction_report = json.loads((prediction_output / "prediction.json").read_text())
    assert prediction_report["summary"]["power_w"] == 487.5
    assert prediction_report["summary"]["power_coverage"] == 0.9
    diagnostics = prediction_report["power_diagnostics"]
    assert diagnostics == prediction_stdout["power_diagnostics"]
    assert diagnostics["schema_version"] == "1.0"
    assert diagnostics["scope"] == "active_forward_pass_per_gpu"
    assert diagnostics["publication_status"] == "available"
    assert diagnostics["power_w"] == diagnostics["energy_wms"] / diagnostics["latency_ms"]
    phase = diagnostics["phases"][0]
    assert phase["power_w"] == phase["energy_wms"] / phase["latency_ms"]
    assert phase["operations"][0]["name"] == "gemm"
    assert phase["operations"][0]["source"] == "silicon"
    assert runner.spec.goal["sla"] == {sla_field: bound}

    withheld_output = tmp_path / "withheld-recommend-output"
    withheld_runner = _Runner(power_w=487.5, power_coverage=0.42)
    monkeypatch.setattr(cli, "resolve_runner_factory", lambda stack: _Factory(withheld_runner))
    assert (
        cli.main(
            [
                "recommend",
                "--config",
                str(config_path),
                "--output-dir",
                str(withheld_output),
                "--format",
                "json",
            ]
        )
        == 0
    )
    withheld_rows = json.loads(capsys.readouterr().out)
    assert withheld_rows[0]["power_w"] is None
    assert withheld_rows[0]["power_coverage"] == 0.42
    withheld_result = json.loads((withheld_output / "recommendation.json").read_text())
    withheld_candidate = withheld_result["candidates"][0]
    assert withheld_candidate["metrics"]["power_w"] is None
    assert withheld_candidate["metrics"]["power_coverage"] == 0.42
    assert withheld_candidate["provenance"]["power"]["publication_status"] == "withheld"
    assert withheld_candidate["provenance"]["power"]["scope"] == "active_forward_pass_per_gpu"
    assert withheld_candidate["provenance"]["power"]["power_w_unit"] == "W"
    assert withheld_candidate["provenance"]["power"]["coverage_gate"] == 0.9
    assert withheld_candidate["provenance"]["power"]["source"] == "modeled"
    assert withheld_candidate["provenance"]["power"]["power_coverage"] == 0.42
    assert withheld_candidate["provenance"]["power"]["power_w"] is None
    with (withheld_output / "recommendation.csv").open() as csv_file:
        withheld_csv = list(csv.DictReader(csv_file))[0]
    assert withheld_csv["power_w"] == ""
    assert withheld_csv["power_coverage"] == "0.42"
    assert withheld_csv["power_source"] == "modeled"

    withheld_prediction_output = tmp_path / "withheld-predict-output"
    assert (
        cli.main(
            [
                "predict",
                "--config",
                str(recommendation),
                "--output-dir",
                str(withheld_prediction_output),
                "--diagnostics",
                "power",
                "--format",
                "json",
            ]
        )
        == 0
    )
    withheld_prediction_stdout = json.loads(capsys.readouterr().out)
    withheld_prediction_summary = withheld_prediction_stdout["summary"]
    assert withheld_prediction_summary["power_w"] is None
    assert withheld_prediction_summary["power_coverage"] == 0.42
    withheld_prediction_report = json.loads((withheld_prediction_output / "prediction.json").read_text())
    assert withheld_prediction_report["summary"]["power_w"] is None
    assert withheld_prediction_report["summary"]["power_coverage"] == 0.42
    withheld_diagnostics = withheld_prediction_report["power_diagnostics"]
    assert withheld_diagnostics == withheld_prediction_stdout["power_diagnostics"]
    assert withheld_diagnostics["schema_version"] == "1.0"
    assert withheld_diagnostics["scope"] == "active_forward_pass_per_gpu"
    assert withheld_diagnostics["publication_status"] == "withheld"
    assert withheld_diagnostics["power_w"] is None
    assert withheld_diagnostics["phases"][0]["operations"][0]["name"] == "gemm"
    assert withheld_diagnostics["phases"][0]["operations"][0]["source"] == "silicon"

    withheld_runner.power_w = None
    withheld_runner.power_coverage = None
    assert (
        cli.main(
            [
                "recommend",
                "--config",
                str(config_path),
                "--output-dir",
                str(withheld_output),
                "--overwrite",
                "--format",
                "json",
            ]
        )
        == 0
    )
    unsupported_rows = json.loads(capsys.readouterr().out)
    assert unsupported_rows[0]["power_w"] is None
    assert unsupported_rows[0]["power_coverage"] is None
    unsupported_candidate = json.loads((withheld_output / "recommendation.json").read_text())["candidates"][0]
    assert unsupported_candidate["provenance"]["power"]["publication_status"] == "unavailable"
    for key in ("power_w", "power_coverage"):
        assert key in unsupported_candidate["metrics"] and unsupported_candidate["metrics"][key] is None
        assert (
            key in unsupported_candidate["provenance"]["power"]
            and unsupported_candidate["provenance"]["power"][key] is None
        )
    assert (
        cli.main(
            ["predict", "--config", str(recommendation), "--output-dir", str(withheld_prediction_output), "--overwrite"]
        )
        == 0
    )
    table = capsys.readouterr().out
    assert "power_w" in table and "power_coverage" in table
    assert "unavailable (energy provider or topology unsupported)" in table
    unsupported_summary = json.loads((withheld_prediction_output / "prediction.json").read_text())["summary"]
    assert unsupported_summary["power_w"] is None
    assert unsupported_summary["power_coverage"] is None


@pytest.mark.parametrize("resource_limited", [0, 1])
def test_recommendation_outputs_each_concrete_prediction_once(tmp_path, monkeypatch, capsys, resource_limited) -> None:
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
    monkeypatch.setattr(cli, "resolve_runner_factory", lambda stack: _Factory(_Runner()))
    partial = _RecommendationResult(candidates)
    partial.counts.resource_limited = resource_limited
    monkeypatch.setattr(
        "aisimulate.recommend.run_recommendation",
        lambda *args, **kwargs: partial,
    )

    output = tmp_path / "out"
    assert cli.main(
        [
            "recommend",
            "--config",
            str(config_path),
            "--output-dir",
            str(output),
            "--format",
            "json",
        ]
    ) == (3 if resource_limited else 0)

    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 1
    assert rows[0]["score"] == 2.0
    assert [path.name for path in (output / "recommendations").iterdir()] == ["0001.yaml"]
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
    (root / "afd-replay-spec.json").write_text("old")
    (root / "afd-qualification.json").write_text("old")
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
    assert not (root / "afd-replay-spec.json").exists()
    assert not (root / "afd-qualification.json").exists()
    assert not (recommendations / "0001.yaml").exists()


@pytest.mark.parametrize("resource_limited", [0, 1])
def test_recommendation_writes_an_empty_result_before_returning_failure(
    tmp_path, monkeypatch, capsys, resource_limited
) -> None:
    import aisimulate.recommend as recommendation_module

    monkeypatch.setattr(
        cli.CoreRecommendationConfig,
        "model_validate",
        staticmethod(lambda raw: object()),
    )
    monkeypatch.setattr(cli, "_resolve_section_adapters", lambda sections, stack: {})
    result = _RecommendationResult([], failed=1)
    result.counts.resource_limited = resource_limited
    monkeypatch.setattr(
        recommendation_module,
        "run_recommendation",
        lambda *args, **kwargs: result,
    )
    monkeypatch.setattr(cli, "_resource_plan", lambda *args: {"status": "admitted"})
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

    assert status == (3 if resource_limited else 1)
    assert json.loads((output / "recommendation.json").read_text())["counts"] == {
        "failed": 1,
        "feasible": 0,
    }
    assert "saved full result" in capsys.readouterr().err


def _detail_config(tmp_path):
    path = tmp_path / "detail.yaml"
    path.write_text(
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
    return path


@pytest.mark.parametrize("nested", [True, False], ids=["nested-native", "flat-native"])
@pytest.mark.parametrize("native_power", [{}, {"power_w": 999.0, "power_coverage": 1.0}], ids=["missing", "stale"])
@pytest.mark.parametrize("watts,coverage", [(487.5, 0.95), (None, 0.42), (None, None)])
def test_predict_uses_validated_power_metrics_over_raw_metadata(
    tmp_path, monkeypatch, capsys, nested, native_power, watts, coverage
):
    class RawMetadataRunner(_Runner):
        def run(self, *args, **kwargs):
            report = super().run(*args, **kwargs)
            native = report.metadata["native_report"]
            summary = native["summary"]
            summary.pop("power_w", None)
            summary.pop("power_coverage", None)
            summary.update(native_power)
            if not nested:
                report.metadata["native_report"] = summary
            return report

    runner = RawMetadataRunner(power_w=watts, power_coverage=coverage)
    monkeypatch.setattr(cli, "resolve_runner_factory", lambda _: _Factory(runner))
    output = tmp_path / "out"
    assert (
        cli.main(["predict", "-c", str(_detail_config(tmp_path)), "--output-dir", str(output), "--format", "json"]) == 0
    )
    stdout = json.loads(capsys.readouterr().out)
    saved = json.loads((output / "prediction.json").read_text())
    saved_summary = saved["summary"] if nested else saved
    for summary in (stdout, saved_summary):
        assert summary["power_w"] == watts
        assert summary["power_coverage"] == coverage


@pytest.mark.parametrize("selector", ["", "unknown", "time,", "all,unknown", "SUMMARY", "power"])
def test_detail_rejects_unsupported_sections_before_loading_config(selector, monkeypatch):
    monkeypatch.setattr(cli, "_load_mapping", lambda *_: pytest.fail("must validate selector first"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["predict", "-c", "missing.yaml", "--detail", selector])
    assert exc.value.code == 2


def _detail_schema():
    from pathlib import Path

    return json.loads((Path(__file__).resolve().parents[1] / "docs/cli/prediction-details.schema.json").read_text())


@pytest.mark.parametrize(
    "watts,coverage,status,valid",
    [
        (450, 0.9, "available", True),
        (450, 1.0, "available", True),
        (None, 0.0, "withheld", True),
        (None, 0.89, "withheld", True),
        (None, 1.0, "missing", True),
        (None, None, "unsupported", True),
        (None, 0.0, "not_observed", True),
        (450, 0.8999999999999999, "available", False),
        (450, None, "available", False),
        (450, True, "available", False),
        (0, 1.0, "available", False),
        (-1, 1.0, "available", False),
        (None, 1.01, "missing", False),
    ],
)
def test_energy_detail_schema_enforces_power_publication(watts, coverage, status, valid):
    from jsonschema import Draft202012Validator

    details = {
        "schema_version": "1.0",
        "sections": {
            "energy": {
                "status": status,
                "scope": "active_forward_pass_per_gpu",
                "diagnostics": {
                    "power_w": watts,
                    "power_coverage": coverage,
                    "publication_status": status,
                    "phases": [],
                },
            }
        },
        "skipped": {},
    }
    assert Draft202012Validator(_detail_schema()).is_valid(details) is valid


@pytest.mark.parametrize("status", ["available", "withheld", "unsupported", "not_observed", "missing", "invented"])
@pytest.mark.parametrize(
    "publication_status", ["available", "withheld", "unsupported", "not_observed", "missing", "invented"]
)
def test_energy_detail_schema_constrains_publication_status(status, publication_status):
    from jsonschema import Draft202012Validator

    schema = _detail_schema()["properties"]["sections"]["properties"]["energy"]
    section = {
        "status": status,
        "scope": "active_forward_pass_per_gpu",
        "diagnostics": {
            "power_w": 400.0 if publication_status == "available" else None,
            "power_coverage": {"available": 1.0, "withheld": 0.8, "missing": 1.0, "not_observed": 0.0}.get(
                publication_status
            ),
            "publication_status": publication_status,
            "phases": [],
        },
    }
    assert Draft202012Validator(schema).is_valid(section) is (status == publication_status and status != "invented")


@pytest.mark.parametrize("selector", ["all", "time,time", "memory"])
def test_detail_selected_json_and_skips_match_saved_report(tmp_path, monkeypatch, capsys, selector):
    from jsonschema import validate

    class TimingRunner(_Runner):
        def run(self, *args, **kwargs):
            report = super().run(*args, **kwargs)
            report.metadata["native_report"]["summary"].update(
                {
                    "mean_ttft_ms": 2.0,
                    "p99_ttft_ms": 3.0,
                    "mean_trajectory_e2e_latency_ms": 9.0,
                    "p50_trajectory_e2e_latency_ms": 8.0,
                    "wall_time_ms": 40.0,
                    "duration_ms": 100.0,
                    "custom_timer_ms": 7.0,
                }
            )
            return report

    runner = TimingRunner()
    monkeypatch.setattr(cli, "resolve_runner_factory", lambda _: _Factory(runner))
    output = tmp_path / "out"
    assert (
        cli.main(
            [
                "predict",
                "-c",
                str(_detail_config(tmp_path)),
                "--output-dir",
                str(output),
                "--format",
                "json",
                "--detail",
                selector,
            ]
        )
        == 0
    )
    stdout = json.loads(capsys.readouterr().out)
    saved = json.loads((output / "prediction.json").read_text())
    assert stdout["details"] == saved["details"]
    assert stdout["summary"]["duration_ms"] == 100.0
    details = stdout["details"]
    validate(details, _detail_schema())
    expected = {"all": {"summary", "time", "energy", "source"}, "time,time": {"time"}, "memory": set()}[selector]
    assert set(details["sections"]) == expected
    assert ("memory" in details["skipped"]) == (selector != "time,time")
    assert runner.output_requirements.capture_memory_diagnostics == (selector != "time,time")
    if "time" in expected:
        assert details["sections"]["time"]["serving_metrics"] == {
            "mean_ttft_ms": 2.0,
            "p99_ttft_ms": 3.0,
            "mean_trajectory_e2e_latency_ms": 9.0,
            "p50_trajectory_e2e_latency_ms": 8.0,
        }


@pytest.mark.parametrize("name", ["duration_ms", "wall_time_ms", "custom_timer_ms"])
def test_detail_schema_rejects_non_latency_timers(name):
    from jsonschema import ValidationError, validate

    details = {
        "schema_version": "1.0",
        "sections": {
            "time": {
                "status": "available",
                "scope": "serving_workload",
                "latency_unit": "ms",
                "serving_metrics": {name: 1.0},
            }
        },
        "skipped": {},
    }
    with pytest.raises(ValidationError):
        validate(details, _detail_schema())


def test_detail_table_skips_missing_evidence(tmp_path, monkeypatch, capsys):
    class DurationOnlyRunner(_Runner):
        def run(self, *args, **kwargs):
            report = super().run(*args, **kwargs)
            report.metadata["native_report"]["summary"]["duration_ms"] = 10.0
            return report

    monkeypatch.setattr(cli, "resolve_runner_factory", lambda _: _Factory(DurationOnlyRunner()))
    assert (
        cli.main(
            ["predict", "-c", str(_detail_config(tmp_path)), "--detail", "all", "--output-dir", str(tmp_path / "out")]
        )
        == 0
    )
    stdout = capsys.readouterr().out
    assert "Skipped memory:" in stdout
    assert "Detail: time" in stdout
    assert "did not export operation timing" in stdout
    assert "Detail: energy" in stdout
    assert "Detail: source" in stdout


@pytest.mark.parametrize(
    "section",
    [
        {"memory": {"status": "available", "scope": "capacity_estimate_per_rank"}},
        {"memory": {"status": "partial", "scope": "capacity_estimate_per_rank", "roles": {}}},
        {"time": {"status": "available", "scope": "serving_workload", "latency_unit": "ms"}},
        {
            "time": {
                "status": "available",
                "scope": "serving_workload",
                "latency_unit": "ms",
                "serving_metrics": {"duration_ms": 1},
            }
        },
        {"energy": {"status": "unavailable"}},
    ],
)
def test_detail_schema_rejects_incomplete_or_unsupported_sections(section):
    from jsonschema import ValidationError, validate

    with pytest.raises(ValidationError):
        validate({"schema_version": "1.0", "sections": section, "skipped": {}}, _detail_schema())


def test_detail_partial_memory_preserves_unavailable_role_reason():
    from jsonschema import ValidationError, validate

    from aisimulate.detail import build_prediction_details, format_prediction_details

    available = {
        "status": "available",
        "scope": "capacity_estimate_per_rank",
        "stage": "before_native_capacity_adjustments",
        "source": "native",
        "total_gpu_capacity_bytes": 4096,
        "total_kv_size_bytes": 1024,
        "total_kv_size_tokens": 128,
        "kv_size_per_token_bytes": 8,
        "scheduler_block_size_tokens": 64,
        "estimated_num_gpu_blocks": 2,
        "memory_breakdown": None,
        "tolerance_adjusted": None,
    }
    unavailable = {
        "status": "unavailable",
        "scope": "capacity_estimate_per_rank",
        "stage": "before_native_capacity_adjustments",
        "unavailable_reason": "explicit KV blocks",
    }
    result = build_prediction_details(
        {"memory_diagnostics": {"prefill": available, "decode": unavailable}}, ("memory",)
    )
    validate(result, _detail_schema())
    assert result["sections"]["memory"]["status"] == "partial"
    assert "explicit KV blocks" in format_prediction_details(result)
    assert result["skipped"] == {}
    result["sections"]["memory"]["status"] = "available"
    with pytest.raises(ValidationError):
        validate(result, _detail_schema())


@pytest.mark.parametrize("watts,coverage", [(487.5, 0.95), (None, 0.42), (None, None)])
def test_energy_detail_preserves_default_summary_and_all_evidence(tmp_path, monkeypatch, capsys, watts, coverage):
    from jsonschema import validate

    config = _detail_config(tmp_path)
    monkeypatch.setattr(
        cli, "resolve_runner_factory", lambda stack: _Factory(_Runner(power_w=watts, power_coverage=coverage))
    )
    plain_dir, detail_dir = tmp_path / "plain", tmp_path / "detail"
    assert cli.main(["predict", "-c", str(config), "--format", "json", "--output-dir", str(plain_dir)]) == 0
    plain = json.loads(capsys.readouterr().out)
    assert (
        cli.main(
            [
                "predict",
                "-c",
                str(config),
                "--detail",
                "energy",
                "--format",
                "json",
                "--diagnostics-top-n",
                "1",
                "--output-dir",
                str(detail_dir),
            ]
        )
        == 0
    )
    detailed = json.loads(capsys.readouterr().out)
    assert detailed["summary"] == plain
    assert plain["power_w"] == watts and plain["power_coverage"] == coverage
    validate(detailed["details"], _detail_schema())
    evidence = detailed["details"]["sections"]["energy"]["diagnostics"]
    assert evidence["power_w"] == watts and evidence["power_coverage"] == coverage
    assert "power_diagnostics" not in plain
    if coverage is not None:
        assert len(evidence["phases"][0]["operations"]) == 2
    saved = json.loads((detail_dir / "prediction.json").read_text())
    assert saved["details"] == detailed["details"]


def test_energy_detail_reports_missing_adapter_export(tmp_path, monkeypatch, capsys):
    class AdapterWithoutDiagnostics(_Runner):
        def run(self, *args, **kwargs):
            report = super().run(*args, **kwargs)
            report.metadata["native_report"].pop("power_diagnostics")
            return report

    monkeypatch.setattr(cli, "resolve_runner_factory", lambda stack: _Factory(AdapterWithoutDiagnostics()))
    assert (
        cli.main(
            [
                "predict",
                "-c",
                str(_detail_config(tmp_path)),
                "--detail",
                "energy",
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "did not export typed timing-energy evidence" in text
    assert "downstream Dynamo adapter export is not qualified" in text


@pytest.mark.parametrize("profile", [False, True])
def test_snapshot_seed_override_reaches_the_existing_predict_path(tmp_path, monkeypatch, capsys, profile) -> None:
    import os

    from aisimulate.runner import EngineReplayRunnerFactory

    class SnapshotFactory(_Factory):
        def capabilities(self):
            return EngineReplayRunnerFactory().capabilities()

    trace_path = tmp_path / "play.json"
    trace_path.write_text(
        json.dumps({"id": "play", "block_size": 64, "requests": [{"t": 0, "type": "s", "in": 64, "out": 1}]})
    )
    config_path = tmp_path / "snapshot.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "engine": {
                    "model": "example/model",
                    "hardware": "h200_sxm",
                    "context_length": 1024,
                    "workers": {"aggregated": {}},
                },
                "traffic": {
                    "source": {"type": "trace", "format": "weka", "paths": [str(trace_path)]},
                    "load": {"type": "trace_timestamps", "agentic_lanes": 2},
                },
            }
        )
    )
    runner = _Runner()
    monkeypatch.setattr(cli, "resolve_runner_factory", lambda stack: SnapshotFactory(runner))
    if profile:
        # This unit test calls the internal child entry point directly. Model
        # its supervisor context; public-process supervision is tested separately.
        monkeypatch.setenv(
            "_AISIMULATE_SUPERVISED_BUDGET",
            json.dumps(
                {
                    "supervisor_pid": os.getpid(),
                    "memory_limit_bytes": 4_000_000_000,
                    "cpu_limit": 1,
                    "reserved_host_memory_bytes": 1_000_000_000,
                    "events_path": str(tmp_path / "events.jsonl"),
                    "ready_path": str(tmp_path / "ready"),
                }
            ),
        )
    assert (
        cli.main(
            [
                "predict",
                "--config",
                str(config_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--set",
                "traffic.load.agentic_snapshot.seed=42",
                *(["--set", "traffic.load.agentic_profile.duration_seconds=5.0"] if profile else []),
                "--format",
                "json",
            ]
        )
        == 0
    )
    assert runner.spec.workload["agentic_snapshot"] == {"seed": 42}
    assert runner.spec.workload["agentic_lanes"] == 2
    if profile:
        assert runner.spec.workload["agentic_profile"]["duration_seconds"] == 5.0
        assert runner.spec.workload["agentic_profile"]["response_grace_seconds"] == 30.0
    else:
        assert "agentic_profile" not in runner.spec.workload
    assert json.loads(capsys.readouterr().out)["completed_requests"] == 1


def test_time_source_tables_are_bounded_but_json_keeps_complete_evidence():
    from copy import deepcopy

    from jsonschema import validate

    from aisimulate.detail import build_prediction_details, format_prediction_details

    operations = [
        {
            "name": f"op{i}",
            "latency_ms": float(i),
            "source": "empirical",
            "fallbacks": [],
            "sol": None,
            "sol_unavailable_reason": "unsupported test operation",
            "latency_to_sol_ratio": None,
        }
        for i in range(20)
    ]
    native = {
        "summary": {"mean_ttft_ms": 8},
        "performance_diagnostics": {
            "status": "available",
            "scope": "accumulated_active_forward_pass_per_gpu",
            "latency_unit": "ms",
            "phases": [
                {
                    "name": "prefill",
                    "latency_ms": 190,
                    "sol": None,
                    "sol_unavailable_reason": "unsupported test operation",
                    "operations": operations,
                }
            ],
        },
    }
    original = deepcopy(native)
    details = build_prediction_details(native, ("time", "source"))
    validate(details, _detail_schema())
    text = format_prediction_details(details, energy_top_n=2)
    assert "op19:" in text and "op0:" not in text
    assert "18 more operations in JSON" in text
    assert len(details["sections"]["time"]["diagnostics"]["phases"][0]["operations"]) == 20
    assert len(details["sections"]["source"]["phases"][0]["operations"]) == 20
    assert native == original


@pytest.fixture
def performance_record():
    return {
        "status": "available",
        "scope": "accumulated_active_forward_pass_per_gpu",
        "latency_unit": "ms",
        "phases": [
            {
                "name": "prefill",
                "latency_ms": 2.0,
                "sol": None,
                "sol_unavailable_reason": "unsupported",
                "operations": [
                    {
                        "name": "dispatch",
                        "latency_ms": 2.0,
                        "source": "estimated",
                        "sol": None,
                        "sol_unavailable_reason": "unsupported",
                        "latency_to_sol_ratio": None,
                        "fallbacks": [
                            {
                                "inference_phase": "context",
                                "comm_backend": "deepep_ll",
                                "requested_ep_size": 16,
                                "requested_node_num": 2,
                                "measurement_ep_size": 8,
                                "measurement_node_num": 1,
                            }
                        ],
                    }
                ],
            }
        ],
    }


@pytest.mark.parametrize(
    "path,value",
    [
        (("status",), "invented"),
        (("scope",), "seconds"),
        (("latency_unit",), "s"),
        (("phases",), {}),
        (("phases", 0), None),
        (("phases", 0, "operations"), "invalid"),
        (("phases", 0, "latency_ms"), float("inf")),
        (("phases", 0, "latency_ms"), float("nan")),
        (("phases", 0, "operations", 0, "name"), ""),
        (("phases", 0, "operations", 0, "source"), 1),
        (("phases", 0, "operations", 0, "latency_ms"), True),
        (("phases", 0, "operations", 0, "sol"), {}),
        (("phases", 0, "operations", 0, "sol_unavailable_reason"), None),
        (("phases", 0, "operations", 0, "latency_to_sol_ratio"), -1),
        (("phases", 0, "operations", 0, "fallbacks"), "bad"),
        (("phases", 0, "operations", 0, "fallbacks", 0, "requested_ep_size"), 0),
        (("phases", 0, "operations", 0, "fallbacks", 0, "measurement_node_num"), True),
    ],
)
def test_performance_detail_rejects_malformed_nested_records(performance_record, path, value):
    from aisimulate.detail import build_prediction_details

    record = performance_record
    for key in path[:-1]:
        record = record[key]
    record[path[-1]] = value
    expected_path = "performance_diagnostics" + "".join(
        f"[{key}]" if isinstance(key, int) else f".{key}" for key in path
    )
    with pytest.raises(ValueError) as error:
        build_prediction_details({"performance_diagnostics": performance_record}, ("source",))
    assert expected_path in str(error.value)


def test_performance_detail_rejects_missing_fields_and_preserves_valid_records(performance_record):
    from copy import deepcopy

    from jsonschema import validate

    from aisimulate.detail import performance_diagnostics

    for path in [(), ("phases", 0), ("phases", 0, "operations", 0), ("phases", 0, "operations", 0, "fallbacks", 0)]:
        record = performance_record
        for key in path:
            record = record[key]
        for field in record:
            invalid = deepcopy(performance_record)
            target = invalid
            for key in path:
                target = target[key]
            del target[field]
            with pytest.raises(ValueError, match=field):
                performance_diagnostics({"performance_diagnostics": invalid})
    result = performance_diagnostics({"performance_diagnostics": performance_record})
    schema = _detail_schema()
    validate(result, {"$defs": schema["$defs"], "$ref": "#/$defs/performanceDiagnostics"})
    assert result == performance_record
    result["phases"][0]["operations"].clear()
    assert performance_record["phases"][0]["operations"]


@pytest.mark.parametrize(
    "path",
    [
        ("unavailable_reason",),
        ("phases", 0, "name"),
        ("phases", 0, "sol_unavailable_reason"),
        ("phases", 0, "operations", 0, "name"),
        ("phases", 0, "operations", 0, "source"),
        ("phases", 0, "operations", 0, "sol_unavailable_reason"),
        ("phases", 0, "operations", 0, "fallbacks", 0, "comm_backend"),
    ],
)
def test_diagnostic_string_contract_rejects_whitespace_in_schema_and_adapter(performance_record, path):
    from jsonschema import Draft202012Validator

    from aisimulate.detail import build_prediction_details, performance_diagnostics

    if path == ("unavailable_reason",):
        performance_record.update(status="unavailable", phases=[], unavailable_reason="unsupported")
    details = build_prediction_details({"performance_diagnostics": performance_record}, ("time", "source"))
    record = details["sections"]["time"]["diagnostics"]
    target = record
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = " \t\n"
    validator = Draft202012Validator(_detail_schema())
    assert not validator.is_valid(details)
    with pytest.raises(ValueError, match="nonempty string"):
        performance_diagnostics({"performance_diagnostics": record})

    # Source uses separate phase/operation definitions, and omits SOL fields.
    if path[-1] != "sol_unavailable_reason":
        del details["sections"]["time"]
        target = details["sections"]["source"]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = " \t\n"
        assert not validator.is_valid(details)


@pytest.mark.parametrize(
    "sol,reason,ratio,valid",
    [
        (None, None, None, False),
        (None, "  ", None, False),
        (None, "unsupported", 1.0, False),
        ({"latency_ms": 1, "math_ms": 0, "memory_ms": 1}, "unsupported", 2.0, False),
        ({"latency_ms": 1, "math_ms": 0, "memory_ms": 1}, None, 2.0, True),
        ({"latency_ms": 0, "math_ms": 0, "memory_ms": 0}, None, None, True),
        ({"latency_ms": 0, "math_ms": 0, "memory_ms": 0}, None, 2.0, False),
    ],
)
def test_performance_sol_states_match_schema(performance_record, sol, reason, ratio, valid):
    from jsonschema import Draft202012Validator

    from aisimulate.detail import performance_diagnostics

    operation = performance_record["phases"][0]["operations"][0]
    operation.update(sol=sol, sol_unavailable_reason=reason, latency_to_sol_ratio=ratio)
    schema = _detail_schema()
    validator = Draft202012Validator({"$defs": schema["$defs"], "$ref": "#/$defs/performanceDiagnostics"})
    assert validator.is_valid(performance_record) is valid
    if valid:
        assert performance_diagnostics({"performance_diagnostics": performance_record}) == performance_record
    else:
        with pytest.raises(ValueError, match="performance_diagnostics.phases"):
            performance_diagnostics({"performance_diagnostics": performance_record})


def test_cli_reports_invalid_adapter_diagnostic_field(tmp_path, monkeypatch, capsys):
    class InvalidDiagnosticsRunner(_Runner):
        def run(self, *args, **kwargs):
            report = super().run(*args, **kwargs)
            report.metadata["native_report"]["performance_diagnostics"] = {"status": "available"}
            return report

    monkeypatch.setattr(cli, "resolve_runner_factory", lambda _: _Factory(InvalidDiagnosticsRunner()))
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "predict",
                "-c",
                str(_detail_config(tmp_path)),
                "--detail",
                "source",
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
    assert error.value.code == 2
    assert "performance_diagnostics.latency_unit must be present" in capsys.readouterr().err
