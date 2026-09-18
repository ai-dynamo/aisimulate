# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public predict/recommend coverage for analytical AFD."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError

import aisimulate.main as cli
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig, CoreRecommendationConfig
from aisimulate.recommend import run_recommendation
from aisimulate.runner import AFDCompanionTiming, AICAFDCompanionPerformanceModel, EngineReplayRunnerFactory
from aisimulate.sweeper import AFDLayerTimes


class _AFDPerformanceModel:
    def __init__(self) -> None:
        self.requests = []

    def measure(self, request):
        self.requests.append(request)
        phases = ("prefill", "decode") if request.topology.phase.value == "both" else (request.topology.phase.value,)
        return tuple(
            AFDLayerTimes(
                phase=phase,
                attention_ms=1.0,
                ffn_ms=2.0,
                a_to_f_ms=0.1,
                f_to_a_ms=0.1,
                num_layers=2,
                provenance={"provider": "test"},
            )
            for phase in phases
        )


class _CompanionPerformanceModel:
    def measure(self, spec):
        del spec
        return AFDCompanionTiming(
            phase="prefill",
            latency_ms=2.0,
            batch_capacity_per_worker=8,
            workers=1,
            provenance={"provider": "test"},
        )


def _traffic() -> dict:
    return {
        "source": {"type": "synthetic", "input_tokens": 8, "output_tokens": 3},
        "load": {"type": "concurrency", "concurrency": 2},
        "stop": {"requests": 4},
    }


def _pure_prediction() -> dict:
    return {
        "traffic": _traffic(),
        "engine": {
            "mode": "afd",
            "model": "Qwen/Qwen3-32B",
            "hardware": "h200_sxm",
            "backend": "trtllm",
            "afd": {
                "phase": "both",
                "combined_with_pd": False,
                "n_a_nodes": 1,
                "n_f_nodes": 1,
                "tp_a": 8,
                "a_batch_size": 8,
            },
        },
    }


@pytest.mark.parametrize("enabled", [True, False])
def test_afd_rejects_unsupported_chunked_prefill_control(enabled) -> None:
    raw = _pure_prediction()
    raw["engine"]["enable_chunked_prefill"] = enabled
    with pytest.raises(ValidationError, match="enable_chunked_prefill is unsupported for AFD"):
        CorePredictionConfig.model_validate(raw)


def test_afd_prediction_lowers_to_measured_replay_contract() -> None:
    performance_model = _AFDPerformanceModel()
    config = CorePredictionConfig.model_validate(_pure_prediction())

    spec = prediction_to_replay_spec(
        config,
        afd_performance_model=performance_model,
    )

    deployment = spec.backend_deployment
    assert deployment.deployment_mode == "afd"
    assert deployment.backend_version
    assert deployment.parallel_config["afd"]["gpus_per_node"] == 8
    assert deployment.parallel_config["afd_provenance"]["gpu_accounting"]["total_gpus"] == 16
    assert deployment.performance_model_metadata["afd"]["measurement_required"] is False
    assert {item["phase"] for item in deployment.performance_model_metadata["afd"]["measurements"]} == {
        "prefill",
        "decode",
    }
    assert len(performance_model.requests) == 1


def test_afd_plus_pd_prediction_requires_and_materializes_opposite_worker() -> None:
    raw = _pure_prediction()
    raw["engine"]["afd"].update(phase="decode", combined_with_pd=True)
    raw["engine"]["workers"] = {
        "prefill": {
            "parallelism": {"replicas": 1, "tensor": 2},
            "timing": {"type": "fixed", "prefill_ms": 2.0, "decode_ms": 2.0},
        }
    }
    spec = prediction_to_replay_spec(
        CorePredictionConfig.model_validate(raw),
        afd_performance_model=_AFDPerformanceModel(),
    )

    deployment = spec.backend_deployment
    assert deployment.deployment_mode == "afd+pd"
    assert deployment.num_prefill_workers == 1
    assert deployment.parallel_config["prefill_tp"] == 2
    assert deployment.prefill_engine_args["timing_model"]["type"] == "fixed"
    assert deployment.parallel_config["afd_provenance"]["gpu_accounting"] == {
        "attention_gpus": 8,
        "ffn_gpus": 8,
        "companion_gpus": 2,
        "total_gpus": 18,
    }


@pytest.mark.parametrize(("phase", "companion_role"), [("decode", "prefill"), ("prefill", "decode")])
@pytest.mark.parametrize("forward_model", ["fpm", "op_level", None])
def test_public_afd_companion_forward_model_reaches_estimator(phase, companion_role, forward_model) -> None:
    raw = _pure_prediction()
    raw["engine"]["afd"].update(phase=phase, combined_with_pd=True)
    timing = {"type": "default"}
    if forward_model is not None:
        timing["forward_model"] = forward_model
    raw["engine"]["workers"] = {
        companion_role: {"parallelism": {"tensor": 2}, "timing": timing},
    }
    spec = prediction_to_replay_spec(
        CorePredictionConfig.model_validate(raw),
        afd_performance_model=_AFDPerformanceModel(),
    )
    calls = []

    def estimator(model, hardware, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(raw={"ttft": 2.0, "tpot": 2.0})

    report = (
        EngineReplayRunnerFactory(afd_companion_model=AICAFDCompanionPerformanceModel(estimator)).create(0).run(spec)
    )

    expected_model = "op_level" if forward_model is None else forward_model
    assert report.metrics["completed_requests"] == 4
    assert (
        spec.backend_deployment.performance_model_metadata[companion_role]["config"]["forward_model"] == expected_model
    )
    assert len(calls) == 1
    assert calls[0].get("forward_model") == expected_model
    assert report.metadata["afd_replay"]["companion"]["forward_model"] == expected_model


@pytest.mark.parametrize("mode", ["auto", "fpm_regression"])
def test_afd_legacy_fpm_does_not_bypass_estimator_policy_validation(mode):
    raw = _pure_prediction()
    raw["engine"]["afd"].update(phase="decode", combined_with_pd=True)
    raw["engine"]["workers"] = {
        "prefill": {"parallelism": {"tensor": 2}, "timing": {"forward_model": "fpm", "estimation_mode": mode}}
    }
    with pytest.raises(ValueError, match="estimator policies"):
        CorePredictionConfig.model_validate(raw)


@pytest.mark.parametrize("forward_model", ["op_level", "fpm"])
def test_afd_recommendation_preserves_legacy_companion_selection(forward_model):
    from aisimulate.recommend import recommendation_to_sweeper
    from aisimulate.sweeper.config import SmartSearchConfig

    raw = _pure_prediction()
    raw["engine"]["afd"].update(phase="decode", combined_with_pd=True)
    raw["engine"]["afd"].pop("n_a_nodes")
    raw["engine"]["afd"].pop("n_f_nodes")
    raw["engine"]["workers"] = {"prefill": {"timing": {"forward_model": forward_model}}}
    config = CoreRecommendationConfig.model_validate({**raw, "optimization": {}})
    search = recommendation_to_sweeper(config, stack="engine")
    for _ in range(2):
        assert search.search_space.prefill_forward_model == forward_model
        assert search.search_space.role_estimator_controls == {}
        search = SmartSearchConfig.model_validate_json(search.model_dump_json())


@pytest.mark.parametrize(
    ("afd", "match"),
    [
        ({"phase": "decode", "combined_with_pd": True}, "a_batch_size"),
        (
            {"phase": "decode", "combined_with_pd": False, "a_batch_size": 8},
            "pure AFD recommendation requires phase='both'",
        ),
    ],
)
def test_afd_recommendation_rejects_ambiguous_or_incomplete_contract(afd, match) -> None:
    with pytest.raises(ValidationError, match=match):
        CoreRecommendationConfig.model_validate(
            {
                "engine": {
                    "mode": "afd",
                    "model": "Qwen/Qwen3-32B",
                    "hardware": "h200_sxm",
                    "backend": "trtllm",
                    "afd": afd,
                },
                "optimization": {},
            }
        )


def test_afd_rejects_trace_traffic_at_public_boundary() -> None:
    raw = _pure_prediction()
    raw["traffic"] = {
        "source": {"type": "trace", "paths": ["trace.jsonl"], "format": "mooncake"},
        "load": {"type": "trace_timestamps"},
    }
    with pytest.raises(ValidationError, match="fixed-length synthetic request traffic"):
        CorePredictionConfig.model_validate(raw)


def test_public_afd_predict_cli_writes_summary_and_per_request(tmp_path, monkeypatch, capsys) -> None:
    config_path = tmp_path / "afd-prediction.yaml"
    config_path.write_text(yaml.safe_dump(_pure_prediction()))
    output = tmp_path / "out"
    performance_model = _AFDPerformanceModel()
    compile_prediction = prediction_to_replay_spec

    monkeypatch.setattr(cli, "resolve_runner_factory", lambda stack: EngineReplayRunnerFactory())
    monkeypatch.setattr(
        cli,
        "prediction_to_replay_spec",
        lambda config, **kwargs: compile_prediction(
            config,
            afd_performance_model=performance_model,
            **kwargs,
        ),
    )

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
    assert json.loads(capsys.readouterr().out)["completed_requests"] == 4.0
    assert len((output / "requests.jsonl").read_text().splitlines()) == 4
    qualification = json.loads((output / "afd-qualification.json").read_text())
    replay_spec = json.loads((output / "afd-replay-spec.json").read_text())
    assert qualification["qualification"]["status"] == "qualified_for_analytical_replay"
    assert qualification["qualification"]["native_deployment_supported"] is False
    assert replay_spec["backend_deployment"]["deployment_mode"] == "afd"


def test_afd_recommendation_emits_prediction_ready_candidate() -> None:
    config = CoreRecommendationConfig.model_validate(
        {
            "traffic": _traffic(),
            "engine": {
                "mode": "afd",
                "model": "Qwen/Qwen3-32B",
                "hardware": "h200_sxm",
                "backend": "trtllm",
                "afd": {
                    "phase": "decode",
                    "combined_with_pd": True,
                    "a_batch_size": 8,
                    "tp_a": 8,
                    "num_microbatches": 3,
                    "pipeline_model": "optimistic",
                },
            },
            "optimization": {
                "target": "throughput_per_gpu",
                "constraints": {"max_candidate_gpus": 32},
            },
            "optimizer": {"algorithm": "random", "max_trials": 1, "parallelism": 1},
        }
    )
    result = run_recommendation(
        config,
        stack="engine",
        runner_factory=EngineReplayRunnerFactory(afd_companion_model=_CompanionPerformanceModel()),
        afd_performance_model=_AFDPerformanceModel(),
        show_progress=False,
    )

    assert result.counts.feasible == 1
    selected = result.selected_candidates[0]
    assert selected.prediction_config["engine"]["mode"] == "afd"
    assert selected.prediction_config["engine"]["afd"]["combined_with_pd"] is True
    assert set(selected.prediction_config["engine"]["workers"]) == {"prefill"}
    CorePredictionConfig.model_validate(selected.prediction_config)
