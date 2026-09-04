# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public predict/recommend coverage for analytical AFD."""

from __future__ import annotations

import json

import pytest
import yaml
from pydantic import ValidationError

import aisimulate.main as cli
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig, CoreRecommendationConfig
from aisimulate.recommend import run_recommendation
from aisimulate.runner import AFDCompanionTiming, EngineReplayRunnerFactory
from aisimulate.sweeper import AFDLayerTimes


class _AFDPerformanceModel:
    def __init__(self) -> None:
        self.requests = []

    def measure(self, request):
        self.requests.append(request)
        phases = (
            ("prefill", "decode")
            if request.topology.phase.value == "both"
            else (request.topology.phase.value,)
        )
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
    assert (
        deployment.parallel_config["afd_provenance"]["gpu_accounting"]["total_gpus"]
        == 16
    )
    assert deployment.performance_model_metadata["afd"]["measurement_required"] is False
    assert {
        item["phase"]
        for item in deployment.performance_model_metadata["afd"]["measurements"]
    } == {
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
def test_afd_recommendation_rejects_ambiguous_or_incomplete_contract(
    afd, match
) -> None:
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


def test_public_afd_predict_cli_writes_summary_and_per_request(
    tmp_path, monkeypatch, capsys
) -> None:
    config_path = tmp_path / "afd-prediction.yaml"
    config_path.write_text(yaml.safe_dump(_pure_prediction()))
    output = tmp_path / "out"
    performance_model = _AFDPerformanceModel()
    compile_prediction = prediction_to_replay_spec

    monkeypatch.setattr(
        cli, "resolve_runner_factory", lambda stack: EngineReplayRunnerFactory()
    )
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
        runner_factory=EngineReplayRunnerFactory(
            afd_companion_model=_CompanionPerformanceModel()
        ),
        afd_performance_model=_AFDPerformanceModel(),
        show_progress=False,
    )

    assert result.counts.feasible == 1
    selected = result.selected_candidates[0]
    assert selected.prediction_config["engine"]["mode"] == "afd"
    assert selected.prediction_config["engine"]["afd"]["combined_with_pd"] is True
    assert set(selected.prediction_config["engine"]["workers"]) == {"prefill"}
    CorePredictionConfig.model_validate(selected.prediction_config)
