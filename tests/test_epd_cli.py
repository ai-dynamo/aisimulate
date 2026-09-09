# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified EPD contracts, including real native recommend -> YAML -> predict replay."""

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config.cli import CorePredictionConfig, CoreRecommendationConfig
from aisimulate.config.epd import encoder_prediction_fields, validate_epd_prediction_mapping
from aisimulate.main import main
from aisimulate.recommend import recommendation_to_sweeper
from aisimulate.sweeper import SweepResult


def _prediction(mode="aggregated"):
    worker = {
        "parallelism": {"replicas": 1, "tensor": 1},
        "scheduler": {"max_batched_tokens": 8192, "max_sequences": 8},
        "kv_cache": {"capacity": {"type": "fixed", "blocks": 4096}},
    }
    roles = ["aggregated"] if mode == "aggregated" else ["prefill", "decode"]
    return {
        "traffic": {
            "source": {
                "type": "synthetic",
                "input_tokens": 128,
                "output_tokens": 4,
                "images": {"height": 448, "width": 448, "count": 1},
            },
            "load": {"type": "concurrency", "concurrency": 2},
            "stop": {"requests": 4},
        },
        "engine": {
            "model": "Qwen/Qwen3-VL-8B-Instruct",
            "hardware": "h200_sxm",
            "backend": "sglang",
            "backend_version": "0.5.14",
            "context_length": 4096,
            "mode": mode,
            "workers": {
                **{role: deepcopy(worker) for role in roles},
                "encoder": {
                    "tensor": 1,
                    "batch_size": 2,
                    "replicas": 2,
                    "latency_correction": 1.25,
                    "rate_degradation": 0.8,
                },
            },
        },
    }


def _recommendation(mode="aggregated"):
    raw = _prediction(mode)
    for role, worker in raw["engine"]["workers"].items():
        if role != "encoder":
            worker["parallelism"] = {
                "preset": False,
                "replicas": 1,
                "tensor": 1,
                "pipeline": 1,
                "attention_data": 1,
                "moe_tensor": 1,
                "moe_expert": 1,
            }
    raw["engine"]["workers"]["encoder"]["replicas"] = {"choices": [1, 2]}
    raw["optimization"] = {"target": "throughput_per_gpu", "constraints": {"max_candidate_gpus": 4}}
    raw["optimizer"] = {
        "algorithm": "random",
        "max_trials": 2,
        "parallelism": 1,
        "candidate_timeout_seconds": 60.0,
        "seed": 13,
    }
    return raw


@pytest.mark.parametrize("mode", ["aggregated", "disaggregated"])
@pytest.mark.parametrize("relative_stop", [False, True])
def test_native_cli_epd_recommend_yaml_predict(tmp_path, capsys, mode, relative_stop):
    raw = _recommendation(mode)
    if relative_stop:
        raw["traffic"]["stop"] = {"requests_per_load_unit": 2.0}
    # Exercise strict aggregate SLA and retention in the selected prediction.
    raw["evaluation"] = {"sla": {"ttft_ms": 10000.0}}
    raw["optimization"]["strict_sla"] = True
    path = tmp_path / "search.yaml"
    path.write_text(yaml.safe_dump(raw))
    root = tmp_path / "recommend"
    assert main(["recommend", "-c", str(path), "--output-dir", str(root), "--format", "json"]) == 0
    capsys.readouterr()
    result = SweepResult.from_json((root / "recommendation.json").read_text())
    assert result.selected_candidates
    candidate = result.selected_candidates[0]
    saved = root / "recommendations" / "0001.yaml"
    concrete = CorePredictionConfig.from_yaml(saved)
    spec = prediction_to_replay_spec(concrete)
    encoder = spec.backend_deployment.encoder
    assert asdict(encoder) == candidate.config["encoder"]
    assert concrete.engine.workers.encoder.model_dump() == encoder_prediction_fields(encoder)
    assert candidate.used_gpus == (1 if mode == "aggregated" else 2) + encoder.total_gpus
    assert candidate.config["prediction_config_supported"] is True
    assert candidate.config["deployment_artifact_generation_supported"] is False
    assert spec.workload["isl"] == 128  # visual context added only by the runner
    output = tmp_path / "predict"
    assert main(["predict", "-c", str(saved), "--output-dir", str(output), "--format", "json"]) == 0
    stdout = json.loads(capsys.readouterr().out)
    report = json.loads((output / "prediction.json").read_text())
    assert stdout["metric_semantics"] == "analytical_epd_overlay"
    assert report["metadata"]["encoder"] == candidate.config["encoder"]
    assert report["metadata"]["total_gpus"] == candidate.used_gpus
    assert report["metadata"]["aggregate_sla_bounds"]["ttft_ms"] == 10000
    for key in (
        "mean_ttft_ms",
        "mean_tpot_ms",
        "mean_e2e_latency_ms",
        "output_throughput_tok_s",
        "duration_ms",
        "gpu_hours",
    ):
        assert report["summary"][key] == pytest.approx(candidate.metrics[key])
    assert report["summary"]["completed_requests"] == 4
    assert "per_request" not in report
    assert not any(key.startswith(("goodput", "p99")) for key in report["summary"])
    table_output = tmp_path / "table"
    assert main(["predict", "-c", str(saved), "--output-dir", str(table_output)]) == 0
    assert "aggregate estimates" in capsys.readouterr().out


@pytest.mark.parametrize(
    "kind", ["missing_encoder", "missing_images", "trace", "rate", "load_search", "fpm", "fixed", "startup"]
)
@pytest.mark.parametrize("recommend", [False, True])
def test_epd_public_schema_rejects_unsupported(kind, recommend):
    raw = _recommendation() if recommend else _prediction()
    if kind == "missing_encoder":
        del raw["engine"]["workers"]["encoder"]
    elif kind == "missing_images":
        del raw["traffic"]["source"]["images"]
    elif kind == "trace":
        raw["traffic"] = {
            "source": {"type": "trace", "paths": ["unused.jsonl"]},
            "load": {"type": "concurrency", "concurrency": 2},
        }
    elif kind == "rate":
        raw["traffic"]["load"] = {"type": "poisson", "requests_per_second": 2.0}
    elif kind == "load_search":
        raw["traffic"]["load"]["concurrency"] = {"choices": [1, 2]}
    else:
        worker = raw["engine"]["workers"]["aggregated"]
        worker.update(
            {"timing": {"forward_model": "fpm"}}
            if kind == "fpm"
            else {"timing": {"type": "fixed", "prefill_ms": 1, "decode_ms": 1}}
            if kind == "fixed"
            else {"startup_seconds": 1}
        )
    with pytest.raises(ValueError):
        (CoreRecommendationConfig if recommend else CorePredictionConfig).model_validate(raw)


@pytest.mark.parametrize(
    "field,value",
    [
        ("tensor", True),
        ("replicas", 0),
        ("batch_size", 9),
        ("latency_correction", float("nan")),
        ("rate_degradation", 1.1),
    ],
)
def test_encoder_schema_negative(field, value):
    raw = _recommendation()
    raw["engine"]["workers"]["encoder"][field] = value
    with pytest.raises(ValueError):
        CoreRecommendationConfig.model_validate(raw)


def test_recommendation_domains_lower_without_loss():
    raw = _recommendation()
    raw["engine"]["workers"]["encoder"].update(
        hardware="h100_sxm",
        backend_version="pinned",
        tensor={"choices": [1, 2]},
        batch_size={"choices": [1, 8]},
    )
    lowered = recommendation_to_sweeper(CoreRecommendationConfig.model_validate(raw))
    assert lowered.search_space.encoder.model_dump() == {
        "hardware_sku": "h100_sxm",
        "backend_version": "pinned",
        "tp": [1, 2],
        "workers": [1, 2],
        "batch_size": [1, 8],
        "latency_correction": 1.25,
        "rate_degradation": 0.8,
    }
    assert lowered.workload.source_type == "synthetic"
    assert lowered.workload.images.height == 448


@pytest.mark.parametrize("flag", ["--capture-per-request", "--online"])
def test_cli_rejects_unsupported_outputs_before_touching_output(tmp_path, capsys, flag):
    path = tmp_path / "prediction.yaml"
    path.write_text(yaml.safe_dump(_prediction()))
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "prediction.json"
    sentinel.write_text("keep me")
    with pytest.raises(SystemExit, match="2"):
        main(["predict", "-c", str(path), "--output-dir", str(output), "--overwrite", flag])
    assert "analytical EPD requires offline" in capsys.readouterr().err
    assert sentinel.read_text() == "keep me"


def test_context_limit_includes_visual_tokens():
    raw = _prediction()
    raw["engine"]["context_length"] = 140
    with pytest.raises(ValueError, match="visual"):
        prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))


@pytest.mark.parametrize(
    "change", ["image", "model", "encoder", "version", "correction", "text", "count", "concurrency", "topology"]
)
def test_prediction_callback_cannot_drop_or_change_epd(change):
    raw = _prediction()
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    raw["engine"]["workers"]["encoder"] = encoder_prediction_fields(spec.backend_deployment.encoder)
    validate_epd_prediction_mapping(raw, spec)
    if change == "image":
        raw["traffic"]["source"]["images"]["count"] = 2
    elif change == "model":
        raw["engine"]["model"] = "other"
    elif change == "encoder":
        del raw["engine"]["workers"]["encoder"]
    elif change == "version":
        raw["engine"]["workers"]["encoder"]["backend_version"] = "latest"
    elif change == "correction":
        raw["engine"]["workers"]["encoder"]["latency_correction"] = 1.0
    elif change == "text":
        raw["traffic"]["source"]["input_tokens"] = 129
    elif change == "count":
        raw["traffic"]["stop"]["requests"] = 5
    elif change == "concurrency":
        raw["traffic"]["load"]["concurrency"] = 3
    else:
        raw["engine"]["workers"]["aggregated"]["parallelism"]["replicas"] = 2
    with pytest.raises(ValueError, match="prediction-ready"):
        validate_epd_prediction_mapping(raw, spec)


def test_cli_examples_parse():
    root = Path(__file__).resolve().parents[1]
    for mode in ("aggregated", "disaggregated"):
        CorePredictionConfig.from_yaml(root / f"examples/cli/epd-predict-{mode}.yaml")
    CoreRecommendationConfig.from_yaml(root / "examples/cli/epd-recommend.yaml")
