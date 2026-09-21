# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native VL replay: image workloads encoded on a host-aware SGLang worker."""

import dataclasses
import json
from copy import deepcopy

import pytest
import yaml
from pydantic import ValidationError

from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config.cli import CorePredictionConfig, CoreRecommendationConfig
from aisimulate.config.vl import validate_vl_prediction_mapping
from aisimulate.main import main
from aisimulate.recommend import recommendation_to_sweeper
from aisimulate.runner import EngineReplayRunnerFactory, InvalidRunnerError
from aisimulate.sweeper import SweepResult


def _prediction():
    return {
        "traffic": {
            "source": {
                "type": "synthetic",
                "input_tokens": 128,
                "output_tokens": 4,
                "images": {
                    "height": 448,
                    "width": 448,
                    "count": 1,
                    "identity": {"pool": 2},
                },
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
            "mode": "aggregated",
            "workers": {
                "aggregated": {
                    "parallelism": {"replicas": 1, "tensor": 1},
                    "scheduler": {"max_batched_tokens": 8192, "max_sequences": 8},
                    "kv_cache": {"capacity": {"type": "fixed", "blocks": 4096}},
                    "host_loop": True,
                    "frontend": {
                        "stages": [
                            {
                                "resource": "pool",
                                "workers": 1,
                                "cost": {"const_ms": 3.0},
                            }
                        ],
                    },
                }
            },
        },
    }


def test_native_vl_lowering_targets_the_host_aware_sglang_rank():
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(_prediction()))
    args = spec.backend_deployment.agg_engine_args
    assert args["vision"] is True
    assert args["sglang"]["chunked_prefill_size"] == 8192
    assert args["sglang"]["vlm_cache_bytes"] == 100 << 20
    assert args["sglang"]["host_loop"] is True
    assert args["frontend"]["stages"][0]["workers"] == 1
    assert spec.workload["images"]["identity"] == {"pool": 2}
    assert (
        spec.workload["isl"] == 128
    )  # placeholders are laid out by the workload driver


def test_native_vl_predict_reports_ttft_milestones(tmp_path, capsys):
    path = tmp_path / "predict.yaml"
    path.write_text(yaml.safe_dump(_prediction()))
    output = tmp_path / "out"
    argv = [
        "predict",
        "-c",
        str(path),
        "--output-dir",
        str(output),
        "--format",
        "json",
        "--capture-per-request",
    ]
    assert main(argv) == 0
    stdout = json.loads(capsys.readouterr().out)
    report = json.loads((output / "prediction.json").read_text())
    assert report["completed_requests"] == 4
    # 448x448 -> 196 visual tokens per image on top of the 128 text tokens.
    assert report["total_input_tokens"] == 4 * (128 + 196)
    for key in ("mean_frontend_ms", "mean_prefill_elapsed_ms"):
        assert stdout[key] == report[key] > 0.0
    # The scheduler thread is free: a received request is selected in the same instant.
    assert report["mean_receive_to_admit_ms"] == 0.0
    assert (
        report["mean_ttft_ms"]
        >= report["mean_frontend_ms"] + report["mean_prefill_elapsed_ms"]
    )
    records = [
        json.loads(line)
        for line in (output / "requests.jsonl").read_text().splitlines()
    ]
    # The two initial arrivals share one pool worker: 3 ms and 6 ms in the frontend.
    delays = sorted(
        record["frontend_ready_ms"] - record["arrival_time_ms"]
        for record in records
        if record["arrival_time_ms"] == 0.0
    )
    assert delays == pytest.approx([3.0, 6.0])
    for record in records:
        assert (
            record["arrival_time_ms"]
            <= record["frontend_ready_ms"]
            <= record["scheduler_received_ms"]
            <= record["selected_ms"]
            < record["prefill_complete_ms"]
            <= record["first_token_ms"]
        )


@pytest.mark.parametrize(
    "kind",
    ["no_host", "encoder_and_host", "frontend_without_host", "vllm", "fixed_timing"],
)
def test_native_vl_schema_rejects_unsupported(kind):
    raw = _prediction()
    worker = raw["engine"]["workers"]["aggregated"]
    if kind == "no_host":
        del worker["host_loop"]
        del worker["frontend"]
    elif kind == "encoder_and_host":
        raw["engine"]["workers"]["encoder"] = {
            "tensor": 1,
            "batch_size": 1,
            "replicas": 1,
        }
    elif kind == "frontend_without_host":
        worker["host_loop"] = False
    elif kind == "vllm":
        raw["engine"]["backend"] = "vllm"
        del worker["frontend"]
    else:
        worker["timing"] = {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0}
    with pytest.raises(ValidationError):
        CorePredictionConfig.model_validate(deepcopy(raw))


def _host_table(**row_overrides):
    worker = _prediction()["engine"]["workers"]["aggregated"]
    row = {
        "identity": {
            "cpu": "example-cpu",
            "sglang_revision": "0bcd822377da7b5718e674eaf9c870d349424dd1",
            "model": "Qwen/Qwen3-VL-8B-Instruct",
            "frontend": "python",
        },
        "shape": {"height": 448, "width": 448, "count": 1, "encoding": "png", "text_tokens": 128},
        "stages": worker["frontend"]["stages"],
        "provenance": {"sampled_on": "example-host"},
    }
    row.update(row_overrides)
    return {"schema_version": 3, "rows": [row]}


def test_host_profile_lowers_to_the_explicit_stages(tmp_path):
    path = tmp_path / "table.json"
    path.write_text(json.dumps(_host_table()))
    explicit = prediction_to_replay_spec(
        CorePredictionConfig.model_validate(_prediction())
    )
    raw = _prediction()
    worker = raw["engine"]["workers"]["aggregated"]
    del worker["host_loop"], worker["frontend"]
    worker["host_profile"] = {"path": str(path), "frontend": "python"}
    profiled = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    assert (
        profiled.backend_deployment.agg_engine_args
        == explicit.backend_deployment.agg_engine_args
    )
    vl = profiled.backend_deployment.performance_model_metadata["aggregated"]["vl"]
    assert vl["frontend"] == "python" and len(vl["host_profile_digest"]) == 16


@pytest.mark.parametrize("kind", ["frontend", "encoding", "text_tokens", "revision"])
def test_host_profile_misses_fail_closed_with_the_collect_command(tmp_path, kind):
    raw = _prediction()
    worker = raw["engine"]["workers"]["aggregated"]
    del worker["host_loop"], worker["frontend"]
    table = _host_table()
    frontend = "python"
    if kind == "frontend":
        frontend = "rust"
    elif kind == "encoding":
        raw["traffic"]["source"]["images"]["encoding"] = "jpeg"
    elif kind == "text_tokens":
        raw["traffic"]["source"]["input_tokens"] = 256
    else:
        table["rows"][0]["identity"]["sglang_revision"] = "0" * 40
    path = tmp_path / "table.json"
    path.write_text(json.dumps(table))
    worker["host_profile"] = {"path": str(path), "frontend": frontend}
    with pytest.raises(ValueError) as error:
        prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    message = str(error.value)
    assert "has no row for" in message
    # The failure names the exact measurement that would add the row.
    assert "-m aisimulate.vl.collect" in message and f"--frontend {frontend}" in message
    expected_shape = "448x448x1 --encoding " + ("jpeg" if kind == "encoding" else "png")
    assert f"--images {expected_shape}" in message
    assert f"--text-tokens {256 if kind == 'text_tokens' else 128}" in message


def _recommendation():
    raw = _prediction()
    worker = raw["engine"]["workers"]["aggregated"]
    worker["parallelism"] = {
        "preset": False,
        "replicas": 1,
        "tensor": 1,
        "pipeline": 1,
        "attention_data": 1,
        "moe_tensor": 1,
        "moe_expert": 1,
    }
    worker["scheduler"]["max_batched_tokens"] = {"choices": [4096, 8192]}
    raw["optimization"] = {
        "target": "throughput_per_gpu",
        "constraints": {"max_candidate_gpus": 2},
    }
    raw["optimizer"] = {
        "algorithm": "random",
        "max_trials": 2,
        "parallelism": 1,
        "candidate_timeout_seconds": 60.0,
        "seed": 13,
    }
    return raw


def test_native_vl_recommend_yaml_predict_roundtrip(tmp_path, capsys):
    path = tmp_path / "search.yaml"
    path.write_text(yaml.safe_dump(_recommendation()))
    root = tmp_path / "recommend"
    assert (
        main(
            [
                "recommend",
                "-c",
                str(path),
                "--output-dir",
                str(root),
                "--format",
                "json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    result = SweepResult.from_json((root / "recommendation.json").read_text())
    candidate = result.selected_candidates[0]
    assert candidate.config["prediction_config_supported"] is True
    assert candidate.config["agg_host_loop"] is True
    assert candidate.config["agg_frontend"]["stages"][0]["cost"]["const_ms"] == 3.0

    saved = root / "recommendations" / "0001.yaml"
    concrete = CorePredictionConfig.from_yaml(saved)
    assert concrete.engine.workers.aggregated.host_loop is True
    assert concrete.engine.workers.aggregated.frontend.stages[0].cost.const_ms == 3.0
    assert concrete.engine.workers.aggregated.vision.cache_mib == 100
    output = tmp_path / "predict"
    assert (
        main(
            [
                "predict",
                "-c",
                str(saved),
                "--output-dir",
                str(output),
                "--format",
                "json",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    for key in ("mean_ttft_ms", "mean_e2e_latency_ms", "output_throughput_tok_s"):
        assert report[key] == pytest.approx(candidate.metrics[key])

    spec = prediction_to_replay_spec(concrete)
    raw = concrete.model_dump(mode="python", exclude_none=True)
    validate_vl_prediction_mapping(raw, spec)
    raw["engine"]["workers"]["aggregated"]["frontend"]["stages"][0]["cost"]["const_ms"] = 1.0
    with pytest.raises(ValueError, match="prediction-ready"):
        validate_vl_prediction_mapping(raw, spec)


def test_profile_backed_recommendation_pins_the_resolved_stages(tmp_path, capsys):
    path = tmp_path / "table.json"
    path.write_text(json.dumps(_host_table()))
    raw = _recommendation()
    worker = raw["engine"]["workers"]["aggregated"]
    del worker["host_loop"]
    explicit_frontend = worker.pop("frontend")
    worker["host_profile"] = {"path": str(path), "frontend": "python"}
    worker["parallelism"]["tensor"] = 2
    lowered = recommendation_to_sweeper(CoreRecommendationConfig.model_validate(raw))
    space = lowered.search_space
    # Candidates score the stages resolved from the table, identified by content;
    # the mutable file path is not part of what a candidate carries or exports.
    assert space.agg_host_loop is True
    assert (
        space.agg_frontend["stages"][0]["cost"]["const_ms"]
        == explicit_frontend["stages"][0]["cost"]["const_ms"]
    )
    assert len(space.agg_host_profile_digest) == 16
    assert not hasattr(space, "agg_host_profile")

    search = tmp_path / "search.yaml"
    search.write_text(yaml.safe_dump(raw))
    root = tmp_path / "recommend"
    assert (
        main(
            [
                "recommend",
                "-c",
                str(search),
                "--output-dir",
                str(root),
                "--format",
                "json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    result = SweepResult.from_json((root / "recommendation.json").read_text())
    assert result.selected_candidates
    for candidate in result.selected_candidates:
        assert candidate.config["prediction_config_supported"] is True
        assert candidate.config["tp"] == 2
    saved = CorePredictionConfig.from_yaml(root / "recommendations" / "0001.yaml")
    assert saved.engine.workers.aggregated.frontend.stages[0].cost.const_ms == 3.0
    assert saved.engine.workers.aggregated.host_profile is None


def test_text_only_host_tables_reach_predict_and_recommend_alike(tmp_path, capsys):
    raw = _prediction()
    del raw["traffic"]["source"]["images"]
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    args = spec.backend_deployment.agg_engine_args
    assert args["sglang"]["host_loop"] is True
    assert "vision" not in args and "images" not in spec.workload

    search = _recommendation()
    del search["traffic"]["source"]["images"]
    space = recommendation_to_sweeper(
        CoreRecommendationConfig.model_validate(search)
    ).search_space
    assert space.agg_host_loop is True
    assert space.agg_frontend == args["frontend"]
    assert space.agg_vision is None

    # The scored candidates validate and save without an image workload.
    path = tmp_path / "search.yaml"
    path.write_text(yaml.safe_dump(search))
    root = tmp_path / "recommend"
    assert (
        main(
            [
                "recommend",
                "-c",
                str(path),
                "--output-dir",
                str(root),
                "--format",
                "json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    result = SweepResult.from_json((root / "recommendation.json").read_text())
    assert result.selected_candidates
    assert all(
        c.config["prediction_config_supported"] is True
        for c in result.selected_candidates
    )
    saved = CorePredictionConfig.from_yaml(root / "recommendations" / "0001.yaml")
    assert saved.engine.workers.aggregated.host_loop is True
    assert saved.traffic.source.images is None


def test_materialized_request_lists_reject_image_workloads():
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(_prediction()))
    legacy = dataclasses.replace(
        spec,
        workload={
            key: value
            for key, value in spec.workload.items()
            if key not in ("source_type", "load_type")
        },
    )
    with pytest.raises(InvalidRunnerError, match="workload-driver"):
        EngineReplayRunnerFactory().create(0).run(legacy)
