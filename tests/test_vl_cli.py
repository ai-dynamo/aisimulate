# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native VL replay: image workloads encoded on a host-aware SGLang worker."""

import json
import subprocess
from copy import deepcopy

import pytest
import yaml
from pydantic import ValidationError

from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config.cli import CorePredictionConfig, CoreRecommendationConfig
from aisimulate.config.vl import validate_vl_prediction_mapping
from aisimulate.main import main
from aisimulate.recommend import recommendation_to_sweeper
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
                    "host": {
                        "receive": {"const_ms": 0.5},
                        "select": {"const_ms": 0.2},
                        "launch_extend": {"const_ms": 5.0},
                        "launch_decode": {"const_ms": 2.0},
                        "result": {"const_ms": 0.3},
                    },
                    "frontend": {
                        "processor_workers": 1,
                        "stages": [
                            {
                                "resource": "processor",
                                "unit": "request",
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
    assert args["sglang"]["host"]["launch_extend"]["const_ms"] == 5.0
    assert args["frontend"]["processor_workers"] == 1
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
    for key in (
        "mean_frontend_ms",
        "mean_scheduler_inbox_wait_ms",
        "mean_receive_to_admit_ms",
        "mean_prefill_elapsed_ms",
    ):
        assert stdout[key] == report[key] > 0.0
    # Two arrivals share one processor worker: 3 ms and 6 ms in the frontend.
    assert report["mean_frontend_ms"] == pytest.approx(4.5)
    assert (
        report["mean_ttft_ms"] >= report["mean_frontend_ms"] + report["mean_prefill_elapsed_ms"]
    )
    for line in (output / "requests.jsonl").read_text().splitlines():
        record = json.loads(line)
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
        del worker["host"]
        del worker["frontend"]
    elif kind == "encoder_and_host":
        raw["engine"]["workers"]["encoder"] = {
            "tensor": 1,
            "batch_size": 1,
            "replicas": 1,
        }
    elif kind == "frontend_without_host":
        del worker["host"]
    elif kind == "vllm":
        raw["engine"]["backend"] = "vllm"
        del worker["frontend"]
    else:
        worker["timing"] = {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0}
    with pytest.raises(ValidationError):
        CorePredictionConfig.model_validate(deepcopy(raw))


def _host_profile(**overrides):
    worker = _prediction()["engine"]["workers"]["aggregated"]
    profile = {
        "schema_version": 1,
        "identity": {
            "sglang_revision": "0bcd822377da7b5718e674eaf9c870d349424dd1",
            "model": "Qwen/Qwen3-VL-8B-Instruct",
            "frontend": "python",
            "image_encoding": "png",
        },
        "host": worker["host"],
        "frontend": worker["frontend"],
        "tp_sync_ms": {"2": 0.4},
        "provenance": {"sampled_on": "example-host"},
    }
    profile.update(overrides)
    return profile


def test_host_profile_lowers_to_the_explicit_tables(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(_host_profile()))
    explicit = prediction_to_replay_spec(
        CorePredictionConfig.model_validate(_prediction())
    )
    raw = _prediction()
    worker = raw["engine"]["workers"]["aggregated"]
    del worker["host"], worker["frontend"]
    worker["host_profile"] = {"path": str(path), "frontend": "python"}
    profiled = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    assert (
        profiled.backend_deployment.agg_engine_args
        == explicit.backend_deployment.agg_engine_args
    )
    vl = profiled.backend_deployment.performance_model_metadata["aggregated"]["vl"]
    assert vl["frontend"] == "python" and len(vl["host_profile_digest"]) == 16

    worker["parallelism"]["tensor"] = 2
    tp2 = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    assert tp2.backend_deployment.agg_engine_args["sglang"]["host"]["tp_sync_ms"] == 0.4


@pytest.mark.parametrize("kind", ["frontend", "encoding", "tp_sync", "missing"])
def test_host_profile_mismatches_fail_closed(tmp_path, kind):
    raw = _prediction()
    worker = raw["engine"]["workers"]["aggregated"]
    del worker["host"], worker["frontend"]
    profile = _host_profile()
    if kind == "frontend":
        worker["host_profile"] = {"path": "", "frontend": "rust"}
        expected = "frontend: profile='python', prediction='rust'"
    elif kind == "encoding":
        raw["traffic"]["source"]["images"]["encoding"] = "jpeg"
        expected = "image_encoding: profile='png', prediction='jpeg'"
    elif kind == "tp_sync":
        worker["parallelism"]["tensor"] = 4
        expected = "no tp_sync_ms entry for tensor parallel 4"
    else:
        profile["missing"] = ["launch_extend"]
        expected = "lacks measured costs for: launch_extend"
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    worker["host_profile"] = {
        "path": str(path),
        "frontend": worker.get("host_profile", {}).get("frontend", "python"),
    }
    with pytest.raises(ValueError, match=expected):
        prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))


def test_missing_profile_is_calibrated_in_an_unsupervised_subprocess(
    tmp_path, monkeypatch
):
    profile_path = tmp_path / "fresh-profile.json"
    raw = _prediction()
    worker = raw["engine"]["workers"]["aggregated"]
    del worker["host"], worker["frontend"]
    worker["host_profile"] = {
        "path": str(profile_path),
        "frontend": "python",
        "on_missing": "calibrate",
    }
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("_AISIMULATE_SUPERVISED_BUDGET", "{}")
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["env"] = kwargs["env"]
        profile_path.write_text(json.dumps(_host_profile()))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    assert seen["command"][1:5] == [
        "-m",
        "aisimulate.vl.calibrate",
        "--frontend",
        "python",
    ]
    assert "--images" in seen["command"] and "448x448x1" in seen["command"]
    assert (
        "OMP_NUM_THREADS" not in seen["env"]
        and "_AISIMULATE_SUPERVISED_BUDGET" not in seen["env"]
    )
    assert (
        spec.backend_deployment.agg_engine_args["sglang"]["host"]["launch_extend"][
            "const_ms"
        ]
        == 5.0
    )


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
    assert candidate.config["agg_host"]["launch_extend"]["const_ms"] == 5.0

    saved = root / "recommendations" / "0001.yaml"
    concrete = CorePredictionConfig.from_yaml(saved)
    assert concrete.engine.workers.aggregated.host.launch_extend.const_ms == 5.0
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
    raw["engine"]["workers"]["aggregated"]["host"]["launch_extend"]["const_ms"] = 1.0
    with pytest.raises(ValueError, match="prediction-ready"):
        validate_vl_prediction_mapping(raw, spec)


def test_profile_backed_recommendation_keeps_naming_its_profile(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(_host_profile()))
    raw = _recommendation()
    worker = raw["engine"]["workers"]["aggregated"]
    del worker["host"], worker["frontend"]
    worker["host_profile"] = {"path": str(path), "frontend": "python"}
    lowered = recommendation_to_sweeper(CoreRecommendationConfig.model_validate(raw))
    space = lowered.search_space
    assert space.agg_host["launch_extend"]["const_ms"] == 5.0
    assert space.agg_tp_sync_ms == {"2": 0.4}
    assert space.agg_host_profile == {**worker["host_profile"], "on_missing": "error"}
    assert len(space.agg_host_profile_digest) == 16
