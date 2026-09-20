# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native VL replay: image workloads encoded on a host-aware SGLang worker."""

import json
from copy import deepcopy

import pytest
import yaml
from pydantic import ValidationError

from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config.cli import CorePredictionConfig
from aisimulate.main import main


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


def test_native_vl_predict_reports_host_stages(tmp_path, capsys):
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
        "mean_scheduler_receive_ms",
        "mean_selection_wait_ms",
        "mean_forward_ms",
    ):
        assert stdout[key] == report[key] > 0.0
    # Two arrivals share one processor worker: 3 ms and 6 ms in the frontend.
    assert report["mean_frontend_ms"] == pytest.approx(4.5)
    assert (
        report["mean_ttft_ms"] >= report["mean_frontend_ms"] + report["mean_forward_ms"]
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
    assert vl["frontend"] == "python" and len(vl["host_profile_id"]) == 16

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
