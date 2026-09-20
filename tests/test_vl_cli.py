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
