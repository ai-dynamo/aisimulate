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
from aisimulate.config.epd import validate_epd_prediction_mapping
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
                    "frontend": {"stages": [{"workers": 1, "service_ms": 3.0}]},
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
    assert spec.workload["isl"] == 128  # placeholders are laid out by the workload driver


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
    # The stage means do not depend on keeping per-request records.
    assert main(argv[:-1] + ["--output-dir", str(tmp_path / "bounded")]) == 0
    bounded = json.loads(capsys.readouterr().out)
    assert bounded["mean_frontend_ms"] == report["mean_frontend_ms"] > 0.0
    assert report["completed_requests"] == 4
    # 448x448 -> 196 visual tokens per image on top of the 128 text tokens.
    assert report["total_input_tokens"] == 4 * (128 + 196)
    for key in ("mean_frontend_ms", "mean_prefill_elapsed_ms"):
        assert stdout[key] == report[key] > 0.0
    # The scheduler thread is free: a received request is selected in the same instant.
    assert report["mean_receive_to_admit_ms"] == 0.0
    assert report["mean_ttft_ms"] >= report["mean_frontend_ms"] + report["mean_prefill_elapsed_ms"]
    records = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
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


def test_images_encode_on_the_language_worker_without_a_host_loop():
    raw = _prediction()
    del raw["engine"]["workers"]["aggregated"]["frontend"]
    args = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw)).backend_deployment.agg_engine_args
    assert args["vision"] is True
    assert "host_loop" not in args["sglang"] and "frontend" not in args


@pytest.mark.parametrize(
    "kind",
    ["encoder_and_host", "frontend_without_host", "vllm", "fixed_timing", "pipeline_without_vision_key"],
)
def test_native_vl_schema_rejects_unsupported(kind):
    raw = _prediction()
    worker = raw["engine"]["workers"]["aggregated"]
    if kind == "pipeline_without_vision_key":
        # Images enable the encoder implicitly; the layout gate must not depend on spelling `vision` out.
        del worker["frontend"]
        worker["parallelism"] = {"replicas": 1, "tensor": 1, "pipeline": 2}
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


def _host_row(frontend="python", feature_transport="shm", service_ms=3.0):
    return {
        "measured_for": {
            "model": "Qwen/Qwen3-VL-8B-Instruct",
            "frontend": frontend,
            "feature_transport": feature_transport,
            "height": 448,
            "width": 448,
            "count": 1,
            "encoding": "png",
        },
        "stages": [{"workers": 1, "service_ms": service_ms}],
        "provenance": {"text_tokens": 128, "host": "example-host", "threads": 16},
    }


def _host_table(sglang_version="0.5.19", rows=None):
    environment = {"cpu": "example-cpu", "sglang_version": sglang_version, "python": "3.10.12"}
    return {"schema_version": 1, "environment": environment, "rows": rows or [_host_row()]}


def test_host_profile_lowers_to_the_explicit_stages(tmp_path):
    path = tmp_path / "table.json"
    path.write_text(json.dumps(_host_table()))
    explicit = prediction_to_replay_spec(CorePredictionConfig.model_validate(_prediction()))
    raw = _prediction()
    worker = raw["engine"]["workers"]["aggregated"]
    del worker["frontend"]
    worker["host_profile"] = {"path": str(path), "frontend": "python"}
    profiled = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    assert profiled.backend_deployment.agg_engine_args == explicit.backend_deployment.agg_engine_args
    vl = profiled.backend_deployment.performance_model_metadata["aggregated"]["vl"]
    assert vl["frontend"] == "python" and len(vl["host_profile_digest"]) == 16


@pytest.mark.parametrize("kind", ["frontend", "encoding", "count", "version"])
def test_host_profile_misses_fail_closed_with_the_collect_command(tmp_path, kind):
    raw = _prediction()
    worker = raw["engine"]["workers"]["aggregated"]
    del worker["frontend"]
    table = _host_table("0.6.0" if kind == "version" else "0.5.19")
    frontend = "python"
    if kind == "frontend":
        frontend = "rust"
    elif kind == "encoding":
        raw["traffic"]["source"]["images"]["encoding"] = "jpeg"
    elif kind == "count":
        raw["traffic"]["source"]["images"]["count"] = 2
    path = tmp_path / "table.json"
    path.write_text(json.dumps(table))
    worker["host_profile"] = {"path": str(path), "frontend": frontend}
    with pytest.raises(ValueError) as error:
        prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    message = str(error.value)
    if kind == "version":
        assert "measured with sglang 0.6.0" in message
        return
    assert "has no row for" in message
    # The failure names the exact measurement that would add the row; a text-only
    # length is not part of it.
    assert "-m aisimulate.vl.collect" in message and f"--frontend {frontend}" in message
    count = 2 if kind == "count" else 1
    assert f"--images 448x448x{count} --encoding " + ("jpeg" if kind == "encoding" else "png") in message
    assert ("--tp 1" in message) == (frontend == "rust")
    assert "--text-tokens" not in message


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
    recommendation = _recommendation()
    # A relative stop is resolved per candidate; the saved prediction must still be
    # recognized as the scored run.
    recommendation["traffic"]["stop"] = {"requests_per_load_unit": 2}
    path = tmp_path / "search.yaml"
    path.write_text(yaml.safe_dump(recommendation))
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
    assert candidate.config["agg_frontend"]["stages"][0]["service_ms"] == 3.0

    saved = root / "recommendations" / "0001.yaml"
    concrete = CorePredictionConfig.from_yaml(saved)
    assert concrete.engine.workers.aggregated.host_loop is True
    assert concrete.engine.workers.aggregated.frontend.stages[0].service_ms == 3.0
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
    raw["engine"]["workers"]["aggregated"]["frontend"]["stages"][0]["service_ms"] = 1.0
    with pytest.raises(ValueError, match="prediction-ready"):
        validate_vl_prediction_mapping(raw, spec)
    # The resolved request count is compared, not the spelling of the stop condition.
    raw = concrete.model_dump(mode="python", exclude_none=True)
    raw["traffic"]["stop"] = {"requests": 99}
    with pytest.raises(ValueError, match="traffic changed: request_count=99"):
        validate_vl_prediction_mapping(raw, spec)


def test_profile_backed_recommendation_pins_the_resolved_stages(tmp_path, capsys):
    path = tmp_path / "table.json"
    path.write_text(json.dumps(_host_table()))
    raw = _recommendation()
    worker = raw["engine"]["workers"]["aggregated"]
    explicit_frontend = worker.pop("frontend")
    worker["host_profile"] = {"path": str(path), "frontend": "python"}
    worker["parallelism"]["tensor"] = 2
    lowered = recommendation_to_sweeper(CoreRecommendationConfig.model_validate(raw))
    space = lowered.search_space
    # Candidates score the row resolved for their feature transport (the Python
    # frontend always uses shared memory), identified by content; the mutable file
    # path is not part of what a candidate carries or exports.
    assert space.agg_host_loop is True
    row = space.agg_frontend_by_transport["shm"]
    assert row["frontend"]["stages"][0]["service_ms"] == explicit_frontend["stages"][0]["service_ms"]
    assert row["frontend"]["measured_for"]["feature_transport"] == "shm"
    assert len(row["digest"]) == 16
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
    saved_path = root / "recommendations" / "0001.yaml"
    saved = CorePredictionConfig.from_yaml(saved_path)
    assert saved.engine.workers.aggregated.frontend.stages[0].service_ms == 3.0
    assert saved.engine.workers.aggregated.frontend.measured_for.feature_transport == "shm"
    assert saved.engine.workers.aggregated.host_profile is None
    # The saved stages carry the workload they were measured for: changing the
    # images makes the candidate refuse to price them with the old constants.
    edited = yaml.safe_load(saved_path.read_text())
    edited["traffic"]["source"]["images"]["count"] = 2
    with pytest.raises(ValueError, match="measured for"):
        prediction_to_replay_spec(CorePredictionConfig.model_validate(edited))
    edited = yaml.safe_load(saved_path.read_text())
    edited["engine"]["model"] = "Qwen/Qwen3-VL-2B-Instruct"
    with pytest.raises(ValueError, match="measured for"):
        prediction_to_replay_spec(CorePredictionConfig.model_validate(edited))


def test_rust_profiles_resolve_only_the_transports_the_tp_domain_uses(tmp_path):
    path = tmp_path / "table.json"
    rows = [_host_row("rust", "inline", 2.0), _host_row("rust", "shm", 5.0)]
    path.write_text(json.dumps(_host_table(rows=rows)))
    raw = _recommendation()
    worker = raw["engine"]["workers"]["aggregated"]
    del worker["frontend"]
    worker["host_profile"] = {"path": str(path), "frontend": "rust"}

    def transports(tensor):
        worker["parallelism"]["tensor"] = tensor
        space = recommendation_to_sweeper(CoreRecommendationConfig.model_validate(deepcopy(raw))).search_space
        return {key: row["frontend"]["stages"][0]["service_ms"] for key, row in space.agg_frontend_by_transport.items()}

    # One rank keeps features inline; every wider candidate shares the shm row.
    assert transports({"choices": [1, 2]}) == {"inline": 2.0, "shm": 5.0}
    assert transports({"range": {"min": 2, "max": 4, "scale": "log"}}) == {"shm": 5.0}
    assert transports(4) == {"shm": 5.0}


def test_text_only_host_tables_reach_predict_and_recommend_alike(tmp_path, capsys):
    raw = _prediction()
    del raw["traffic"]["source"]["images"]
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    args = spec.backend_deployment.agg_engine_args
    assert args["sglang"]["host_loop"] is True
    assert "vision" not in args and "images" not in spec.workload

    search = _recommendation()
    del search["traffic"]["source"]["images"]
    space = recommendation_to_sweeper(CoreRecommendationConfig.model_validate(search)).search_space
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
    assert all(c.config["prediction_config_supported"] is True for c in result.selected_candidates)
    saved = CorePredictionConfig.from_yaml(root / "recommendations" / "0001.yaml")
    assert saved.engine.workers.aggregated.host_loop is True
    assert saved.traffic.source.images is None


def test_materialized_request_lists_reject_image_workloads():
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(_prediction()))
    legacy = dataclasses.replace(
        spec,
        workload={key: value for key, value in spec.workload.items() if key not in ("source_type", "load_type")},
    )
    with pytest.raises(InvalidRunnerError, match="workload-driver"):
        EngineReplayRunnerFactory().create(0).run(legacy)


def _disaggregated_prediction():
    raw = _prediction()
    prefill = raw["engine"]["workers"].pop("aggregated")
    prefill["scheduler"] = {"max_batched_tokens": 8192, "max_sequences": 1}
    raw["engine"]["mode"] = "disaggregated"
    raw["engine"]["workers"] = {
        "prefill": prefill,
        "decode": {
            "parallelism": {"replicas": 1, "tensor": 1},
            "scheduler": {"max_batched_tokens": 8192, "max_sequences": 8},
            "kv_cache": {"capacity": {"type": "fixed", "blocks": 4096}},
        },
    }
    return raw


def test_native_vl_disaggregated_prefill_hosts_the_loop_and_the_encoder(tmp_path, capsys):
    raw = _disaggregated_prediction()
    deployment = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw)).backend_deployment
    prefill, decode = deployment.prefill_engine_args, deployment.decode_engine_args
    assert prefill["vision"] is True and prefill["sglang"]["host_loop"] is True
    assert prefill["frontend"]["stages"][0]["service_ms"] == 3.0
    # The decode rank neither encodes images nor runs the modeled loop: SGLang's
    # decode-side preprocessing runs concurrently with the prefill rank's.
    assert "vision" not in decode and "frontend" not in decode and "host_loop" not in decode.get("sglang", {})

    path = tmp_path / "predict.yaml"
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "out"
    argv = ["predict", "-c", str(path), "--output-dir", str(output), "--format", "json", "--capture-per-request"]
    assert main(argv) == 0
    capsys.readouterr()
    report = json.loads((output / "prediction.json").read_text())
    assert report["completed_requests"] == 4
    assert report["total_input_tokens"] == 4 * (128 + 196)
    records = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    # The prefill rank's frontend deadlines wake the disaggregated replay: one
    # pool worker serializes the two initial arrivals.
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
            < record["first_token_ms"]
        )


def test_native_vl_schema_rejects_host_fields_on_the_decode_worker():
    raw = _disaggregated_prediction()
    raw["engine"]["workers"]["decode"]["host_loop"] = True
    with pytest.raises(ValidationError):
        CorePredictionConfig.model_validate(raw)


def test_native_vl_disaggregated_recommend_keys_the_prefill_worker(tmp_path, capsys):
    raw = _disaggregated_prediction()
    for role in ("prefill", "decode"):
        raw["engine"]["workers"][role]["parallelism"] = {
            "preset": False,
            "replicas": 1,
            "tensor": 1,
            "pipeline": 1,
            "attention_data": 1,
            "moe_tensor": 1,
            "moe_expert": 1,
        }
    raw["engine"]["workers"]["prefill"]["scheduler"]["max_batched_tokens"] = {"choices": [4096, 8192]}
    raw["optimization"] = {"target": "throughput_per_gpu", "constraints": {"max_candidate_gpus": 2}}
    raw["optimizer"] = {
        "algorithm": "random",
        "max_trials": 2,
        "parallelism": 1,
        "candidate_timeout_seconds": 60.0,
        "seed": 13,
    }
    space = recommendation_to_sweeper(CoreRecommendationConfig.model_validate(raw)).search_space
    assert space.prefill_host_loop is True
    assert space.prefill_frontend["stages"][0]["service_ms"] == 3.0
    assert space.prefill_vision["cache_mib"] == 100
    assert space.agg_host_loop is None and space.agg_vision is None

    path = tmp_path / "search.yaml"
    path.write_text(yaml.safe_dump(raw))
    root = tmp_path / "recommend"
    assert main(["recommend", "-c", str(path), "--output-dir", str(root), "--format", "json"]) == 0
    capsys.readouterr()
    result = SweepResult.from_json((root / "recommendation.json").read_text())
    candidate = result.selected_candidates[0]
    assert candidate.config["prediction_config_supported"] is True
    assert candidate.config["prefill_host_loop"] is True
    saved_path = root / "recommendations" / "0001.yaml"
    saved = CorePredictionConfig.from_yaml(saved_path)
    assert saved.engine.workers.prefill.frontend.stages[0].service_ms == 3.0
    assert saved.engine.workers.prefill.vision.cache_mib == 100
    assert "host_loop" not in yaml.safe_load(saved_path.read_text())["engine"]["workers"]["decode"]


def _native_epd_prediction(table_path):
    raw = _prediction()
    worker = raw["engine"]["workers"]["aggregated"]
    del worker["frontend"]
    worker["host_loop"] = True
    raw["engine"]["workers"]["encoder"] = {
        "mode": "native",
        "tensor": 1,
        "replicas": 1,
        "batch_size": 2,
        "host_profile": {"path": str(table_path), "frontend": "python"},
        "transfer": {"bandwidth_gb_per_second": 10},
    }
    return raw


def test_native_encoder_pool_gates_the_language_worker(tmp_path, capsys):
    table = tmp_path / "table.json"
    table.write_text(json.dumps(_host_table()))
    raw = _native_epd_prediction(table)
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    encoder = spec.backend_deployment.encoder
    native = encoder.native
    # The `process` stage of the Python frontend, per image, extrapolates the encoder's CPU
    # preprocessing; the forward is priced at replay by the canonical timing model at the pool's width.
    assert encoder.mode == "native" and native.preprocess_ms_per_image == 3.0
    assert native.preprocess_source == "frontend_process_extrapolated"
    assert native.shape["output_tokens"] == 196 and native.transfer_bytes_per_image > 0
    assert native.timing_model["config"]["tp"] == 1 and native.timing_model["config"]["encoder_parallel"] == "tp"
    metadata = spec.backend_deployment.performance_model_metadata["encoder"]
    assert metadata["host_profile_digest"] == native.host_profile_digest
    # The language worker runs --language-only: no tower, no image frontend stages, the loop stays.
    args = spec.backend_deployment.agg_engine_args
    assert "vision" not in args and "frontend" not in args and args["sglang"]["host_loop"] is True

    path = tmp_path / "predict.yaml"
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "out"
    argv = ["predict", "-c", str(path), "--output-dir", str(output), "--format", "json", "--capture-per-request"]
    assert main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    # The encoder mean does not depend on keeping per-request records.
    assert main(argv[:-1] + ["--output-dir", str(tmp_path / "bounded")]) == 0
    bounded = json.loads(capsys.readouterr().out)
    assert bounded["encoder_latency_ms"] == report["encoder_latency_ms"] > 3.0
    # The pool's GPU joins the language worker's in the totals, and is charged for the
    # whole duration in gpu_hours on top of the language worker's uptime.
    assert report["encoder_gpus"] == 1 and report["total_gpus"] == 2
    duration_hours = report["duration_ms"] / 3_600_000
    assert 2 * duration_hours <= report["gpu_hours"] < 3 * duration_hours
    assert report["completed_requests"] == 4
    # 448x448 -> 196 visual tokens per image occupy the language prompt.
    assert report["total_input_tokens"] == 4 * (128 + 196)
    records = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    # The two concurrent arrivals share one encoder batch and are delivered together;
    # the scheduler receives a request only after that.
    delivered = sorted(record["encoder_ready_ms"] for record in records if record["arrival_time_ms"] == 0.0)
    assert delivered[0] == delivered[1] > 0.0
    for record in records:
        assert record["arrival_time_ms"] < record["encoder_ready_ms"] <= record["scheduler_received_ms"]


def test_native_encoder_pool_requires_its_host_table_and_link(tmp_path):
    raw = _native_epd_prediction(tmp_path / "table.json")
    del raw["engine"]["workers"]["encoder"]["host_profile"]
    with pytest.raises(ValidationError):
        CorePredictionConfig.model_validate(raw)


def test_native_encoder_pool_honors_the_pixel_budget(tmp_path):
    table = tmp_path / "table.json"
    table.write_text(json.dumps(_host_table()))
    default = prediction_to_replay_spec(
        CorePredictionConfig.model_validate(_native_epd_prediction(table))
    ).backend_deployment
    row = _host_row()
    row["measured_for"]["max_pixels"] = 65536
    table.write_text(json.dumps(_host_table(rows=[row])))
    raw = _native_epd_prediction(table)
    raw["traffic"]["source"]["images"]["max_pixels"] = 65536
    encoder = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw)).backend_deployment.encoder
    # 448x448 rescaled into a 256x256 budget: the 64 visual tokens are what the prompt
    # holds, the encoder forwards and the transfer carries.
    assert encoder.visual_tokens == 64 and encoder.native.shape["output_tokens"] == 64
    assert encoder.native.transfer_bytes_per_image < default.encoder.native.transfer_bytes_per_image


def test_native_encoder_pool_recommend_yaml_predict_roundtrip(tmp_path, capsys):
    table = tmp_path / "table.json"
    table.write_text(json.dumps(_host_table()))
    raw = _recommendation()
    worker = raw["engine"]["workers"]["aggregated"]
    del worker["frontend"]
    worker["host_loop"] = True
    raw["engine"]["workers"]["encoder"] = {
        "mode": "native",
        "tensor": 1,
        "replicas": {"choices": [1, 2]},
        "batch_size": 2,
        "host_profile": {"path": str(table), "frontend": "python"},
        "transfer": {"bandwidth_gb_per_second": 10},
    }
    raw["optimization"]["constraints"]["max_candidate_gpus"] = 4
    path = tmp_path / "search.yaml"
    path.write_text(yaml.safe_dump(raw))
    root = tmp_path / "recommend"
    assert main(["recommend", "-c", str(path), "--output-dir", str(root), "--format", "json"]) == 0
    capsys.readouterr()
    result = SweepResult.from_json((root / "recommendation.json").read_text())
    candidate = result.selected_candidates[0]
    encoder = candidate.config["encoder"]
    assert encoder["native"]["preprocess_ms_per_image"] == 3.0
    # Encoder GPUs are in the candidate's totals and in gpu_hours, so throughput_per_gpu
    # ranks encoder replicas against language workers.
    assert candidate.used_gpus == 1 + encoder["tp"] * encoder["workers"] == candidate.metrics["total_gpus"]
    duration_hours = candidate.metrics["duration_ms"] / 3_600_000
    assert (
        candidate.used_gpus * duration_hours
        <= candidate.metrics["gpu_hours"]
        < (candidate.used_gpus + 1) * duration_hours
    )

    saved = root / "recommendations" / "0001.yaml"
    concrete = CorePredictionConfig.from_yaml(saved)
    assert concrete.engine.workers.encoder.mode == "native"
    output = tmp_path / "predict"
    assert main(["predict", "-c", str(saved), "--output-dir", str(output), "--format", "json"]) == 0
    report = json.loads(capsys.readouterr().out)
    for key in ("mean_ttft_ms", "encoder_latency_ms", "gpu_hours"):
        assert report[key] == pytest.approx(candidate.metrics[key])

    spec = prediction_to_replay_spec(concrete)
    raw_saved = concrete.model_dump(mode="python", exclude_none=True)
    validate_epd_prediction_mapping(raw_saved, spec)
    # A host table that changed underneath the saved recommendation no longer reproduces it.
    table.write_text(json.dumps(_host_table(rows=[_host_row(service_ms=1.0)])))
    with pytest.raises(ValueError, match="encoder cost terms changed"):
        validate_epd_prediction_mapping(raw_saved, spec)


@pytest.mark.parametrize("case", ["constant_rate", "kv_capacity_fraction", "fixed_timing"])
def test_prediction_mapping_accepts_the_scored_spellings(case):
    """The callback compares resolved execution meaning, not the compiler's derived keys."""
    raw = _prediction()
    if case == "constant_rate":
        raw["traffic"]["load"] = {"type": "constant_rate", "requests_per_second": 4.0}
    elif case == "fixed_timing":
        del raw["traffic"]["source"]["images"]
        raw["engine"]["workers"]["aggregated"]["timing"] = {"type": "fixed", "prefill_ms": 5.0, "decode_ms": 1.0}
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    if case == "kv_capacity_fraction":
        # The sweeper resolves a KV-capacity load into this concurrency before scoring.
        spec = dataclasses.replace(
            spec,
            concurrency=spec.workload["concurrency"],
            workload={**spec.workload, "load_type": "kv_capacity_fraction"},
        )
    validate_vl_prediction_mapping(raw, spec)
