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
from aisimulate.recommend import _candidate_prediction, recommendation_to_sweeper
from aisimulate.runner import EngineReplayRunnerFactory
from aisimulate.sweeper import SmartSearchConfig, Sweeper, SweepResult


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


@pytest.mark.parametrize("mode", ["aggregated", "disaggregated", "heterogeneous"])
@pytest.mark.parametrize("relative_stop", [False, True])
def test_native_cli_epd_recommend_yaml_predict(tmp_path, capsys, mode, relative_stop):
    raw = _recommendation("disaggregated" if mode == "heterogeneous" else mode)
    if mode == "heterogeneous":
        raw["engine"]["workers"]["decode"]["hardware"] = "gb200"
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


@pytest.mark.parametrize(
    "mode,role,change",
    [
        (mode, None, change)
        for mode in ("aggregated", "disaggregated")
        for change in ("hardware", "backend_version", "missing_version", "context", "sla")
    ]
    + [
        (mode, role, change)
        for mode, role in (
            ("aggregated", "aggregated"),
            ("disaggregated", "prefill"),
            ("disaggregated", "decode"),
        )
        for change in ("scheduler", "prefill_interval", "prefill_decode_interval", "cache", "prefix")
    ],
)
def test_epd_sweeper_rejects_changed_language_prediction(mode, role, change):
    raw = _recommendation(mode)
    raw["engine"]["workers"]["encoder"]["replicas"] = 1
    raw["optimizer"]["max_trials"] = 1
    raw["evaluation"] = {"sla": {"ttft_ms": 1000.0}}
    raw["optimization"]["strict_sla"] = True
    source = CoreRecommendationConfig.model_validate(raw)
    config = recommendation_to_sweeper(source)
    config.sweep.max_eval_seconds = None

    def callback(sample, spec):
        value = _candidate_prediction(source, sample, spec, adapter_sections={})
        worker = value["engine"]["workers"][role] if role else None
        if change == "hardware":
            value["engine"]["hardware"] = "h100_sxm"
        elif change == "backend_version":
            value["engine"]["backend_version"] = "changed-version"
        elif change == "missing_version":
            del value["engine"]["backend_version"]
        elif change == "scheduler":
            worker["scheduler"]["max_sequences"] = 1
        elif change == "prefill_interval":
            worker["scheduler"]["prefill_schedule_interval"] = 2
        elif change == "prefill_decode_interval":
            worker["scheduler"]["prefill_decode_interval"] = 2
        elif change == "cache":
            worker["kv_cache"]["capacity"]["blocks"] = 8192
        elif change == "prefix":
            worker["kv_cache"]["prefix_caching"] = False
        elif change == "context":
            value["engine"]["context_length"] = 140
        else:
            value["evaluation"]["sla"]["ttft_ms"] = 0.1
        return value

    result = Sweeper(
        runner_factory=EngineReplayRunnerFactory(), show_progress=False, prediction_config_factory=callback
    ).run(config)
    assert not result.selected_candidates
    assert result.candidates
    assert all("prediction-ready" in candidate.reason for candidate in result.candidates)


@pytest.mark.parametrize("change", ["bytes_per_token", "bandwidth_gb_per_second", "timing_mode"])
def test_epd_callback_rejects_changed_transfer(change):
    raw = _prediction("disaggregated")
    raw["engine"]["kv_transfer"] = {
        "bytes_per_token": 1024,
        "bandwidth_gb_per_second": 25.0,
        "timing_mode": "destination_missing",
    }
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    raw["engine"]["workers"]["encoder"] = encoder_prediction_fields(spec.backend_deployment.encoder)
    raw["engine"]["kv_transfer"][change] = {
        "bytes_per_token": 2048,
        "bandwidth_gb_per_second": 50.0,
        "timing_mode": "full_prompt",
    }[change]
    with pytest.raises(ValueError, match="prediction-ready"):
        validate_epd_prediction_mapping(raw, spec)


@pytest.mark.parametrize("mode", ["aggregated", "disaggregated"])
@pytest.mark.parametrize("inferred_capacity", [False, True])
def test_epd_callback_preserves_equivalent_defaults_and_stops(mode, inferred_capacity):
    raw = _recommendation(mode)
    raw["engine"]["workers"]["encoder"]["replicas"] = 1
    raw["optimizer"]["max_trials"] = 1
    if inferred_capacity:
        for role, worker in raw["engine"]["workers"].items():
            if role != "encoder":
                worker["kv_cache"]["capacity"] = {"type": "default"}
    source = CoreRecommendationConfig.model_validate(raw)
    config = recommendation_to_sweeper(source)
    config.sweep.max_eval_seconds = None

    def callback(sample, spec):
        value = _candidate_prediction(source, sample, spec, adapter_sections={})
        value["traffic"]["stop"] = {"requests_per_load_unit": 2.0}
        value["engine"]["backend_version"] = "current"
        for role, worker in value["engine"]["workers"].items():
            if role == "encoder":
                continue
            worker["scheduler"].pop("prefill_schedule_interval", None)
            worker["scheduler"].pop("prefill_decode_interval", None)
            worker["kv_cache"].pop("block_size", None)
            worker.pop("startup_seconds", None)
        return value

    result = Sweeper(
        runner_factory=EngineReplayRunnerFactory(), show_progress=False, prediction_config_factory=callback
    ).run(config)
    assert len(result.selected_candidates) == 1, result.to_json()
    candidate = result.selected_candidates[0]
    replay = prediction_to_replay_spec(CorePredictionConfig.model_validate(candidate.prediction_config))
    report = EngineReplayRunnerFactory().create(0).run(replay)
    assert report.metrics == pytest.approx(candidate.metrics)


@pytest.mark.parametrize("backends", [["sglang", "vllm"], ["vllm", "sglang"], ["vllm"]])
@pytest.mark.parametrize("absence", ["database", "version"])
def test_epd_native_search_preserves_available_backends(monkeypatch, caplog, backends, absence):
    from aisimulate.sweeper import kv_estimate
    from aisimulate_core.sdk import perf_database

    original_database = perf_database.get_database_view
    original_version = kv_estimate.get_latest_database_version

    def database(system, backend, version, **kwargs):
        return None if backend == "vllm" else original_database(system, backend, version, **kwargs)

    def latest_version(system, backend):
        return None if backend == "vllm" else original_version(system, backend)

    raw = _recommendation()
    raw["engine"]["workers"]["encoder"]["replicas"] = 1
    raw["engine"]["backend_version"] = None
    raw["optimizer"]["max_trials"] = 2
    config = recommendation_to_sweeper(CoreRecommendationConfig.model_validate(raw))
    payload = config.model_dump()
    payload["search_space"]["backend"] = backends
    config = SmartSearchConfig.model_validate(payload)
    config.sweep.max_eval_seconds = None
    if absence == "database":
        monkeypatch.setattr(perf_database, "get_database_view", database)
    else:
        # Keep resolve_backend_version real: a missing version raises its
        # documented NoPerfDatabase before get_database_view can return None.
        monkeypatch.setattr(kv_estimate, "get_latest_database_version", latest_version)
    sweeper = Sweeper(runner_factory=EngineReplayRunnerFactory(), show_progress=False)
    if backends == ["vllm"]:
        with pytest.raises(ValueError, match="no feasible encoder pool"):
            sweeper.run(config)
    else:
        result = sweeper.run(config)
        payload["search_space"]["backend"] = ["sglang"]
        payload["sweep"]["max_eval_seconds"] = None
        control = sweeper.run(SmartSearchConfig.model_validate(payload))
        assert len(result.selected_candidates) == len(control.selected_candidates) == 1
        assert all(candidate.config["backend"] == "sglang" for candidate in result.selected_candidates)
        for candidate, expected in zip(result.selected_candidates, control.selected_candidates, strict=True):
            assert candidate.metrics == pytest.approx(expected.metrics, rel=1e-12, abs=1e-12)
    assert "no encoder database for h200_sxm/vllm" in caplog.text


def test_cli_examples_parse():
    root = Path(__file__).resolve().parents[1]
    for mode in ("aggregated", "disaggregated"):
        CorePredictionConfig.from_yaml(root / f"examples/cli/epd-predict-{mode}.yaml")
    CoreRecommendationConfig.from_yaml(root / "examples/cli/epd-recommend.yaml")


@pytest.mark.parametrize("command", ["predict", "recommend"])
@pytest.mark.parametrize("combined_with_pd", [False, True])
def test_public_cli_rejects_afd_encoder_composition(tmp_path, capsys, command, combined_with_pd):
    raw = _prediction() if command == "predict" else _recommendation()
    encoder = raw["engine"]["workers"]["encoder"]
    raw["engine"]["mode"] = "afd"
    raw["engine"]["afd"] = {
        "phase": "decode" if combined_with_pd else "both",
        "combined_with_pd": combined_with_pd,
    }
    if command == "predict":
        raw["engine"]["afd"].update(n_a_nodes=1, n_f_nodes=1, tp_a=8, a_batch_size=8)
    else:
        raw["engine"]["afd"]["a_batch_size"] = {"choices": [8]}
    raw["engine"]["workers"] = {"encoder": encoder}
    if combined_with_pd:
        raw["engine"]["workers"]["prefill"] = {}
    path = tmp_path / "hybrid.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(SystemExit) as exc:
        main([command, "-c", str(path), "--output-dir", str(tmp_path / "out")])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "AFD does not support analytical EPD encoder pools" in captured.err


@pytest.mark.parametrize("mode", ["aggregated", "disaggregated"])
@pytest.mark.parametrize("selector", ["memory", "all"])
@pytest.mark.parametrize("explicit_blocks", [False, True])
def test_native_epd_detail_preserves_language_capacity_and_encoder_gap(
    tmp_path, capsys, mode, selector, explicit_blocks
):
    from jsonschema import validate

    raw = _prediction(mode)
    roles = [role for role in raw["engine"]["workers"] if role != "encoder"]
    if not explicit_blocks:
        for role in roles:
            raw["engine"]["workers"][role]["kv_cache"] = {"block_size": 1}
    path = tmp_path / "epd.yaml"
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "detailed"
    assert (
        main(["predict", "-c", str(path), "--detail", selector, "--format", "json", "--output-dir", str(output)]) == 0
    )
    stdout = json.loads(capsys.readouterr().out)
    saved = json.loads((output / "prediction.json").read_text())
    assert stdout["details"] == saved["details"]
    assert stdout["summary"] == saved["summary"]
    assert saved["summary"]["metric_semantics"] == "analytical_epd_overlay"
    assert "native_report" not in saved["metadata"]
    assert "per_request" not in saved
    assert not any(key.startswith(("goodput", "p99")) for key in saved["summary"])
    details = saved["details"]
    validate(
        details,
        json.loads((Path(__file__).resolve().parents[1] / "docs/cli/prediction-details.schema.json").read_text()),
    )
    if explicit_blocks:
        assert "memory" not in details["sections"]
        assert "encoder" in details["skipped"]["memory"]
        assert all(role in details["skipped"]["memory"] for role in roles)
    else:
        memory = details["sections"]["memory"]
        assert memory["status"] == "partial"
        assert memory["roles"]["encoder"]["status"] == "unavailable"
        assert "memory_breakdown" not in memory["roles"]["encoder"]
        for role in roles:
            estimate = memory["roles"][role]
            assert estimate["status"] == "available"
            assert estimate["stage"] == "before_native_capacity_adjustments"
            assert estimate["total_gpu_capacity_bytes"] > estimate["total_kv_size_bytes"] > 0
            assert estimate == saved["memory_diagnostics"][role]
    plain = tmp_path / "plain"
    assert main(["predict", "-c", str(path), "--format", "json", "--output-dir", str(plain)]) == 0
    plain_summary = json.loads(capsys.readouterr().out)
    assert plain_summary.keys() == stdout["summary"].keys()
    for name, value in stdout["summary"].items():
        # Native floating-point reductions can differ in their final bits.
        expected = pytest.approx(value, rel=1e-12, abs=1e-12) if isinstance(value, (int, float)) else value
        assert plain_summary[name] == expected
    assert "memory_diagnostics" not in json.loads((plain / "prediction.json").read_text())
