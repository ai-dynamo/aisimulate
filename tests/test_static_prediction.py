# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static prediction input, output, and failure contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import yaml

import aiconfigurator.cli.api as estimator
import aisimulate.main as cli
from aiconfigurator.cli.api import EstimateResult
from aisimulate.config.cli import CorePredictionConfig
from aisimulate.static_prediction import run_static_prediction, static_prediction_kwargs


@pytest.fixture
def config():
    return {
        "traffic": {
            "source": {"type": "synthetic", "input_tokens": 16, "output_tokens": 4},
            "load": {"type": "concurrency", "concurrency": 99},
            "stop": {"requests": 100},
        },
        "engine": {
            "model": "tests/e2e/configs/unified_cli/fixtures/tiny-model",
            "hardware": "h200_sxm",
            "backend": "vllm",
            "backend_version": "0.24.0",
            "workers": {"aggregated": {"parallelism": {"tensor": 2}}},
        },
    }


def _result(kwargs):
    return EstimateResult(
        ttft=10.0,
        tpot=2.0,
        power_w=None,
        isl=kwargs["isl"],
        osl=kwargs["osl"],
        batch_size=kwargs["batch_size"],
        ctx_tokens=kwargs["isl"],
        tp_size=kwargs["tp_size"],
        pp_size=kwargs["pp_size"],
        model_path=kwargs["model_path"],
        system_name=kwargs["system_name"],
        backend_name=kwargs["backend_name"],
        backend_version="0.24.0",
        mode=kwargs["mode"],
        raw={"memory": 12.0, "request_latency": 16.0, "generation_latency": 6.0},
        summary=SimpleNamespace(
            get_memory=lambda: {"total": 12.0, "weights": 10.0, "kvcache": 2.0},
            get_mem_capacity_bytes=lambda: 128 * 1024**3,
            get_context_latency_dict=lambda: {"context_gemm": 10.0},
            get_generation_latency_dict=lambda: {"generation_gemm": 6.0},
        ),
    )


def _args(tmp_path, config, *extra):
    path = tmp_path / "prediction.yaml"
    path.write_text(yaml.safe_dump(config))
    return [
        "predict",
        "-c",
        str(path),
        "--output-dir",
        str(tmp_path / "out"),
        "--estimate-mode",
        "static",
        "--batch-size",
        "8",
        *extra,
    ]


@pytest.mark.parametrize(
    ("mode", "phases", "metrics"),
    [
        ("static", {"prefill", "decode"}, {"ttft_ms", "tpot_ms", "generation_latency_ms", "request_latency_ms"}),
        ("static_ctx", {"prefill"}, {"ttft_ms"}),
        ("static_gen", {"decode"}, {"tpot_ms", "generation_latency_ms"}),
    ],
)
def test_static_predict_uses_yaml_and_writes_same_json(mode, phases, metrics, config, tmp_path, monkeypatch, capsys):
    calls = []

    def estimate(**kwargs):
        calls.append(kwargs)
        print("model resolution notice")
        return _result(kwargs)

    monkeypatch.setattr(estimator, "cli_estimate", estimate)
    monkeypatch.setattr(cli, "resolve_runner_factory", lambda _: object())  # Cannot create/replay a runner.
    assert (
        cli.main(
            _args(
                tmp_path,
                config,
                "--estimate-mode",
                mode,
                "--detail",
                "all",
                "--format",
                "json",
                "--set",
                "traffic.source.input_tokens=32",
                "--set",
                "engine.workers.aggregated.parallelism.tensor=4",
            )
        )
        == 0
    )
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report == json.loads((tmp_path / "out/prediction.json").read_text())
    assert "model resolution notice" in captured.err
    assert report["prediction_kind"] == "static_estimate"
    assert report["estimate_mode"] == mode
    assert report["inputs"]["backend_version"] == "0.24.0"
    assert set(report["summary"]) == metrics | {"memory_gib"}
    assert set(report["details"]["sections"]["time"]["per_operation"]) == phases
    assert report["details"]["sections"]["memory"]["components"]["total"] == 12.0
    assert report["details"]["sections"]["time"]["scope"] == "phase_total"
    assert report["assumptions"]["serving_controls_not_modeled"] == [
        "traffic.load",
        "traffic.stop",
        "engine.workers.aggregated.scheduler",
        "evaluation",
    ]
    assert calls == [
        {
            "model_path": config["engine"]["model"],
            "system_name": "h200_sxm",
            "backend_name": "vllm",
            "backend_version": "0.24.0",
            "mode": mode,
            "batch_size": 8,
            "isl": 32,
            "osl": 4,
            "tp_size": 4,
            "pp_size": 1,
            "attention_dp_size": 1,
            "moe_tp_size": None,
            "moe_ep_size": None,
            "forward_model": "op_level",
            "nextn": 0,
            "prefix": 0,
        }
    ]
    assert not (tmp_path / "out/requests.jsonl").exists()


def test_static_parallelism_and_forward_model_projection(config):
    config["engine"]["workers"]["aggregated"] = {
        "parallelism": {"tensor": 4, "pipeline": 2, "attention_data": 2, "moe_tensor": 2, "moe_expert": 4},
        "timing": {"forward_model": "fpm"},
    }
    kwargs = static_prediction_kwargs(CorePredictionConfig.model_validate(config), "static_gen", 8)
    assert {key: kwargs[key] for key in ("tp_size", "pp_size", "attention_dp_size", "moe_tp_size", "moe_ep_size")} == {
        "tp_size": 4,
        "pp_size": 2,
        "attention_dp_size": 2,
        "moe_tp_size": 2,
        "moe_ep_size": 4,
    }
    assert kwargs["forward_model"] == "fpm"


@pytest.mark.parametrize(
    "extra",
    [
        ["--estimate-mode", "static"],
        ["--estimate-mode", "static", "--batch-size", "0"],
        ["--estimate-mode", "static", "--batch-size", "-1"],
        ["--batch-size", "4"],
        ["--estimate-mode", "agg", "--batch-size", "4"],
        ["--estimate-mode", "static", "--batch-size", "4", "--detail", "source"],
    ],
)
def test_invalid_static_flags_fail_before_loading_config(extra, monkeypatch):
    monkeypatch.setattr(cli, "_load_mapping", lambda _: pytest.fail("invalid flags reached config loading"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["predict", "-c", "unused.yaml", *extra])
    assert exc.value.code == 2


def test_estimate_is_not_a_new_command_and_recommend_rejects_modes():
    for args in (["estimate"], ["recommend", "-c", "unused.yaml", "--estimate-mode", "static"]):
        with pytest.raises(SystemExit) as exc:
            cli.main(args)
        assert exc.value.code == 2


@pytest.mark.parametrize(
    "override",
    [
        "engine.mode=disaggregated",
        "engine.workers.aggregated.parallelism.replicas=2",
        "engine.workers.aggregated.startup_seconds=1",
        "engine.workers.aggregated.timing={type: fixed, prefill_ms: 1, decode_ms: 1}",
        "engine.workers.aggregated.timing={type: polynomial}",
        "engine.workers.aggregated.kv_cache.capacity={type: fixed, blocks: 256}",
        "engine.workers.aggregated.kv_cache.capacity.memory_fraction=0.8",
        "engine.workers.aggregated.kv_cache.prefix_caching=false",
        "engine.context_length=2048",
        "traffic.source={type: trace, format: mooncake, paths: [unused.jsonl]}",
        "traffic.source={type: synthetic-session, session: {turns: 2}}",
        "router={}",
    ],
)
def test_unsupported_config_fails_before_output_or_estimation(override, config, tmp_path, monkeypatch):
    monkeypatch.setattr(estimator, "cli_estimate", lambda **_: pytest.fail("unsupported config reached estimator"))
    with pytest.raises(SystemExit) as exc:
        cli.main(_args(tmp_path, config, "--set", override))
    assert exc.value.code == 2
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("flag", ["--online", "--capture-per-request"])
def test_static_rejects_serving_execution_flags(flag, config, tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli.main(_args(tmp_path, config, flag))
    assert exc.value.code == 2
    assert not (tmp_path / "out").exists()


def test_static_table_and_oom_warning(config, tmp_path, monkeypatch, capsys):
    def estimate(**kwargs):
        result = _result(kwargs)
        result.kv_cache_warning = "OOM: reduce batch size"
        return result

    monkeypatch.setattr(estimator, "cli_estimate", estimate)
    assert cli.main(_args(tmp_path, config)) == 0
    text = capsys.readouterr().out
    assert "Static prediction (static)" in text
    assert "Fixed batch: 8" in text
    assert "no request arrivals, queueing, scheduling, or SLA evaluation" in text
    assert "OOM: reduce batch size" in text
    report = json.loads((tmp_path / "out/prediction.json").read_text())
    assert report["warnings"] == ["OOM: reduce batch size"]
    assert "details" not in report


def test_missing_detail_evidence_is_explicit(config, monkeypatch):
    def estimate(**kwargs):
        result = _result(kwargs)
        result.summary = None
        return result

    monkeypatch.setattr(estimator, "cli_estimate", estimate)
    kwargs = static_prediction_kwargs(CorePredictionConfig.model_validate(config), "static", 8)
    report = run_static_prediction(kwargs, ("summary", "memory", "time"))
    assert set(report["details"]["sections"]) == {"summary"}
    assert set(report["details"]["skipped"]) == {"memory", "time"}


@pytest.mark.parametrize(("error", "code"), [(RuntimeError("missing profile"), 1), (KeyboardInterrupt(), 130)])
def test_estimator_failures_use_predict_exit_codes(error, code, config, tmp_path, monkeypatch):
    def estimate(**_):
        raise error

    monkeypatch.setattr(estimator, "cli_estimate", estimate)
    assert cli.main(_args(tmp_path, config)) == code
    assert not (tmp_path / "out/prediction.json").exists()


def test_nonfinite_result_is_not_published(config, tmp_path, monkeypatch):
    def estimate(**kwargs):
        result = _result(kwargs)
        result.tpot = float("nan")
        return result

    monkeypatch.setattr(estimator, "cli_estimate", estimate)
    assert cli.main(_args(tmp_path, config, "--format", "json")) == 1
    assert not (tmp_path / "out/prediction.json").exists()
