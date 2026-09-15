# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import yaml
from pydantic import ValidationError
from test_host_offload_cli import (
    _host_offload,
    _prediction_engine,
    _recommendation_engine,
    _RecordingRuntime,
)

from aisimulate import main as cli
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config.cli import CorePredictionConfig, CoreRecommendationConfig
from aisimulate.config.engine import G3OffloadConfig
from aisimulate.runner import EngineReplayRunnerFactory


def _g3_offload() -> dict:
    return {
        "scope": "cluster_shared",
        "num_g3_blocks": 4096,
        "latency_to_first_byte_ms": 0,
        "read_bandwidth_gbps": 10,
        "write_bandwidth_gbps": 10,
        "shared_read_bandwidth_gbps": 20,
        "shared_write_bandwidth_gbps": 20,
    }


@pytest.mark.parametrize("scope", ["worker_local", "cluster_shared"])
@pytest.mark.parametrize("override", [None, 0.0, 3.0])
def test_g3_defaults_and_overrides_reach_replay(scope, override) -> None:
    g3 = {"scope": scope, "num_g3_blocks": 4096}
    defaults = {
        "latency_to_first_byte_ms": 0.1,
        "read_bandwidth_gbps": 10.0,
        "write_bandwidth_gbps": 10.0,
        "shared_read_bandwidth_gbps": 80.0,
        "shared_write_bandwidth_gbps": 80.0,
    }
    if override is not None:
        g3.update(dict.fromkeys(defaults, override))
    expected = {**g3, **(defaults if override is None else dict.fromkeys(defaults, override))}
    assert G3OffloadConfig.model_validate(g3).model_dump() == expected
    assert set(G3OffloadConfig.model_json_schema()["required"]) == {"scope", "num_g3_blocks"}
    engine = _prediction_engine()
    engine["workers"]["aggregated"]["kv_cache"].update(bytes_per_token=256, host_offload=_host_offload(), g3_offload=g3)
    deployment = prediction_to_replay_spec(CorePredictionConfig.model_validate({"engine": engine})).backend_deployment
    assert deployment.agg_engine_args["g3_offload"] == expected


def test_g3_prediction_preserves_replica_count_and_flat_capacity(tmp_path) -> None:
    engine = _prediction_engine()
    worker = engine["workers"]["aggregated"]
    worker["parallelism"]["replicas"] = 3
    worker["kv_cache"].update(bytes_per_token=256, host_offload=_host_offload(), g3_offload=_g3_offload())
    path = tmp_path / "g3.yaml"
    path.write_text(yaml.safe_dump({"engine": engine}), encoding="utf-8")
    config = CorePredictionConfig.from_yaml(path)
    deployment = prediction_to_replay_spec(config).backend_deployment
    assert deployment.num_workers == 3
    assert deployment.agg_engine_args["g3_offload"] == _g3_offload()


@pytest.mark.parametrize("mutation", ["missing_g2", "nested_capacity", "attention_dp"])
def test_g3_rejects_unsupported_configuration(tmp_path, mutation) -> None:
    engine = _prediction_engine()
    worker = engine["workers"]["aggregated"]
    worker["kv_cache"].update(host_offload=_host_offload(), g3_offload=_g3_offload())
    if mutation == "missing_g2":
        worker["kv_cache"].pop("host_offload")
    elif mutation == "nested_capacity":
        worker["kv_cache"]["g3_offload"]["capacity"] = {"num_g3_blocks": 4096}
    else:
        worker["parallelism"]["attention_data"] = 2
    path = tmp_path / "g3-invalid.yaml"
    path.write_text(yaml.safe_dump({"engine": engine}), encoding="utf-8")
    with pytest.raises(ValidationError):
        CorePredictionConfig.from_yaml(path)


@pytest.mark.parametrize("scope", ["worker_local", "cluster_shared"])
def test_predict_cli_passes_g3_to_runtime(tmp_path, monkeypatch, capsys, scope) -> None:
    engine = _prediction_engine()
    g3 = {**_g3_offload(), "scope": scope}
    engine["workers"]["aggregated"]["kv_cache"].update(bytes_per_token=256, host_offload=_host_offload(), g3_offload=g3)
    path = tmp_path / "prediction.yaml"
    path.write_text(yaml.safe_dump({"engine": engine}), encoding="utf-8")
    runtime = _RecordingRuntime()
    monkeypatch.setattr(
        cli,
        "resolve_runner_factory",
        lambda _stack: EngineReplayRunnerFactory(runtime=runtime),
    )
    assert (
        cli.main(
            [
                "predict",
                "--stack",
                "engine",
                "--config",
                str(path),
                "--output-dir",
                str(tmp_path / "out"),
                "--format",
                "json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["completed_requests"] == 1
    rank = runtime.execution_spec["spec"]["engine"]["rank"]
    assert rank["g3_offload"] == g3
    assert rank["native_host_offload"] == _host_offload()
    assert rank["kv_cache_bytes_per_token"] == 256
    assert (tmp_path / "out" / "prediction.json").is_file()


@pytest.mark.parametrize("scope", ["worker_local", "cluster_shared"])
@pytest.mark.parametrize("capacity", [1, 2, 4096])
def test_predict_cli_runs_g3_through_real_rust_runtime(tmp_path, capsys, scope, capacity) -> None:
    from aisimulate import _runtime

    assert callable(_runtime.run_replay_json)
    engine = _prediction_engine()
    engine["workers"]["aggregated"]["kv_cache"].update(
        bytes_per_token=256,
        host_offload=_host_offload(),
        g3_offload={**_g3_offload(), "scope": scope, "num_g3_blocks": capacity},
    )
    config = {
        "engine": engine,
        "traffic": {
            "source": {"type": "synthetic", "input_tokens": 33, "output_tokens": 1},
            "load": {"type": "concurrency", "concurrency": 1},
            "stop": {"requests": 1},
        },
    }
    path = tmp_path / "prediction.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "out"
    assert (
        cli.main(
            ["predict", "--stack", "engine", "--config", str(path), "--output-dir", str(output), "--format", "json"]
        )
        == 0
    )
    stdout = json.loads(capsys.readouterr().out)
    saved = json.loads((output / "prediction.json").read_text())
    assert stdout == saved
    assert stdout["completed_requests"] == 1
    # The 33-token prompt stores two full 16-token blocks. Even a one-block
    # G3 must retain its leading block instead of discarding the whole cohort.
    stats = stdout["g3_offload"]
    assert stats["write"]["submitted_jobs"] == stats["write"]["completed_jobs"] == 1
    assert stats["write"]["completed_bytes"] == min(capacity, 2) * 16 * 256
    assert stats["resident_blocks"] == min(capacity, 2)
    assert stats["pending_blocks"] == 0


@pytest.mark.parametrize("field", ["scope", "num_g3_blocks"])
def test_g3_requires_scope_and_capacity(field) -> None:
    config = _g3_offload()
    config.pop(field)
    with pytest.raises(ValidationError):
        G3OffloadConfig.model_validate(config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("scope", "unknown"),
        ("num_g3_blocks", 0),
        ("num_g3_blocks", True),
        ("latency_to_first_byte_ms", -1),
        ("read_bandwidth_gbps", -1),
        ("write_bandwidth_gbps", float("inf")),
        ("shared_read_bandwidth_gbps", float("nan")),
        ("shared_write_bandwidth_gbps", -1),
    ],
)
def test_g3_rejects_invalid_scalar_values(field, value) -> None:
    with pytest.raises(ValidationError):
        G3OffloadConfig.model_validate({**_g3_offload(), field: value})


@pytest.mark.parametrize("mutation", ["sglang", "trtllm", "no_prefix", "disaggregated"])
def test_g3_rejects_unsupported_engine_scope(mutation) -> None:
    engine = _prediction_engine(mode="disaggregated" if mutation == "disaggregated" else "aggregated")
    role = "prefill" if mutation == "disaggregated" else "aggregated"
    cache = engine["workers"][role]["kv_cache"]
    cache.update(host_offload=_host_offload(), g3_offload=_g3_offload())
    if mutation in {"sglang", "trtllm"}:
        engine["backend"] = mutation
    elif mutation == "no_prefix":
        cache["prefix_caching"] = False
    with pytest.raises(ValidationError, match="host_offload"):
        CorePredictionConfig.model_validate({"engine": engine})


def test_g3_recommendation_is_rejected_not_silently_dropped() -> None:
    engine = _recommendation_engine()
    engine["workers"]["aggregated"]["kv_cache"].update(host_offload=_host_offload(), g3_offload=_g3_offload())
    with pytest.raises(ValidationError, match="g3_offload"):
        CoreRecommendationConfig.model_validate({"engine": engine})


def test_documented_g3_extension_validates_and_lowers_with_host_example(monkeypatch) -> None:
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "docs/cli/user-guide.md").read_text()
    host = text.split("### Native vLLM host-offload prediction\n", 1)[1]
    host, optional = host.split("#### Optional G3 offload\n", 1)
    optional = optional.split("## Router (Dynamo Adapter)", 1)[0]
    config = yaml.safe_load(host.split("```yaml\n", 1)[1].split("```", 1)[0])
    extension = yaml.safe_load(optional.split("```yaml\n", 1)[1].split("```", 1)[0])
    assert list(extension) == ["g3_offload"]
    assert set(extension["g3_offload"]) == {"scope", "num_g3_blocks"}
    cache = config["engine"]["workers"]["aggregated"]["kv_cache"]
    assert "host_offload" in cache
    cache.update(extension)
    prediction = CorePredictionConfig.model_validate(config)
    # Validate lowering without downloading the example's gated model metadata.
    monkeypatch.setattr("aisimulate.compiler.estimate_kv_bytes_per_token", lambda *_args, **_kwargs: 256)
    deployment = prediction_to_replay_spec(prediction).backend_deployment
    assert deployment.agg_engine_args["kv_cache_bytes_per_token"] == 256
    assert (
        deployment.agg_engine_args["g3_offload"] == G3OffloadConfig.model_validate(extension["g3_offload"]).model_dump()
    )
