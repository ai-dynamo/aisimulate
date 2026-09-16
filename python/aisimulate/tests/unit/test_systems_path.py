# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public prediction and recommendation use an isolated local systems root."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from importlib.resources import files

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from pydantic import ValidationError

from aiconfigurator_core.sdk import models, perf_database
from aiconfigurator_core.sdk.config import ModelConfig
from aiconfigurator_core.sdk.operations.fpm_forward import _CELL_MATCH_COLUMNS
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig, CoreRecommendationConfig
from aisimulate.output import write_recommendations
from aisimulate.recommend import run_recommendation
from aisimulate.runner import EngineReplayRunnerFactory

pytestmark = pytest.mark.unit

_SYSTEM = "local_fpm_test"
_VERSION = "0.0.1"


def _request() -> dict:
    return {
        "engine": {
            "mode": "aggregated",
            "model": "Qwen/Qwen3-0.6B",
            "hardware": "h200_sxm",
            "backend": "vllm",
            "context_length": 128,
            "workers": {"aggregated": {}},
        },
    }


@pytest.mark.parametrize("config_type", [CorePredictionConfig, CoreRecommendationConfig])
def test_systems_root_is_absolute_and_reloadable_before_collection(config_type, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = _request()
    raw["engine"]["systems_path"] = "local profiles"
    if config_type is CoreRecommendationConfig:
        raw["optimization"] = {"target": "throughput"}

    config = config_type.model_validate(raw)
    saved = config.model_dump(mode="json", exclude_none=True)
    monkeypatch.chdir(tmp_path.parent)

    assert saved["engine"]["systems_path"] == str(tmp_path / "local profiles")
    assert config_type.model_validate(saved).engine.systems_path == config.engine.systems_path


@pytest.mark.parametrize("config_type", [CorePredictionConfig, CoreRecommendationConfig])
@pytest.mark.parametrize("value", ["", "   ", 7, ["one"], {"choices": ["one"]}, "one,two"])
def test_systems_root_rejects_empty_or_multiple_roots(config_type, value):
    raw = _request()
    raw["engine"]["systems_path"] = value
    if config_type is CoreRecommendationConfig:
        raw["optimization"] = {"target": "throughput"}

    with pytest.raises(ValidationError, match="systems_path"):
        config_type.model_validate(raw)


@pytest.mark.parametrize("config_type", [CorePredictionConfig, CoreRecommendationConfig])
@pytest.mark.parametrize("unsupported", ["AFD", "analytical encoder pools"])
def test_systems_root_rejects_modes_that_cannot_consume_it(config_type, unsupported, tmp_path):
    raw = _request()
    if unsupported == "AFD":
        raw["engine"]["mode"] = "afd"
        raw["engine"]["workers"] = {}
        raw["engine"]["afd"] = {"phase": "both", "combined_with_pd": False, "a_batch_size": 1}
        if config_type is CorePredictionConfig:
            raw["engine"]["afd"].update(n_a_nodes=1, n_f_nodes=1, tp_a=1)
    else:
        raw["engine"]["workers"]["encoder"] = {}
        raw["traffic"] = {
            "source": {"type": "synthetic", "images": {"height": 32, "width": 32}},
            "load": {"type": "concurrency", "concurrency": 1},
            "stop": {"requests": 1},
        }
    if config_type is CoreRecommendationConfig:
        raw["optimization"] = {"target": "throughput"}

    config_type.model_validate(raw)
    raw["engine"]["systems_path"] = str(tmp_path)
    with pytest.raises(ValidationError, match=f"systems_path.*{unsupported}"):
        config_type.model_validate(raw)


@pytest.fixture
def local_profiles(tmp_path):
    """Two roots share an identity but have distinct timings and GPU memory."""
    model = models.get_model("Qwen/Qwen3-0.6B", ModelConfig(forward_model="fpm"), "vllm")
    identity = dict(zip(_CELL_MATCH_COLUMNS, model.context_ops[0]._match_identity, strict=True))
    for name in ("tp", "pp", "dp", "moe_tp", "moe_ep", "cp"):
        identity[name] = int(identity[name])
    for name in ("enable_wideep", "enable_eplb"):
        identity[name] = identity[name] == "True"
    roots = []
    for label, latency, memory in (("fast", 20.0, 16 << 30), ("slow", 200.0, 32 << 30)):
        root = tmp_path / label
        data = root / "data" / _SYSTEM / "vllm" / _VERSION
        data.mkdir(parents=True)
        system = yaml.safe_load((files("aiconfigurator_core") / "systems/h200_sxm.yaml").read_text())
        system["data_dir"] = f"data/{_SYSTEM}"
        system["gpu"]["mem_capacity"] = memory
        (root / f"{_SYSTEM}.yaml").write_text(yaml.safe_dump(system))
        rows = []
        for phase, tokens, kv in (("prefill", 16, 0), ("decode", 0, 0), ("decode", 0, 128)):
            rows.append(
                identity
                | {
                    "cell_id": phase,
                    "model_path": model.model_path,
                    "system": _SYSTEM,
                    "backend": "vllm",
                    "backend_version": _VERSION,
                    "weight_quantization": identity["gemm_quant_mode"],
                    "workload_kind": phase,
                    "batch_size": 1,
                    "total_prefill_tokens": tokens,
                    "total_kv_read_tokens": kv,
                    "partition_policy": "balanced_v1",
                    "latency_ms": latency,
                }
            )
        parquet = data / "fpm_forward_perf.parquet"
        pq.write_table(pa.Table.from_pylist(rows), parquet)
        (data / "fpm_forward_perf.metadata.json").write_text(
            json.dumps(
                {
                    "schema_name": "aic_fpm_forward_perf",
                    "schema_version": 6,
                    "coordinate_system": "iteration_totals_balanced_v1",
                    "measurement_policy": "dynamo_native_single_sample_v1",
                    "row_count": len(rows),
                    "parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
                    "system": _SYSTEM,
                    "backend": "vllm",
                    "backend_version": _VERSION,
                }
            )
        )
        roots.append(root)
    return roots


def _local_request(root, mode="aggregated") -> dict:
    raw = _request()
    raw["engine"].update(mode=mode, hardware=_SYSTEM, systems_path=str(root), backend_version=_VERSION)
    worker = {
        "scheduler": {"max_batched_tokens": 16, "max_sequences": 1},
        "kv_cache": {"prefix_caching": False},
        "timing": {"forward_model": "fpm"},
    }
    roles = ("aggregated",) if mode == "aggregated" else ("prefill", "decode")
    raw["engine"]["workers"] = {role: deepcopy(worker) for role in roles}
    raw["traffic"] = {
        "source": {"type": "synthetic", "input_tokens": 16, "output_tokens": 2},
        "load": {"type": "concurrency", "concurrency": 1},
        "stop": {"requests": 1},
    }
    return raw


def _predict(raw):
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    runner = EngineReplayRunnerFactory().create(0)
    try:
        return runner.run(spec)
    finally:
        runner.close()


@pytest.mark.parametrize("mode", ["aggregated", "disaggregated"])
def test_prediction_consumes_local_fpm_timings_without_changing_default_roots(local_profiles, mode):
    default_roots = perf_database.get_systems_paths()
    fast, slow = local_profiles

    fast_report = _predict(_local_request(fast, mode))
    slow_report = _predict(_local_request(slow, mode))
    repeat_report = _predict(_local_request(fast, mode))

    assert fast_report.metrics["completed_requests"] == slow_report.metrics["completed_requests"] == 1
    assert slow_report.metrics["mean_ttft_ms"] > 5 * fast_report.metrics["mean_ttft_ms"]
    assert repeat_report.metrics["duration_ms"] == fast_report.metrics["duration_ms"]
    assert perf_database.get_systems_paths() == default_roots
    deployment = prediction_to_replay_spec(
        CorePredictionConfig.model_validate(_local_request(fast, mode))
    ).backend_deployment
    assert deployment.performance_model_metadata
    for metadata in deployment.performance_model_metadata.values():
        assert metadata["config"]["systems_path"] == str(fast)


@pytest.mark.parametrize("timing", [{"type": "fixed", "prefill_ms": 1, "decode_ms": 2}, {"type": "polynomial"}])
def test_custom_timing_can_use_local_profiles_for_inferred_kv_capacity(local_profiles, timing):
    raw = _local_request(local_profiles[0])
    raw["engine"]["workers"]["aggregated"]["timing"] = timing

    assert _predict(raw).metrics["completed_requests"] == 1


@pytest.mark.parametrize("explicit_version", [False, True])
def test_worker_hardware_override_resolves_in_local_systems_root(local_profiles, explicit_version):
    raw = _local_request(local_profiles[0], "disaggregated")
    raw["engine"]["hardware"] = "h200_sxm"
    for worker in raw["engine"]["workers"].values():
        worker["hardware"] = _SYSTEM
    if not explicit_version:
        raw["engine"].pop("backend_version")

    assert _predict(raw).metrics["completed_requests"] == 1


def test_default_prediction_still_uses_bundled_op_level_data_after_a_local_request(local_profiles):
    local = _local_request(local_profiles[0])
    default = deepcopy(local)
    default["engine"].pop("systems_path")
    default["engine"]["hardware"] = "h200_sxm"
    default["engine"]["backend_version"] = perf_database.get_latest_database_version("h200_sxm", "vllm")
    default["engine"]["workers"]["aggregated"].pop("timing")

    before = _predict(default)
    assert _predict(local).metrics["completed_requests"] == 1
    after = _predict(default)

    assert before.metrics["completed_requests"] == after.metrics["completed_requests"] == 1
    assert after.metrics["duration_ms"] == before.metrics["duration_ms"]
    config = CorePredictionConfig.model_validate(default)
    assert "systems_path" not in config.model_dump(mode="json", exclude_none=True)["engine"]
    assert config.engine.workers.aggregated.timing.forward_model == "op_level"


@pytest.mark.parametrize("mode", ["aggregated", "disaggregated"])
def test_recommendation_resolves_local_capacity_and_exports_replayable_profiles(local_profiles, mode, tmp_path):
    default_roots = perf_database.get_systems_paths()
    candidates = []
    for root in local_profiles:
        raw = _local_request(root, mode)
        del raw["engine"]["backend_version"]
        raw["traffic"]["load"] = {"type": "kv_capacity_fraction", "fraction": 0.000001}
        raw["optimization"] = {
            "target": "throughput",
            "constraints": {"max_candidate_gpus": 1 if mode == "aggregated" else 2},
        }
        raw["optimizer"] = {"algorithm": "random", "max_trials": 1, "parallelism": 1}
        for worker in raw["engine"]["workers"].values():
            worker["kv_cache"].update(block_size=64, capacity={"memory_fraction": 0.9})
            worker["parallelism"] = {
                "preset": [
                    {"replicas": 1, "tensor": 1, "pipeline": 1, "attention_data": 1, "moe_tensor": 1, "moe_expert": 1}
                ]
            }

        result = run_recommendation(
            CoreRecommendationConfig.model_validate(raw),
            stack="engine",
            runner_factory=EngineReplayRunnerFactory(),
            show_progress=False,
        )

        assert result.counts.feasible == 1, [record.reason for record in result.candidates]
        candidate = result.selected_candidates[0]
        assert candidate.config["systems_path"] == str(root)
        assert candidate.prediction_config["engine"]["systems_path"] == str(root)
        assert candidate.prediction_config["engine"]["backend_version"] == _VERSION
        assert result.candidates[0].provenance.performance_data
        for metadata in result.candidates[0].provenance.performance_data:
            assert metadata["config"]["systems_path"] == str(root)
        [saved] = write_recommendations(tmp_path / f"export-{root.name}", [candidate.prediction_config])
        rerun = subprocess.run(
            [
                sys.executable,
                "-m",
                "aisimulate",
                "predict",
                "--config",
                str(saved),
                "--output-dir",
                str(tmp_path / f"predict-{root.name}"),
                "--format",
                "json",
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert rerun.returncode == 0, rerun.stderr
        assert json.loads(rerun.stdout)["mean_ttft_ms"] == candidate.metrics["mean_ttft_ms"]
        candidates.append(candidate)

    fast, slow = candidates
    assert slow.metrics["mean_ttft_ms"] > 5 * fast.metrics["mean_ttft_ms"]
    assert slow.config["kv_load_capacity_tokens"] > fast.config["kv_load_capacity_tokens"]
    assert perf_database.get_systems_paths() == default_roots
