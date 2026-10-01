# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small offline functionality check; this does not run an optimization sweep."""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import math
import tempfile
from pathlib import Path


def main() -> None:
    import dynamo._core
    import jax
    from dynamo.llm import AisPerfConfig, KvRouterConfig
    from dynamo.mocker import MockEngineArgs
    from dynamo.planner.config.planner_config import PlannerConfig
    from dynamo.planner.core.load.predictors import LOAD_PREDICTORS
    from dynamo.planner.simulation.load_predictor import (
        LOAD_PREDICTOR_PRESETS,
        complete_predictor_preset,
    )
    from dynamo.replay import run_synthetic_trace_replay
    from dynamo.replay.simulation import DynamoReplayRunnerFactory
    from transformers import AutoConfig, AutoTokenizer
    from vizier import pyvizier as vz
    from vizier.service import clients
    from vizier.service import pyvizier as service_vz

    expected = {
        "aisimulate": "0.13.0.dev202609300000000061",
        "ai-dynamo": "1.6.0",
        "ai-dynamo-runtime": "1.6.0",
        "google-vizier": "0.1.21",
        "jax": "0.4.38",
        "jaxlib": "0.4.38",
    }
    for name, version in expected.items():
        assert metadata.version(name) == version, (name, metadata.version(name))
    binding_hash = hashlib.sha256(Path(dynamo._core.__file__).read_bytes()).hexdigest()
    assert binding_hash == "2f47cbb1def5e6b71af0271f0e189dd00e65ee739e801232928ce6dd2254277b"
    assert all(device.platform == "cpu" for device in jax.devices())
    entry_points = {
        group: [ep.name for ep in metadata.entry_points(group=group)]
        for group in ["aisimulate.config_adapters", "aisimulate.runner_factories"]
    }
    assert {"dynamo.router", "dynamo.planner"} <= set(entry_points["aisimulate.config_adapters"])
    assert "dynamo" in entry_points["aisimulate.runner_factories"]
    for group in entry_points:
        for entry_point in metadata.entry_points(group=group):
            entry_point.load()
    capabilities = DynamoReplayRunnerFactory().capabilities()
    assert len(capabilities.supported_hooks) == 2
    print(json.dumps({"phase": "versions", "versions": expected, "binding_sha256": binding_hash, "entry_points": entry_points}), flush=True)

    model = "deepseek-ai/DeepSeek-V3"
    config = AutoConfig.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    assert config.model_type == "deepseek_v3"
    assert tokenizer.encode("Offline environment smoke")
    print(json.dumps({"phase": "hf_metadata", "model": model, "model_type": config.model_type, "tokenizer": type(tokenizer).__name__}), flush=True)

    forecasts = {}
    for name in LOAD_PREDICTOR_PRESETS:
        fields = complete_predictor_preset(name)
        predictor = LOAD_PREDICTORS[fields["load_predictor"]](
            PlannerConfig(mode="disagg", throughput_adjustment_interval_seconds=5, **fields)
        )
        for value in [10.0, 12.0, 11.0, 14.0, 13.0, 15.0, 16.0, 14.0]:
            predictor.add_data_point(value)
        forecast = predictor.predict_next()
        assert math.isfinite(forecast) and forecast >= 0, (name, forecast)
        if fields["load_predictor"] == "arima":
            assert predictor.model is not None, "ARIMA must fit rather than silently fall back"
        forecasts[name] = forecast
    print(json.dumps({"phase": "load_predictors", "forecasts": forecasts}), flush=True)

    def perf(role):
        return {
            "model": model,
            "system": "h200_sxm",
            "backend": "vllm",
            "backend_version": "0.24.0",
            "worker_type": role,
            "tp": 8,
            "pp": 1,
            "attention_dp": 1,
            "moe_tp_size": 8,
            "moe_ep_size": 1,
            "fmha_quant_mode": "bfloat16",
            "kvcache_quant_mode": "fp8",
            "estimation_mode": "op_level",
        }

    def engine(role):
        return MockEngineArgs(
            worker_type=role,
            engine_type="vllm",
            ais_perf_config=perf(role),
            block_size=64,
            num_gpu_blocks=1000,
            max_num_batched_tokens=2048,
            max_num_seqs=32,
            max_model_len=8192,
        )

    with tempfile.TemporaryDirectory(prefix="ais-environment-smoke-") as tmp:
        history = Path(tmp) / "history.jsonl"
        history.write_text("".join(json.dumps({"timestamp": i * 1000, "input_length": 128 + i, "output_length": 8, "hash_ids": [i]}) + "\n" for i in range(40)))
        for mode in ["static", "router", "planner"]:
            kwargs = {}
            if mode != "static":
                kwargs.update(
                    router_mode="kv_router",
                    router_config=KvRouterConfig(router_prefill_load_model="ais"),
                    ais_perf_config=AisPerfConfig(config=perf("prefill")),
                )
            if mode == "planner":
                kwargs.update(
                    planner_config={
                        "mode": "disagg",
                        "enable_load_scaling": True,
                        "enable_throughput_scaling": True,
                        "load_predictor": "constant",
                        "load_predictor_warmup_trace": str(history),
                        "throughput_adjustment_interval_seconds": 5,
                        "load_adjustment_interval_seconds": 1,
                        "min_endpoint": 1,
                        "max_gpu_budget": 32,
                        "optimization_target": "sla",
                        "served_model_name": model,
                    },
                    benchmark_granularity=2,
                )
            report = run_synthetic_trace_replay(
                input_tokens=128,
                output_tokens=8,
                request_count=8,
                prefill_engine_args=engine("prefill"),
                decode_engine_args=engine("decode"),
                num_prefill_workers=1,
                num_decode_workers=1,
                replay_mode="offline",
                arrival_interval_ms=1000,
                capture_per_request=True,
                **kwargs,
            )
            assert report.summary["completed_requests"] == 8, report.summary
            assert report.summary["mean_ttft_ms"] > 0
            assert report.summary["mean_tpot_ms"] > 0
            planner_details = {}
            if mode == "planner":
                assert report.planner is not None
                assert report.planner.total_ticks > 0
                bootstrap = report.planner.metadata["bootstrap"]
                assert bootstrap["status"] == "installed", bootstrap
                assert bootstrap["prefill_fpm_count"] > 0 and bootstrap["decode_fpm_count"] > 0
                planner_details = {"ticks": report.planner.total_ticks, "bootstrap": bootstrap}
            print(json.dumps({"phase": "replay", "mode": mode, "completed_requests": report.summary["completed_requests"], "mean_ttft_ms": report.summary["mean_ttft_ms"], "mean_tpot_ms": report.summary["mean_tpot_ms"], "history_bootstrap": mode == "planner", "planner": planner_details}), flush=True)

    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param("x", 0.0, 1.0)
    problem.metric_information.append(vz.MetricInformation(name="value", goal=vz.ObjectiveMetricGoal.MAXIMIZE))
    study_config = service_vz.StudyConfig.from_problem(problem)
    study_config.algorithm = "GAUSSIAN_PROCESS_BANDIT"
    clients.environment_variables.servicer_kwargs["database_url"] = "sqlite:///:memory:"
    study = clients.Study.from_study_config(
        study_config,
        owner="environment-smoke",
        study_id="offline-cpu",
    )
    trial = study.suggest(count=1)[0]
    trial.complete(vz.Measurement({"value": 1.0}))
    print(json.dumps({"phase": "vizier", "algorithm": "GAUSSIAN_PROCESS_BANDIT", "suggest_complete": "passed"}), flush=True)
    print(json.dumps({"status": "passed", "scope": "CPU-only offline smoke, 24 synthetic requests, 11 predictor presets, no search or GPU workload"}), flush=True)


if __name__ == "__main__":
    main()
