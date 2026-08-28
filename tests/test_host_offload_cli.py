# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest
import yaml
from pydantic import ValidationError

from aisimulate import main as cli
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config.cli import CorePredictionConfig, CoreRecommendationConfig
from aisimulate.recommend import _candidate_prediction, recommendation_to_sweeper
from aisimulate.runner import EngineReplayRunnerFactory
from aisimulate.sweeper.deploy import build_backend_deployment
from aisimulate.sweeper.parallel_enum import ParallelShape, ReplicaParallelConfig
from aisimulate.sweeper.replay import ReplaySpec
from aisimulate.sweeper.sample import unroll_sample


def _prediction_worker(*, tp: int = 1, attention_dp: int = 1) -> dict:
    return {
        "parallelism": {
            "replicas": 1,
            "tensor": tp,
            "pipeline": 1,
            "attention_data": attention_dp,
            "moe_tensor": 1,
            "moe_expert": 1,
        },
        "scheduler": {"max_batched_tokens": 8192, "max_sequences": 4},
        "kv_cache": {
            "block_size": 16,
            "prefix_caching": True,
            "capacity": {"type": "fixed", "blocks": 128},
        },
        "timing": {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0},
    }


def _host_offload() -> dict:
    return {
        "num_host_blocks": 4096,
        "d2h_bandwidth_gbps": 7.0,
        "h2d_bandwidth_gbps": 38.0,
    }


def _prediction_engine(*, mode: str = "aggregated") -> dict:
    workers = (
        {"aggregated": _prediction_worker()}
        if mode == "aggregated"
        else {
            "prefill": _prediction_worker(tp=2),
            "decode": _prediction_worker(tp=1),
        }
    )
    return {
        "mode": mode,
        "model": "example/model",
        "hardware": "h200_sxm",
        "backend": "vllm",
        "context_length": 4096,
        "workers": workers,
    }


def _recommendation_engine() -> dict:
    worker = _prediction_worker()
    worker["parallelism"] = {"preset": False, **worker["parallelism"]}
    return {
        "mode": "aggregated",
        "model": "example/model",
        "hardware": "h200_sxm",
        "backend": "vllm",
        "context_length": 4096,
        "workers": {"aggregated": worker},
    }


class _RecordingRuntime:
    def __init__(self) -> None:
        self.execution_spec = None

    def run_replay_json(self, execution_spec_json: str) -> str:
        self.execution_spec = json.loads(execution_spec_json)
        return json.dumps(
            {
                "duration_ms": 1.0,
                "output_throughput_tok_s": 1.0,
                "gpu_hours": 0.0,
                "completed_requests": 1,
            }
        )


def test_predict_yaml_accepts_canonical_host_offload_schema(tmp_path) -> None:
    engine = _prediction_engine()
    cache = engine["workers"]["aggregated"]["kv_cache"]
    cache["bytes_per_token"] = 131_072
    cache["host_offload"] = {"num_host_blocks": 4096}
    path = tmp_path / "host-offload.yaml"
    path.write_text(yaml.safe_dump({"engine": engine}), encoding="utf-8")

    config = CorePredictionConfig.from_yaml(path)
    rank = prediction_to_replay_spec(config).backend_deployment.agg_engine_args

    assert config.engine.workers.aggregated.kv_cache.bytes_per_token == 131_072
    assert rank["kv_bytes_per_token"] == 131_072
    assert rank["native_host_offload"] == {
        "num_host_blocks": 4096,
        "d2h_bandwidth_gbps": 32.0,
        "h2d_bandwidth_gbps": 32.0,
    }


def test_predict_cli_reaches_native_rank_host_offload(
    tmp_path, monkeypatch, capsys
) -> None:
    engine = _prediction_engine()
    cache = engine["workers"]["aggregated"]["kv_cache"]
    cache["bytes_per_token"] = 131_072
    cache["host_offload"] = _host_offload()
    path = tmp_path / "host-offload.yaml"
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
    assert rank["kv_bytes_per_token"] == 131_072
    assert rank["native_host_offload"] == _host_offload()


def test_prediction_host_offload_auto_geometry_uses_aggregated_shape(
    monkeypatch,
) -> None:
    engine = _prediction_engine()
    engine["workers"]["aggregated"]["parallelism"]["tensor"] = 4
    engine["workers"]["aggregated"]["kv_cache"]["host_offload"] = _host_offload()
    calls: list[dict] = []

    def estimate(_model: str, **shape) -> int:
        calls.append(shape)
        return 444_444

    monkeypatch.setattr("aisimulate.compiler.estimate_kv_bytes_per_token", estimate)
    rank = prediction_to_replay_spec(
        CorePredictionConfig.model_validate({"engine": engine})
    ).backend_deployment.agg_engine_args

    assert rank["kv_bytes_per_token"] == 444_444
    assert rank["native_host_offload"] == _host_offload()
    assert calls == [
        {
            "tp_size": 4,
            "pp_size": 1,
            "moe_tp_size": 1,
            "moe_ep_size": 1,
        }
    ]


def test_prediction_auto_geometry_resolves_per_pd_role(monkeypatch) -> None:
    engine = _prediction_engine(mode="disaggregated")
    engine["kv_transfer"] = {
        "bandwidth_gb_per_second": 400.0,
        "timing_mode": "destination_missing",
    }
    calls: list[dict] = []

    def estimate(_model: str, **shape) -> int:
        calls.append(shape)
        return 10_000 * shape["tp_size"] + shape["pp_size"]

    monkeypatch.setattr("aisimulate.compiler.estimate_kv_bytes_per_token", estimate)
    deployment = prediction_to_replay_spec(
        CorePredictionConfig.model_validate({"engine": engine})
    ).backend_deployment

    assert deployment.prefill_engine_args["kv_bytes_per_token"] == 20_001
    assert deployment.decode_engine_args["kv_bytes_per_token"] == 10_001
    assert [call["tp_size"] for call in calls] == [2, 1]
    assert deployment.prefill_engine_args["kv_transfer_bandwidth"] == 400.0
    assert (
        deployment.decode_engine_args["kv_transfer_timing_mode"]
        == "destination_missing"
    )


def test_prediction_explicit_geometry_passes_through_per_pd_role(monkeypatch) -> None:
    engine = _prediction_engine(mode="disaggregated")
    engine["kv_transfer"] = {"bandwidth_gb_per_second": 400.0}
    engine["workers"]["prefill"]["kv_cache"]["bytes_per_token"] = 111
    engine["workers"]["decode"]["kv_cache"]["bytes_per_token"] = 222
    monkeypatch.setattr(
        "aisimulate.compiler.estimate_kv_bytes_per_token",
        lambda *_args, **_kwargs: pytest.fail(
            "explicit geometry must not be estimated"
        ),
    )

    deployment = prediction_to_replay_spec(
        CorePredictionConfig.model_validate({"engine": engine})
    ).backend_deployment

    assert deployment.prefill_engine_args["kv_bytes_per_token"] == 111
    assert deployment.decode_engine_args["kv_bytes_per_token"] == 222


def test_explicit_legacy_kv_transfer_geometry_yaml_remains_supported(tmp_path) -> None:
    engine = _prediction_engine(mode="disaggregated")
    engine["kv_transfer"] = {
        "bytes_per_token": 333,
        "bandwidth_gb_per_second": 400.0,
    }
    path = tmp_path / "legacy-kv-transfer.yaml"
    path.write_text(yaml.safe_dump({"engine": engine}), encoding="utf-8")

    deployment = prediction_to_replay_spec(
        CorePredictionConfig.from_yaml(path)
    ).backend_deployment

    assert deployment.prefill_engine_args["kv_bytes_per_token"] == 333
    assert deployment.decode_engine_args["kv_bytes_per_token"] == 333


def test_duplicate_legacy_and_canonical_geometry_is_rejected() -> None:
    engine = _prediction_engine(mode="disaggregated")
    engine["kv_transfer"] = {"bytes_per_token": 333}
    engine["workers"]["prefill"]["kv_cache"]["bytes_per_token"] = 111

    with pytest.raises(ValidationError, match="explicitly configured in both"):
        CorePredictionConfig.model_validate({"engine": engine})


def test_recommendation_carries_fixed_host_descriptor_without_search_dimension() -> (
    None
):
    engine = _recommendation_engine()
    cache = engine["workers"]["aggregated"]["kv_cache"]
    cache["bytes_per_token"] = 131_072
    cache["host_offload"] = _host_offload()
    config = CoreRecommendationConfig.model_validate(
        {
            "engine": engine,
            "optimization": {"constraints": {"max_candidate_gpus": 8}},
        }
    )

    space = recommendation_to_sweeper(config).search_space

    assert space.deployment_mode == ["agg"]
    assert space.backend == ["vllm"]
    assert space.agg_kv_bytes_per_token == 131_072
    assert space.agg_native_host_offload == _host_offload()
    assert "host" not in space.engine_float_ranges


def test_recommendation_materializes_concrete_host_offload_prediction() -> None:
    engine = _recommendation_engine()
    cache = engine["workers"]["aggregated"]["kv_cache"]
    cache["bytes_per_token"] = 131_072
    cache["host_offload"] = _host_offload()
    config = CoreRecommendationConfig.model_validate(
        {
            "engine": engine,
            "optimization": {"constraints": {"max_candidate_gpus": 8}},
        }
    )
    smart = recommendation_to_sweeper(config)
    sample = unroll_sample(
        search_space=smart.search_space,
        selection={
            "deployment_mode": "agg",
            "backend": "vllm",
            "agg_max_num_batched_tokens": 8192,
            "agg_max_num_seqs": 4,
        },
        parallel_config=ReplicaParallelConfig(
            ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1
        ),
    )
    deployment = build_backend_deployment(sample, backend_version="test")
    prediction = _candidate_prediction(
        config,
        sample,
        ReplaySpec(
            backend_deployment=deployment,
            workload={},
            goal={},
        ),
        adapter_sections={},
    )

    generated_cache = prediction["engine"]["workers"]["aggregated"]["kv_cache"]
    assert generated_cache["bytes_per_token"] == 131_072
    assert generated_cache["host_offload"] == _host_offload()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda engine: engine.update(backend="sglang"), "backend=vllm"),
        (
            lambda engine: engine["workers"]["aggregated"]["kv_cache"].update(
                prefix_caching=False
            ),
            "prefix_caching=true",
        ),
        (
            lambda engine: engine["workers"]["aggregated"].update(
                parallelism={
                    **engine["workers"]["aggregated"]["parallelism"],
                    "attention_data": 2,
                }
            ),
            "attention_data=1",
        ),
    ],
)
def test_prediction_host_offload_rejects_unsupported_runtime_scope(
    mutation, message: str
) -> None:
    engine = _prediction_engine()
    engine["workers"]["aggregated"]["kv_cache"]["host_offload"] = _host_offload()
    mutation(engine)

    with pytest.raises(ValidationError, match=message):
        CorePredictionConfig.model_validate({"engine": engine})


def test_prediction_host_offload_rejects_disaggregated_role() -> None:
    engine = _prediction_engine(mode="disaggregated")
    engine["workers"]["prefill"]["kv_cache"]["host_offload"] = _host_offload()

    with pytest.raises(ValidationError, match="aggregated worker"):
        CorePredictionConfig.model_validate({"engine": engine})


def test_recommendation_host_offload_rejects_mixed_backend_domain() -> None:
    engine = _recommendation_engine()
    engine["backend"] = {"choices": ["vllm", "sglang"]}
    engine["workers"]["aggregated"]["kv_cache"]["host_offload"] = _host_offload()

    with pytest.raises(ValidationError, match="concrete backend=vllm"):
        CoreRecommendationConfig.model_validate(
            {
                "engine": engine,
                "optimization": {"constraints": {"max_candidate_gpus": 8}},
            }
        )


def test_recommendation_host_offload_rejects_attention_dp_domain() -> None:
    engine = _recommendation_engine()
    engine["workers"]["aggregated"]["parallelism"]["attention_data"] = {
        "choices": [1, 2]
    }
    engine["workers"]["aggregated"]["kv_cache"]["host_offload"] = _host_offload()

    with pytest.raises(ValidationError, match="fixed parallelism"):
        CoreRecommendationConfig.model_validate(
            {
                "engine": engine,
                "optimization": {"constraints": {"max_candidate_gpus": 8}},
            }
        )


def test_native_speculative_field_is_not_part_of_public_engine_schema() -> None:
    engine = _prediction_engine()
    engine["aic_nextn"] = 1

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CorePredictionConfig.model_validate({"engine": engine})
