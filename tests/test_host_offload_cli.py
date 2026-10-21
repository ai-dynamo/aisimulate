# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

pytest.importorskip(
    "aisimulate.runner",
    reason="standalone AISimulate runtime is not installed",
    exc_type=ImportError,
)

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


def _mtp_host_engine(*, mode="aggregated", expected=1.5) -> dict:
    engine = _prediction_engine(mode=mode)
    engine.update(
        model="meta-llama/Meta-Llama-3.1-8B",
        backend_version="0.24.0",
        estimation_mode="op_level",
        speculation={"kind": "mtp", "params": {"depth": 2}, "expected_accepted_tokens": expected, "seed": 17},
    )
    for worker in engine["workers"].values():
        worker["timing"] = {"type": "default"}
        worker["kv_cache"].update(bytes_per_token=250_000, host_offload=_host_offload())
    return engine


@pytest.mark.parametrize("agentic", [False, True])
def test_canonical_mtp_host_config_roundtrips_and_preserves_cost_identity(agentic) -> None:
    raw = {"engine": _mtp_host_engine()}
    if agentic:
        raw["traffic"] = {
            "source": {"type": "trace", "paths": ["self-authored-weka.jsonl"], "format": "weka"},
            "load": {"type": "trace_timestamps", "agentic_lanes": 1},
        }
    config = CorePredictionConfig.model_validate(raw)
    restored = CorePredictionConfig.model_validate_json(config.model_dump_json())
    assert restored == config
    runtime = _RecordingRuntime()
    EngineReplayRunnerFactory(runtime=runtime).create(0).run(prediction_to_replay_spec(restored))
    rank = runtime.execution_spec["spec"]["engine"]["rank"]
    assert rank["native_host_offload"] == _host_offload()
    assert rank["aic_nextn"] == 2
    assert [float(rate) for rate in rank["aic_nextn_accept_rates"].split(",")] == [1.0, 0.5]
    assert rank["aic_mtp_seed"] == 17
    assert rank["timing_model"]["config"]["speculation"] == {"kind": "mtp", "params": {"depth": 2}}
    assert rank["timing_model"]["config"]["backend_version"] == "0.24.0"


@pytest.mark.parametrize(
    "change,error",
    [
        ({"parallelism": {"attention_data": 2}}, "attention_data=1"),
        ({"kv_cache": {"host_offload": {"scope": "cluster_shared", "num_host_blocks": 16}}}, "dp_rank_local"),
        ({"kv_cache": {"g3_offload": {"scope": "worker_local", "num_g3_blocks": 16}}}, "no g3_offload"),
        ({"kv_cache": {"prefix_caching": False}}, "prefix_caching=true"),
        ({"kv_cache": {"capacity": {"type": "default"}}}, "fixed KV capacity"),
        ({"timing": {"type": "fixed", "prefill_ms": 1, "decode_ms": 1}}, "default timing"),
    ],
)
def test_canonical_mtp_host_config_preserves_composition_limits(change, error) -> None:
    engine = _mtp_host_engine()
    worker = engine["workers"]["aggregated"]
    for section, values in change.items():
        worker[section].update(values)
    with pytest.raises(ValidationError, match=error):
        CorePredictionConfig.model_validate({"engine": engine})


def test_canonical_mtp_host_config_rejects_disaggregated_workers() -> None:
    with pytest.raises(ValidationError, match="aggregated vllm"):
        CorePredictionConfig.model_validate({"engine": _mtp_host_engine(mode="disaggregated")})


@pytest.mark.parametrize("expected", [0.0, 2.0])
def test_canonical_mtp_native_replay_restores_an_evicted_host_prefix(expected) -> None:
    from aisimulate import _runtime

    engine = _mtp_host_engine(expected=expected)
    worker = engine["workers"]["aggregated"]
    worker["scheduler"].update(max_batched_tokens=16, max_sequences=1)
    worker["kv_cache"].update(
        block_size=4,
        capacity={"type": "fixed", "blocks": 4},
        host_offload={"num_host_blocks": 16, "d2h_bandwidth_gbps": 1.0, "h2d_bandwidth_gbps": 1.0},
    )
    config = CorePredictionConfig.model_validate({"engine": engine})
    recording = _RecordingRuntime()
    EngineReplayRunnerFactory(runtime=recording).create(0).run(prediction_to_replay_spec(config))
    # Use the public compiler/runner's complete canonical engine identity with
    # explicit request IDs to prove eviction, host restoration and MTP output.
    payload = {
        "version": 1,
        "topology": {"kind": "aggregated", "workers": {"initial_workers": 1}},
        "engine": recording.execution_spec["spec"]["engine"],
        "record_per_request": True,
        "requests": [
            {
                "id": request_id,
                "arrival_time_ms": arrival,
                "input_tokens": 8,
                "input_token_ids": tokens,
                "output_tokens": 5,
            }
            for request_id, arrival, tokens in [
                ("seed", 0.0, list(range(1, 9))),
                ("evict", 1000.0, list(range(21, 29))),
                ("restore", 2000.0, list(range(1, 9))),
            ]
        ],
    }
    report = json.loads(_runtime.run_replay_json(json.dumps(payload)))
    repeated = json.loads(_runtime.run_replay_json(json.dumps(payload)))
    assert report["completed_requests"] == 3
    assert report["total_output_tokens"] == 15
    records = {record["request_id"]: record for record in report["per_request"]}
    repeated_records = {record["request_id"]: record for record in repeated["per_request"]}
    # Replay assigns new internal UUIDs; authored request identity, progress,
    # timings and every other recorded field must remain deterministic.
    assert {
        request_id: {key: value for key, value in record.items() if key != "uuid"}
        for request_id, record in records.items()
    } == {
        request_id: {key: value for key, value in record.items() if key != "uuid"}
        for request_id, record in repeated_records.items()
    }
    assert records.keys() == {"seed", "evict", "restore"}
    assert all(record["output_length"] == record["requested_output_length"] == 5 for record in records.values())
    restored = records["restore"]
    assert restored["first_admission_g1_reused_input_tokens"] == 0
    assert restored["first_admission_host_reused_input_tokens"] > 0
    assert restored["first_admit_ms"] > restored["arrival_time_ms"]


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
    assert rank["kv_cache_bytes_per_token"] == 131_072
    assert rank["native_host_offload"] == {
        "num_host_blocks": 4096,
        "d2h_bandwidth_gbps": 32.0,
        "h2d_bandwidth_gbps": 32.0,
    }


def test_predict_cli_reaches_native_rank_host_offload(tmp_path, monkeypatch, capsys) -> None:
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
    assert rank["kv_cache_bytes_per_token"] == 131_072
    assert rank["native_host_offload"] == _host_offload()


@pytest.mark.parametrize(
    "scopes",
    [("default",), ("dp_rank_local",), ("cluster_shared",)]
    + [
        (prefill, decode)
        for prefill in (None, "dp_rank_local", "cluster_shared")
        for decode in (None, "dp_rank_local", "cluster_shared")
        if prefill is not None or decode is not None
    ],
)
def test_agentic_g2_public_config_preserves_role_scopes_through_native_payload(scopes) -> None:
    mode = "aggregated" if len(scopes) == 1 else "disaggregated"
    engine = _prediction_engine(mode=mode)
    if mode == "disaggregated":
        engine["kv_transfer"] = {"bytes_per_token": 131_072}
    roles = ("aggregated",) if mode == "aggregated" else ("prefill", "decode")
    for role, scope in zip(roles, scopes, strict=True):
        worker = engine["workers"][role]
        worker["parallelism"]["tensor"] = 1
        if scope is not None:
            offload = _host_offload()
            if scope != "default":
                offload["scope"] = scope
            worker["kv_cache"].update(bytes_per_token=131_072, host_offload=offload)
    public = CorePredictionConfig.model_validate(
        {
            "engine": engine,
            "traffic": {
                "source": {"type": "trace", "paths": ["self-authored-weka.jsonl"], "format": "weka"},
                "load": {"type": "trace_timestamps", "agentic_lanes": 1},
            },
        }
    )
    runtime = _RecordingRuntime()
    EngineReplayRunnerFactory(runtime=runtime).create(0).run(prediction_to_replay_spec(public))
    native = runtime.execution_spec["spec"]["engine"]
    for role, scope in zip(roles, scopes, strict=True):
        rank = native["rank"] if role == "aggregated" else native[role]["rank"]
        if scope is None:
            assert rank.get("native_host_offload") is None
        else:
            offload = rank["native_host_offload"]
            assert offload.get("scope", "dp_rank_local") == ("dp_rank_local" if scope == "default" else scope)
            assert {key: offload[key] for key in _host_offload()} == _host_offload()
            assert rank["kv_cache_bytes_per_token"] == 131_072
            if scope == "cluster_shared":
                assert offload["kv_layout_id"]


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

    assert rank["kv_cache_bytes_per_token"] == 444_444
    assert rank["native_host_offload"] == _host_offload()
    assert calls == [
        {
            "tp_size": 4,
            "pp_size": 1,
            "moe_tp_size": 1,
            "moe_ep_size": 1,
            "kvcache_quant_mode": None,
        }
    ]


def test_prediction_auto_transfer_geometry_uses_prefill_source_shape(
    monkeypatch,
) -> None:
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
    deployment = prediction_to_replay_spec(CorePredictionConfig.model_validate({"engine": engine})).backend_deployment

    assert deployment.prefill_engine_args["kv_transfer_bytes_per_token"] == 20_001
    assert deployment.decode_engine_args["kv_transfer_bytes_per_token"] == 20_001
    assert [call["tp_size"] for call in calls] == [2]
    assert deployment.prefill_engine_args["kv_transfer_bandwidth"] == 400.0
    assert deployment.decode_engine_args["kv_transfer_timing_mode"] == "destination_missing"


def test_prediction_transfer_and_cache_geometry_are_independent(monkeypatch) -> None:
    engine = _prediction_engine(mode="disaggregated")
    engine["kv_transfer"] = {
        "bytes_per_token": 333,
        "bandwidth_gb_per_second": 400.0,
    }
    engine["workers"]["prefill"]["kv_cache"]["bytes_per_token"] = 111
    engine["workers"]["decode"]["kv_cache"]["bytes_per_token"] = 222
    monkeypatch.setattr(
        "aisimulate.compiler.estimate_kv_bytes_per_token",
        lambda *_args, **_kwargs: pytest.fail("explicit geometry must not be estimated"),
    )

    deployment = prediction_to_replay_spec(CorePredictionConfig.model_validate({"engine": engine})).backend_deployment

    assert deployment.prefill_engine_args["kv_transfer_bytes_per_token"] == 333
    assert deployment.decode_engine_args["kv_transfer_bytes_per_token"] == 333
    assert "kv_cache_bytes_per_token" not in deployment.prefill_engine_args
    assert "kv_cache_bytes_per_token" not in deployment.decode_engine_args


def test_explicit_legacy_kv_transfer_geometry_yaml_remains_supported(tmp_path) -> None:
    engine = _prediction_engine(mode="disaggregated")
    engine["kv_transfer"] = {
        "bytes_per_token": 333,
        "bandwidth_gb_per_second": 400.0,
    }
    path = tmp_path / "legacy-kv-transfer.yaml"
    path.write_text(yaml.safe_dump({"engine": engine}), encoding="utf-8")

    deployment = prediction_to_replay_spec(CorePredictionConfig.from_yaml(path)).backend_deployment

    assert deployment.prefill_engine_args["kv_transfer_bytes_per_token"] == 333
    assert deployment.decode_engine_args["kv_transfer_bytes_per_token"] == 333


def test_recommendation_keeps_transfer_and_cache_geometry_independent() -> None:
    engine = _prediction_engine(mode="disaggregated")
    for worker in engine["workers"].values():
        worker["parallelism"] = {"preset": False, **worker["parallelism"]}
    engine["kv_transfer"] = {
        "bytes_per_token": 333,
        "bandwidth_gb_per_second": 400.0,
    }
    engine["workers"]["prefill"]["kv_cache"]["bytes_per_token"] = 111
    engine["workers"]["decode"]["kv_cache"]["bytes_per_token"] = 222
    config = CoreRecommendationConfig.model_validate(
        {
            "engine": engine,
            "optimization": {"constraints": {"max_candidate_gpus": 8}},
        }
    )

    reparsed = CoreRecommendationConfig.model_validate(config.model_dump(mode="json"))
    space = recommendation_to_sweeper(reparsed).search_space

    assert reparsed.engine.kv_transfer.bytes_per_token == 333
    assert reparsed.engine.workers.prefill.kv_cache.bytes_per_token == 111
    assert reparsed.engine.workers.decode.kv_cache.bytes_per_token == 222
    assert space.kv_transfer_bytes_per_token == 333
    assert space.prefill_kv_bytes_per_token == 111
    assert space.decode_kv_bytes_per_token == 222


def test_recommendation_carries_fixed_host_descriptor_without_search_dimension() -> None:
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
        parallel_config=ReplicaParallelConfig(ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1),
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
            lambda engine: engine["workers"]["aggregated"]["kv_cache"].update(prefix_caching=False),
            "prefix_caching=true",
        ),
    ],
)
def test_prediction_host_offload_rejects_unsupported_runtime_scope(mutation, message: str) -> None:
    engine = _prediction_engine()
    engine["workers"]["aggregated"]["kv_cache"]["host_offload"] = _host_offload()
    mutation(engine)

    with pytest.raises(ValidationError, match=message):
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


def test_native_speculative_field_is_not_part_of_public_engine_schema() -> None:
    engine = _prediction_engine()
    engine["aic_nextn"] = 1

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CorePredictionConfig.model_validate({"engine": engine})
