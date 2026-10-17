# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

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
from aisimulate.config.engine import KvCachePredictionConfig
from aisimulate.recommend import _candidate_prediction, recommendation_to_sweeper
from aisimulate.runner import EngineReplayRunnerFactory
from aisimulate.sweeper.deploy import build_backend_deployment
from aisimulate.sweeper.parallel_enum import ParallelShape, ReplicaParallelConfig
from aisimulate.sweeper.replay import ReplayOutputRequirements, ReplaySpec
from aisimulate.sweeper.sample import unroll_sample

ROOT = Path(__file__).resolve().parents[1]


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


def _kvcr_defaults() -> dict:
    return {"latency_to_first_byte_ms": 1.0, "bandwidth_gbps": 50.0}


@pytest.mark.parametrize(
    ("cache", "message"),
    [
        ({"kvcr": {}}, "kvcr requires host_offload with scope dp_rank_local"),
        (
            {"host_offload": {"num_host_blocks": 8, "scope": "cluster_shared"}, "kvcr": {}},
            "kvcr requires host_offload with scope dp_rank_local",
        ),
        (
            {
                "host_offload": {"num_host_blocks": 8},
                "g3_offload": {"scope": "worker_local", "num_g3_blocks": 8},
                "kvcr": {},
            },
            "kvcr cannot be combined with g3_offload",
        ),
        ({"host_offload": {"num_host_blocks": 8}, "kvcr": {"bandwidth_gbps": -1}}, "greater than or equal to 0"),
        (
            {"host_offload": {"num_host_blocks": 8}, "kvcr": {"policy": "x"}},
            r"kvcr\.policy\s+Extra inputs are not permitted",
        ),
    ],
)
def test_kvcr_requires_private_host_offload_and_no_g3(cache, message) -> None:
    with pytest.raises(ValueError, match=message):
        KvCachePredictionConfig.model_validate(cache)


def test_kvcr_defaults_and_overrides() -> None:
    cache = {"host_offload": {"num_host_blocks": 8}, "kvcr": {}}
    assert KvCachePredictionConfig.model_validate(cache).kvcr.model_dump() == _kvcr_defaults()
    cache["kvcr"] = {"latency_to_first_byte_ms": 0, "bandwidth_gbps": 0}
    assert KvCachePredictionConfig.model_validate(cache).kvcr.model_dump() == {
        "latency_to_first_byte_ms": 0.0,
        "bandwidth_gbps": 0.0,
    }


def test_kvcr_defaults_match_the_rust_runtime() -> None:
    from aisimulate import _runtime
    from aisimulate.config.engine import KvcrConfig

    # Rust fills the fields an empty `kvcr` block omits; Python must default to the same values.
    role = {"rank": {"kv_cache_bytes_per_token": 256, "native_host_offload": {"num_host_blocks": 8}, "kvcr": {}}}
    launch = json.loads(_runtime.engine_launch_from_replay_role_json(json.dumps(role)))
    assert launch["engine"]["kvcr"] == KvcrConfig().model_dump() == _kvcr_defaults()


def test_kvcr_supports_the_prefill_role_and_rejects_the_decode_role() -> None:
    engine = _prediction_engine(mode="disaggregated")
    engine["kv_transfer"] = {"bytes_per_token": 131_072}
    prefill, decode = (engine["workers"][role]["kv_cache"] for role in ("prefill", "decode"))
    prefill.update(bytes_per_token=131_072, host_offload=_host_offload(), kvcr={"bandwidth_gbps": 25})
    deployment = prediction_to_replay_spec(CorePredictionConfig.model_validate({"engine": engine})).backend_deployment
    assert deployment.prefill_engine_args["kvcr"] == {**_kvcr_defaults(), "bandwidth_gbps": 25.0}
    assert "kvcr" not in deployment.decode_engine_args

    decode.update(bytes_per_token=131_072, host_offload=_host_offload(), kvcr={})
    with pytest.raises(ValidationError, match="kvcr supports aggregated and prefill workers only"):
        CorePredictionConfig.model_validate({"engine": engine})


def test_kvcr_rejects_pipeline_parallelism() -> None:
    engine = _prediction_engine()
    worker = engine["workers"]["aggregated"]
    worker["kv_cache"].update(host_offload=_host_offload(), kvcr={})
    worker["parallelism"] = {**worker.get("parallelism", {}), "pipeline": 2}
    with pytest.raises(ValidationError, match="kvcr does not support pipeline parallelism"):
        CorePredictionConfig.model_validate({"engine": engine})


def test_kvcr_recommendation_is_rejected_not_silently_dropped() -> None:
    engine = _recommendation_engine()
    engine["workers"]["aggregated"]["kv_cache"].update(host_offload=_host_offload(), kvcr={})
    with pytest.raises(ValidationError, match="kvcr"):
        CoreRecommendationConfig.model_validate({"engine": engine})


def test_predict_cli_runs_kvcr_through_the_real_runtime(tmp_path, capsys) -> None:
    from aisimulate import _runtime

    assert callable(_runtime.run_replay_json)
    engine = _prediction_engine()
    engine["workers"]["aggregated"]["kv_cache"].update(bytes_per_token=256, host_offload=_host_offload(), kvcr={})
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
    argv = ["predict", "--stack", "engine", "--config", str(path), "--format", "json"]
    assert cli.main([*argv, "--output-dir", str(tmp_path / "kvcr")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["completed_requests"] == 1
    # CLI traffic carries no hints, so KVCR is idle; `bypassed` is reported only when nonzero.
    assert report["kvcr"] == {
        "pulls": {"success": 0, "partial": 0},
        "blocks": 0,
        "bytes": 0,
        "remote_deliver_ms": 0.0,
        "source_wait_ms": 0.0,
        "loaded_blocks": 0,
    }


def _hinted_replay(*, kvcr: bool) -> dict:
    """Run a ReplaySpec with an authored hint directly on the Rust runtime. Request `a`
    computes four blocks on DP rank 0; `b` extends them on DP rank 1 and names rank 0."""
    from aisimulate import _runtime

    rank = {
        "num_gpu_blocks": 64,
        "block_size": 4,
        "max_num_seqs": 4,
        "max_num_batched_tokens": 128,
        "kv_cache_bytes_per_token": 250_000,
        "timing_model": {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0},
        "native_host_offload": {"num_host_blocks": 64, "d2h_bandwidth_gbps": 32.0, "h2d_bandwidth_gbps": 32.0},
    }
    if kvcr:
        rank["kvcr"] = {}

    def request(request_id: str, arrival_ms: float, dp_rank: int, tokens: list[int], **fields) -> dict:
        return {
            "id": request_id,
            "arrival_time_ms": arrival_ms,
            "input_tokens": len(tokens),
            "input_token_ids": tokens,
            "output_tokens": 1,
            "dp_rank": dp_rank,
            **fields,
        }

    spec = {
        "version": 1,
        "topology": {"kind": "aggregated", "workers": {"initial_workers": 1}},
        "engine": {"dp_size": 2, "rank": rank},
        "requests": [
            request("a", 0.0, 0, list(range(1, 17))),
            request("b", 10.0, 1, list(range(1, 20)), kv_hints={"source_worker": 0, "source_dp_rank": 0}),
        ],
        "record_per_request": True,
    }
    report = json.loads(_runtime.run_replay_json(json.dumps(spec)))
    report["per_request"] = {record["request_id"]: record for record in report["per_request"]}
    return report


def test_authored_kv_hints_pull_through_the_python_runtime() -> None:
    report = _hinted_replay(kvcr=True)
    kvcr = report["kvcr"]
    assert kvcr["pulls"] == {"success": 1, "partial": 0}
    # Four blocks of 4 tokens at 250,000 bytes per token.
    assert (kvcr["blocks"], kvcr["bytes"]) == (4, 4_000_000)
    # Default timing: the pull starts at b's arrival (rank 0 is idle), waits 1 ms
    # for its first byte, then moves 4 MB at 50 GB/s.
    assert kvcr["source_wait_ms"] == 0.0
    assert kvcr["remote_deliver_ms"] == 1.08
    assert kvcr["loaded_blocks"] == 4
    assert "bypassed" not in kvcr
    # b reuses all four blocks from its own G2, where the pull put them.
    assert report["per_request"]["b"]["first_admission_host_reused_input_tokens"] == 16

    # Without KVCR the same hint is ignored silently, and b computes its whole prompt.
    plain = _hinted_replay(kvcr=False)
    assert "kvcr" not in plain
    assert plain["per_request"]["b"]["first_admission_host_reused_input_tokens"] == 0
    assert plain["per_request"]["b"]["admission_history"][0]["reused_input_tokens"] == 0


def test_agentx_accepts_kvcr_and_stays_idle() -> None:
    config = yaml.safe_load((ROOT / "examples/cli/agentx-g2-local.yaml").read_text(encoding="utf-8"))
    config["traffic"]["source"]["paths"] = [
        str(ROOT / "tests/e2e/configs/unified_cli/fixtures/traces/weka-g2-restore.jsonl")
    ]
    config["engine"]["model"] = str(ROOT / "tests/e2e/configs/unified_cli/fixtures/tiny-model")
    config["engine"]["workers"]["aggregated"]["kv_cache"]["kvcr"] = {}
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(config))
    result = (
        EngineReplayRunnerFactory()
        .create(0)
        .run(
            spec,
            output_requirements=ReplayOutputRequirements(include_raw_report=True, capture_per_request=True),
        )
    )
    report = result.metadata["native_report"]
    assert report["completed_requests"] == 4
    # One aggregated rank has no peer, and the trace carries no hints.
    assert report["kvcr"]["pulls"] == {"success": 0, "partial": 0}
    # G2 still restores the parent's evicted prefix.
    assert sum(row["first_admission_host_reused_input_tokens"] for row in report["per_request"]) == 8


def test_kvcr_docs_example_validates(monkeypatch) -> None:
    text = (ROOT / "docs/replay/engine/kv-cache.md").read_text(encoding="utf-8")
    host = text.split('<a id="native-vllm-host-offload-prediction"></a>', 1)[1]
    config = yaml.safe_load(host.split("```yaml\n", 1)[1].split("```", 1)[0])
    section = text.split('<a id="kvcr-peer-fetch"></a>', 1)[1]
    extension = yaml.safe_load(section.split("```yaml\n", 1)[1].split("```", 1)[0])
    assert list(extension) == ["kvcr"]
    cache = config["engine"]["workers"]["aggregated"]["kv_cache"]
    assert cache["host_offload"].get("scope", "dp_rank_local") == "dp_rank_local"
    cache.update(extension)
    prediction = CorePredictionConfig.model_validate(config)
    # Validate lowering without downloading the example's gated model metadata.
    monkeypatch.setattr("aisimulate.compiler.estimate_kv_bytes_per_token", lambda *_args, **_kwargs: 256)
    deployment = prediction_to_replay_spec(prediction).backend_deployment
    # The documented values are the defaults in the knob table.
    assert deployment.agg_engine_args["kvcr"] == _kvcr_defaults()
