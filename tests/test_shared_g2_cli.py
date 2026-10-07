# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import dataclasses
import json
import re
from pathlib import Path

import pytest

pytest.importorskip(
    "aisimulate.runner",
    reason="standalone AISimulate runtime is not installed",
    exc_type=ImportError,
)

import yaml
from pydantic import ValidationError
from test_host_offload_cli import _host_offload, _prediction_engine, _recommendation_engine, _RecordingRuntime

from aisimulate import main as cli
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config.cli import CorePredictionConfig, CoreRecommendationConfig
from aisimulate.config.engine import HostOffloadConfig
from aisimulate.recommend import recommendation_to_sweeper
from aisimulate.runner import EngineReplayRunnerFactory, _kv_layout_id, _materialize_engine_execution_spec
from aisimulate.sweeper.deploy import build_backend_deployment
from aisimulate.sweeper.forward_pass_estimator import ForwardPassEstimatorResolver
from aisimulate.sweeper.parallel_enum import ParallelShape, ReplicaParallelConfig
from aisimulate.sweeper.replay import ReplaySpec
from aisimulate.sweeper.sample import unroll_sample

SHARED = {"scope": "cluster_shared", "shared_d2h_bandwidth_gbps": 40.0, "latency_to_first_byte_ms": 0.5}
OPTIMIZATION = {"constraints": {"max_candidate_gpus": 8}}
EXPLICIT_PARALLELISM = "cluster_shared host_offload on both roles requires explicit integer tensor and pipeline"
EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "cli" / "shared-g2-predict.yaml"
# Default AIC timing resolves against the checked-in h200_sxm vLLM 0.24.0 data.
AIC_LAYOUT = {
    "backend": "vllm",
    "backend_version": "0.24.0",
    "block_size": 16,
    "bytes_per_token": 131_072,
    "model": "Qwen/Qwen3-32B-FP8",
    "pp": 1,
    "tp": 2,
}


def _recorded_prediction(tmp_path, monkeypatch, engine: dict) -> dict:
    """Run ``predict`` through the real compiler and runner; return the runtime input."""
    path = tmp_path / "prediction.yaml"
    path.write_text(yaml.safe_dump({"engine": engine}), encoding="utf-8")
    runtime = _RecordingRuntime()
    monkeypatch.setattr(cli, "resolve_runner_factory", lambda _stack: EngineReplayRunnerFactory(runtime=runtime))
    argv = ["predict", "--stack", "engine", "--config", str(path), "--output-dir", str(tmp_path / "out")]
    assert cli.main([*argv, "--overwrite", "--format", "json"]) == 0
    return runtime.execution_spec


def _layouts(execution: dict) -> dict[str, dict]:
    engine = execution.get("spec", execution)["engine"]
    roles = {"aggregated": engine} if "rank" in engine else {role: engine[role] for role in ("prefill", "decode")}
    return {role: json.loads(spec["rank"]["native_host_offload"]["kv_layout_id"]) for role, spec in roles.items()}


def _aic_engine(mode: str = "disaggregated", **engine_fields) -> dict:
    engine = _prediction_engine(mode=mode)
    engine.update({"model": AIC_LAYOUT["model"], "backend_version": "0.24.0", **engine_fields})
    for worker in engine["workers"].values():
        worker["parallelism"]["tensor"] = 2
        worker["timing"] = {"type": "default"}
        worker["kv_cache"].update(bytes_per_token=131_072, host_offload={**_host_offload(), **SHARED})
    return engine


def _sweeper_layouts(engine: dict, edit_estimator=None) -> dict[str, dict]:
    """Replay one TP2 aggregated sweeper candidate; return its layouts."""
    engine["workers"]["aggregated"]["parallelism"]["preset"] = False
    space = recommendation_to_sweeper(
        CoreRecommendationConfig.model_validate({"engine": engine, "optimization": OPTIMIZATION})
    ).search_space
    sample = unroll_sample(
        search_space=space,
        selection={
            "deployment_mode": "agg",
            "backend": "vllm",
            "agg_max_num_batched_tokens": 8192,
            "agg_max_num_seqs": 4,
        },
        parallel_config=ReplicaParallelConfig(ParallelShape(tp=2, dp=1, moe_tp=1, moe_ep=1), replicas=1),
    )
    estimators = ForwardPassEstimatorResolver(space).resolve_candidate(sample)
    if edit_estimator is not None:
        estimators = {"agg": edit_estimator(estimators["agg"])}
    deployment = build_backend_deployment(
        sample, backend_version=estimators["agg"].backend_version, forward_pass_estimators=estimators
    )
    runtime = _RecordingRuntime()
    workload = {"isl": 64, "osl": 4, "concurrency": 1, "num_request_ratio": 1}
    EngineReplayRunnerFactory(runtime=runtime).create(0).run(
        ReplaySpec(backend_deployment=deployment, workload=workload, goal={"target": "throughput"})
    )
    return _layouts(runtime.execution_spec)


def _shared_pd_recommendation(**parallelism) -> dict:
    engine = _recommendation_engine()
    worker = engine["workers"].pop("aggregated")
    worker["parallelism"].update(parallelism)
    worker["kv_cache"]["host_offload"] = {**_host_offload(), **SHARED}
    engine.update(mode="disaggregated", workers={"prefill": worker, "decode": copy.deepcopy(worker)})
    return {"engine": engine, "optimization": OPTIMIZATION}


def test_g2_scope_defaults_keep_the_descriptor_shape_and_redirect_the_g3_name() -> None:
    assert HostOffloadConfig.model_validate(_host_offload()).model_dump(mode="json") == _host_offload()
    shared = HostOffloadConfig.model_validate({**_host_offload(), **SHARED})
    assert shared.model_dump(mode="json") == {**_host_offload(), **SHARED}
    assert HostOffloadConfig.model_validate(shared.model_dump()) == shared
    with pytest.raises(
        ValidationError,
        match="host_offload.scope `worker_local` is a G3 scope; use `dp_rank_local` for per-DP-rank G2 caches",
    ):
        HostOffloadConfig.model_validate({**_host_offload(), "scope": "worker_local"})


def test_disaggregated_attention_dp_roles_lower_with_one_shared_layout(tmp_path, monkeypatch) -> None:
    engine = _prediction_engine(mode="disaggregated")
    for role in ("prefill", "decode"):
        worker = engine["workers"][role]
        worker["parallelism"].update(tensor=1, attention_data=2)
        worker["kv_cache"].update(bytes_per_token=131_072, host_offload={**_host_offload(), **SHARED})
    native = _recorded_prediction(tmp_path, monkeypatch, engine)["spec"]["engine"]

    prefill, decode = (native[role] for role in ("prefill", "decode"))
    assert (prefill["dp_size"], decode["dp_size"]) == (2, 2)
    layout = json.loads(prefill["rank"]["native_host_offload"].pop("kv_layout_id"))
    # Fixed timing carries no model timing identity; geometry still participates.
    assert layout == {"backend": "vllm", "block_size": 16, "bytes_per_token": 131_072, "tp": 1}
    assert decode["rank"]["native_host_offload"].pop("kv_layout_id") == json.dumps(
        layout, sort_keys=True, separators=(",", ":")
    )
    assert prefill["rank"]["native_host_offload"] == {**_host_offload(), **SHARED}


@pytest.mark.parametrize(
    "change",
    [
        {"tp": 2},
        {"pp": 2},
        {"dcp": 1},
        {"dcp": 2},
        {"kvcache_quant_mode": "fp8"},
        {"attention_backend": "flashinfer"},
        {"backend_version": "0.25.1"},
        {"model": "other/model"},
    ],
)
def test_layout_identity_covers_resolved_timing_identity(change: dict) -> None:
    config = {"model": "example/model", "tp": 1, "pp": 1, "backend_version": "0.24.0"}
    rank = {"backend": "vllm", "block_size": 16, "kv_cache_bytes_per_token": 131_072}

    def layout(timing_config: dict) -> str:
        return _kv_layout_id({**rank, "timing_model": {"type": "external", "config": timing_config}}, None, 1)

    assert layout(config) == layout(dict(reversed(config.items()))), "key order does not matter"
    assert layout({**config, **change}) != layout(config)


def test_recommendation_searches_attention_dp_but_pins_a_shared_pd_layout() -> None:
    engine = _recommendation_engine()
    engine["workers"]["aggregated"]["parallelism"]["attention_data"] = {"choices": [1, 2]}
    engine["workers"]["aggregated"]["kv_cache"]["host_offload"] = _host_offload()
    CoreRecommendationConfig.model_validate({"engine": engine, "optimization": OPTIMIZATION})

    payload = _shared_pd_recommendation(attention_data={"choices": [1, 2]})
    CoreRecommendationConfig.model_validate(payload)
    for tensor, message in [
        (2, "require matching tensor, pipeline"),
        ({"choices": [1, 2]}, EXPLICIT_PARALLELISM),
        ({"range": {"min": 1, "max": 2, "step": 1}}, EXPLICIT_PARALLELISM),
    ]:
        changed = copy.deepcopy(payload)
        changed["engine"]["workers"]["decode"]["parallelism"]["tensor"] = tensor
        with pytest.raises(ValidationError, match=message):
            CoreRecommendationConfig.model_validate(changed)


@pytest.mark.parametrize("roles", [("decode",), ("prefill", "decode")], ids=["decode", "both_roles"])
@pytest.mark.parametrize(
    "omitted", [("tensor",), ("pipeline",), ("tensor", "pipeline")], ids=["tensor", "pipeline", "both"]
)
def test_shared_pd_recommendation_requires_explicit_tensor_and_pipeline(omitted: tuple, roles: tuple) -> None:
    # An omitted knob is searched per role, so candidates could split the pool.
    payload = _shared_pd_recommendation()
    for role in roles:
        for knob in omitted:
            del payload["engine"]["workers"][role]["parallelism"][knob]
    with pytest.raises(ValidationError, match=EXPLICIT_PARALLELISM):
        CoreRecommendationConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("change", "mismatch"),
    [
        ({"num_host_blocks": 2048}, "num_host_blocks: prefill=4096, decode=2048"),
        ({"shared_d2h_bandwidth_gbps": 80.0}, "shared_d2h_bandwidth_gbps: prefill=40.0, decode=80.0"),
        ({"shared_h2d_bandwidth_gbps": 40.0}, "shared_h2d_bandwidth_gbps: prefill=80.0, decode=40.0"),
        # Per-role access links and a private decode cache are outside the pool contract.
        ({"d2h_bandwidth_gbps": 11.0, "h2d_bandwidth_gbps": 19.0, "latency_to_first_byte_ms": 2.0}, None),
        ({"scope": "dp_rank_local", "num_host_blocks": 2048, "shared_d2h_bandwidth_gbps": 80.0}, None),
    ],
)
def test_shared_pd_recommendation_rejects_a_split_pool_before_execution(change: dict, mismatch: str | None) -> None:
    payload = _shared_pd_recommendation()
    payload["engine"]["workers"]["decode"]["kv_cache"]["host_offload"].update(change)
    if mismatch is None:
        CoreRecommendationConfig.model_validate(payload)
        return
    message = f"cluster_shared host_offload roles require matching host_offload.{mismatch}"
    with pytest.raises(ValidationError, match=re.escape(message)):
        CoreRecommendationConfig.model_validate(payload)


TIMINGS = {
    "default": {"type": "default"},
    "fixed": {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0},
    "polynomial": {"type": "polynomial"},
}


@pytest.mark.parametrize(
    ("prefill", "decode"),
    [("default", "fixed"), ("polynomial", "default"), ("fixed", "polynomial"), ("default", "default")],
)
def test_shared_pd_recommendation_matches_the_runtime_timing_identity(
    tmp_path, monkeypatch, prefill: str, decode: str
) -> None:
    # The lowered layouts decide whether the runtime builds one pool; the
    # preflight must reject exactly the pairs that would split it.
    engine = _aic_engine()
    payload = _shared_pd_recommendation()
    for role, timing in (("prefill", prefill), ("decode", decode)):
        engine["workers"][role]["timing"] = TIMINGS[timing]
        payload["engine"]["workers"][role]["timing"] = TIMINGS[timing]
    layouts = _layouts(_recorded_prediction(tmp_path, monkeypatch, engine))
    one_pool = layouts["prefill"] == layouts["decode"]
    assert one_pool == ((prefill == "default") == (decode == "default"))
    if one_pool:
        CoreRecommendationConfig.model_validate(payload)
        return
    message = f"require default timing on both roles or neither: prefill={prefill!r}, decode={decode!r}"
    with pytest.raises(ValidationError, match=re.escape(message)):
        CoreRecommendationConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("section", "field", "value", "identity"),
    [
        ("timing", "kvcache_quant_mode", "fp8", {"kvcache_quant_mode": "fp8"}),
        # Checked-in data resolves only the default attention backend and has no DCP > 1 FPM.
        ("timing", "attention_backend", "default", {"attention_backend": "default"}),
        ("parallelism", "decode_context", 1, {"dcp": 1}),
    ],
)
def test_default_timing_layout_follows_each_workers_resolved_identity(
    tmp_path, monkeypatch, section: str, field: str, value, identity: dict
) -> None:
    from aisimulate import _runtime

    engine = _aic_engine()
    engine["workers"]["decode"][section][field] = value
    split = _recorded_prediction(tmp_path, monkeypatch, engine)
    assert _layouts(split) == {"prefill": AIC_LAYOUT, "decode": {**AIC_LAYOUT, **identity}}
    # The recorded input is exactly what the runtime receives; the pool refuses the split.
    with pytest.raises(RuntimeError, match="cluster_shared host_offload participants are incompatible"):
        _runtime.run_replay_json(json.dumps(split))

    engine["workers"]["prefill"][section][field] = value
    layout = {**AIC_LAYOUT, **identity}
    assert _layouts(_recorded_prediction(tmp_path, monkeypatch, engine)) == {"prefill": layout, "decode": layout}


@pytest.mark.parametrize("version", [None, "current", "0.24.0"])
def test_default_timing_layout_uses_the_canonical_backend_version(tmp_path, monkeypatch, version: str | None) -> None:
    native = _recorded_prediction(tmp_path, monkeypatch, _aic_engine(backend_version=version))["spec"]["engine"]
    layouts = {native[role]["rank"]["native_host_offload"]["kv_layout_id"] for role in ("prefill", "decode")}
    assert layouts == {json.dumps(AIC_LAYOUT, sort_keys=True, separators=(",", ":"))}


@pytest.mark.parametrize(
    ("controls", "identity"),
    [
        ({}, {}),
        ({"backend_version": "current"}, {}),
        ({"kvcache_quant_mode": "fp8"}, {"kvcache_quant_mode": "fp8"}),
    ],
)
def test_sweeper_candidate_layout_matches_the_prediction_layout(
    tmp_path, monkeypatch, controls: dict, identity: dict
) -> None:
    engine = _aic_engine(mode="aggregated", **controls)
    expected = {"aggregated": {**AIC_LAYOUT, **identity}}
    assert _layouts(_recorded_prediction(tmp_path, monkeypatch, engine)) == expected
    assert _sweeper_layouts(engine) == expected


def test_sweeper_layout_reads_a_legacy_kv_cache_dtype() -> None:
    # A forward-pass estimator config may still name KV quantization
    # `kv_cache_dtype`; it must reach the layout identity.
    def legacy(estimator):
        config = {**estimator.config, "kv_cache_dtype": "fp8"}
        del config["kvcache_quant_mode"]
        return dataclasses.replace(estimator, config=config)

    layouts = _sweeper_layouts(_aic_engine(mode="aggregated"), edit_estimator=legacy)
    assert layouts == {"aggregated": {**AIC_LAYOUT, "kvcache_quant_mode": "fp8"}}


@pytest.mark.parametrize(
    ("override", "scope", "reuse", "host_tokens", "domains"),
    [
        (
            [],
            "cluster_shared",
            0.861328125,
            2688,
            [{"capacity_blocks": 1024, "resident_blocks": 568, "used_blocks": 568}],
        ),
        (
            ["--set", "engine.workers.aggregated.kv_cache.host_offload.scope=dp_rank_local"],
            "dp_rank_local",
            0.8203125,
            0,
            None,
        ),
    ],
)
def test_shared_g2_example_runs_for_both_scopes(
    tmp_path, capsys, override: list[str], scope: str, reuse: float, host_tokens: int, domains
) -> None:
    from aisimulate import _runtime

    argv = ["predict", "--stack", "engine", "--config", str(EXAMPLE), *override, "--output-dir", str(tmp_path)]
    assert cli.main([*argv, "--format", "json"]) == 0
    report = json.loads(capsys.readouterr().out)
    # The same configuration also runs directly on the Rust runtime.
    raw = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    raw["engine"]["workers"]["aggregated"]["kv_cache"]["host_offload"]["scope"] = scope
    execution = _materialize_engine_execution_spec(
        prediction_to_replay_spec(CorePredictionConfig.model_validate(raw)),
        trace_block_size=16,
        record_per_request=True,
    )
    direct = json.loads(_runtime.run_replay_json(json.dumps(execution)))
    for result in (report, direct):
        assert (result["completed_requests"], result["prefix_cache_reused_ratio"], result.get("g2_domains")) == (
            64,
            reuse,
            domains,
        )
    assert sum(row["first_admission_host_reused_input_tokens"] for row in direct["per_request"]) == host_tokens


def test_prediction_rejects_host_offload_outside_vllm_language_workers() -> None:
    engine = _prediction_engine(mode="disaggregated")
    engine["workers"]["decode"]["kv_cache"]["host_offload"] = {**_host_offload(), **SHARED}
    CorePredictionConfig.model_validate({"engine": engine})
    engine["backend"] = "sglang"
    with pytest.raises(ValidationError, match="host_offload is supported only for backend=vllm"):
        CorePredictionConfig.model_validate({"engine": engine})
