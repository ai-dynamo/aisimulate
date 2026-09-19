# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Class-independent public plumbing; real timing validation is separate."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from aisimulate.capacity import materialize_aic_num_gpu_blocks
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig, CoreRecommendationConfig
from aisimulate.config.common import ENGINE_MODEL_CONTROL_FIELDS
from aisimulate.recommend import _candidate_prediction, recommendation_to_sweeper
from aisimulate.support.fpm import fpm_cli_args
from aisimulate.support.plan import check_plan, create_plan
from aisimulate.support.schema import SupportRequest
from aisimulate.sweeper.config import SearchSpace
from aisimulate.sweeper.deploy import build_backend_deployment
from aisimulate.sweeper.forward_pass_estimator import (
    ForwardPassEstimatorResolutionError,
    ForwardPassEstimatorResolver,
)
from aisimulate.sweeper.model_hw import parallel_configs_for
from aisimulate.sweeper.replay import ReplaySpec
from aisimulate.sweeper.sample import unroll_sample

pytestmark = pytest.mark.unit


@pytest.fixture
def profile():
    # Deliberately fictional metadata, with declared byte bounds, proves the
    # workflow cannot recover resources through an existing model registration.
    common = {
        "system": "h200_sxm",
        "backend": "vllm",
        "backend_version": "0.25.1",
        "gemm_quant_mode": "fp8_block",
        "moe_quant_mode": "fp8_block",
        "fmha_quant_mode": "bfloat16",
        "comm_quant_mode": "half",
        "kv_cache_dtype": "fp8",
        "resources": {
            "weights_bytes": 20 * 1024**3,
            "activations_bytes": 1024**3,
            "runtime_overhead_bytes": 1024**3,
            "comm_overhead_bytes": 1024**3,
            "kv_bytes_per_token": 512,
            "cache_layout": "linear",
            "max_num_tokens": 8192,
            "max_batch_size": 8,
            "provenance": "Synthetic rank-local bounds for workflow tests; excludes CUDA graph reservation.",
        },
    }
    return {
        "schema_version": 1,
        "model": "test-only/UnregisteredMoe",
        "model_revision": "workflow-fixture-v1",
        "architecture": "UnregisteredMoeForCausalLM",
        "context_length": 4096,
        "num_experts": 8,
        "provenance": "Fictional model, no timing or silicon accuracy claim.",
        "deployments": [
            {**deepcopy(common), "tp": 2, "dp": 1, "moe_tp": 2, "moe_ep": 1},
            {
                **deepcopy(common),
                "tp": 1,
                "dp": 2,
                "moe_tp": 1,
                "moe_ep": 2,
                "resources": {**common["resources"], "kv_bytes_per_token": 1024, "weights_bytes": 30 * 1024**3},
            },
        ],
    }


@pytest.fixture(autouse=True)
def forbid_registered_model(monkeypatch):
    from aisimulate import capacity, compiler, recommend
    from aisimulate.sweeper import model_hw
    from aisimulate_core.sdk import engine, memory

    def fail(*args, **kwargs):
        pytest.fail("class-independent workflow reached model/config construction")

    monkeypatch.setattr(engine, "get_model", fail)
    monkeypatch.setattr(memory.KVCacheEstimator, "from_request", fail)
    monkeypatch.setattr(memory.NaiveKVCacheEstimator, "from_model_path", fail)
    monkeypatch.setattr(capacity, "resolve_model_context_length", fail)
    monkeypatch.setattr(compiler, "resolve_model_context_length", fail)
    monkeypatch.setattr(recommend, "resolve_model_context_length", fail)
    monkeypatch.setattr(model_hw, "get_model_config_from_model_path", fail)
    monkeypatch.setattr(model_hw, "_estimate_model_weight_bytes", fail)


@pytest.fixture
def timing_systems(profile, tmp_path, monkeypatch):
    """Small synthetic timing cells exercise the public constructor and replay transport."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    import aisimulate_core
    from aisimulate_core.sdk.fpm_profile import load_fpm_profile

    root = tmp_path / "systems"
    root.mkdir()
    packaged = Path(aisimulate_core.__file__).parent / "systems"
    (root / "h200_sxm.yaml").write_bytes((packaged / "h200_sxm.yaml").read_bytes())
    rows = []
    for index, deployment in enumerate(load_fpm_profile(profile).deployments):
        identity = deployment.model_dump(mode="json", exclude={"resources"})
        for phase, kv_tokens in (("prefill", 0), ("decode", 0), ("decode", 1), ("decode", 64)):
            rows.append(
                {
                    **identity,
                    "cell_id": f"synthetic-{index}-{phase}-{kv_tokens}",
                    "model_path": profile["model"],
                    "weight_quantization": "synthetic",
                    "workload_kind": phase,
                    "partition_policy": "balanced_v1",
                    "batch_size": 1,
                    "total_prefill_tokens": 1 if phase == "prefill" else 0,
                    "total_kv_read_tokens": kv_tokens,
                    "latency_ms": 1.0,
                    "kv_seed_regime": "real_kv",
                }
            )
    path = root / "data/h200_sxm/vllm/0.25.1/fpm_forward_perf.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    path.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "schema_name": "aic_fpm_forward_perf",
                "schema_version": 6,
                "coordinate_system": "iteration_totals_balanced_v1",
                "measurement_policy": "dynamo_native_single_sample_v1",
                "system": "h200_sxm",
                "backend": "vllm",
                "backend_version": "0.25.1",
                "row_count": len(rows),
                "parquet_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    )
    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    return str(root)


def _worker(*, dep=False):
    return {
        "parallelism": {
            "tensor": 1 if dep else 2,
            "attention_data": 2 if dep else 1,
            "moe_tensor": 1 if dep else 2,
            "moe_expert": 2 if dep else 1,
        },
        "scheduler": {"max_batched_tokens": 1024, "max_sequences": 4},
        "timing": {
            "estimation_mode": "fpm_interpolation",
            "fallback_policy": "deny",
            "estimator_config": {"fpm_interpolation": {"method": "direct"}},
        },
    }


def _engine(profile):
    return {
        "model": profile["model"],
        "fpm_profile": profile,
        "hardware": "h200_sxm",
        "backend": "vllm",
        "backend_version": "0.25.1",
        "workers": {"aggregated": _worker()},
    }


def test_prediction_materializes_rank_local_capacity_without_a_model(profile, timing_systems):
    config = CorePredictionConfig.model_validate({"engine": {**_engine(profile), "systems_paths": [timing_systems]}})
    spec = prediction_to_replay_spec(config)
    args = spec.backend_deployment.agg_engine_args
    assert args["max_model_len"] == 4096
    canonical = args["timing_model"]["config"]
    assert canonical["estimator_config"]["fpm_interpolation"]["method"] == "direct"
    assert canonical["fpm_profile"] == config.engine.fpm_profile.model_dump(mode="json")
    metadata = spec.backend_deployment.performance_model_metadata["aggregated"]["config"]
    assert metadata["fpm_profile"] == canonical["fpm_profile"]
    assert metadata["estimator_config"] == canonical["estimator_config"]
    diagnostics = {}
    lowered = materialize_aic_num_gpu_blocks(args, memory_diagnostics=diagnostics)
    assert lowered["num_gpu_blocks"] > 0
    assert diagnostics["source"] == "profile"
    reserved_args = deepcopy(args)
    reserved_args["cuda_graph_reserved_bytes"] = 1024**3
    reserved_args["timing_model"]["config"]["cuda_graph_reserved_bytes"] = 1024**3
    reserved = materialize_aic_num_gpu_blocks(reserved_args)
    assert reserved["num_gpu_blocks"] < lowered["num_gpu_blocks"]


def test_disaggregated_transfer_uses_profile_cache_geometry(profile):
    engine = _engine(profile)
    engine.update(
        mode="disaggregated",
        workers={"prefill": _worker(dep=True), "decode": _worker()},
        kv_transfer={"bytes_per_token": "auto"},
    )
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate({"engine": engine}))
    assert spec.backend_deployment.prefill_engine_args["kv_transfer_bytes_per_token"] == 1024
    assert spec.backend_deployment.decode_engine_args["kv_transfer_bytes_per_token"] == 1024


@pytest.mark.parametrize("mode,budget,expected", [("agg", 2, 2), ("disagg", 4, 4)])
def test_profile_candidates_use_only_declared_topologies(profile, mode, budget, expected, monkeypatch):
    from aisimulate_core.sdk import RustForwardPassPerfModel

    def no_timing(*args, **kwargs):
        pytest.fail("candidate enumeration constructed a timing estimator before collection")

    monkeypatch.setattr(RustForwardPassPerfModel, "best_available", no_timing)
    candidates = parallel_configs_for(
        profile["model"],
        "h200_sxm",
        backend="vllm",
        backend_version="0.25.1",
        deployment_mode=mode,
        gpu_budget=budget,
        max_num_tokens=1024,
        max_batch_size=4,
        fpm_profile=profile,
    )
    assert len(candidates) == expected
    assert all(candidate.total_gpus <= budget for candidate in candidates)
    if mode == "agg":
        assert {candidate.shape.strategy for candidate in candidates} == {"tp", "dep"}


def test_dense_profile_preserves_unsharded_moe_dimensions(profile):
    profile["num_experts"] = 0
    profile["deployments"] = [{**profile["deployments"][0], "moe_tp": 1, "moe_ep": 1}]
    candidates = parallel_configs_for(
        profile["model"],
        "h200_sxm",
        backend="vllm",
        backend_version="0.25.1",
        deployment_mode="agg",
        gpu_budget=2,
        max_num_tokens=1024,
        max_batch_size=4,
        fpm_profile=profile,
    )
    assert [(candidate.shape.tp, candidate.shape.moe_tp, candidate.shape.moe_ep) for candidate in candidates] == [
        (2, 1, 1)
    ]


@pytest.mark.parametrize("source", ["sample", "resolved"])
def test_profile_kv_load_cache_keeps_resource_identity_without_model_construction(profile, monkeypatch, source):
    from aisimulate.sweeper import kv_load
    from aisimulate.sweeper.config import Workload
    from aisimulate.sweeper.parallel_enum import ParallelShape, ReplicaParallelConfig

    sample = {
        "deployment_mode": "agg",
        "model_name": profile["model"],
        "hardware_sku": "h200_sxm",
        "backend": "vllm",
        "agg_block_size": 64,
        "agg_max_num_batched_tokens": 1024,
        "agg_max_num_seqs": 4,
        "agg_gpu_memory_utilization": 0.9,
    }
    parallel = ReplicaParallelConfig(ParallelShape(tp=2, dp=1, moe_tp=2, moe_ep=1), replicas=1)
    larger = deepcopy(profile)
    larger["deployments"][0]["resources"]["weights_bytes"] += 10 * 1024**3
    original = kv_load.estimate_kv_tokens
    seen = []

    def estimate(*args, fpm_profile, **kwargs):
        seen.append(deepcopy(fpm_profile))
        return original(*args, fpm_profile=fpm_profile, **kwargs)

    monkeypatch.setattr(kv_load, "estimate_kv_tokens", estimate)
    kv_load._per_rank_capacity_tokens.cache_clear()
    capacities = []
    try:
        for resources in (profile, larger, profile):
            if source == "resolved":
                sample["forward_pass_estimators"] = {"agg": {"config": {"fpm_profile": resources}}}
                sample["fpm_profile"] = larger
            else:
                sample["fpm_profile"] = resources
            result = kv_load.resolve_kv_load(
                sample,
                workload=Workload(isl=1024, osl=128, kv_load_ratio=1.0, request_count=1),
                parallel_config=parallel,
                ratio=1.0,
                backend_version="0.25.1",
            )
            capacities.append(result.role_capacity_tokens["agg"])
        assert capacities[0] == capacities[2] > capacities[1]
        assert seen == [profile, larger]
    finally:
        kv_load._per_rank_capacity_tokens.cache_clear()


@pytest.mark.parametrize("method", ["auto", "direct"])
@pytest.mark.parametrize("backend_version", ["0.25.1", {"vllm": " 0.25.1 "}])
def test_recommendation_preserves_profile_in_exported_prediction(
    profile, monkeypatch, method, timing_systems, backend_version
):
    from aisimulate_core.sdk import RustForwardPassPerfModel

    engine = _engine(profile)
    engine["backend_version"] = backend_version
    engine["workers"]["aggregated"]["parallelism"] = {"preset": "default"}
    engine["workers"]["aggregated"]["timing"]["estimator_config"]["fpm_interpolation"]["method"] = method
    engine["systems_paths"] = [timing_systems]
    engine["mode"] = "aggregated"
    source = CoreRecommendationConfig.model_validate(
        {"engine": engine, "optimization": {"constraints": {"max_candidate_gpus": 2}}}
    )
    smart = recommendation_to_sweeper(source)
    assert smart.search_space.requested_backend_version("vllm") == "0.25.1"
    candidates = parallel_configs_for(
        profile["model"],
        "h200_sxm",
        backend="vllm",
        backend_version="0.25.1",
        deployment_mode="agg",
        gpu_budget=2,
        max_num_tokens=1024,
        max_batch_size=4,
        fpm_profile=profile,
    )
    sample = unroll_sample(
        search_space=smart.search_space,
        selection={
            "deployment_mode": "agg",
            "backend": "vllm",
            "agg_max_num_batched_tokens": 1024,
            "agg_max_num_seqs": 4,
        },
        parallel_config=candidates[1],
    )
    sample["backend_version"] = "0.25.1"
    estimators = ForwardPassEstimatorResolver(smart.search_space).resolve_candidate(sample)
    resolved = estimators["agg"].config
    assert resolved["estimator_config"]["fpm_interpolation"]["method"] == "direct"
    assert resolved["fpm_profile"] == source.engine.fpm_profile.model_dump(mode="json")
    controls = {
        "gemm_quant_mode": "fp8_block",
        "moe_quant_mode": "fp8_block",
        "fmha_quant_mode": "bfloat16",
        "kvcache_quant_mode": "fp8",
        "comm_quant_mode": "half",
        "attention_backend": "auto",
        "moe_backend": None,
        "enable_eplb": False,
        "wideep_num_slots": None,
    }
    assert {name: resolved[name] for name in ENGINE_MODEL_CONTROL_FIELDS} == controls
    assert resolved["systems_paths"] == [timing_systems]
    from aisimulate_core.sdk.models.base import _MODEL_REGISTRY

    # A fresh environment may now have an analytical class for the same model.
    # Export must retain the method actually selected for the evaluated sample.
    monkeypatch.setitem(_MODEL_REGISTRY, profile["architecture"], object())
    replay = ReplaySpec(
        backend_deployment=build_backend_deployment(
            sample, backend_version="0.25.1", forward_pass_estimators=estimators
        ),
        workload={},
        goal={},
    )
    exported = _candidate_prediction(source, sample, replay, adapter_sections={})
    reloaded = CorePredictionConfig.model_validate(exported)
    assert reloaded.engine.fpm_profile == source.engine.fpm_profile
    assert reloaded.engine.workers.aggregated.timing.estimator_config["fpm_interpolation"]["method"] == "direct"
    assert "forward_model" not in exported["engine"]["workers"]["aggregated"]["timing"]
    assert "fpm_interpolation" not in exported["engine"]["workers"]["aggregated"]["timing"]
    assert reloaded.engine.workers.aggregated.parallelism.attention_data == 2
    deployment = prediction_to_replay_spec(reloaded).backend_deployment
    args = deployment.agg_engine_args
    model = RustForwardPassPerfModel.best_available(deployment.performance_model_metadata["aggregated"]["config"])
    try:
        identity = model.diagnostics()["provenance"]["config"]
    finally:
        model.close()
    assert {name: identity[name] for name in ENGINE_MODEL_CONTROL_FIELDS} == controls
    assert identity["worker_type"] == "aggregated"
    assert tuple(identity[name] for name in ("tp", "pp", "attention_dp", "moe_tp_size", "moe_ep_size")) == (
        1,
        1,
        2,
        1,
        2,
    )
    assert materialize_aic_num_gpu_blocks(args)["num_gpu_blocks"] > 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("gemm_quant_mode", "bfloat16"),
        ("moe_quant_mode", "bfloat16"),
        ("fmha_quant_mode", "fp8"),
        ("kvcache_quant_mode", "bfloat16"),
        ("comm_quant_mode", "fp8"),
        ("attention_backend", "fa3"),
        ("moe_backend", "deepep_moe"),
        ("enable_eplb", True),
        ("wideep_num_slots", 128),
    ],
)
def test_profile_resolver_normalizes_cache_identity_and_rejects_conflicting_overrides(
    profile, timing_systems, monkeypatch, field, value
):
    from dataclasses import replace

    from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

    resolver = ForwardPassEstimatorResolver(SearchSpace(model_name=profile["model"], hardware_sku="h200_sxm"))
    request = ForwardPassPerfModelConfig(
        model=profile["model"],
        fpm_profile=profile,
        system="h200_sxm",
        backend="vllm",
        backend_version="0.25.1",
        worker_type="aggregated",
        tp=2,
        moe_tp_size=2,
        moe_ep_size=1,
        kv_block_size=64,
        systems_paths=(timing_systems,),
        estimation_mode="fpm_interpolation",
        estimator_config={"fpm_interpolation": {"method": "direct"}},
    )
    first = resolver._resolve(request, "agg")

    def no_build(*args, **kwargs):
        pytest.fail("equivalent defaults or conflicting profile override reached estimator construction")

    monkeypatch.setattr(RustForwardPassPerfModel, "best_available", no_build)
    explicit = replace(request, **{name: first.config[name] for name in ENGINE_MODEL_CONTROL_FIELDS})
    assert resolver._resolve(explicit, "agg").config == first.config
    with pytest.raises(ForwardPassEstimatorResolutionError, match="identity conflict|override|EPLB"):
        resolver._resolve(replace(explicit, **{field: value}), "agg")


@pytest.mark.parametrize("field", [*ENGINE_MODEL_CONTROL_FIELDS, "worker_type", "tp"])
def test_profile_resolver_rejects_changed_resolved_identity(profile, timing_systems, monkeypatch, field):
    from aisimulate_core.sdk import RustForwardPassPerfModel

    engine = _engine(profile)
    engine.update(mode="aggregated", systems_paths=[timing_systems])
    engine["workers"]["aggregated"]["parallelism"] = {"preset": "default"}
    smart = recommendation_to_sweeper(CoreRecommendationConfig.model_validate({"engine": engine, "optimization": {}}))
    sample = {
        "deployment_mode": "agg",
        "hardware_sku": "h200_sxm",
        "backend": "vllm",
        "tp": 2,
        "pp": 1,
        "attention_dp": 1,
        "moe_tp": 2,
        "moe_ep": 1,
        "agg_block_size": 64,
    }
    diagnostics = RustForwardPassPerfModel.diagnostics

    def changed(model):
        result = diagnostics(model)
        result["provenance"]["config"][field] = "unexpected"
        return result

    monkeypatch.setattr(RustForwardPassPerfModel, "diagnostics", changed)
    with pytest.raises(ForwardPassEstimatorResolutionError, match=f"changed exact candidate field '{field}'"):
        ForwardPassEstimatorResolver(smart.search_space).resolve_candidate(sample)


@pytest.mark.parametrize("role", ["agg", "prefill", "decode"])
@pytest.mark.parametrize("timing", [{"type": "fixed", "prefill_ms": 1, "decode_ms": 1}, {"type": "polynomial"}])
def test_profile_search_space_rejects_custom_timing(profile, role, timing):
    with pytest.raises(ValidationError, match="fpm_profile requires default timing"):
        SearchSpace(
            model_name=profile["model"],
            hardware_sku="h200_sxm",
            backend_version="0.25.1",
            fpm_profile=profile,
            **{f"{role}_timing_model": timing, f"{role}_num_gpu_blocks": 1024},
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"deployment_mode": ["afd"]},
        {"deployment_mode": ["afd+pd"]},
        {"encoder": {"hardware_sku": "h200_sxm"}},
        {"backend": ["sglang"]},
    ],
)
def test_profile_search_space_rejects_unsupported_providers(profile, updates):
    with pytest.raises(ValidationError, match="FPM profiles support vLLM aggregated/disaggregated"):
        SearchSpace(
            model_name=profile["model"],
            hardware_sku="h200_sxm",
            backend_version="0.25.1",
            fpm_profile=profile,
            **updates,
        )


@pytest.mark.parametrize(
    "backend_version,error",
    [
        (None, "literal"),
        ({}, "literal"),
        ({"sglang": "0.25.1"}, "literal"),
        ({"vllm": " "}, "literal"),
        ({"vllm": "current"}, "literal"),
        ({"vllm": "previous"}, "literal"),
        ({"vllm": "next"}, "literal"),
        ({"vllm": "0.24.0"}, "does not match"),
    ],
)
def test_profile_recommendation_cli_reports_invalid_version_mapping(profile, tmp_path, capsys, backend_version, error):
    import aisimulate.main as cli

    engine = _engine(profile)
    engine.update(mode="aggregated", backend_version=backend_version)
    engine["workers"]["aggregated"]["parallelism"] = {"preset": "default"}
    path = tmp_path / "recommend.json"
    path.write_text(json.dumps({"engine": engine}))
    with pytest.raises(SystemExit) as exc:
        cli.main(["recommend", "--config", str(path), "--output-dir", str(tmp_path / "output")])
    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert error in stderr
    assert "TypeError" not in stderr
    assert not (tmp_path / "output").exists()


def test_profile_recommendation_cli_accepts_version_mapping_and_exports_literal(
    profile, timing_systems, tmp_path, monkeypatch
):
    import yaml

    import aisimulate.main as cli
    from aisimulate.recommend import _run_recommendation

    monkeypatch.setattr("aisimulate.recommend.run_recommendation", _run_recommendation)
    engine = _engine(profile)
    engine.update(mode="aggregated", backend_version={"vllm": "0.25.1"}, systems_paths=[timing_systems])
    engine["workers"]["aggregated"]["parallelism"] = {"preset": "default"}
    raw = {
        "engine": engine,
        "traffic": {
            "source": {"type": "synthetic", "input_tokens": 1, "output_tokens": 2},
            "load": {"type": "concurrency", "concurrency": 1},
            "stop": {"requests": 1},
        },
        "optimization": {"constraints": {"max_candidate_gpus": 2}},
        "optimizer": {"algorithm": "random", "max_trials": 1, "parallelism": 1},
    }
    path = tmp_path / "recommend.json"
    path.write_text(json.dumps(raw))
    output = tmp_path / "output"
    assert cli.main(["recommend", "--config", str(path), "--output-dir", str(output), "--format", "json"]) == 0
    exported = yaml.safe_load((output / "recommendations/0001.yaml").read_text())
    assert exported["engine"]["backend_version"] == "0.25.1"
    assert CorePredictionConfig.model_validate(exported).engine.fpm_profile.model == profile["model"]


@pytest.mark.parametrize(
    "update,error",
    [
        ({"model": "another/model"}, "match"),
        ({"backend_version": None}, "literal"),
        ({"context_length": 8192}, "exceeds"),
        ({"workers": {"aggregated": {**_worker(), "scheduler": {"max_sequences": 32}}}}, "envelope exceeded"),
        ({"workers": {"aggregated": {**_worker(), "parallelism": {"tensor": 4, "moe_tensor": 4}}}}, "no matching"),
    ],
)
def test_invalid_profile_predictions_fail_before_runtime(profile, update, error):
    with pytest.raises(ValidationError, match=error):
        CorePredictionConfig.model_validate({"engine": {**_engine(profile), **update}})


def test_onboard_dep_keeps_full_identity_and_checks_resources_before_collection(profile, tmp_path, monkeypatch):
    from aisimulate_core.sdk import RustForwardPassPerfModel

    def no_timing(*args, **kwargs):
        pytest.fail("planning constructed a timing estimator before collection")

    monkeypatch.setattr(RustForwardPassPerfModel, "best_available", no_timing)
    request = SupportRequest.model_validate(
        {
            "identity": {
                "model": profile["model"],
                "model_revision": profile["model_revision"],
                "model_kind": "moe",
                "framework_version": "0.25.1",
                "gpu": "h200_sxm",
                "gpu_count": 2,
                "gpus_per_node": 2,
                "interconnect": "NVLink",
            },
            "search": {
                "tensor_parallel": 1,
                "attention_data_parallel": 2,
                "moe_tensor_parallel": 1,
                "moe_expert_parallel": 2,
                "context_length": 4096,
            },
            "fpm_profile": profile,
        }
    )
    plan = create_plan(request, tmp_path)
    assert plan["fpm"]["worker_gpus"] == 2
    assert plan["search"]["candidates"][0]["total_gpus"] == 2
    assert plan["resources"]["source"] == "profile"
    assert plan["resources"]["total_kv_size_tokens"] > 4096
    loaded = CorePredictionConfig.from_yaml(tmp_path / "predict/pilot.yaml")
    assert loaded.engine.workers.aggregated.parallelism.attention_data == 2
    assert loaded.engine.workers.aggregated.timing.estimator_config["fpm_interpolation"]["method"] == "direct"
    assert loaded.engine.workers.aggregated.timing.estimation_mode == "fpm_interpolation"
    assert loaded.engine.workers.aggregated.timing.fallback_policy == "deny"
    command = fpm_cli_args(request, output_dir=tmp_path, plan_only=True)
    assert command[command.index("--fpm-parallel-presets") + 1] == "dep"
    assert command[command.index("--fpm-gpu-counts") + 1] == "2"
    assert command[command.index("--fpm-kv-cache-dtypes") + 1] == "fp8"
    profile_path = tmp_path / "fpm-model-profile.json"
    assert command[command.index("--fpm-model-profile") + 1] == str(profile_path)
    tampered = json.loads(profile_path.read_text())
    tampered["deployments"][0]["fmha_quant_mode"] = "fp8"
    profile_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="differs from the requested identity"):
        check_plan(request, tmp_path)


@pytest.mark.parametrize(
    "preset,max_tokens,max_sequences,moe_backend",
    [
        ("dep", 4096, 8, "auto"),
        ("dep", 8192, 4, "auto"),
        ("dep", 8192, 8, "flashinfer_cutlass"),
        ("tep", 4096, 4, "flashinfer_cutlass"),
    ],
)
def test_generated_glm_collector_plan_preserves_backend_and_rank_local_bounds(
    profile, tmp_path, monkeypatch, capsys, preset, max_tokens, max_sequences, moe_backend
):
    from collector.fpm_forward import cli as collector_cli

    # Real GLM identity, with synthetic resource bounds: this exercises the
    # generated command and collector admission without claiming GPU accuracy.
    profile.update(
        model="nvidia/GLM-5.2-NVFP4",
        architecture="GlmMoeDsaForCausalLM",
        context_length=8192,
        num_experts=256,
        provenance="Real GLM identity with synthetic resource bounds for guided collection tests.",
    )
    deployment = profile["deployments"][0]
    deployment.update(
        system="b200_sxm",
        tp=1 if preset == "dep" else 8,
        dp=8 if preset == "dep" else 1,
        moe_tp=1,
        moe_ep=8,
        gemm_quant_mode="nvfp4",
        moe_quant_mode="nvfp4",
        fmha_quant_mode="fp8",
        moe_backend=moe_backend,
    )
    deployment["resources"].update(max_num_tokens=max_tokens, max_batch_size=max_sequences)
    profile["deployments"] = [deployment]
    request = SupportRequest.model_validate(
        {
            "identity": {
                "model": profile["model"],
                "model_revision": profile["model_revision"],
                "model_kind": "moe",
                "framework_version": "0.25.1",
                "gpu": "b200_sxm",
                "gpu_count": 8,
                "gpus_per_node": 8,
                "interconnect": "NVLink",
            },
            "search": {
                "tensor_parallel": deployment["tp"],
                "attention_data_parallel": deployment["dp"],
                "moe_tensor_parallel": 1,
                "moe_expert_parallel": 8,
                "context_length": 8192,
            },
            "workload": {"input_tokens": 1024, "output_tokens": 128, "concurrency": 8, "request_count": 8},
            "fpm_profile": profile,
        }
    )
    plan = create_plan(request, tmp_path)
    command = json.loads((tmp_path / "commands.json").read_text())["fpm_plan_local"]
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    assert collector_cli.main(command[3:]) == 0
    collected = json.loads(capsys.readouterr().out)
    assert collected["counts"]["cells"] == 2
    assert {cell["parallel_strategy"] for cell in collected["cells"]} == {preset}
    assert collected["options"]["moe_backend"] == moe_backend
    sampling = collected["options"]["prefill_sampling"]
    expected_tokens = min(max_tokens, 1024 * max_sequences)
    assert sampling["max_total_prefill_tokens"] == expected_tokens
    assert sampling["max_batch_size"] == max_sequences
    assert collected["topology_memory_admission"][0]["activation_envelope"]["scope"] == "rank_local"
    prediction = CorePredictionConfig.from_yaml(tmp_path / "predict/pilot.yaml")
    scheduler = prediction.engine.workers.aggregated.scheduler
    assert scheduler.max_batched_tokens == max_tokens
    assert scheduler.max_sequences == max_sequences
    assert prediction.traffic.load.concurrency == 8
    assert prediction.traffic.stop.requests == 8
    assert "timing coverage" in plan["fpm"]["sampling"]


def test_onboard_rejects_profile_below_collector_minimum_token_axis(profile, tmp_path):
    profile["deployments"][0]["resources"]["max_num_tokens"] = 1
    request = SupportRequest.model_validate(
        {
            "identity": {
                "model": profile["model"],
                "model_revision": profile["model_revision"],
                "model_kind": "moe",
                "framework_version": "0.25.1",
                "gpu": "h200_sxm",
                "gpu_count": 2,
                "gpus_per_node": 2,
                "interconnect": "NVLink",
            },
            "search": {"tensor_parallel": 2, "context_length": 4096},
            "fpm_profile": profile,
        }
    )
    with pytest.raises(ValueError, match="FPM collection requires.*at least 2"):
        create_plan(request, tmp_path)


@pytest.mark.parametrize(
    "tp,dp,moe_tp,moe_ep,preset", [(2, 1, 2, 1, "pure_tp"), (1, 2, 1, 2, "dep"), (2, 1, 1, 2, "tep")]
)
def test_onboard_cli_embeds_profile_and_preserves_topology(profile, tmp_path, tp, dp, moe_tp, moe_ep, preset):
    import aisimulate.main as cli

    profile["deployments"].append({**deepcopy(profile["deployments"][0]), "moe_tp": 1, "moe_ep": 2})
    source = tmp_path / "model-profile.json"
    source.write_text(json.dumps(profile))
    request_path = tmp_path / "request.yaml"
    assert (
        cli.main(
            [
                "onboard",
                "init",
                "--model",
                profile["model"],
                "--model-revision",
                profile["model_revision"],
                "--model-kind",
                "moe",
                "--framework-version",
                "0.25.1",
                "--gpu",
                "h200_sxm",
                "--gpu-count",
                "2",
                "--interconnect",
                "NVLink",
                "--context-length",
                "4096",
                "--fpm-profile",
                str(source),
                "--tensor-parallel",
                str(tp),
                "--attention-data-parallel",
                str(dp),
                "--moe-tensor-parallel",
                str(moe_tp),
                "--moe-expert-parallel",
                str(moe_ep),
                "--output",
                str(request_path),
            ]
        )
        == 0
    )
    source.unlink()
    request = SupportRequest.from_yaml(request_path)
    assert request.fpm_profile.model == profile["model"]
    assert request.worker_gpus == 2
    assert request.parallel_preset == preset
    assert request.profile_deployment().parallel_tuple == (tp, 1, dp, moe_tp, moe_ep, 1)
    plan = tmp_path / "plan"
    assert cli.main(["onboard", "plan", "--config", str(request_path), "--output-dir", str(plan)]) == 0
    prediction = CorePredictionConfig.from_yaml(plan / "predict/pilot.yaml")
    assert prediction.engine.fpm_profile == request.fpm_profile
    profile_output = plan / "fpm-model-profile.json"
    original = profile_output.read_bytes()
    profile_output.unlink()
    create_plan(request, plan, overwrite=True)
    assert profile_output.read_bytes() == original
