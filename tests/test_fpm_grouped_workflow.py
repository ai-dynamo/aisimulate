# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public grouped-cache workflows with fictional resources and synthetic timings."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from aisimulate import main as cli
from aisimulate.capacity import materialize_aic_num_gpu_blocks
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig, CoreRecommendationConfig
from aisimulate.support.plan import create_plan
from aisimulate.support.schema import SupportRequest
from aisimulate.sweeper.config import Workload
from aisimulate.sweeper.kv_load import resolve_kv_load
from aisimulate.sweeper.model_hw import parallel_configs_for

pytestmark = pytest.mark.unit


@pytest.fixture
def grouped_case(tmp_path, monkeypatch):
    import aisimulate_core
    from aisimulate_core.sdk import engine, memory

    def no_graph(*_args, **_kwargs):
        pytest.fail("grouped FPM workflow constructed an analytical model")

    monkeypatch.setattr(engine, "get_model", no_graph)
    monkeypatch.setattr(engine, "build_model_config", no_graph)
    monkeypatch.setattr(memory.KVCacheEstimator, "from_request", no_graph)
    monkeypatch.setattr(memory.NaiveKVCacheEstimator, "from_model_path", no_graph)
    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    resources = {
        "weights_bytes": 100,
        "activations_bytes": 20,
        "runtime_overhead_bytes": 30,
        "comm_overhead_bytes": 50,
        "cache_layout": "grouped",
        "cache_groups": [
            {
                "name": "full",
                "kind": "attention",
                "num_layers": 1,
                "block_size_tokens": 64,
                "page_size_bytes": 64,
            },
            {
                "name": "window",
                "kind": "attention",
                "num_layers": 2,
                "block_size_tokens": 16,
                "page_size_bytes": 16,
                "sliding_window": 512,
            },
            {
                "name": "conv",
                "kind": "convolution",
                "num_layers": 3,
                "block_size_tokens": 4,
                "page_size_bytes": 8,
                "sliding_window": 4,
            },
        ],
        "max_num_tokens": 128,
        "max_batch_size": 8,
        "provenance": "Hand-sized fake cache pages including aggregate group layers; not measured GPU resources.",
    }
    deployment = {
        "system": "h200_sxm",
        "backend": "vllm",
        "backend_version": "0.25.1",
        "gemm_quant_mode": "fp8",
        "moe_quant_mode": "fp8",
        "fmha_quant_mode": "bfloat16",
        "comm_quant_mode": "half",
        "kv_cache_dtype": "fp8",
        "resources": resources,
    }
    profile = {
        "schema_version": 1,
        "model": "test-only/UnregisteredWindowMoe",
        "model_revision": "synthetic-v1",
        "architecture": "UnregisteredWindowMoeForCausalLM",
        "context_length": 2048,
        "num_experts": 4,
        "provenance": "Fictional model and timings; functional coverage only, no silicon accuracy claim.",
        "deployments": [
            {**deepcopy(deployment), "tp": 2, "dp": 1, "moe_tp": 2, "moe_ep": 1},
            {**deepcopy(deployment), "tp": 1, "dp": 2, "moe_tp": 1, "moe_ep": 2},
        ],
    }
    root = tmp_path / "systems"
    root.mkdir()
    system = yaml.safe_load((Path(aisimulate_core.__file__).parent / "systems/h200_sxm.yaml").read_text())
    # A deliberately tiny fake GPU creates meaningful byte pressure in CPU tests.
    system["gpu"]["mem_capacity"] = 4096
    (root / "h200_sxm.yaml").write_text(yaml.safe_dump(system))
    _write_timings(root, profile)
    return profile, root


def _write_timings(root, profile, *, decode_ceiling=8192):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from aisimulate_core.sdk.fpm_profile import load_fpm_profile

    rows = []
    for index, deployment in enumerate(load_fpm_profile(profile).deployments):
        identity = deployment.model_dump(mode="json", exclude={"resources"})
        for batch in (1, 2, 4, 8):
            for phase, tokens, kvs in (
                (
                    "prefill",
                    (batch, 128, 256, 512, 1024),
                    (0, 1, 512, 1024, 2048, 8192),
                ),
                ("decode", (0,), (0, 1, 512, decode_ceiling)),
            ):
                for new_tokens in sorted(set(tokens)):
                    for kv in sorted(set(kvs)):
                        rows.append(
                            {
                                **identity,
                                "cell_id": f"synthetic-{index}-{phase}-{batch}-{new_tokens}-{kv}",
                                "model_path": profile["model"],
                                "weight_quantization": "synthetic",
                                "workload_kind": phase,
                                "partition_policy": "balanced_v1",
                                "batch_size": batch,
                                "total_prefill_tokens": new_tokens,
                                "total_kv_read_tokens": kv,
                                "latency_ms": 1 + batch + new_tokens / 128 + kv / 1024,
                                "kv_seed_regime": "real_kv",
                            }
                        )
    path = root / "data/h200_sxm/vllm/0.25.1/fpm_forward_perf.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _prediction(profile, root, *, dep=False):
    return {
        "engine": {
            "mode": "aggregated",
            "model": profile["model"],
            "fpm_profile": profile,
            "hardware": "h200_sxm",
            "backend": "vllm",
            "backend_version": "0.25.1",
            "context_length": 2048,
            "systems_paths": [str(root)],
            "workers": {
                "aggregated": {
                    "parallelism": {
                        "tensor": 1 if dep else 2,
                        "attention_data": 2 if dep else 1,
                        "moe_tensor": 1 if dep else 2,
                        "moe_expert": 2 if dep else 1,
                    },
                    "scheduler": {"max_batched_tokens": 128, "max_sequences": 4},
                    "kv_cache": {"prefix_caching": False},
                    "timing": {
                        "estimation_mode": "fpm_interpolation",
                        "fallback_policy": "deny",
                        "estimator_config": {
                            "fpm_interpolation": {
                                "method": "direct",
                                "collect_coverage": True,
                            }
                        },
                    },
                }
            },
        },
        "traffic": {
            "source": {"type": "synthetic", "input_tokens": 1152, "output_tokens": 4},
            "load": {"type": "concurrency", "concurrency": 2},
            "stop": {"requests": 2},
        },
    }


def _mixed_profile(profile):
    profile = deepcopy(profile)
    resources = profile["deployments"][0]["resources"]
    resources["cache_layout"] = "linear"
    resources["kv_bytes_per_token"] = 1
    resources.pop("cache_groups")
    # A full-only grouped deployment is valid beside a linear deployment.
    profile["deployments"][1]["resources"]["cache_groups"] = profile["deployments"][1]["resources"]["cache_groups"][:1]
    return profile


@pytest.mark.parametrize("unrelated", ["topology", "hardware", "version"])
def test_mixed_profile_predict_preserves_selected_linear_prefix_cache(grouped_case, tmp_path, unrelated):
    profile, root = grouped_case
    profile = _mixed_profile(profile)
    if unrelated != "topology":
        profile["deployments"][1].update(tp=2, dp=1, moe_tp=2, moe_ep=1)
        profile["deployments"][1].update(
            {"system": "h100_sxm"} if unrelated == "hardware" else {"backend_version": "0.25.2"}
        )
    raw = _prediction(profile, root)
    raw["engine"]["workers"]["aggregated"]["kv_cache"]["prefix_caching"] = True
    CorePredictionConfig.model_validate(raw)
    path = tmp_path / "linear.yaml"
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "linear-prediction"
    assert cli.main(["predict", "-c", str(path), "--output-dir", str(output), "--detail", "memory"]) == 0
    report = json.loads((output / "prediction.json").read_text())
    assert report["completed_requests"] == 2
    assert report["details"]["sections"]["memory"]["roles"]["aggregated"]["total_kv_size_tokens"] > 0


@pytest.mark.parametrize("preset", [False, True])
def test_mixed_profile_recommendation_preserves_pinned_linear_prefix_cache(grouped_case, tmp_path, monkeypatch, preset):
    from aisimulate.recommend import _run_recommendation, recommendation_to_sweeper
    from aisimulate.sweeper.search_space import enumerate_branches

    profile, root = grouped_case
    raw = _prediction(_mixed_profile(profile), root)
    worker = raw["engine"]["workers"]["aggregated"]
    worker["kv_cache"]["prefix_caching"] = True
    worker["parallelism"]["preset"] = False
    if preset:
        worker["parallelism"] = {"preset": [{**worker["parallelism"], "replicas": 1, "pipeline": 1}]}
        worker["parallelism"]["preset"][0].pop("preset")
    raw["optimization"] = {"constraints": {"max_candidate_gpus": 2}}
    raw["optimizer"] = {"algorithm": "random", "max_trials": 1, "parallelism": 1}
    smart = recommendation_to_sweeper(CoreRecommendationConfig.model_validate(raw))
    branches = enumerate_branches(smart)
    assert len(branches) == 1
    assert {candidate.shape.tp for candidate in branches[0].parallel_configs} == {2}
    monkeypatch.setattr("aisimulate.recommend.run_recommendation", _run_recommendation)
    path = tmp_path / "linear-recommend.yaml"
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "linear-recommendation"
    assert cli.main(["recommend", "-c", str(path), "--output-dir", str(output)]) == 0
    exported = CorePredictionConfig.from_yaml(output / "recommendations/0001.yaml")
    assert exported.engine.workers.aggregated.parallelism.tensor == 2
    assert exported.engine.workers.aggregated.kv_cache.prefix_caching is True


def _mixed_search(profile, root):
    return {
        "deployment_mode": ["agg"],
        "backend": ["vllm"],
        "backend_version": "0.25.1",
        "model_name": profile["model"],
        "hardware_sku": "h200_sxm",
        "fpm_profile": _mixed_profile(profile),
        "systems_paths": [str(root)],
        "gpu_budget": 2,
        "agg_max_num_batched_tokens": [128],
        "agg_max_num_seqs": [4],
        "agg_enable_prefix_caching": True,
    }


@pytest.mark.parametrize("filter_by", ["hardware", "version", "budget", "pin", "mode_pin", "choices", "log_range"])
def test_mixed_search_ignores_unreachable_grouped_deployments(grouped_case, filter_by):
    from aisimulate.sweeper.config import SmartSearchConfig
    from aisimulate.sweeper.search_space import enumerate_branches

    profile, root = grouped_case
    raw = _mixed_search(profile, root)
    grouped = raw["fpm_profile"]["deployments"][1]
    if filter_by == "hardware":
        grouped["system"] = "h100_sxm"
    elif filter_by == "version":
        grouped["backend_version"] = "0.25.2"
        raw["backend_version"] = {"vllm": " 0.25.1 "}
    elif filter_by == "budget":
        grouped.update(tp=4, dp=1, moe_tp=1, moe_ep=4)
    elif filter_by in {"pin", "mode_pin"}:
        pinned = [{"tp": 2, "moe_tp": 2}]
        raw["parallel_configs" if filter_by == "pin" else "parallel_configs_by_mode"] = (
            pinned if filter_by == "pin" else {"agg": pinned}
        )
    elif filter_by == "choices":
        raw["parallel_independent_by_mode"] = {"agg": {"tp": [2]}}
    else:
        raw["parallel_independent_by_mode"] = {"agg": {"tp": None}}
        raw["parallel_independent_log_ranges_by_mode"] = {"agg": {"tp": [2, 4]}}
    config = SmartSearchConfig(search_space=raw, workload=Workload(isl=1152, osl=4, concurrency=2, request_count=2))
    branches = enumerate_branches(config)
    assert len(branches) == 1
    assert {candidate.shape.tp for candidate in branches[0].parallel_configs} == {2}


@pytest.mark.parametrize("placement", ["aggregated", "disaggregated", "heterogeneous"])
def test_mixed_recommendation_linear_fixed_capacity_and_disaggregation(grouped_case, placement):
    from aisimulate.recommend import recommendation_to_sweeper
    from aisimulate.sweeper.search_space import enumerate_branches

    profile, root = grouped_case
    raw = _prediction(_mixed_profile(profile), root)
    # Unselected deployments must not reach resource-envelope validation either.
    raw["engine"]["fpm_profile"]["deployments"][1]["resources"]["max_num_tokens"] = 64
    worker = raw["engine"]["workers"]["aggregated"]
    worker["kv_cache"].update(prefix_caching=True, capacity={"type": "fixed", "blocks": 64})
    disaggregated = placement != "aggregated"
    if disaggregated:
        raw["engine"]["mode"] = "disaggregated"
        raw["engine"]["workers"] = {"prefill": deepcopy(worker), "decode": deepcopy(worker)}
    if placement == "heterogeneous":
        linear = deepcopy(raw["engine"]["fpm_profile"]["deployments"][0])
        linear["system"] = "h100_sxm"
        raw["engine"]["fpm_profile"]["deployments"].append(linear)
        raw["engine"]["workers"]["decode"]["hardware"] = "h100_sxm"
        (root / "h100_sxm.yaml").write_bytes((root / "h200_sxm.yaml").read_bytes())
    CorePredictionConfig.model_validate(raw)
    for worker in raw["engine"]["workers"].values():
        worker["parallelism"]["preset"] = False
        worker["parallelism"]["replicas"] = 1
    if placement == "heterogeneous":
        parallel = raw["engine"]["workers"]["prefill"]["parallelism"]
        parallel.pop("preset")
        raw["engine"]["workers"]["prefill"]["parallelism"] = {"preset": [{**parallel, "pipeline": 1}]}
    raw["optimization"] = {"constraints": {"max_candidate_gpus": 4 if disaggregated else 2}}
    branches = enumerate_branches(recommendation_to_sweeper(CoreRecommendationConfig.model_validate(raw)))
    assert len(branches) == 1
    assert len(branches[0].parallel_configs) == 1
    candidate = branches[0].parallel_configs[0]
    assert (candidate.prefill.shape.tp if disaggregated else candidate.shape.tp) == 2


def test_mixed_search_keeps_both_layouts_when_grouped_settings_are_supported(grouped_case):
    from aisimulate.sweeper.config import SmartSearchConfig
    from aisimulate.sweeper.search_space import enumerate_branches

    profile, root = grouped_case
    raw = _mixed_search(profile, root)
    raw["agg_enable_prefix_caching"] = False
    config = SmartSearchConfig(search_space=raw, workload=Workload(isl=1152, osl=4, concurrency=2, request_count=2))
    assert {candidate.shape.tp for candidate in enumerate_branches(config)[0].parallel_configs} == {1, 2}


@pytest.mark.parametrize(
    "update,error",
    [
        ({}, "prefix_caching=false"),
        ({"agg_enable_prefix_caching": False, "agg_native_host_offload": {"num_host_blocks": 64}}, "only HBM"),
        ({"agg_enable_prefix_caching": False, "agg_num_gpu_blocks": 64}, "not scalar capacity"),
        ({"agg_enable_prefix_caching": False, "agg_kv_bytes_per_token": 1}, "not scalar capacity"),
        ({"deployment_mode": ["disagg"], "gpu_budget": 4}, "without speculative decoding"),
    ],
)
def test_mixed_search_rejects_reachable_grouped_unsupported_settings(grouped_case, update, error):
    from aisimulate.sweeper.config import SearchSpace

    profile, root = grouped_case
    raw = _mixed_search(profile, root)
    raw.update(update)
    with pytest.raises(ValidationError, match=error):
        SearchSpace.model_validate(raw)


def test_public_recommendation_rejects_reachable_grouped_before_estimating(grouped_case, tmp_path, monkeypatch, capsys):
    from aisimulate.recommend import _run_recommendation
    from aisimulate_core.sdk import RustForwardPassPerfModel

    def no_estimator(*_args, **_kwargs):
        pytest.fail("unsupported grouped recommendation reached estimator construction")

    profile, root = grouped_case
    raw = _prediction(_mixed_profile(profile), root)
    raw["engine"]["workers"]["aggregated"]["parallelism"] = {"preset": "default"}
    raw["engine"]["workers"]["aggregated"]["kv_cache"]["prefix_caching"] = True
    raw["optimization"] = {"constraints": {"max_candidate_gpus": 2}}
    raw["optimizer"] = {"algorithm": "random", "max_trials": 1, "parallelism": 1}
    CoreRecommendationConfig.model_validate(raw)
    monkeypatch.setattr("aisimulate.recommend.run_recommendation", _run_recommendation)
    monkeypatch.setattr(RustForwardPassPerfModel, "best_available", no_estimator)
    path = tmp_path / "unsupported-recommend.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(SystemExit) as error:
        cli.main(["recommend", "-c", str(path), "--output-dir", str(tmp_path / "unsupported")])
    assert error.value.code == 2
    output = capsys.readouterr()
    assert "grouped FPM cache requires agg_enable_prefix_caching=false" in output.out + output.err


@pytest.mark.parametrize("dep", [False, True])
@pytest.mark.parametrize("layout", ["mixed", "window_only", "full_only"])
def test_public_predict_uses_group_bytes_with_tp_and_dep(grouped_case, tmp_path, dep, layout):
    profile, root = grouped_case
    if layout != "mixed":
        for deployment in profile["deployments"]:
            groups = deployment["resources"]["cache_groups"]
            deployment["resources"]["cache_groups"] = groups[:1] if layout == "full_only" else groups[1:]
    raw = _prediction(profile, root, dep=dep)
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    diagnostics = {}
    lowered = materialize_aic_num_gpu_blocks(spec.backend_deployment.agg_engine_args, memory_diagnostics=diagnostics)
    assert "num_gpu_blocks" not in lowered
    assert lowered["kv_cache_capacity_bytes"] == 3486  # floor(4096 * .9 - 200)
    assert lowered["kv_cache_groups"] == diagnostics["cache_groups"]
    assert diagnostics["total_kv_size_tokens"] is None
    path = tmp_path / "predict.yaml"
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "prediction"
    assert (
        cli.main(
            [
                "predict",
                "-c",
                str(path),
                "--output-dir",
                str(output),
                "--detail",
                "memory",
                "--capture-per-request",
            ]
        )
        == 0
    )
    report = json.loads((output / "prediction.json").read_text())
    assert report["completed_requests"] == 2
    assert all(row["output_length"] == 4 for row in report["per_request"])
    assert report["fpm_query_coverage"]["status"] == "covered"
    memory = report["details"]["sections"]["memory"]["roles"]["aggregated"]
    assert memory["total_kv_size_bytes"] == 3486
    assert memory["total_kv_size_tokens"] is None
    assert "estimated_num_gpu_blocks" not in memory


def test_public_predict_does_not_clamp_logical_context_to_window(grouped_case, tmp_path):
    profile, root = grouped_case
    # Window-only allocation is bounded, but a decode at context 1152 must ask
    # for 1151 past KV tokens plus its current input. A table ending at 512 must fail, not return a
    # plausible short-context result or shrink the physical byte budget.
    for deployment in profile["deployments"]:
        deployment["resources"]["cache_groups"] = deployment["resources"]["cache_groups"][1:]
    _write_timings(root, profile, decode_ceiling=512)
    raw = _prediction(profile, root)
    raw["traffic"]["load"]["concurrency"] = 1
    raw["traffic"]["stop"]["requests"] = 1
    path = tmp_path / "uncovered.yaml"
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "uncovered"
    assert cli.main(["predict", "-c", str(path), "--output-dir", str(output)]) != 0
    coverage = json.loads((output / "fpm-coverage.json").read_text())
    gaps = coverage["roles"][0]["coverage"]["gaps"]
    assert any(gap["coordinates"]["total_kv_read_tokens"] == 1151 for gap in gaps)


@pytest.mark.parametrize("relative_load", [False, True])
@pytest.mark.parametrize("observed_memory", [False, True])
def test_grouped_recommendation_exports_a_runnable_prediction(
    grouped_case, tmp_path, monkeypatch, relative_load, observed_memory
):
    from aisimulate.recommend import _run_recommendation

    monkeypatch.setattr("aisimulate.recommend.run_recommendation", _run_recommendation)
    profile, root = grouped_case
    if observed_memory:
        for deployment in profile["deployments"]:
            resources = deployment["resources"]
            for name in ("weights_bytes", "activations_bytes", "runtime_overhead_bytes", "comm_overhead_bytes"):
                resources.pop(name)
            resources["max_batch_size"] = 4
            resources["runtime_memory"] = {
                "kv_cache_bytes": 3486,
                "max_model_len": 2048,
                "gpu_memory_utilization": 0.9,
                "provenance": "Synthetic initialized cache pool; no GPU qualification.",
            }
    raw = _prediction(profile, root)
    raw["engine"]["workers"]["aggregated"]["parallelism"] = {"preset": "default"}
    raw["optimization"] = {"constraints": {"max_candidate_gpus": 2}}
    raw["optimizer"] = {"algorithm": "random", "max_trials": 2, "parallelism": 1}
    if relative_load:
        raw["traffic"]["load"] = {"type": "kv_capacity_fraction", "fraction": 1.0}
    CoreRecommendationConfig.model_validate(raw)
    path = tmp_path / "recommend.yaml"
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "recommendation"
    assert cli.main(["recommend", "-c", str(path), "--output-dir", str(output)]) == 0
    exported_path = output / "recommendations/0001.yaml"
    exported = CorePredictionConfig.from_yaml(exported_path)
    assert exported.engine.fpm_profile.deployments[0].resources.cache_layout == "grouped"
    assert exported.engine.workers.aggregated.kv_cache.prefix_caching is False
    assert (
        cli.main(
            [
                "predict",
                "-c",
                str(exported_path),
                "--output-dir",
                str(tmp_path / "exported-prediction"),
            ]
        )
        == 0
    )


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("kv_cache_capacity_bytes", 9999, "byte capacity conflicts"),
        ("kv_cache_groups", None, "cache groups conflict"),
        ("tensor_parallel_size", 4, "tensor_parallel_size|topology"),
        ("dp_size", 2, "dp_size|topology"),
    ],
)
def test_native_grouped_rematerialization_checks_identity_and_budget(grouped_case, field, value, error):
    from aisimulate import _runtime
    from aisimulate.runner import EngineReplayRunnerFactory

    profile, root = grouped_case
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(_prediction(profile, root)))

    class ChangedNativeInput:
        def run_replay_json(self, serialized):
            payload = json.loads(serialized)
            engine = payload.get("spec", payload)["engine"]
            if field in {"tensor_parallel_size", "dp_size"}:
                engine[field] = value
            elif field == "kv_cache_groups":
                # Wrong but internally valid physical page geometry must not
                # replace the canonical profile after Python materialization.
                engine["rank"][field][0]["page_size_bytes"] += 1
            else:
                engine["rank"][field] = value
            return _runtime.run_replay_json(json.dumps(payload))

    with pytest.raises(RuntimeError, match=error):
        EngineReplayRunnerFactory(runtime=ChangedNativeInput()).create(0).run(spec)


def test_grouped_capacity_preserves_graph_reservation_and_rejects_fixed_blocks(grouped_case):
    profile, root = grouped_case
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(_prediction(profile, root)))
    args = deepcopy(spec.backend_deployment.agg_engine_args)
    args["cuda_graph_reserved_bytes"] = 1000
    args["timing_model"]["config"]["cuda_graph_reserved_bytes"] = 1000
    budget = materialize_aic_num_gpu_blocks(args)
    assert budget["kv_cache_capacity_bytes"] == 2486
    args["num_gpu_blocks"] = 1
    with pytest.raises(ValueError, match="fixed num_gpu_blocks"):
        materialize_aic_num_gpu_blocks(args)


def test_grouped_topology_admission_uses_each_rank_byte_budget_before_collection(grouped_case, monkeypatch):
    from aisimulate_core.sdk import RustForwardPassPerfModel

    profile, root = grouped_case
    # DEP has an explicitly larger rank-local weight reservation. It leaves
    # 1086 bytes, insufficient for even the 2048-byte full-history group.
    profile["deployments"][1]["resources"]["weights_bytes"] = 2500

    def no_timing(*_args, **_kwargs):
        pytest.fail("candidate fit constructed a timing estimator before collection")

    monkeypatch.setattr(RustForwardPassPerfModel, "best_available", no_timing)
    candidates = parallel_configs_for(
        profile["model"],
        "h200_sxm",
        backend="vllm",
        backend_version="0.25.1",
        deployment_mode="agg",
        gpu_budget=2,
        max_num_tokens=128,
        max_batch_size=4,
        fpm_profile=profile,
        systems_paths=[str(root)],
    )
    assert [candidate.shape.strategy for candidate in candidates] == ["tp"]


def test_grouped_kv_load_uses_bytes_without_inventing_token_capacity(grouped_case):
    profile, root = grouped_case
    candidates = parallel_configs_for(
        profile["model"],
        "h200_sxm",
        backend="vllm",
        backend_version="0.25.1",
        deployment_mode="agg",
        gpu_budget=2,
        max_num_tokens=128,
        max_batch_size=4,
        fpm_profile=profile,
        systems_paths=[str(root)],
    )
    assert len(candidates) == 2
    for candidate in candidates:
        sample = {
            "model_name": profile["model"],
            "hardware_sku": "h200_sxm",
            "backend": "vllm",
            "fpm_profile": profile,
            "systems_paths": [str(root)],
            "agg_block_size": 64,
            "agg_max_num_batched_tokens": 128,
            "agg_max_num_seqs": 4,
            "agg_gpu_memory_utilization": 0.9,
        }
        result = resolve_kv_load(
            sample,
            workload=Workload(isl=1152, osl=4, kv_load_ratio=1.0, request_count=2),
            parallel_config=candidate,
            ratio=1.0,
            backend_version="0.25.1",
        )
        # 1216 full-history + 528 sliding-window + 16 convolution bytes.
        assert result.role_request_cache_bytes == {"agg": 1760}
        assert result.role_capacity_bytes == {"agg": 3486 * candidate.shape.dp}
        assert result.concurrency_capacity == candidate.shape.dp
        assert result.role_capacity_tokens == {}


@pytest.mark.parametrize(
    "update,error",
    [
        ({"prefix_caching": True}, "prefix_caching=false"),
        ({"capacity": {"type": "fixed", "blocks": 1024}}, "byte budget"),
        ({"bytes_per_token": 8}, "byte budget"),
        ({"host_offload": {"num_host_blocks": 64}}, "only HBM"),
    ],
)
def test_grouped_incompatible_cache_settings_fail_before_replay(grouped_case, update, error):
    profile, root = grouped_case
    raw = _prediction(profile, root)
    raw["engine"]["workers"]["aggregated"]["kv_cache"].update(update)
    with pytest.raises(ValidationError, match=error):
        CorePredictionConfig.model_validate(raw)


@pytest.mark.parametrize(
    "changes",
    [
        {"mode": "disaggregated"},
        {"nextn": 1, "nextn_accepted": 1.0},
        {"speculation": {"kind": "ngram", "num_speculative_tokens": 1, "acceptance_rates": [0.5]}},
    ],
)
def test_grouped_unsupported_execution_modes_fail_before_replay(grouped_case, changes):
    profile, root = grouped_case
    raw = _prediction(profile, root)
    raw["engine"].update(changes)
    if changes.get("mode") == "disaggregated":
        worker = raw["engine"]["workers"].pop("aggregated")
        raw["engine"]["workers"] = {"prefill": deepcopy(worker), "decode": worker}
    with pytest.raises(ValidationError, match="grouped FPM cache supports only aggregated vLLM"):
        CorePredictionConfig.model_validate(raw)


def test_grouped_onboarding_profile_reaches_real_collector_preview_and_render(
    grouped_case, tmp_path, monkeypatch, capsys
):
    from collector.fpm_forward import cli as collector_cli
    from collector.fpm_forward import runner as collector_runner

    profile, _root = grouped_case
    profile = deepcopy(profile)
    for deployment in profile["deployments"]:
        deployment["gemm_quant_mode"] = "fp8_static"
        deployment["fmha_quant_mode"] = "fp8"
    local_config = tmp_path / "config.json"
    local_config.write_text(
        json.dumps(
            {
                "architectures": [profile["architecture"]],
                "model_type": "example_moe",
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 3,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 1024,
                "n_routed_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 128,
                "max_position_embeddings": 2048,
                "torch_dtype": "bfloat16",
                "quantization_config": {"quant_method": "fp8", "kv_cache_scheme": "FP8"},
                "sliding_window": 512,
                "layer_types": ["full_attention", "sliding_attention", "sliding_attention"],
            }
        )
    )
    request = SupportRequest.model_validate(
        {
            "identity": {
                "model": profile["model"],
                "model_revision": profile["model_revision"],
                "model_kind": "moe",
                "framework_version": "0.25.1",
                "gpu": "h200_sxm",
                "interconnect": "nvswitch",
            },
            "search": {
                "tensor_parallel": 2,
                "attention_data_parallel": 1,
                "moe_tensor_parallel": 2,
                "moe_expert_parallel": 1,
                "context_length": 2048,
            },
            "collection": {"max_num_tokens": 128, "max_batch_size": 4},
            "fpm_profile": profile,
        }
    )
    plan_dir = tmp_path / "collection"
    create_plan(request, plan_dir)
    command = json.loads((plan_dir / "commands.json").read_text())["fpm_plan_local"][3:]
    command += ["--fpm-model-config", str(local_config)]
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    assert collector_cli.main(command) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["fpm_profile"] == request.fpm_profile.model_dump(mode="json")
    assert preview["point_generation"]["method"] == "native_self_benchmark"
    assert preview["point_generation"]["planned_point_count"] is None
    assert {cell["parallel_strategy"] for cell in preview["cells"]} == {"pure_tp"}
    assert (
        preview["topology_memory_admission"][0]["estimates"][0]["provenance"]
        == (profile["deployments"][0]["resources"]["provenance"])
    )

    def render_without_launch(_args, resolved):
        plan, overrides = resolved
        assert plan.to_dict()["fpm_profile"] == preview["fpm_profile"]
        for cell in plan.cells:
            target = tmp_path / f"render-{cell.workload_kind}"
            target.mkdir()
            collector_runner._render_cell(plan, cell, target, overrides)
            script = (target / "run.sh").read_text()
            assert f"--model {profile['model']}" in script
            assert "--max-num-batched-tokens 128" in script
            assert "--max-num-seqs 4" in script
        return []

    monkeypatch.setattr(collector_cli, "run_resolved", render_without_launch)
    assert collector_cli.main([arg for arg in command if arg != "--plan-only"]) == 0


@pytest.mark.parametrize("dep", [False, True])
def test_onboard_validation_replays_windowed_weka_with_full_context(grouped_case, tmp_path, monkeypatch, dep):
    from aisimulate import supervision

    profile, _root = grouped_case
    request = SupportRequest.model_validate(
        {
            "identity": {
                "model": profile["model"],
                "model_revision": profile["model_revision"],
                "model_kind": "moe",
                "framework_version": "0.25.1",
                "gpu": "h200_sxm",
                "interconnect": "nvswitch",
            },
            "search": {
                "tensor_parallel": 1 if dep else 2,
                "attention_data_parallel": 2 if dep else 1,
                "moe_tensor_parallel": 1 if dep else 2,
                "moe_expert_parallel": 2 if dep else 1,
                "context_length": 2048,
            },
            "collection": {"max_num_tokens": 128, "max_batch_size": 4},
            "fpm_profile": profile,
        }
    )
    plan = tmp_path / "collection"
    create_plan(request, plan)
    _write_timings(plan / "systems", profile)
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "id": "window-boundary-play",
                "models": ["trace/source-model"],
                "block_size": 64,
                "hash_id_scope": "local",
                "requests": [
                    {
                        "t": 0.0,
                        "type": "s",
                        "model": "trace/source-model",
                        "in": 512,
                        "out": 3,
                        "hash_ids": list(range(8)),
                    },
                    {
                        "t": 0.1,
                        "type": "s",
                        "model": "trace/source-model",
                        "in": 1152,
                        "out": 4,
                        "hash_ids": list(range(18)),
                    },
                ],
            }
        )
        + "\n"
    )
    monkeypatch.setattr(supervision, "main", cli.main)
    output = tmp_path / "validation"
    assert (
        cli.main(
            [
                "onboard",
                "validate-fpm",
                "--config",
                str(plan / "request.yaml"),
                "--output-dir",
                str(plan),
                "--trace",
                str(trace),
                "--validation-output-dir",
                str(output),
            ]
        )
        == 0
    )
    validation = json.loads((output / "validation.json").read_text())
    assert validation["status"] == "covered"
    assert validation["accuracy"] == "not_assessed"
    report = json.loads((output / "prediction/prediction.json").read_text())
    assert report["completed_requests"] == 2
    assert report["agentic_model_projection"]["target_model"] == profile["model"]
    saved = CorePredictionConfig.from_yaml(output / "predict.yaml")
    assert saved.engine.workers.aggregated.kv_cache.prefix_caching is False
    assert saved.engine.fpm_profile.deployments[0].resources.cache_layout == "grouped"
