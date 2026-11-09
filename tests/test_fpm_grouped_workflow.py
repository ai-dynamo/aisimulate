# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public grouped-cache workflows with fictional resources and synthetic timings."""

from __future__ import annotations

import hashlib
import json
import os
import re
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
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"architectures": ["UnregisteredWindowMoeForCausalLM"]}))
    profile = {
        "schema_version": 1,
        "model": str(checkpoint),
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


def _write_timings(
    root,
    profile,
    *,
    decode_ceiling=8192,
    version="0.25.1",
    prefill_tokens=(128, 256, 512, 1024),
    prefill_kvs=(0, 1, 512, 1024, 2048, 8192),
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from aisimulate_core.sdk.fpm_profile import load_fpm_profile

    rows = []
    for index, deployment in enumerate(load_fpm_profile(profile).deployments):
        identity = deployment.model_dump(mode="json", exclude={"resources"})
        for batch in (1, 2, 4, 8):
            for phase, tokens, kvs in (
                ("prefill", (batch, *prefill_tokens), prefill_kvs),
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
    path = root / f"data/h200_sxm/vllm/{version}/fpm_forward_perf.parquet"
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
                "backend_version": version,
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


# Two pool blocks of 64 bytes for every running request, whatever its length.
_STATE_GROUP = {
    "name": "state",
    "kind": "state",
    "num_layers": 2,
    "block_size_tokens": 64,
    "page_size_bytes": 64,
    "blocks_per_request": 2,
}


def _with_state_group(profile):
    profile = deepcopy(profile)
    for deployment in profile["deployments"]:
        deployment["resources"]["cache_groups"].append(dict(_STATE_GROUP))
    return profile


# A convolution group as the probe import writes one for Inkling: a 4-token window, marked cacheable. Its block and
# page fit _with_prefix_evidence.
_CONVOLUTION_GROUP = {
    "name": "conv",
    "kind": "convolution",
    "num_layers": 3,
    "block_size_tokens": 64,
    "page_size_bytes": 64,
    "sliding_window": 4,
    "prefix_cacheable": True,
}


def _with_prefix_evidence(profile, *, hash_block=64, version="0.31.0"):
    """Runtime memory from a probe that ran with prefix caching on, and every group marked cacheable.

    The Rust gate needs one page size, a hash block that divides every cacheable block and one running block
    per state group, so every page is 64 bytes and every block is one hash block. Grouped prefix caching refuses
    convolution groups, so the fictional convolution group goes.
    """
    profile = deepcopy(profile)
    for deployment in profile["deployments"]:
        deployment["backend_version"] = version
        resources = deployment["resources"]
        resources["cache_groups"] = [group for group in resources["cache_groups"] if group["kind"] != "convolution"]
        for name in ("weights_bytes", "activations_bytes", "runtime_overhead_bytes", "comm_overhead_bytes"):
            resources.pop(name)
        # Runtime memory requires the exact recorded scheduler envelope of _prediction.
        resources["max_batch_size"] = 4
        resources["runtime_memory"] = {
            "kv_cache_bytes": 3456,
            "gpu_memory_utilization": 0.9,
            # The Rust gate admits this pool at 1024 tokens, as vLLM's startup check would.
            "max_model_len": 1024,
            "provenance": "Synthetic pool of 54 blocks of 64 bytes from a fictional probe. No GPU qualification.",
            "enable_prefix_caching": True,
            "mamba_cache_mode": "align",
            "hash_block_size_tokens": hash_block,
        }
        for group in resources["cache_groups"]:
            group.update(block_size_tokens=hash_block, page_size_bytes=64, prefix_cacheable=True)
            if group["kind"] == "state":
                group["blocks_per_request"] = 1
                if version == "0.32.0":
                    group["prefill_checkpoint_blocks"] = 0
    return profile


def _prefix_caching_prediction(profile, root, version):
    """Two identical 520-token requests in sequence on a grouped rank with prefix caching."""
    raw = _prediction(profile, root)
    raw["engine"]["backend_version"] = version
    # The rank runs at the probe's max_model_len. The Rust gate needs a pool for one prompt of max_model_len plus
    # the state blocks of one request: 16 + 13 + 2 = 31 of the 54 blocks at 1024 tokens.
    raw["engine"]["context_length"] = profile["deployments"][0]["resources"]["runtime_memory"]["max_model_len"]
    raw["engine"]["workers"]["aggregated"]["kv_cache"]["prefix_caching"] = True
    raw["traffic"] = {
        "source": {"type": "synthetic", "input_tokens": 520, "output_tokens": 4, "cached_prefix_tokens": 520},
        "load": {"type": "concurrency", "concurrency": 1},
        "stop": {"requests": 2},
    }
    return raw


_PD_PAIRING = (
    "grouped FPM cache P/D requires prefill and decode deployments with equal tensor parallelism and identical "
    "cache groups"
)
_PD_MARKERS = (
    "grouped FPM cache P/D requires every cache group to declare prefix_cacheable and transfer_bytes_per_block; "
    "re-import the runtime probe"
)
_PD_VERSION = "grouped FPM cache P/D is modeled for vLLM 0.31.0 and 0.32.0 only"
_PD_TRANSFER = "grouped FPM cache derives kv_transfer.bytes_per_token from its cache groups; set it to auto"


def _with_transfer_markers(profile, *, version="0.31.0"):
    """The P/D markers of a probe import on every grouped deployment, with the page as the transfer bytes."""
    profile = deepcopy(profile)
    for deployment in profile["deployments"]:
        deployment["backend_version"] = version
        for group in deployment["resources"].get("cache_groups", []):
            group.setdefault("prefix_cacheable", True)
            group["transfer_bytes_per_block"] = group["page_size_bytes"]
    return profile


def _pd_prediction(profile, root, *, bandwidth=None):
    """A prefill and a decode worker of tensor 2 on the profile's vLLM version, with a KV transfer at a bandwidth."""
    raw = _prediction(profile, root)
    engine = raw["engine"]
    engine["mode"] = "disaggregated"
    engine["backend_version"] = profile["deployments"][0]["backend_version"]
    worker = engine["workers"].pop("aggregated")
    engine["workers"] = {"prefill": deepcopy(worker), "decode": worker}
    if bandwidth is not None:
        engine["kv_transfer"] = {"bytes_per_token": "auto", "bandwidth_gb_per_second": bandwidth}
    return raw


def _pd_search(profile, root):
    """The search of _pd_prediction: disaggregated workers without prefix caching."""
    return {
        "deployment_mode": ["disagg"],
        "backend": ["vllm"],
        "backend_version": profile["deployments"][0]["backend_version"],
        "model_name": profile["model"],
        "hardware_sku": "h200_sxm",
        "fpm_profile": profile,
        "systems_paths": [str(root)],
        "gpu_budget": 4,
        "prefill_max_num_batched_tokens": [128],
        "prefill_max_num_seqs": [4],
        "prefill_enable_prefix_caching": False,
        "decode_max_num_batched_tokens": [128],
        "decode_max_num_seqs": [4],
    }


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
        (
            {
                "agg_enable_prefix_caching": False,
                "speculation": {"kind": "ngram", "num_speculative_tokens": 1, "acceptance_rates": [0.5]},
            },
            "grouped FPM cache requires vLLM without speculative decoding",
        ),
        ({"deployment_mode": ["disagg"], "gpu_budget": 4}, "grouped FPM cache requires prefill_enable_prefix_caching"),
        (
            {
                "deployment_mode": ["disagg"],
                "gpu_budget": 4,
                "prefill_enable_prefix_caching": False,
                "decode_native_host_offload": {"num_host_blocks": 64},
            },
            "only HBM",
        ),
        (
            {
                "deployment_mode": ["disagg"],
                "gpu_budget": 4,
                "prefill_enable_prefix_caching": False,
                "decode_num_gpu_blocks": 64,
            },
            "not scalar capacity",
        ),
        ({"deployment_mode": ["disagg"], "gpu_budget": 4, "prefill_enable_prefix_caching": False}, _PD_MARKERS),
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


@pytest.mark.parametrize("dep", [False, True])
def test_public_predict_runs_with_a_state_group(grouped_case, tmp_path, dep):
    profile, root = grouped_case
    raw = _prediction(_with_state_group(profile), root, dep=dep)
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    diagnostics = {}
    lowered = materialize_aic_num_gpu_blocks(spec.backend_deployment.agg_engine_args, memory_diagnostics=diagnostics)
    # The state group reaches the engine configuration with its block count. No other group has the key.
    assert lowered["kv_cache_groups"] == diagnostics["cache_groups"]
    groups = {group["name"]: group for group in lowered["kv_cache_groups"]}
    assert groups["state"] == {**_STATE_GROUP, "sliding_window": None}
    assert all("blocks_per_request" not in group for name, group in groups.items() if name != "state")
    assert lowered["kv_cache_capacity_bytes"] == 3486
    # 2048 full-history + 656 sliding-window + 272 convolution + 128 state bytes (2 blocks of 64).
    assert diagnostics["request_peak_cache_bytes"] == 3104
    path = tmp_path / "predict.yaml"
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "prediction"
    arguments = ["predict", "-c", str(path), "--output-dir", str(output), "--detail", "memory", "--capture-per-request"]
    assert cli.main(arguments) == 0
    report = json.loads((output / "prediction.json").read_text())
    assert report["completed_requests"] == 2
    assert all(row["output_length"] == 4 for row in report["per_request"])
    assert report["fpm_query_coverage"]["status"] == "covered"
    memory = report["details"]["sections"]["memory"]["roles"]["aggregated"]
    assert memory["total_kv_size_bytes"] == 3486
    assert memory["total_kv_size_tokens"] is None


def test_public_predict_does_not_clamp_logical_context_to_window(grouped_case, tmp_path):
    profile, root = grouped_case
    # Window-only allocation is bounded, but the first decode after the
    # prefill-emitted token must read all 1152 prompt KV tokens. A table ending
    # at 512 must fail, not shrink the logical context or physical byte budget.
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
    assert any(gap["coordinates"]["total_kv_read_tokens"] == 1152 for gap in gaps)


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


def test_grouped_kv_load_sizes_each_pd_role_by_its_bytes(grouped_case):
    profile, root = grouped_case
    profile = _with_transfer_markers(profile)
    search = {
        "backend": "vllm",
        "backend_version": "0.31.0",
        "deployment_mode": "disagg",
        "gpu_budget": 4,
        "max_num_tokens": 128,
        "max_batch_size": 4,
        "fpm_profile": profile,
        "systems_paths": [str(root)],
    }
    candidates = parallel_configs_for(profile["model"], "h200_sxm", **search)
    # The shared deployments pair only with their own shape.
    assert {(candidate.prefill.shape.tp, candidate.decode.shape.tp) for candidate in candidates} == {(2, 2), (1, 1)}
    sample = {
        "model_name": profile["model"],
        "hardware_sku": "h200_sxm",
        "backend": "vllm",
        "fpm_profile": profile,
        "systems_paths": [str(root)],
        **{
            f"{role}_{name}": value
            for role in ("prefill", "decode")
            for name, value in (
                ("block_size", 64),
                ("max_num_batched_tokens", 128),
                ("max_num_seqs", 4),
                ("gpu_memory_utilization", 0.9),
            )
        },
    }
    workload = Workload(isl=1152, osl=4, kv_load_ratio=1.0, request_count=2)
    for candidate in candidates:
        result = resolve_kv_load(
            sample, workload=workload, parallel_config=candidate, ratio=1.0, backend_version="0.31.0"
        )
        # Each role holds the 1,760 bytes per request of the aggregated rank, and the decode role sets the load.
        assert result.role_request_cache_bytes == {"prefill": 1760, "decode": 1760}
        assert result.role_capacity_bytes == {
            "prefill": 3486 * candidate.prefill.shape.dp,
            "decode": 3486 * candidate.decode.shape.dp,
        }
        assert result.concurrency_capacity == candidate.decode.shape.dp
        assert result.role_capacity_tokens == {}
    # A P/D role refuses its own fixed blocks, and the topology budget refuses speculation.
    with pytest.raises(ValueError, match="^grouped FPM cache requires replay without fixed scalar blocks$"):
        resolve_kv_load(
            {**sample, "decode_num_gpu_blocks": 64},
            workload=workload,
            parallel_config=candidates[0],
            ratio=1.0,
            backend_version="0.31.0",
        )
    with pytest.raises(ValueError, match="^grouped FPM cache requires replay without speculation$"):
        parallel_configs_for(profile["model"], "h200_sxm", nextn=1, **search)


def test_grouped_kv_load_sizes_each_pd_role_with_its_own_context_length(grouped_case, monkeypatch):
    profile, root = grouped_case
    profile = _with_transfer_markers(profile)
    for deployment in profile["deployments"]:
        resources = deployment["resources"]
        for name in ("weights_bytes", "activations_bytes", "runtime_overhead_bytes", "comm_overhead_bytes"):
            resources.pop(name)
        resources["max_batch_size"] = 4
        resources["runtime_memory"] = {
            "kv_cache_bytes": 3486,
            "max_model_len": 1024,
            "gpu_memory_utilization": 0.9,
            "provenance": "Synthetic initialized cache pool; no GPU qualification.",
        }
    role_context = {"prefill": 1024, "decode": 512}
    candidates = parallel_configs_for(
        profile["model"],
        "h200_sxm",
        backend="vllm",
        backend_version="0.31.0",
        deployment_mode="disagg",
        gpu_budget=4,
        max_num_tokens=128,
        max_batch_size=4,
        fpm_profile=profile,
        systems_paths=[str(root)],
        max_seq_len=2048,
        role_max_seq_len=role_context,
    )
    # The global context is above the probe's max_model_len, and each role sets its own context within it, as a
    # role deployment reads it. The two roles differ, so one role's context cannot size both.
    sample = {
        "model_name": profile["model"],
        "hardware_sku": "h200_sxm",
        "backend": "vllm",
        "fpm_profile": profile,
        "systems_paths": [str(root)],
        "context_length": 2048,
        **{
            f"{role}_{name}": value
            for role in ("prefill", "decode")
            for name, value in (
                ("block_size", 64),
                ("max_num_batched_tokens", 128),
                ("max_num_seqs", 4),
                ("gpu_memory_utilization", 0.9),
                ("context_length", role_context[role]),
            )
        },
    }
    from aisimulate.sweeper import kv_load

    sized = {}
    estimate = kv_load.estimate_grouped_cache_budget

    def record(*args, **kwargs):
        sized[kwargs["worker_type"]] = kwargs["context_length"]
        return estimate(*args, **kwargs)

    monkeypatch.setattr(kv_load, "estimate_grouped_cache_budget", record)
    result = resolve_kv_load(
        sample,
        workload=Workload(isl=256, osl=4, kv_load_ratio=1.0, request_count=2),
        parallel_config=candidates[0],
        ratio=1.0,
        backend_version="0.31.0",
    )
    assert sized == {"prefill": 1024, "decode": 512}
    assert set(result.role_request_cache_bytes) == {"prefill", "decode"}


def test_grouped_state_group_adds_fixed_bytes_per_request(grouped_case):
    profile, root = grouped_case
    profile = _with_state_group(profile)
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
        # The 1760 bytes of the other groups plus 2 state blocks of 64 bytes. The capacity does not change.
        assert result.role_request_cache_bytes == {"agg": 1888}
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
        {"nextn": 1, "nextn_accepted": 1.0},
        {"speculation": {"kind": "ngram", "num_speculative_tokens": 1, "acceptance_rates": [0.5]}},
    ],
)
def test_grouped_unsupported_execution_modes_fail_before_replay(grouped_case, changes):
    profile, root = grouped_case
    raw = _prediction(profile, root)
    raw["engine"].update(changes)
    with pytest.raises(ValidationError, match="grouped FPM cache supports only vLLM without speculative decoding"):
        CorePredictionConfig.model_validate(raw)


@pytest.mark.parametrize(
    "case,error",
    [
        ("vllm_0310", None),
        ("vllm_0320", None),
        ("no_markers", _PD_MARKERS),
        ("vllm_0300", _PD_VERSION),
        ("unequal_tp", _PD_PAIRING),
        ("other_groups", _PD_PAIRING),
        ("linear_decode", _PD_PAIRING),
        ("transfer_bytes", _PD_TRANSFER),
    ],
)
def test_grouped_pd_requires_equal_tp_identical_groups_and_markers(grouped_case, case, error):
    from aisimulate.sweeper.config import SearchSpace

    profile, root = grouped_case
    version = {"vllm_0320": "0.32.0", "vllm_0300": "0.30.0"}.get(case, "0.31.0")
    if case == "linear_decode":
        profile = _mixed_profile(profile)
    profile = _with_transfer_markers(profile, version=version)
    dep = {"tensor": 1, "attention_data": 2, "moe_tensor": 1, "moe_expert": 2}
    if case == "no_markers":
        for deployment in profile["deployments"]:
            for group in deployment["resources"]["cache_groups"]:
                del group["prefix_cacheable"], group["transfer_bytes_per_block"]
    elif case == "other_groups":
        # Role-tagged deployments of one shape: the decode deployment lacks the convolution group.
        prefill = profile["deployments"][0]
        decode = {**deepcopy(prefill), "worker_type": "decode"}
        decode["resources"]["cache_groups"] = decode["resources"]["cache_groups"][:-1]
        profile["deployments"][:1] = [{**prefill, "worker_type": "prefill"}, decode]
    raw = _pd_prediction(profile, root)
    search = _pd_search(profile, root)
    workers = raw["engine"]["workers"]
    # A search drops the enumerated pairs that fail the pairing rule, so it names such a pair with a pin.
    tensor_pin, dep_pin = {"tp": 2, "moe_tp": 2}, {"tp": 1, "attention_dp": 2, "moe_ep": 2}
    if case == "unequal_tp":
        workers["decode"]["parallelism"] = dep
        search["parallel_configs_by_mode"] = {"disagg": [{"prefill": tensor_pin, "decode": dep_pin}]}
    elif case == "other_groups":
        search["parallel_configs_by_mode"] = {"disagg": [{"prefill": tensor_pin, "decode": tensor_pin}]}
    elif case == "linear_decode":
        # The mixed profile groups the DEP deployment and keeps the tensor-parallel one linear.
        workers["prefill"]["parallelism"] = dep
        search["parallel_configs_by_mode"] = {"disagg": [{"prefill": dep_pin, "decode": tensor_pin}]}
    elif case == "transfer_bytes":
        raw["engine"]["kv_transfer"] = {"bytes_per_token": 7}
        search["kv_transfer_bytes_per_token"] = 7
    if error is None:
        CorePredictionConfig.model_validate(raw)
        SearchSpace.model_validate(search)
        return
    with pytest.raises(ValidationError, match=re.escape(error)):
        CorePredictionConfig.model_validate(raw)
    with pytest.raises(ValidationError, match=re.escape(error)):
        SearchSpace.model_validate(search)


def test_grouped_pd_recommendation_with_fixed_unequal_tp_gives_the_pairing_text(grouped_case):
    from aisimulate.recommend import recommendation_to_sweeper

    profile, root = grouped_case
    raw = _pd_prediction(_with_transfer_markers(profile), root)
    workers = raw["engine"]["workers"]
    workers["decode"]["parallelism"] = {"tensor": 1, "attention_data": 2, "moe_tensor": 1, "moe_expert": 2}
    # A preset list for each role lowers to the pinned pairs of the two lists.
    for worker in workers.values():
        worker["parallelism"] = {"preset": [{**worker["parallelism"], "replicas": 1, "pipeline": 1}]}
    raw["optimization"] = {"constraints": {"max_candidate_gpus": 4}}
    config = CoreRecommendationConfig.model_validate(raw)
    with pytest.raises(ValidationError, match=re.escape(_PD_PAIRING)):
        recommendation_to_sweeper(config)


def test_disaggregated_prediction_with_a_grouped_aggregated_worker_keeps_the_role_error(grouped_case):
    # Only the prefill and the decode deployment decide the P/D rule. Here both are linear, so the grouped aggregated
    # worker fails on its role, not on a P/D text.
    profile, root = grouped_case
    raw = _pd_prediction(_mixed_profile(profile), root)
    workers = raw["engine"]["workers"]
    workers["aggregated"] = deepcopy(workers["prefill"])
    workers["aggregated"]["parallelism"] = {"tensor": 1, "attention_data": 2, "moe_tensor": 1, "moe_expert": 2}
    with pytest.raises(ValidationError, match="disaggregated mode rejects aggregated workers"):
        CorePredictionConfig.model_validate(raw)


@pytest.mark.parametrize("version", ["0.31.0", "0.32.0"])
def test_grouped_pd_prediction_replays_one_p_one_d(grouped_case, tmp_path, version):
    profile, root = grouped_case
    profile = _with_transfer_markers(_with_state_group(profile), version=version)
    _write_timings(root, profile, version=version)
    raw = _pd_prediction(profile, root, bandwidth=1.0)
    deployment = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw)).backend_deployment
    prefill, decode = deployment.prefill_engine_args, deployment.decode_engine_args
    for args in (prefill, decode):
        # A grouped rank derives the transfer payload from its groups, so it has no scalar bytes per token.
        assert "kv_transfer_bytes_per_token" not in args
        assert (args["kv_transfer_bandwidth"], args["kv_transfer_timing_mode"]) == (1.0, "destination_missing")
        # The Rust gate refuses a grouped prefill or decode rank without vLLM 0.31.0 or 0.32.0, also with prefix
        # caching off.
        assert (args["enable_prefix_caching"], args["vllm_version"]) == (False, version)
        assert "prefix_match_unit" not in args
    groups = materialize_aic_num_gpu_blocks(prefill)["kv_cache_groups"]
    assert [group["name"] for group in groups] == ["full", "window", "conv", "state"]
    assert materialize_aic_num_gpu_blocks(decode)["kv_cache_groups"] == groups
    path = tmp_path / "predict.yaml"
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "prediction"
    assert cli.main(["predict", "-c", str(path), "--output-dir", str(output)]) == 0
    assert json.loads((output / "prediction.json").read_text())["completed_requests"] == 2


@pytest.mark.parametrize("version", ["0.31.0", "0.32.0"])
def test_grouped_pd_prediction_with_prefix_caching_on_both_ranks(grouped_case, tmp_path, version):
    # Planning writes prefix_caching: true into an accepted role's worker.yaml, so a composed P/D prediction of two
    # planned roles runs both ranks in block mode.
    profile, root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile), version=version)
    profile = _with_transfer_markers(profile, version=version)
    _write_timings(root, profile, version=version)
    raw = _pd_prediction(profile, root, bandwidth=1.0)
    aggregated = _prefix_caching_prediction(profile, root, version)
    raw["engine"]["context_length"] = aggregated["engine"]["context_length"]
    raw["traffic"] = aggregated["traffic"]
    for worker in raw["engine"]["workers"].values():
        worker["kv_cache"]["prefix_caching"] = True
    deployment = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw)).backend_deployment
    for args in (deployment.prefill_engine_args, deployment.decode_engine_args):
        assert (args["enable_prefix_caching"], args["prefix_match_unit"], args["vllm_version"]) == (True, 64, version)
    path = tmp_path / "predict.yaml"
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "prediction"
    assert cli.main(["predict", "-c", str(path), "--output-dir", str(output), "--capture-per-request"]) == 0
    report = json.loads((output / "prediction.json").read_text())
    assert report["completed_requests"] == 2
    rows = report["per_request"]
    # The prefill rank computes 519 tokens and keeps the state at 512, so the identical second request reuses 512.
    assert [row["reused_input_tokens"] for row in rows] == [0, 512]
    # The first transfer moves 9 full-attention blocks, 9 window blocks and 1 state block, 64 bytes each: 1,216
    # bytes. The 512-token window skips 519 - 512 + 1 = 8 of the 519 tokens, less than a block, so all 9 window
    # blocks lie inside it. The decode rank keeps the 8 full blocks that the first load cached, so the second moves
    # 1 full block and the same fixed bytes: 704 bytes. At 1 GB/s a byte takes 1e-6 ms.
    delays = [row["destination_activated_ms"] - row["destination_reserved_ms"] for row in rows]
    assert delays == [pytest.approx(1216e-6, abs=1e-9), pytest.approx(704e-6, abs=1e-9)]


@pytest.mark.parametrize("version", ["0.31.0", "0.32.0"])
def test_grouped_pd_prediction_with_prefix_caching_on_the_prefill_rank_only(grouped_case, tmp_path, version):
    # A search with prefix evidence defaults to prefix caching on prefill workers and off on decode workers.
    profile, root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile), version=version)
    profile = _with_transfer_markers(profile, version=version)
    _write_timings(root, profile, version=version)
    raw = _pd_prediction(profile, root, bandwidth=1.0)
    aggregated = _prefix_caching_prediction(profile, root, version)
    raw["engine"]["context_length"] = aggregated["engine"]["context_length"]
    raw["traffic"] = aggregated["traffic"]
    raw["engine"]["workers"]["prefill"]["kv_cache"]["prefix_caching"] = True
    deployment = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw)).backend_deployment
    prefill, decode = deployment.prefill_engine_args, deployment.decode_engine_args
    assert (prefill["enable_prefix_caching"], prefill["prefix_match_unit"]) == (True, 64)
    assert (decode["enable_prefix_caching"], "prefix_match_unit" in decode) == (False, False)
    assert prefill["vllm_version"] == decode["vllm_version"] == version
    path = tmp_path / "predict.yaml"
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "prediction"
    assert cli.main(["predict", "-c", str(path), "--output-dir", str(output), "--capture-per-request"]) == 0
    report = json.loads((output / "prediction.json").read_text())
    assert report["completed_requests"] == 2
    rows = report["per_request"]
    # The prefill rank reuses 512 tokens as with prefix caching on both ranks. The cold decode rank keeps no blocks,
    # so both transfers move the full 1,216 bytes.
    assert [row["reused_input_tokens"] for row in rows] == [0, 512]
    delays = [row["destination_activated_ms"] - row["destination_reserved_ms"] for row in rows]
    assert delays == [pytest.approx(1216e-6, abs=1e-9), pytest.approx(1216e-6, abs=1e-9)]


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


def _plan_and_validate_weka(profile, tmp_path, monkeypatch, *, dep=False, version="0.25.1", context_length=2048):
    """Plan the onboarding of a grouped profile and replay a windowed Weka trace with validate-fpm.

    A shorter context, such as a probe's max_model_len, gets a workload and a second request that fit in it. The
    second request still crosses the 512-token window.
    """
    from aisimulate import supervision

    # The default workload of 1,024 input and 128 output tokens needs a context of 1,152 tokens.
    workload = {} if context_length >= 1152 else {"workload": {"input_tokens": 512, "output_tokens": 4}}
    second = min(1152, context_length - 64)
    request = SupportRequest.model_validate(
        {
            "identity": {
                "model": profile["model"],
                "model_revision": profile["model_revision"],
                "model_kind": "moe",
                "framework_version": version,
                "gpu": "h200_sxm",
                "interconnect": "nvswitch",
            },
            "search": {
                "tensor_parallel": 1 if dep else 2,
                "attention_data_parallel": 2 if dep else 1,
                "moe_tensor_parallel": 1 if dep else 2,
                "moe_expert_parallel": 2 if dep else 1,
                "context_length": context_length,
            },
            **workload,
            "collection": {"max_num_tokens": 128, "max_batch_size": 4},
            "fpm_profile": profile,
        }
    )
    plan = tmp_path / "collection"
    create_plan(request, plan)
    _write_timings(plan / "systems", profile, version=version)
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
                        "in": second,
                        "out": 4,
                        "hash_ids": list(range(second // 64)),
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
    return plan, output


@pytest.mark.parametrize("dep", [False, True])
def test_onboard_validation_replays_windowed_weka_with_full_context(grouped_case, tmp_path, monkeypatch, dep):
    profile, _root = grouped_case
    _plan, output = _plan_and_validate_weka(profile, tmp_path, monkeypatch, dep=dep)
    validation = json.loads((output / "validation.json").read_text())
    assert validation["status"] == "covered"
    assert validation["accuracy"] == "not_assessed"
    report = json.loads((output / "prediction/prediction.json").read_text())
    assert report["completed_requests"] == 2
    assert report["agentic_model_projection"]["target_model"] == profile["model"]
    saved = CorePredictionConfig.from_yaml(output / "predict.yaml")
    assert saved.engine.workers.aggregated.kv_cache.prefix_caching is False
    assert saved.engine.fpm_profile.deployments[0].resources.cache_layout == "grouped"


@pytest.mark.parametrize("profile_field", ["aic_fpm_profile", "fpm_profile"])
def test_grouped_runner_capability_rejects_unsupported_stack(grouped_case, profile_field):
    from aisimulate.runner import EngineReplayRunnerFactory
    from aisimulate.sweeper.replay import RunnerCapabilities

    profile, root = grouped_case
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(_prediction(profile, root)))
    with pytest.raises(ValueError, match="grouped FPM caches; use --stack engine"):
        RunnerCapabilities(supported_backend_topologies=(("vllm", "agg"),)).require_compatible(spec)
    EngineReplayRunnerFactory().capabilities().require_compatible(spec)
    linear_spec = prediction_to_replay_spec(
        CorePredictionConfig.model_validate(_prediction(_mixed_profile(profile), root))
    )
    RunnerCapabilities(supported_backend_topologies=(("vllm", "agg"),)).require_compatible(linear_spec)

    from dataclasses import replace

    for profile_variant, grouped in ((profile, True), (_mixed_profile(profile), False)):
        selected = profile_variant["deployments"][0]
        flat = {
            "aic_model_path": profile_variant["model"],
            "aic_system": selected["system"],
            profile_field: profile_variant,
            "aic_tp_size": selected["tp"],
            "aic_attention_dp_size": selected["dp"],
            "aic_moe_tp_size": selected["moe_tp"],
            "aic_moe_ep_size": selected["moe_ep"],
        }
        flat_spec = replace(spec, backend_deployment=replace(spec.backend_deployment, agg_engine_args=flat))
        capabilities = RunnerCapabilities(supported_backend_topologies=(("vllm", "agg"),))
        if grouped:
            with pytest.raises(ValueError, match="grouped FPM caches"):
                capabilities.require_compatible(flat_spec)
        else:
            capabilities.require_compatible(flat_spec)


def test_aggregated_only_grouped_profile_skips_unavailable_disagg(grouped_case):
    from aisimulate.sweeper.config import SmartSearchConfig
    from aisimulate.sweeper.search_space import enumerate_branches

    profile, root = grouped_case
    for deployment in profile["deployments"]:
        deployment["worker_type"] = "aggregated"
    raw = _mixed_search(profile, root)
    raw.update(fpm_profile=profile, agg_enable_prefix_caching=False)
    raw.pop("deployment_mode")
    config = SmartSearchConfig(search_space=raw, workload=Workload(isl=1152, osl=4, concurrency=1, request_count=2))
    with pytest.warns(UserWarning, match="disagg.*skipped"):
        branches = enumerate_branches(config, max_seq_len=2048)
    assert [branch.deployment_mode for branch in branches] == ["agg"]


def test_grouped_profile_without_a_valid_pair_skips_disagg(grouped_case):
    from aisimulate.sweeper.config import SmartSearchConfig
    from aisimulate.sweeper.search_space import enumerate_branches

    profile, root = grouped_case
    profile = _with_transfer_markers(profile)
    tensor, dep = profile["deployments"]
    # Prefill workers run only at tensor 2 and decode workers only at DEP, so no pair has equal tensor parallelism.
    profile["deployments"] = [
        {**deepcopy(tensor), "worker_type": "aggregated"},
        {**tensor, "worker_type": "prefill"},
        {**dep, "worker_type": "decode"},
    ]
    raw = {
        **_pd_search(profile, root),
        "deployment_mode": ["disagg", "agg"],
        "agg_max_num_batched_tokens": [128],
        "agg_max_num_seqs": [4],
        "agg_enable_prefix_caching": False,
    }
    # Validation warns about nothing. The root configuration turns any warning outside pytest.warns into an error.
    config = SmartSearchConfig(search_space=raw, workload=Workload(isl=1152, osl=4, concurrency=1, request_count=2))
    reason = "no prefill/decode deployment pair with equal tensor parallelism and identical cache groups"
    # One skip warning carries the reason. pytest.warns raises any other warning again, so a second one fails.
    with pytest.warns(UserWarning, match=f"^smart-sweep: deployment_mode='disagg' skipped — {reason}$"):
        branches = enumerate_branches(config, max_seq_len=2048)
    assert [branch.deployment_mode for branch in branches] == ["agg"]


def test_grouped_pd_recommendation_enumerates_equal_tp_pairs(grouped_case):
    from aisimulate.recommend import recommendation_to_sweeper
    from aisimulate.sweeper.search_space import enumerate_branches

    profile, root = grouped_case
    profile = _with_transfer_markers(profile)
    tensor, dep = profile["deployments"]
    # The role-tagged deployments at tensor 2 pair. The DEP deployment serves only decode workers and pairs with none.
    profile["deployments"] = [
        {**deepcopy(tensor), "worker_type": "prefill"},
        {**tensor, "worker_type": "decode"},
        {**dep, "worker_type": "decode"},
    ]
    raw = _pd_prediction(profile, root, bandwidth=1.0)
    for worker in raw["engine"]["workers"].values():
        worker["parallelism"] = {"preset": "default"}
    raw["optimization"] = {"constraints": {"max_candidate_gpus": 4}}
    branches = enumerate_branches(recommendation_to_sweeper(CoreRecommendationConfig.model_validate(raw)))
    assert [branch.deployment_mode for branch in branches] == ["disagg"]
    candidates = branches[0].parallel_configs
    assert candidates
    assert all(candidate.prefill.shape.tp == candidate.decode.shape.tp == 2 for candidate in candidates)


def test_grouped_heterogeneous_pd_search_keeps_only_valid_pairs(grouped_case):
    from aisimulate.sweeper.config import SmartSearchConfig
    from aisimulate.sweeper.search_space import enumerate_branches

    profile, root = grouped_case
    profile = _with_transfer_markers(profile)
    profile["deployments"] += [{**deepcopy(deployment), "system": "h100_sxm"} for deployment in profile["deployments"]]
    (root / "h100_sxm.yaml").write_bytes((root / "h200_sxm.yaml").read_bytes())
    raw = {**_pd_search(profile, root), "decode_hardware_sku": "h100_sxm"}
    config = SmartSearchConfig(search_space=raw, workload=Workload(isl=1152, osl=4, concurrency=1, request_count=2))
    (branch,) = enumerate_branches(config, max_seq_len=2048)
    # Each role is enumerated on its own hardware, and the pairs keep equal tensor parallelism.
    assert {(candidate.prefill.shape, candidate.decode.shape) for candidate in branch.parallel_configs} == {
        (shape, shape) for shape in {candidate.prefill.shape for candidate in branch.parallel_configs}
    }
    assert {candidate.prefill.shape.tp for candidate in branch.parallel_configs} == {1, 2}


@pytest.mark.parametrize(
    "prefix_caching",
    [{"prefill": False, "decode": False}, {"prefill": True, "decode": True}, {}],
    ids=["off", "on", "defaults"],
)
def test_grouped_sweeper_pd_candidate_ranks_take_the_vllm_version(grouped_case, prefix_caching):
    from aisimulate.sweeper.config import SearchSpace
    from aisimulate.sweeper.deploy import build_backend_deployment
    from aisimulate.sweeper.parallel_enum import DisaggParallelConfig, ParallelShape, ReplicaParallelConfig
    from aisimulate.sweeper.sample import unroll_sample

    profile, root = grouped_case
    profile = _with_transfer_markers(_with_prefix_evidence(_with_state_group(profile)))
    raw = {
        **_pd_search(profile, root),
        # A recommendation with a KV transfer lowers bytes_per_token "auto" into the search.
        "kv_transfer_bytes_per_token": "auto",
        "kv_transfer_bandwidth": 1.0,
    }
    # Without settings, the search keeps its defaults: prefix caching on prefill workers and off on decode workers.
    raw.pop("prefill_enable_prefix_caching")
    raw.update({f"{role}_enable_prefix_caching": value for role, value in prefix_caching.items()})
    worker = ReplicaParallelConfig(ParallelShape(tp=2, dp=1, moe_tp=2, moe_ep=1), replicas=1)
    sample = unroll_sample(
        search_space=SearchSpace.model_validate(raw),
        selection={
            "deployment_mode": "disagg",
            "backend": "vllm",
            "prefill_max_num_batched_tokens": 128,
            "prefill_max_num_seqs": 4,
            "decode_max_num_batched_tokens": 128,
            "decode_max_num_seqs": 4,
        },
        parallel_config=DisaggParallelConfig(prefill=worker, decode=worker),
    )
    deployment = build_backend_deployment(sample, backend_version="0.31.0")
    defaults = {"prefill": True, "decode": False}
    for role, args in (("prefill", deployment.prefill_engine_args), ("decode", deployment.decode_engine_args)):
        # Each grouped rank reads the transfer rules of its version and derives the payload from its groups.
        assert (args["vllm_version"], args["kv_transfer_bandwidth"]) == ("0.31.0", 1.0)
        assert "kv_transfer_bytes_per_token" not in args
        assert args["kv_transfer_timing_mode"] == "destination_missing"
        assert args["enable_prefix_caching"] is prefix_caching.get(role, defaults[role])
        if args["enable_prefix_caching"]:
            # Without a search context length, the rank takes the profile's.
            assert (args["prefix_match_unit"], args["max_model_len"]) == (64, 2048)
        else:
            assert not {"prefix_match_unit", "max_model_len"} & args.keys()


@pytest.mark.parametrize("prefix_caching", [False, True])
def test_grouped_pd_recommendation_replays_every_candidate(grouped_case, tmp_path, monkeypatch, prefix_caching):
    from aisimulate.recommend import _run_recommendation
    from aisimulate.sweeper import SweepResult
    from aisimulate.sweeper.result import CandidateStatus

    monkeypatch.setattr("aisimulate.recommend.run_recommendation", _run_recommendation)
    profile, root = grouped_case
    if prefix_caching:
        profile = _with_prefix_evidence(_with_state_group(profile))
    profile = _with_transfer_markers(profile)
    _write_timings(root, profile, version="0.31.0")
    raw = _pd_prediction(profile, root, bandwidth=1.0)
    if prefix_caching:
        aggregated = _prefix_caching_prediction(profile, root, "0.31.0")
        raw["engine"]["context_length"] = aggregated["engine"]["context_length"]
        raw["traffic"] = aggregated["traffic"]
    for worker in raw["engine"]["workers"].values():
        worker["parallelism"] = {"preset": "default"}
        worker["kv_cache"]["prefix_caching"] = prefix_caching
    raw["optimization"] = {"constraints": {"max_candidate_gpus": 4}}
    raw["optimizer"] = {"algorithm": "random", "max_trials": 2, "parallelism": 1}
    path = tmp_path / "recommend.yaml"
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "recommendation"
    assert cli.main(["recommend", "-c", str(path), "--output-dir", str(output)]) == 0
    result = SweepResult.from_json((output / "recommendation.json").read_text())
    # Both equal-TP pairs replay. A candidate that failed its build or the Rust gate would not be feasible.
    assert len(result.candidates) == 2
    assert all(candidate.status is CandidateStatus.FEASIBLE for candidate in result.candidates)
    exported = CorePredictionConfig.from_yaml(output / "recommendations/0001.yaml")
    assert exported.engine.kv_transfer.bandwidth_gb_per_second == 1.0
    assert exported.engine.workers.prefill.kv_cache.prefix_caching is prefix_caching


def _prefix_search(profile, root, version):
    return {**_mixed_search(profile, root), "fpm_profile": profile, "backend_version": version}


@pytest.mark.parametrize(
    "case,error",
    [
        ("no_evidence", "prefix_caching=false for cold replay"),
        ("probe_without_prefix_caching", "prefix_caching=false for cold replay"),
        ("unmarked_group", "every cache group to declare prefix_cacheable; re-import the runtime probe"),
        ("convolution_group", "grouped prefix caching does not support convolution cache groups"),
        ("state_without_align", "requires mamba_cache_mode=align for a prefix-cacheable state group"),
        ("vllm_0320", None),
        ("vllm_0300", "is modeled for vLLM 0.31.0 and 0.32.0 only"),
        ("unequal_pages", "requires equal page_size_bytes across cache groups"),
        ("other_unit", "must equal the profile's hash_block_size_tokens for a grouped FPM cache"),
    ],
)
def test_grouped_prefix_caching_needs_vllm_evidence_with_align_mode(grouped_case, case, error):
    from aisimulate.sweeper.config import SearchSpace

    profile, root = grouped_case
    version = {"no_evidence": "0.25.1", "vllm_0320": "0.32.0", "vllm_0300": "0.30.0"}.get(case, "0.31.0")
    profile = _with_state_group(profile)
    if case != "no_evidence":
        profile = _with_prefix_evidence(profile, version=version)
    for deployment in profile["deployments"]:
        resources = deployment["resources"]
        if case == "probe_without_prefix_caching":
            resources["runtime_memory"].update(enable_prefix_caching=False, mamba_cache_mode="none")
        elif case == "unmarked_group":
            del resources["cache_groups"][1]["prefix_cacheable"]
        elif case == "convolution_group":
            resources["cache_groups"].append(dict(_CONVOLUTION_GROUP))
        elif case == "state_without_align":
            resources["runtime_memory"]["mamba_cache_mode"] = "none"
        elif case == "unequal_pages":
            resources["cache_groups"][1]["page_size_bytes"] = 16
    raw = _prediction(profile, root)
    raw["engine"]["backend_version"] = version
    cache = raw["engine"]["workers"]["aggregated"]["kv_cache"]
    cache["prefix_caching"] = True
    if case == "other_unit":
        cache["prefix_match_unit"] = 32
    # The search space has no prefix_match_unit, so only the prediction can state another unit.
    search = None if case == "other_unit" else _prefix_search(profile, root, version)
    if error is None:
        CorePredictionConfig.model_validate(raw)
        SearchSpace.model_validate(search)
        return
    with pytest.raises(ValidationError, match=error):
        CorePredictionConfig.model_validate(raw)
    if search is not None:
        with pytest.raises(ValidationError, match=error):
            SearchSpace.model_validate(search)


@pytest.mark.parametrize("version", ["0.31.0", "0.32.0"])
def test_grouped_prefix_caching_replays_with_reuse(grouped_case, tmp_path, version):
    profile, root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile), version=version)
    _write_timings(root, profile, version=version)
    raw = _prefix_caching_prediction(profile, root, version)
    args = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw)).backend_deployment.agg_engine_args
    assert (args["enable_prefix_caching"], args["prefix_match_unit"], args["vllm_version"]) == (True, 64, version)
    path = tmp_path / "predict.yaml"
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "prediction"
    assert cli.main(["predict", "-c", str(path), "--output-dir", str(output), "--capture-per-request"]) == 0
    report = json.loads((output / "prediction.json").read_text())
    assert report["completed_requests"] == 2
    # Both versions keep the state at 512, the last 64-token boundary below the 520-token prompt, so the
    # identical second request reuses 512 tokens.
    assert [row["reused_input_tokens"] for row in report["per_request"]] == [0, 512]


def test_grouped_prefix_caching_rank_takes_the_probe_hash_block(grouped_case):
    profile, root = grouped_case
    raw = _prefix_caching_prediction(_with_prefix_evidence(_with_state_group(profile)), root, "0.31.0")

    def engine_args():
        return prediction_to_replay_spec(CorePredictionConfig.model_validate(raw)).backend_deployment.agg_engine_args

    implicit = engine_args()
    raw["engine"]["workers"]["aggregated"]["kv_cache"]["prefix_match_unit"] = 64
    assert engine_args() == implicit
    # The Rust gate also refuses a rank without max_model_len. The compiler writes it for every vLLM rank.
    assert (implicit["prefix_match_unit"], implicit["vllm_version"], implicit["max_model_len"]) == (64, "0.31.0", 1024)
    raw["engine"]["workers"]["aggregated"]["kv_cache"] = {"prefix_caching": False}
    assert not {"prefix_match_unit", "vllm_version"} & engine_args().keys()


@pytest.mark.parametrize(
    "hash_block,budget,refused",
    [(2176, 8190, True), (2176, 8192, False), (2174, 8192, True)],
)
def test_glm_grid_refuses_off_grid_units_and_budgets(grouped_case, hash_block, budget, refused):
    from aisimulate.sweeper.config import SearchSpace

    profile, root = grouped_case
    profile = _with_prefix_evidence(profile, hash_block=hash_block)
    profile["architecture"] = "Glm5NextForConditionalGeneration"
    for deployment in profile["deployments"]:
        # Runtime memory requires the exact recorded budget.
        deployment["resources"]["max_num_tokens"] = budget
    raw = _prediction(profile, root)
    raw["engine"]["backend_version"] = "0.31.0"
    worker = raw["engine"]["workers"]["aggregated"]
    worker["kv_cache"]["prefix_caching"] = True
    worker["scheduler"]["max_batched_tokens"] = budget
    search = {**_prefix_search(profile, root, "0.31.0"), "agg_max_num_batched_tokens": [budget]}
    if not refused:
        CorePredictionConfig.model_validate(raw)
        SearchSpace.model_validate(search)
        return
    error = "GLM-5.3-Flash prefix caching requires prefix_match_unit and max_batched_tokens that are multiples of 4"
    with pytest.raises(ValidationError, match=error):
        CorePredictionConfig.model_validate(raw)
    with pytest.raises(ValidationError, match=error):
        SearchSpace.model_validate(search)


def test_grouped_plan_keeps_prefix_caching_on_with_evidence_and_validation_cold(grouped_case, tmp_path, monkeypatch):
    profile, _root = grouped_case
    profile = _with_prefix_evidence(profile)
    # Onboarding keeps the context within the probe's max_model_len (support/cli.py:317-320).
    plan, output = _plan_and_validate_weka(profile, tmp_path, monkeypatch, version="0.31.0", context_length=1024)
    for name in ("predict/pilot.yaml", "recommend/pilot.yaml"):
        worker = yaml.safe_load((plan / name).read_text())["engine"]["workers"]["aggregated"]
        assert worker["kv_cache"]["prefix_caching"] is True
    validation = json.loads((output / "validation.json").read_text())
    assert validation["status"] == "covered"
    # validate-fpm tests FPM timing, so its replay stays cold.
    saved = CorePredictionConfig.from_yaml(output / "predict.yaml")
    assert saved.engine.workers.aggregated.kv_cache.prefix_caching is False


def test_cold_validation_drops_the_unit_that_only_prefix_caching_reads(grouped_case, tmp_path, monkeypatch):
    from aisimulate.support.validation import _prediction_config

    profile, _root = grouped_case
    plan, _output = _plan_and_validate_weka(
        _with_prefix_evidence(profile), tmp_path, monkeypatch, version="0.31.0", context_length=1024
    )
    # The probe's hash block is a valid prefix_match_unit for a grouped rank with prefix caching. The plan does not
    # write it, and validate-fpm refuses a changed pilot, so the test gives such a pilot to _prediction_config.
    pilot = yaml.safe_load((plan / "predict/pilot.yaml").read_text())
    cache = pilot["engine"]["workers"]["aggregated"]["kv_cache"]
    cache["prefix_match_unit"] = 64
    CorePredictionConfig.model_validate(pilot)
    edited = tmp_path / "edited"
    (edited / "predict").mkdir(parents=True)
    (edited / "predict/pilot.yaml").write_text(yaml.safe_dump(pilot))
    # On a grouped rank only prefix caching reads the unit, so the cold replay turns prefix caching off and drops it.
    cold = _prediction_config(SupportRequest.from_yaml(plan / "request.yaml"), edited, tmp_path / "trace.jsonl")
    del cache["prefix_match_unit"]
    assert cold["engine"]["workers"]["aggregated"]["kv_cache"] == {**cache, "prefix_caching": False}


def test_grouped_plan_without_the_probe_hash_block_stays_cold(grouped_case, tmp_path, monkeypatch):
    # Replay hashes prompts in blocks of the probe's hash block, so a probe record without it is no evidence.
    profile, _root = grouped_case
    profile = _with_prefix_evidence(profile)
    for deployment in profile["deployments"]:
        del deployment["resources"]["runtime_memory"]["hash_block_size_tokens"]
    plan, _output = _plan_and_validate_weka(profile, tmp_path, monkeypatch, version="0.31.0", context_length=1024)
    for name in ("predict/pilot.yaml", "recommend/pilot.yaml"):
        worker = yaml.safe_load((plan / name).read_text())["engine"]["workers"]["aggregated"]
        assert worker["kv_cache"]["prefix_caching"] is False


@pytest.mark.parametrize("version", ["0.30.0", "0.25.1"])
def test_grouped_plan_on_an_unmodeled_vllm_version_stays_cold(grouped_case, tmp_path, monkeypatch, version):
    # Replay models grouped prefix caching for vLLM 0.31.0 and 0.32.0 only, so a probe of another version with
    # evidence and markers plans the cold replay that it planned before.
    profile, _root = grouped_case
    profile = _with_prefix_evidence(profile, version=version)
    plan, output = _plan_and_validate_weka(profile, tmp_path, monkeypatch, version=version, context_length=1024)
    for name in ("predict/pilot.yaml", "recommend/pilot.yaml"):
        worker = yaml.safe_load((plan / name).read_text())["engine"]["workers"]["aggregated"]
        assert worker["kv_cache"]["prefix_caching"] is False
    assert json.loads((output / "validation.json").read_text())["status"] == "covered"


@pytest.mark.parametrize("unsupported", ["state_without_align", "convolution_group"])
def test_grouped_plan_that_fails_a_prefix_caching_requirement_stays_cold(
    grouped_case, tmp_path, monkeypatch, unsupported
):
    # A probe with evidence and markers on a modeled version still plans the cold replay that it planned before
    # when the deployment fails another grouped prefix-caching requirement, and that replay runs.
    profile, _root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile))
    for deployment in profile["deployments"]:
        resources = deployment["resources"]
        if unsupported == "state_without_align":
            resources["runtime_memory"]["mamba_cache_mode"] = "none"
        else:
            resources["cache_groups"].append(dict(_CONVOLUTION_GROUP))
    plan, output = _plan_and_validate_weka(profile, tmp_path, monkeypatch, version="0.31.0", context_length=1024)
    for name in ("predict/pilot.yaml", "recommend/pilot.yaml"):
        worker = yaml.safe_load((plan / name).read_text())["engine"]["workers"]["aggregated"]
        assert worker["kv_cache"]["prefix_caching"] is False
    assert json.loads((output / "validation.json").read_text())["status"] == "covered"
    assert json.loads((output / "prediction/prediction.json").read_text())["completed_requests"] == 2


@pytest.mark.parametrize("prefix_caching", [False, True])
def test_grouped_sweeper_candidate_rank_takes_the_probe_hash_block(grouped_case, prefix_caching):
    from aisimulate.sweeper.config import SearchSpace
    from aisimulate.sweeper.deploy import build_backend_deployment
    from aisimulate.sweeper.parallel_enum import ParallelShape, ReplicaParallelConfig
    from aisimulate.sweeper.sample import unroll_sample

    profile, root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile))
    raw = {**_prefix_search(profile, root, "0.31.0"), "agg_enable_prefix_caching": prefix_caching}
    sample = unroll_sample(
        search_space=SearchSpace.model_validate(raw),
        selection={
            "deployment_mode": "agg",
            "backend": "vllm",
            "agg_max_num_batched_tokens": 128,
            "agg_max_num_seqs": 4,
        },
        parallel_config=ReplicaParallelConfig(ParallelShape(tp=2, dp=1, moe_tp=2, moe_ep=1), replicas=1),
    )
    args = build_backend_deployment(sample, backend_version="0.31.0").agg_engine_args
    if prefix_caching:
        # Without a search context length, the rank takes the profile's.
        assert (args["prefix_match_unit"], args["vllm_version"], args["max_model_len"]) == (64, "0.31.0", 2048)
    else:
        assert not {"prefix_match_unit", "vllm_version", "max_model_len"} & args.keys()


def test_grouped_prefix_caching_recommendation_replays_every_candidate(grouped_case, tmp_path, monkeypatch):
    from aisimulate.recommend import _run_recommendation
    from aisimulate.sweeper import SweepResult
    from aisimulate.sweeper.result import CandidateStatus

    monkeypatch.setattr("aisimulate.recommend.run_recommendation", _run_recommendation)
    profile, root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile))
    _write_timings(root, profile, version="0.31.0")
    raw = _prefix_caching_prediction(profile, root, "0.31.0")
    raw["engine"]["workers"]["aggregated"]["parallelism"] = {"preset": "default"}
    raw["optimization"] = {"constraints": {"max_candidate_gpus": 2}}
    raw["optimizer"] = {"algorithm": "random", "max_trials": 2, "parallelism": 1}
    path = tmp_path / "recommend.yaml"
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "recommendation"
    assert cli.main(["recommend", "-c", str(path), "--output-dir", str(output)]) == 0
    result = SweepResult.from_json((output / "recommendation.json").read_text())
    # Both topologies replay. A candidate that the Rust gate refused would be failed with "replay failed".
    assert len(result.candidates) == 2
    assert all(candidate.status is CandidateStatus.FEASIBLE for candidate in result.candidates)
    exported = CorePredictionConfig.from_yaml(output / "recommendations/0001.yaml")
    assert exported.engine.workers.aggregated.kv_cache.prefix_caching is True


def test_grouped_sweeper_checks_each_candidate_budget_before_replay(grouped_case):
    from aisimulate.sweeper.config import SearchSpace
    from aisimulate.sweeper.deploy import build_backend_deployment
    from aisimulate.sweeper.parallel_enum import ParallelShape, ReplicaParallelConfig
    from aisimulate.sweeper.sample import unroll_sample

    profile, root = grouped_case
    profile = _with_prefix_evidence(profile, hash_block=2176)
    profile["architecture"] = "Glm5NextForConditionalGeneration"
    # The search gate checks the largest budget, 8192. The 8190 candidate is refused when its rank is built.
    search = SearchSpace.model_validate(
        {**_prefix_search(profile, root, "0.31.0"), "agg_max_num_batched_tokens": [8190, 8192]}
    )
    sample = unroll_sample(
        search_space=search,
        selection={
            "deployment_mode": "agg",
            "backend": "vllm",
            "agg_max_num_batched_tokens": 8190,
            "agg_max_num_seqs": 4,
        },
        parallel_config=ReplicaParallelConfig(ParallelShape(tp=2, dp=1, moe_tp=2, moe_ep=1), replicas=1),
    )
    with pytest.raises(ValueError, match="max_batched_tokens that are multiples of 4"):
        build_backend_deployment(sample, backend_version="0.31.0")


_GLM_PAGE = 13_045_120


def _glm_shaped_profile(profile, version):
    """The six cache groups of the GLM-5.3-Flash TP 4 probe on vLLM 0.31.0, with a pool of 64 blocks.

    The 0.32.0 profile keeps prefill_checkpoint_blocks 0 on its state groups, because no vLLM 0.32.0 probe exists yet.
    """
    profile = deepcopy(profile)
    profile["architecture"] = "Glm5NextForConditionalGeneration"
    profile["context_length"] = 32768
    groups = [
        {
            "name": "runtime_group_0",
            "kind": "attention",
            "num_layers": 22,
            "block_size_tokens": 2176,
            "page_size_bytes": _GLM_PAGE,
            "prefix_cacheable": True,
        },
        {
            "name": "runtime_group_1",
            "kind": "state",
            "num_layers": 11,
            "block_size_tokens": 4,
            "page_size_bytes": _GLM_PAGE,
            "blocks_per_request": 1,
            "prefix_cacheable": False,
        },
    ]
    for index, layers in ((2, 9), (3, 9), (4, 8), (5, 8)):
        groups.append(
            {
                "name": f"runtime_group_{index}",
                "kind": "state",
                "num_layers": layers,
                "block_size_tokens": 2176,
                "page_size_bytes": _GLM_PAGE,
                "blocks_per_request": 1,
                "prefix_cacheable": True,
                "prefill_checkpoint_blocks": 0,
            }
        )
    for deployment in profile["deployments"]:
        deployment["backend_version"] = version
        resources = deployment["resources"]
        for name in ("weights_bytes", "activations_bytes", "runtime_overhead_bytes", "comm_overhead_bytes"):
            resources.pop(name)
        resources.update(cache_groups=deepcopy(groups), max_num_tokens=8192, max_batch_size=4)
        resources["runtime_memory"] = {
            "kv_cache_bytes": 64 * _GLM_PAGE,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 32768,
            "provenance": "Synthetic pool of 64 GLM-5.3-Flash pages. Not a probe result.",
            "enable_prefix_caching": True,
            "mamba_cache_mode": "align",
            "hash_block_size_tokens": 2176,
        }
    return profile


@pytest.mark.parametrize("version", ["0.31.0", "0.32.0"])
def test_glm_shaped_grouped_profile_replays_a_multi_turn_trace_with_hits(grouped_case, tmp_path, monkeypatch, version):
    # A trace has no known host memory peak, so the CLI replays it only in a supervised child. Model the
    # supervisor context of that child, as test_unified_cli.py does.
    monkeypatch.setenv(
        "_AISIMULATE_SUPERVISED_BUDGET",
        json.dumps(
            {
                "supervisor_pid": os.getpid(),
                "memory_limit_bytes": 4_000_000_000,
                "cpu_limit": 1,
                "reserved_host_memory_bytes": 1_000_000_000,
                "events_path": str(tmp_path / "events.jsonl"),
                "ready_path": str(tmp_path / "ready"),
            }
        ),
    )
    profile, root = grouped_case
    profile = _glm_shaped_profile(profile, version)
    (Path(profile["model"]) / "config.json").write_text(json.dumps({"architectures": [profile["architecture"]]}))
    import aisimulate_core

    # The real H200 memory holds the 64 GLM pages that the tiny fake GPU of grouped_case cannot.
    system = (Path(aisimulate_core.__file__).parent / "systems/h200_sxm.yaml").read_text()
    (root / "h200_sxm.yaml").write_text(system)
    _write_timings(
        root,
        profile,
        decode_ceiling=16384,
        version=version,
        prefill_tokens=(2048, 4096, 8192),
        prefill_kvs=(0, 1, 8192, 16384),
    )
    # One conversation of three turns, each resending the earlier prompt. A trace block of 2,176 tokens fills
    # block i with hash id i, so the turns share the blocks of the Rust test
    # grouped_glm_shaped_multi_turn_replay_reports_hit_tokens_and_block_counts. A fourth turn shares only the
    # first three blocks, so it branches below every state that the earlier turns keep.
    trace = tmp_path / "turns.jsonl"
    turns = [
        (9000, [1, 2, 3, 4, 5]),
        (13000, [1, 2, 3, 4, 5, 6]),
        (13100, [1, 2, 3, 4, 5, 6, 7]),
        (8704, [1, 2, 3, 8]),
    ]
    trace.write_text(
        "".join(
            json.dumps({"timestamp": 1000 * index, "input_length": length, "output_length": 4, "hash_ids": hashes})
            + "\n"
            for index, (length, hashes) in enumerate(turns)
        )
    )
    raw = _prediction(profile, root)
    raw["engine"]["backend_version"] = version
    raw["engine"]["context_length"] = 32768
    worker = raw["engine"]["workers"]["aggregated"]
    worker["scheduler"] = {"max_batched_tokens": 8192, "max_sequences": 4}
    worker["kv_cache"]["prefix_caching"] = True
    raw["traffic"] = {
        "source": {"type": "trace", "paths": [str(trace)], "format": "mooncake", "block_size": 2176},
        "load": {"type": "concurrency", "concurrency": 1},
    }
    path = tmp_path / "predict.yaml"
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "prediction"
    assert cli.main(["predict", "-c", str(path), "--output-dir", str(output), "--capture-per-request"]) == 0
    report = json.loads((output / "prediction.json").read_text())
    assert report["completed_requests"] == 4
    # Turn two hits the state that turn one keeps at (9000 - 1) / 2176 * 2176 = 8704, and turn three the one that
    # turn two keeps at (13000 - 1) / 2176 * 2176 = 10880. Both versions reuse 19,584 tokens. Turn four finds
    # 6,528 tokens of cached full-attention blocks, which a linear pool would reuse. Under retention 0 no state is
    # kept at 6,528 or below, so the grouped path reuses nothing.
    assert [row["reused_input_tokens"] for row in report["per_request"]] == [0, 8704, 10880, 0]
    assert report["prefix_cache_reused_ratio"] == pytest.approx((8704 + 10880) / (9000 + 13000 + 13100 + 8704))


# The payload of one block of each GLM-5.3-Flash group at TP 4 with an fp8 cache, without the page padding:
# 11 x 1,114,112 + 11 x 71,808 bytes of attention, 11 x 2,048 bytes of ring, and 9 or 8 x 1,085,440 bytes of KDA state.
_GLM_TRANSFER_BYTES = {
    "runtime_group_0": 13_045_120,
    "runtime_group_1": 22_528,
    "runtime_group_2": 9_768_960,
    "runtime_group_3": 9_768_960,
    "runtime_group_4": 8_683_520,
    "runtime_group_5": 8_683_520,
}


@pytest.mark.parametrize("version", ["0.31.0", "0.32.0"])
def test_glm_shaped_grouped_pd_transfer_time_matches_the_oracle(grouped_case, tmp_path, version):
    profile, root = grouped_case
    profile = _glm_shaped_profile(profile, version)
    for deployment in profile["deployments"]:
        for group in deployment["resources"]["cache_groups"]:
            group["transfer_bytes_per_block"] = _GLM_TRANSFER_BYTES[group["name"]]
    (Path(profile["model"]) / "config.json").write_text(json.dumps({"architectures": [profile["architecture"]]}))
    import aisimulate_core

    # The real H200 memory holds the 64 GLM pages that the tiny fake GPU of grouped_case cannot.
    system = (Path(aisimulate_core.__file__).parent / "systems/h200_sxm.yaml").read_text()
    (root / "h200_sxm.yaml").write_text(system)
    _write_timings(
        root,
        profile,
        decode_ceiling=16384,
        version=version,
        prefill_tokens=(2048, 4096, 8192),
        prefill_kvs=(0, 1, 8192, 16384),
    )
    mean_ttft = {}
    for bandwidth in (None, 1.0):
        raw = _pd_prediction(profile, root, bandwidth=bandwidth)
        raw["engine"]["context_length"] = 32768
        for worker in raw["engine"]["workers"].values():
            worker["scheduler"] = {"max_batched_tokens": 8192, "max_sequences": 4}
        raw["traffic"] = {
            "source": {"type": "synthetic", "input_tokens": 1000, "output_tokens": 2},
            "load": {"type": "concurrency", "concurrency": 1},
            "stop": {"requests": 1},
        }
        path = tmp_path / f"predict-{bandwidth}.yaml"
        path.write_text(yaml.safe_dump(raw))
        output = tmp_path / f"prediction-{bandwidth}"
        assert cli.main(["predict", "-c", str(path), "--output-dir", str(output)]) == 0
        mean_ttft[bandwidth] = json.loads((output / "prediction.json").read_text())["mean_ttft_ms"]
    # The prefill computes 999 tokens, so one 2,176-token attention block of 13,045,120 bytes moves with one block
    # of each state group, 36,927,488 bytes: 49,972,608 bytes take 49.972608 ms at 1 GB/s.
    assert mean_ttft[1.0] - mean_ttft[None] == pytest.approx(49.972608, abs=1e-3)
