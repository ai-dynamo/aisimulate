# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend sample to runner-neutral deployment contract."""

import pytest

import aisimulate.sweeper.deploy as deploy_module
from aisimulate.sweeper.config import SearchSpace
from aisimulate.sweeper.deploy import build_backend_deployment
from aisimulate.sweeper.parallel_enum import (
    DisaggParallelConfig,
    ParallelShape,
    ReplicaParallelConfig,
)
from aisimulate.sweeper.sample import unroll_sample

BACKEND_VERSION = "1.3.0rc10"


@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm"])
def test_deployment_preserves_context_limit_for_all_backends(backend):
    deployment = _agg_deployment(space=_space(context_length=4096), selection=_agg_selection(backend=backend))
    assert deployment.agg_engine_args["max_model_len"] == 4096


def _space(**overrides) -> SearchSpace:
    values = {"model_name": "example/model", "hardware_sku": "example_sku"}
    values.update(overrides)
    return SearchSpace(**values)


def _agg_selection(**overrides) -> dict:
    values = {
        "deployment_mode": "agg",
        "backend": "trtllm",
        "agg_max_num_batched_tokens": 16384,
        "agg_max_num_seqs": 512,
    }
    values.update(overrides)
    return values


AGG_MOE = ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2)


def _agg_deployment(*, space=None, selection=None, parallel_config=AGG_MOE):
    sample = unroll_sample(
        search_space=space or _space(),
        selection=selection or _agg_selection(),
        parallel_config=parallel_config,
    )
    return build_backend_deployment(sample, backend_version=BACKEND_VERSION)


def test_zero_speculation_keeps_the_non_speculative_runtime():
    deployment = _agg_deployment(space=_space(aic_nextn=0))
    assert "aic_nextn" not in deployment.agg_engine_args
    assert "aic_nextn_accepted" not in deployment.agg_engine_args


def test_agg_backend_deployment_preserves_engine_payload():
    deployment = _agg_deployment()
    engine = deployment.agg_engine_args

    assert deployment.deployment_mode == "agg"
    assert deployment.backend == "trtllm"
    assert deployment.backend_version == BACKEND_VERSION
    assert deployment.num_workers == 2
    assert deployment.num_prefill_workers == 0
    assert deployment.num_decode_workers == 0
    assert deployment.prefill_engine_args is None
    assert deployment.decode_engine_args is None
    assert deployment.parallel_config == {
        "tp": 4,
        "pp": 1,
        "attention_dp": 1,
        "moe_tp": 1,
        "moe_ep": 4,
        "strategy": "tep",
        "replicas": 2,
    }
    assert engine == {
        "worker_type": "aggregated",
        "engine_type": "trtllm",
        "aic_backend": "trtllm",
        "aic_backend_version": BACKEND_VERSION,
        "aic_system": "example_sku",
        "aic_model_path": "example/model",
        "aic_tp_size": 4,
        "aic_attention_dp_size": 1,
        "aic_moe_tp_size": 1,
        "aic_moe_ep_size": 4,
        "max_num_batched_tokens": 16384,
        "max_num_seqs": 512,
        "block_size": 64,
        "free_gpu_memory_fraction": 0.9,
        "enable_prefix_caching": True,
    }


def test_disagg_backend_deployment_preserves_both_roles():
    parallel = DisaggParallelConfig(
        prefill=ReplicaParallelConfig(ParallelShape(tp=8, dp=1, moe_tp=1, moe_ep=8), 1),
        decode=ReplicaParallelConfig(ParallelShape(tp=1, dp=8, moe_tp=1, moe_ep=8), 2),
    )
    selection = _agg_selection(
        deployment_mode="disagg",
        backend="sglang",
        prefill_max_num_batched_tokens=32768,
        prefill_max_num_seqs=4,
        decode_max_num_batched_tokens=8192,
        decode_max_num_seqs=1024,
    )
    sample = unroll_sample(search_space=_space(), selection=selection, parallel_config=parallel)

    deployment = build_backend_deployment(sample, backend_version=BACKEND_VERSION)

    assert deployment.deployment_mode == "disagg"
    assert deployment.backend == "sglang"
    assert deployment.num_workers == 0
    assert deployment.num_prefill_workers == 1
    assert deployment.num_decode_workers == 2
    assert deployment.agg_engine_args is None
    assert deployment.prefill_engine_args["worker_type"] == "prefill"
    assert deployment.prefill_engine_args["aic_tp_size"] == 8
    assert deployment.prefill_engine_args["engine_type"] == "sglang"
    assert deployment.decode_engine_args["worker_type"] == "decode"
    assert deployment.decode_engine_args["aic_attention_dp_size"] == 8
    assert deployment.decode_engine_args["engine_type"] == "sglang"


def test_disagg_backend_deployment_uses_role_hardware():
    parallel = DisaggParallelConfig(
        prefill=ReplicaParallelConfig(ParallelShape(tp=2, dp=1, moe_tp=1, moe_ep=1), 1),
        decode=ReplicaParallelConfig(ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), 2),
    )
    selection = _agg_selection(
        deployment_mode="disagg",
        backend="vllm",
        prefill_max_num_batched_tokens=8192,
        prefill_max_num_seqs=4,
        decode_max_num_batched_tokens=8192,
        decode_max_num_seqs=256,
    )
    sample = unroll_sample(
        search_space=_space(
            prefill_hardware_sku="h200_sxm",
            decode_hardware_sku="gb200",
        ),
        selection=selection,
        parallel_config=parallel,
    )

    deployment = build_backend_deployment(sample, backend_version=BACKEND_VERSION)

    assert deployment.prefill_engine_args["aic_system"] == "h200_sxm"
    assert deployment.decode_engine_args["aic_system"] == "gb200"
    assert deployment.performance_model_metadata["prefill"]["config"]["system"] == "h200_sxm"
    assert deployment.performance_model_metadata["decode"]["config"]["system"] == "gb200"
    assert deployment.parallel_config["prefill_hardware_sku"] == "h200_sxm"
    assert deployment.parallel_config["decode_hardware_sku"] == "gb200"


def test_dense_shape_omits_moe_sizes():
    dense = ReplicaParallelConfig(ParallelShape(tp=2, dp=1, moe_tp=1, moe_ep=1), replicas=1)
    engine = _agg_deployment(
        space=_space(model_name="example/dense"),
        parallel_config=dense,
    ).agg_engine_args

    assert engine["aic_tp_size"] == 2
    assert "aic_moe_tp_size" not in engine
    assert "aic_moe_ep_size" not in engine


@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm"])
def test_engine_type_tracks_swept_backend(backend):
    engine = _agg_deployment(selection=_agg_selection(backend=backend)).agg_engine_args

    assert engine["engine_type"] == backend
    assert engine["aic_backend"] == backend


@pytest.mark.parametrize(
    ("backend", "memory_field"),
    [
        ("vllm", "gpu_memory_utilization"),
        ("sglang", "mem_fraction_static"),
        ("trtllm", "free_gpu_memory_fraction"),
    ],
)
def test_memory_fraction_uses_the_backend_native_field(backend, memory_field):
    engine = _agg_deployment(selection=_agg_selection(backend=backend)).agg_engine_args

    assert engine[memory_field] == 0.9
    assert (
        len(
            {
                "gpu_memory_utilization",
                "mem_fraction_static",
                "free_gpu_memory_fraction",
            }
            & engine.keys()
        )
        == 1
    )


def test_optional_backend_runtime_values_are_forwarded():
    engine = _agg_deployment(space=_space(startup_time=45.0, aic_nextn=2, nextn_accepted=1.5)).agg_engine_args

    assert engine["startup_time"] == 45.0
    assert engine["aic_nextn"] == 2
    assert engine["aic_nextn_accepted"] == 1.5


def test_fixed_host_offload_descriptor_lowers_into_aggregated_engine_args():
    host_offload = {
        "num_host_blocks": 4096,
        "d2h_bandwidth_gbps": 7.0,
        "h2d_bandwidth_gbps": 38.0,
    }
    dense = ReplicaParallelConfig(ParallelShape(tp=2, dp=1, moe_tp=1, moe_ep=1), replicas=1)

    engine = _agg_deployment(
        space=_space(
            agg_kv_bytes_per_token=131_072,
            agg_native_host_offload=host_offload,
        ),
        selection=_agg_selection(backend="vllm"),
        parallel_config=dense,
    ).agg_engine_args

    assert engine["kv_cache_bytes_per_token"] == 131_072
    assert engine["native_host_offload"] == host_offload


def test_disagg_auto_transfer_geometry_is_deferred_to_canonical_role_models(monkeypatch):
    monkeypatch.setattr(
        deploy_module,
        "estimate_kv_bytes_per_token",
        lambda _model, **shape: 10_000 * shape["tp_size"] + shape["pp_size"],
    )
    parallel = DisaggParallelConfig(
        prefill=ReplicaParallelConfig(ParallelShape(tp=2, dp=1, moe_tp=1, moe_ep=1), 1),
        decode=ReplicaParallelConfig(ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), 1),
    )
    selection = _agg_selection(
        deployment_mode="disagg",
        backend="vllm",
        prefill_max_num_batched_tokens=8192,
        prefill_max_num_seqs=1,
        decode_max_num_batched_tokens=8192,
        decode_max_num_seqs=256,
    )
    sample = unroll_sample(
        search_space=_space(
            kv_transfer_bytes_per_token="auto",
            kv_transfer_bandwidth=400.0,
        ),
        selection=selection,
        parallel_config=parallel,
    )

    deployment = build_backend_deployment(sample, backend_version=BACKEND_VERSION)

    assert deployment.prefill_engine_args.get("kv_transfer_bytes_per_token") is None
    assert deployment.decode_engine_args.get("kv_transfer_bytes_per_token") is None


def _role_geometry_profile(roles):
    return {
        "schema_version": 1,
        "model": "example/role-geometry",
        "model_revision": "synthetic-v1",
        "architecture": "SyntheticDecoder",
        "context_length": 4096,
        "num_experts": 0,
        "provenance": "Synthetic role geometry; not measured.",
        "deployments": [
            {
                "system": "h200_sxm",
                "backend": "vllm",
                "backend_version": "0.25.1",
                "tp": 1,
                "dp": 1,
                "moe_tp": 1,
                "moe_ep": 1,
                "gemm_quant_mode": "bfloat16",
                "moe_quant_mode": "bfloat16",
                "fmha_quant_mode": "bfloat16",
                "comm_quant_mode": "half",
                "kv_cache_dtype": "bfloat16",
                **({"worker_type": role} if role is not None else {}),
                "resources": {
                    "weights_bytes": 1024,
                    "activations_bytes": 128,
                    "runtime_overhead_bytes": 128,
                    "comm_overhead_bytes": 0,
                    "kv_bytes_per_token": {None: 512, "prefill": 512, "decode": 1024, "aggregated": 2048}[role],
                    "cache_layout": "linear",
                    "max_num_tokens": 8192,
                    "max_batch_size": 256,
                    "provenance": "Synthetic role geometry; not measured.",
                },
            }
            for role in roles
        ],
    }


@pytest.mark.parametrize("roles", [(None,), ("prefill", "decode"), ("prefill", "decode", "aggregated")])
@pytest.mark.parametrize("feature", ["transfer", "host_offload"])
def test_disagg_auto_geometry_selects_matching_profile_role(roles, feature):
    profile = _role_geometry_profile(roles)
    host_offload = {"num_host_blocks": 4096, "d2h_bandwidth_gbps": 7.0, "h2d_bandwidth_gbps": 38.0}
    settings = (
        {"kv_transfer_bytes_per_token": "auto", "kv_transfer_bandwidth": 400.0}
        if feature == "transfer"
        else {
            "prefill_native_host_offload": host_offload,
            "decode_native_host_offload": host_offload,
            "prefill_kv_bytes_per_token": "auto",
            "decode_kv_bytes_per_token": "auto",
        }
    )
    parallel = ReplicaParallelConfig(ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), 1)
    sample = unroll_sample(
        search_space=_space(model_name=profile["model"], hardware_sku="h200_sxm", fpm_profile=profile, **settings),
        selection=_agg_selection(
            deployment_mode="disagg",
            backend="vllm",
            prefill_max_num_batched_tokens=8192,
            prefill_max_num_seqs=1,
            decode_max_num_batched_tokens=8192,
            decode_max_num_seqs=256,
        ),
        parallel_config=DisaggParallelConfig(prefill=parallel, decode=parallel),
    )

    deployment = build_backend_deployment(sample, backend_version="0.25.1")

    field = "kv_transfer_bytes_per_token" if feature == "transfer" else "kv_cache_bytes_per_token"
    assert deployment.prefill_engine_args[field] == 512
    assert deployment.decode_engine_args[field] == (512 if feature == "transfer" or roles == (None,) else 1024)


@pytest.mark.parametrize("roles,expected", [((None,), 512), (("prefill", "decode", "aggregated"), 2048)])
def test_agg_auto_host_offload_geometry_selects_aggregated_profile_role(roles, expected):
    profile = _role_geometry_profile(roles)
    sample = unroll_sample(
        search_space=_space(
            model_name=profile["model"],
            hardware_sku="h200_sxm",
            fpm_profile=profile,
            agg_native_host_offload={"num_host_blocks": 4096, "d2h_bandwidth_gbps": 7.0, "h2d_bandwidth_gbps": 38.0},
            agg_kv_bytes_per_token="auto",
        ),
        selection=_agg_selection(backend="vllm"),
        parallel_config=ReplicaParallelConfig(ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), 1),
    )

    deployment = build_backend_deployment(sample, backend_version="0.25.1")

    assert deployment.agg_engine_args["worker_type"] == "aggregated"
    assert deployment.agg_engine_args["kv_cache_bytes_per_token"] == expected


def test_serialized_search_space_transfer_geometry_enables_pd_transfer():
    legacy = _space(
        kv_transfer_bytes_per_token=333,
        kv_transfer_bandwidth=400.0,
    ).model_dump(mode="json")
    parallel = DisaggParallelConfig(
        prefill=ReplicaParallelConfig(ParallelShape(tp=2, dp=1, moe_tp=1, moe_ep=1), 1),
        decode=ReplicaParallelConfig(ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), 1),
    )
    sample = unroll_sample(
        search_space=SearchSpace.model_validate(legacy),
        selection=_agg_selection(
            deployment_mode="disagg",
            backend="vllm",
            prefill_max_num_batched_tokens=8192,
            prefill_max_num_seqs=1,
            decode_max_num_batched_tokens=8192,
            decode_max_num_seqs=256,
        ),
        parallel_config=parallel,
    )

    deployment = build_backend_deployment(sample, backend_version=BACKEND_VERSION)

    assert deployment.prefill_engine_args["kv_transfer_bytes_per_token"] == 333
    assert deployment.decode_engine_args["kv_transfer_bytes_per_token"] == 333
    assert deployment.prefill_engine_args["kv_transfer_bandwidth"] == 400.0
    assert deployment.decode_engine_args["kv_transfer_bandwidth"] == 400.0


def test_backend_deployment_contains_no_dynamo_policy_fields():
    deployment = _agg_deployment()

    assert not hasattr(deployment, "planner_config")
    assert not hasattr(deployment, "router_config")
    assert not hasattr(deployment, "is_static")


def test_fixed_timing_preserves_aic_identity_for_stack_adapters(monkeypatch):
    monkeypatch.setattr(
        deploy_module,
        "materialize_aic_num_gpu_blocks",
        lambda payload: {**payload, "num_gpu_blocks": 321},
    )
    deployment = _agg_deployment(
        selection=_agg_selection(
            agg_timing_model={
                "type": "fixed",
                "prefill_ms": 1.0,
                "decode_ms": 1.0,
            }
        )
    )
    engine = deployment.agg_engine_args

    assert engine["timing_model"]["type"] == "fixed"
    assert engine["num_gpu_blocks"] == 321
    assert "aic_backend_version" not in engine
    assert "aic_system" not in engine
    assert "aic_model_path" not in engine
    assert deployment.performance_model_metadata == {
        "aggregated": {
            "provider": "aic",
            "config": {
                "backend": "trtllm",
                "backend_version": BACKEND_VERSION,
                "system": "example_sku",
                "model_path": "example/model",
                "tp_size": 4,
                "attention_dp_size": 1,
                "moe_tp_size": 1,
                "moe_ep_size": 4,
                "nextn": None,
                "forward_model": "op_level",
            },
        }
    }


def test_fpm_forward_model_lowers_onto_the_engine_payload():
    deployment = _agg_deployment(space=_space(agg_forward_model="fpm"))
    engine = deployment.agg_engine_args

    assert engine["aic_forward_model"] == "fpm"
    assert "timing_model" not in engine
    assert deployment.performance_model_metadata["aggregated"]["config"]["forward_model"] == "fpm"


def test_op_level_forward_model_adds_no_engine_field():
    deployment = _agg_deployment()

    assert "aic_forward_model" not in deployment.agg_engine_args
    assert deployment.performance_model_metadata["aggregated"]["config"]["forward_model"] == "op_level"


def test_disagg_forward_model_is_lowered_per_role():
    parallel = DisaggParallelConfig(
        prefill=ReplicaParallelConfig(ParallelShape(tp=8, dp=1, moe_tp=1, moe_ep=8), 1),
        decode=ReplicaParallelConfig(ParallelShape(tp=1, dp=8, moe_tp=1, moe_ep=8), 2),
    )
    selection = _agg_selection(
        deployment_mode="disagg",
        backend="vllm",
        prefill_max_num_batched_tokens=32768,
        prefill_max_num_seqs=4,
        decode_max_num_batched_tokens=8192,
        decode_max_num_seqs=1024,
    )
    sample = unroll_sample(
        search_space=_space(decode_forward_model="fpm"), selection=selection, parallel_config=parallel
    )

    deployment = build_backend_deployment(sample, backend_version=BACKEND_VERSION)

    assert "aic_forward_model" not in deployment.prefill_engine_args
    assert deployment.decode_engine_args["aic_forward_model"] == "fpm"
    assert deployment.performance_model_metadata["prefill"]["config"]["forward_model"] == "op_level"
    assert deployment.performance_model_metadata["decode"]["config"]["forward_model"] == "fpm"


def test_fixed_timing_drops_the_forward_model_field(monkeypatch):
    monkeypatch.setattr(
        deploy_module,
        "materialize_aic_num_gpu_blocks",
        lambda payload: {**payload, "num_gpu_blocks": 321},
    )
    deployment = _agg_deployment(
        space=_space(agg_forward_model="fpm"),
        selection=_agg_selection(agg_timing_model={"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0}),
    )

    assert "aic_forward_model" not in deployment.agg_engine_args
    assert deployment.agg_engine_args["timing_model"]["type"] == "fixed"
    assert deployment.performance_model_metadata["aggregated"]["config"]["forward_model"] == "op_level"
