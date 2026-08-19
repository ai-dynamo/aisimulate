# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend sample to runner-neutral deployment contract."""

import pytest

from aisimulate.sweeper.config import SearchSpace
from aisimulate.sweeper.deploy import build_backend_deployment
from aisimulate.sweeper.parallel_enum import (
    DisaggParallelConfig,
    ParallelShape,
    ReplicaParallelConfig,
)
from aisimulate.sweeper.replay import EstimatorSpec
from aisimulate.sweeper.sample import unroll_sample

BACKEND_VERSION = "1.3.0rc10"


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


AGG_MOE = ReplicaParallelConfig(
    ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2
)


def _agg_deployment(*, space=None, selection=None, parallel_config=AGG_MOE):
    sample = unroll_sample(
        search_space=space or _space(),
        selection=selection or _agg_selection(),
        parallel_config=parallel_config,
    )
    return build_backend_deployment(sample, backend_version=BACKEND_VERSION)


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
        "cp": 1,
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
        "aic_pp_size": 1,
        "aic_attention_dp_size": 1,
        "aic_cp_size": 1,
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
    sample = unroll_sample(
        search_space=_space(), selection=selection, parallel_config=parallel
    )

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


def test_dense_shape_omits_moe_sizes():
    dense = ReplicaParallelConfig(
        ParallelShape(tp=2, dp=1, moe_tp=1, moe_ep=1), replicas=1
    )
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
    engine = _agg_deployment(
        space=_space(startup_time=45.0, aic_nextn=2)
    ).agg_engine_args

    assert engine["startup_time"] == 45.0
    assert engine["aic_nextn"] == 2


def test_pipeline_and_context_parallelism_reach_aic_execution():
    parallel = ReplicaParallelConfig(
        ParallelShape(tp=1, pp=2, dp=1, moe_tp=1, moe_ep=4, cp=4), replicas=1
    )
    engine = _agg_deployment(
        selection=_agg_selection(backend="sglang"), parallel_config=parallel
    ).agg_engine_args

    assert engine["aic_tp_size"] == 1
    assert engine["aic_pp_size"] == 2
    assert engine["aic_cp_size"] == 4


def test_resolved_estimator_contract_is_forwarded_to_every_engine():
    estimator = EstimatorSpec(
        model_path="example/model",
        model_architecture="ExampleForCausalLM",
        system="example_sku",
        backend="trtllm",
        backend_version=BACKEND_VERSION,
        performance_data_version=BACKEND_VERSION,
        database_mode="HYBRID",
        transfer_policy=("xshape", "xquant"),
        forward_model="fpm",
        engine_step_backend="rust",
        systems_paths=("/custom/systems",),
        performance_data_root="/custom/systems",
    )
    sample = unroll_sample(
        search_space=_space(),
        selection=_agg_selection(),
        parallel_config=AGG_MOE,
    )

    deployment = build_backend_deployment(
        sample,
        backend_version=BACKEND_VERSION,
        estimator=estimator,
    )

    assert deployment.estimator is estimator
    expected = {
        "aic_database_mode": "HYBRID",
        "aic_transfer_policy": ["xshape", "xquant"],
        "aic_forward_model": "fpm",
        "aic_engine_step_backend": "rust",
        "aic_systems_paths": ["/custom/systems"],
        "aic_performance_data_version": BACKEND_VERSION,
    }
    engine = deployment.agg_engine_args
    assert engine is not None
    assert {key: engine[key] for key in expected} == expected


def test_backend_deployment_contains_no_dynamo_policy_fields():
    deployment = _agg_deployment()

    assert not hasattr(deployment, "planner_config")
    assert not hasattr(deployment, "router_config")
    assert not hasattr(deployment, "is_static")
