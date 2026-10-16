# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shlex

import pytest
import yaml

from aisimulate.generator.api import generate_from_request
from aisimulate.generator.request import (
    ModelFacts,
    SweeperCandidateError,
    from_sweeper_candidate,
    to_legacy_params,
)


def _agg_candidate(**overrides):
    config = {
        "deployment_mode": "agg",
        "model_name": "Qwen/Qwen3-32B-FP8",
        "backend": "vllm",
        "backend_version": "0.20.1",
        "hardware_sku": "h200_sxm",
        "context_length": 8192,
        "tp": 2,
        "pp": 1,
        "attention_dp": 1,
        "moe_tp": 1,
        "moe_ep": 1,
        "replicas": 4,
        "used_gpus": 8,
        "agg_max_num_batched_tokens": 8192,
        "agg_max_num_seqs": 256,
        "agg_block_size": 64,
        "agg_gpu_memory_utilization": 0.9,
        "agg_enable_prefix_caching": True,
    }
    config.update(overrides)
    return {
        "config": config,
        "used_gpus": config["used_gpus"],
        "score": 123.0,
        "metrics": {"output_throughput_tok_s": 123.0},
    }


def _disagg_candidate(**overrides):
    config = {
        "deployment_mode": "disagg",
        "model_name": "deepseek-ai/DeepSeek-V3",
        "backend": "trtllm",
        "backend_version": "1.3.0rc14",
        "hardware_sku": "gb200",
        "context_length": 16384,
        "prefill_tp": 1,
        "prefill_pp": 1,
        "prefill_attention_dp": 1,
        "prefill_moe_tp": 1,
        "prefill_moe_ep": 1,
        "prefill_replicas": 4,
        "prefill_max_num_batched_tokens": 16384,
        "prefill_max_num_seqs": 4,
        "prefill_block_size": 64,
        "prefill_gpu_memory_utilization": 0.9,
        "prefill_enable_prefix_caching": True,
        "decode_tp": 4,
        "decode_pp": 1,
        "decode_attention_dp": 1,
        "decode_moe_tp": 4,
        "decode_moe_ep": 1,
        "decode_replicas": 1,
        "decode_max_num_batched_tokens": 8192,
        "decode_max_num_seqs": 512,
        "decode_block_size": 64,
        "decode_gpu_memory_utilization": 0.85,
        "decode_enable_prefix_caching": False,
        "used_gpus": 8,
        "concurrency": 256,
        "adapters": {
            "dynamo.router": {
                "mode": "kv",
                "overlap_score_credit": 0.75,
                "router_temperature": 0.2,
            },
            "dynamo.planner": {
                "scaling_policy": "throughput",
                "enable_throughput_scaling": True,
                "enable_load_scaling": False,
                "environment": "kubernetes",
                "mode": "disagg",
            },
            "dynamo.kvbm": {"cpu_cache_gb": 64},
        },
    }
    config.update(overrides)
    return {
        "config": config,
        "used_gpus": config["used_gpus"],
        "score": 456.0,
        "metrics": {"output_throughput_tok_s": 456.0},
    }


def test_agg_candidate_lowers_evaluated_engine_limits_and_deployment_overrides():
    request = from_sweeper_candidate(
        _agg_candidate(),
        workload={"isl": 4000, "osl": 1000},
        deployment_target="dynamo-j2",
        output_dir="./results/agg/top1",
        environment_profile="cluster.yaml",
        generator_overrides={
            "K8sConfig": {"k8s_namespace": "inference"},
            "ModelConfig": {"is_moe": True},
        },
        model_facts=ModelFacts(is_moe=False, architecture="Qwen3ForCausalLM"),
    )

    assert request.backend.generated_config_version == "0.20.1"
    assert request.topology.workers == {"agg": 4}
    assert request.platform.environment_profile == "cluster.yaml"
    assert request.emit.output_dir == "./results/agg/top1"

    params = to_legacy_params(request)
    agg = params["params"]["agg"]
    assert agg["tensor_parallel_size"] == 2
    assert agg["gpus_per_worker"] == 2
    assert agg["max_batch_size"] == 256
    assert agg["max_num_tokens"] == 8192
    assert agg["tokens_per_block"] == 64
    assert agg["max_seq_len"] == 8192
    assert agg["kv_cache_free_gpu_memory_fraction"] == 0.9
    assert agg["disable_prefix_cache"] is False
    assert params["WorkerConfig"]["agg_workers"] == 4
    assert params["K8sConfig"]["k8s_namespace"] == "inference"
    assert params["NodeConfig"]["system_name"] == "h200_sxm"
    assert params["ModelConfig"] == {
        "is_moe": False,
        "architecture": "Qwen3ForCausalLM",
    }
    assert params["rule"] == "benchmark"
    assert params["preserve_engine_limits"] is True


def test_disagg_candidate_preserves_dynamo_adapter_configs_and_concurrency():
    request = from_sweeper_candidate(
        _disagg_candidate(prefill_context_length=64000, decode_context_length=128000),
        workload={"isl": 8192, "osl": 1024},
        model_facts=ModelFacts(is_moe=True, architecture="DeepseekV3ForCausalLM"),
    )

    params = to_legacy_params(request)
    assert params["WorkerConfig"]["prefill_workers"] == 4
    assert params["WorkerConfig"]["decode_workers"] == 1
    assert params["params"]["prefill"]["max_batch_size"] == 4
    assert params["params"]["prefill"]["max_seq_len"] == 64000
    assert params["params"]["decode"]["max_batch_size"] == 512
    assert params["params"]["decode"]["max_seq_len"] == 128000
    assert params["params"]["decode"]["kv_cache_free_gpu_memory_fraction"] == 0.85
    assert params["BenchConfig"]["estimated_concurrency"] == 256
    assert params["DynConfig"]["enable_router"] is True
    assert params["DynConfig"]["router_mode"] == "kv"
    assert params["DynConfig"]["router_config"] == {
        "mode": "kv",
        "overlap_score_credit": 0.75,
        "router_temperature": 0.2,
    }
    assert params["DynConfig"]["planner_config"] == {
        "enable_throughput_scaling": True,
        "enable_load_scaling": False,
        "environment": "kubernetes",
        "mode": "disagg",
    }
    assert params["DynConfig"]["kvbm_config"] == {"cpu_cache_gb": 64}
    assert request.backend.generated_config_version == "1.3.0rc14"


def test_disagg_candidate_accepts_matching_effective_role_hardware():
    request = from_sweeper_candidate(
        _disagg_candidate(
            hardware_sku="h200_sxm",
            prefill_hardware_sku="gb200",
            decode_hardware_sku="gb200",
        ),
        workload={"isl": 8192, "osl": 1024},
        model_facts=ModelFacts(is_moe=True, architecture="DeepseekV3ForCausalLM"),
    )

    assert request.platform.hardware_profile == "gb200"
    assert to_legacy_params(request)["NodeConfig"]["system_name"] == "gb200"


def test_disagg_candidate_rejects_heterogeneous_deployment_artifact_generation():
    with pytest.raises(
        SweeperCandidateError,
        match="heterogeneous P/D deployment artifact generation is unsupported",
    ):
        from_sweeper_candidate(
            _disagg_candidate(
                prefill_hardware_sku="h200_sxm",
                decode_hardware_sku="gb200",
            ),
            workload={"isl": 8192, "osl": 1024},
            model_facts=ModelFacts(is_moe=True, architecture="DeepseekV3ForCausalLM"),
        )


def test_candidate_gpu_count_must_match_lowered_topology():
    with pytest.raises(SweeperCandidateError, match="used_gpus=16, topology=8"):
        from_sweeper_candidate(
            _agg_candidate(used_gpus=16),
            workload={"isl": 4000, "osl": 1000},
            model_facts=ModelFacts(is_moe=False),
        )


@pytest.mark.parametrize("mode", ["afd", "afd+pd"])
def test_afd_candidate_rejects_native_generation_with_analytical_artifact_path(mode):
    with pytest.raises(
        SweeperCandidateError,
        match="aisimulate predict.*afd-replay-spec.json.*afd-qualification.json",
    ):
        from_sweeper_candidate(
            {"config": {"deployment_mode": mode}},
            model_facts=ModelFacts(is_moe=False),
        )


def test_unknown_nonempty_adapter_fails_closed():
    with pytest.raises(SweeperCandidateError, match="has no generator mapping"):
        from_sweeper_candidate(
            _agg_candidate(adapters={"custom.feature": {"enabled": True}}),
            workload={"isl": 4000, "osl": 1000},
            model_facts=ModelFacts(is_moe=False),
        )


def test_disabled_planner_selection_does_not_emit_a_planner_service():
    candidate = _agg_candidate(
        adapters={
            "dynamo.planner": {
                "scaling_policy": "none",
                "enable_throughput_scaling": False,
                "enable_load_scaling": False,
            }
        }
    )
    request = from_sweeper_candidate(
        candidate,
        workload={"isl": 4000, "osl": 1000},
        model_facts=ModelFacts(is_moe=False),
    )

    assert "planner_config" not in to_legacy_params(request)["DynConfig"]


def test_workload_lengths_can_come_from_generator_overrides():
    request = from_sweeper_candidate(
        _agg_candidate(),
        generator_overrides={"SlaConfig.isl": 2048, "SlaConfig.osl": 512},
        model_facts=ModelFacts(is_moe=False),
    )

    assert request.sla.isl == 2048
    assert request.sla.osl == 512


def test_model_facts_are_lowered_by_typed_request():
    request = from_sweeper_candidate(
        _agg_candidate(aic_nextn=2),
        workload={"isl": 4000, "osl": 1000},
        model_facts=ModelFacts(is_moe=True, nextn=1, prefix=128, extra={"nextn_accepted": 1.5}),
    )

    assert to_legacy_params(request)["ModelConfig"] == {
        "is_moe": True,
        "nextn": 2,
        "prefix": 128,
        "nextn_accepted": 1.5,
    }


def test_candidate_renders_deployable_artifacts_with_evaluated_limits():
    request = from_sweeper_candidate(
        _agg_candidate(),
        workload={"isl": 4000, "osl": 1000},
        model_facts=ModelFacts(is_moe=False),
        generator_overrides={"K8sConfig": {"k8s_image": "example/vllm:0.20.1"}},
    )

    artifacts = generate_from_request(request)

    assert "k8s_deploy.yaml" in artifacts
    assert "bench_run.sh" in artifacts
    assert "k8s_bench.yaml" in artifacts
    assert '--max-model-len "8192"' in artifacts["cli_args_agg"]
    assert '--max-num-seqs "256"' in artifacts["cli_args_agg"]
    assert "--max-num-batched-tokens 8192" in artifacts["cli_args_agg"]
    assert _cli_flag_value(artifacts["cli_args_agg"], "--gpu-memory-utilization") == "0.9"


def test_disagg_candidate_renders_role_specific_vllm_gpu_memory_utilization():
    request = from_sweeper_candidate(
        _disagg_candidate(backend="vllm", backend_version="0.20.1"),
        workload={"isl": 8192, "osl": 1024},
        model_facts=ModelFacts(is_moe=True, architecture="DeepseekV3ForCausalLM"),
        generator_overrides={"K8sConfig": {"k8s_image": "example/vllm:0.20.1"}},
    )

    artifacts = generate_from_request(request)

    assert _cli_flag_value(artifacts["cli_args_prefill"], "--gpu-memory-utilization") == "0.9"
    assert _cli_flag_value(artifacts["cli_args_decode"], "--gpu-memory-utilization") == "0.85"
    assert artifacts["k8s_deploy.yaml"].count("--gpu-memory-utilization") == 2


@pytest.mark.parametrize(
    ("backend", "backend_version", "expected_fraction"),
    [
        ("vllm", "0.24.0", 0.9),
        ("sglang", "0.5.14", 0.88),
        ("trtllm", "1.3.0rc20", 0.9),
    ],
)
def test_agg_candidate_materializes_backend_default_kv_capacity(backend, backend_version, expected_fraction):
    request = from_sweeper_candidate(
        _agg_candidate(
            backend=backend,
            backend_version=backend_version,
            agg_gpu_memory_utilization=None,
        ),
        workload={"isl": 4000, "osl": 1000},
        model_facts=ModelFacts(is_moe=False, architecture="Qwen3ForCausalLM"),
    )

    extra = request.topology.roles["agg"].extra
    assert extra["kv_cache_free_gpu_memory_fraction"] == expected_fraction
    assert "num_gpu_blocks" not in extra
    assert "kv_cache_max_tokens" not in extra


@pytest.mark.parametrize(
    ("backend", "backend_version"),
    [
        ("vllm", "0.24.0"),
        ("sglang", "0.5.14"),
        ("trtllm", "1.3.0rc20"),
        ("trtllm", "1.3.0rc23"),
    ],
)
def test_agg_candidate_preserves_fixed_kv_capacity_in_generated_artifacts(backend, backend_version):
    request = from_sweeper_candidate(
        _agg_candidate(
            backend=backend,
            backend_version=backend_version,
            agg_gpu_memory_utilization=None,
            agg_num_gpu_blocks=256,
        ),
        workload={"isl": 4000, "osl": 1000},
        model_facts=ModelFacts(is_moe=False, architecture="Qwen3ForCausalLM"),
        generator_overrides={"K8sConfig": {"k8s_image": f"example/{backend}:{backend_version}"}},
    )

    extra = request.topology.roles["agg"].extra
    assert extra["num_gpu_blocks"] == 256
    assert "kv_cache_free_gpu_memory_fraction" not in extra

    artifacts = generate_from_request(request)
    if backend == "vllm":
        assert _cli_flag_value(artifacts["cli_args_agg"], "--num-gpu-blocks-override") == "256"
        assert "--gpu-memory-utilization" not in shlex.split(artifacts["cli_args_agg"])
    elif backend == "sglang":
        assert extra["kv_cache_max_tokens"] == 16384
        assert _cli_flag_value(artifacts["cli_args_agg"], "--max-total-tokens") == "16384"
        assert "--mem-fraction-static" not in shlex.split(artifacts["cli_args_agg"])
    else:
        assert extra["kv_cache_max_tokens"] == 16384
        engine_args = yaml.safe_load(artifacts["extra_engine_args_agg.yaml"])
        assert engine_args["kv_cache_config"]["max_tokens"] == 16384
        assert "free_gpu_memory_fraction" not in engine_args["kv_cache_config"]


@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm"])
def test_disagg_candidate_preserves_role_specific_fixed_kv_capacity(backend):
    request = from_sweeper_candidate(
        _disagg_candidate(
            backend=backend,
            prefill_gpu_memory_utilization=None,
            prefill_num_gpu_blocks=256,
            decode_gpu_memory_utilization=None,
            decode_num_gpu_blocks=512,
        ),
        workload={"isl": 8192, "osl": 1024},
        model_facts=ModelFacts(is_moe=True, architecture="DeepseekV3ForCausalLM"),
    )

    prefill = request.topology.roles["prefill"].extra
    decode = request.topology.roles["decode"].extra
    assert prefill["num_gpu_blocks"] == 256
    assert decode["num_gpu_blocks"] == 512
    assert "kv_cache_free_gpu_memory_fraction" not in prefill
    assert "kv_cache_free_gpu_memory_fraction" not in decode
    if backend in {"sglang", "trtllm"}:
        assert prefill["kv_cache_max_tokens"] == 16384
        assert decode["kv_cache_max_tokens"] == 32768


def test_candidate_rejects_conflicting_fixed_and_fractional_kv_capacity():
    with pytest.raises(SweeperCandidateError, match="are mutually exclusive"):
        from_sweeper_candidate(
            _agg_candidate(agg_gpu_memory_utilization=0.9, agg_num_gpu_blocks=256),
            workload={"isl": 4000, "osl": 1000},
            model_facts=ModelFacts(is_moe=False),
        )


def _cli_flag_value(cli_args: str, flag: str) -> str:
    tokens = shlex.split(cli_args)
    return tokens[tokens.index(flag) + 1]
