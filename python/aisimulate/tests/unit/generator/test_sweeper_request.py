# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import re
import shlex
from pathlib import Path

import pytest
import yaml

from aisimulate.generator import api as generator_api
from aisimulate.generator.api import generate_from_request
from aisimulate.generator.request import (
    ModelFacts,
    SweeperCandidateError,
    from_sweeper_candidate,
    to_legacy_params,
)
from aisimulate.sweeper.config import Workload

_TRACE_WORKLOAD = {
    "source_type": "trace",
    "trace_path": "/traces/day.jsonl",
    "trace_paths": ["/traces/day.jsonl"],
}
_TRTLLM_TRANSFER_BUFFER_LINE = re.compile(r"^[ \t]*max_tokens_in_buffer: \d+\n", re.MULTILINE)


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


@pytest.mark.parametrize(
    ("backend", "backend_version"),
    [("vllm", "0.20.1"), ("sglang", "0.5.11"), ("trtllm", "1.3.0rc14")],
)
@pytest.mark.parametrize("mode", ["agg", "disagg"])
@pytest.mark.parametrize("block_size", [None, "omitted", 16], ids=["null", "missing", "explicit"])
def test_candidate_preserves_block_size_selection_in_artifacts(backend, backend_version, mode, block_size):
    candidate = (_agg_candidate if mode == "agg" else _disagg_candidate)(
        backend=backend, backend_version=backend_version, adapters={}
    )
    roles = ("agg",) if mode == "agg" else ("prefill", "decode")
    for role in roles:
        if block_size == "omitted":
            candidate["config"].pop(f"{role}_block_size")
        else:
            candidate["config"][f"{role}_block_size"] = block_size
    request = from_sweeper_candidate(
        candidate,
        workload={"isl": 4000, "osl": 1000},
        model_facts=ModelFacts(is_moe=False, architecture="Qwen3ForCausalLM"),
        generator_overrides={"K8sConfig": {"k8s_image": f"example/{backend}:{backend_version}"}},
    )

    params = to_legacy_params(request)
    artifacts = generate_from_request(request)
    expected = 16 if block_size == 16 else None
    for role in roles:
        if expected is None:
            assert "tokens_per_block" not in params["params"][role]
        else:
            assert params["params"][role]["tokens_per_block"] == expected
        _assert_artifact_block_size(artifacts, backend, role, expected)
    if expected is None:
        for parameter in ("--block-size", "--page-size", "tokens_per_block:"):
            assert parameter not in artifacts["k8s_deploy.yaml"]


@pytest.mark.parametrize(
    ("backend", "backend_version"),
    [("vllm", "0.20.1"), ("sglang", "0.5.11"), ("trtllm", "1.3.0rc14")],
)
def test_disagg_candidate_keeps_block_size_selection_role_local(backend, backend_version):
    request = from_sweeper_candidate(
        _disagg_candidate(
            backend=backend,
            backend_version=backend_version,
            prefill_block_size=None,
            decode_block_size=16,
            adapters={},
        ),
        workload={"isl": 4000, "osl": 1000},
        model_facts=ModelFacts(is_moe=False, architecture="Qwen3ForCausalLM"),
    )

    artifacts = generate_from_request(request)

    _assert_artifact_block_size(artifacts, backend, "prefill", None)
    _assert_artifact_block_size(artifacts, backend, "decode", 16)


@pytest.mark.parametrize("block_size", [0, -1, True, 32.5, "invalid"])
def test_candidate_rejects_invalid_explicit_block_size(block_size):
    with pytest.raises(SweeperCandidateError, match="agg_block_size must be a positive integer"):
        from_sweeper_candidate(
            _agg_candidate(agg_block_size=block_size),
            workload={"isl": 4000, "osl": 1000},
            model_facts=ModelFacts(is_moe=False),
        )


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


def test_missing_workload_lengths_fall_back_to_generator_overrides():
    request = from_sweeper_candidate(
        _agg_candidate(),
        workload={"isl": None, "osl": None},
        generator_overrides={"SlaConfig": {"isl": 2048, "osl": 512}},
        model_facts=ModelFacts(is_moe=False),
    )

    assert (request.sla.isl, request.sla.osl) == (2048, 512)


def test_workload_lengths_take_precedence_over_generator_overrides():
    request = from_sweeper_candidate(
        _agg_candidate(),
        workload={"isl": 4000, "osl": 1000},
        generator_overrides={"SlaConfig": {"isl": 2048, "osl": 512}},
        model_facts=ModelFacts(is_moe=False),
    )

    assert (request.sla.isl, request.sla.osl) == (4000, 1000)


def test_synthetic_workload_requires_request_lengths():
    with pytest.raises(SweeperCandidateError, match="a synthetic workload must provide isl and osl"):
        from_sweeper_candidate(_agg_candidate(), workload={"isl": 4000}, model_facts=ModelFacts(is_moe=False))


@pytest.mark.parametrize(
    "workload",
    [
        _TRACE_WORKLOAD,
        Workload(trace_path="/traces/day.jsonl"),
        Workload(trace_path="/traces/day.jsonl").model_dump(mode="json"),
        {"trace_paths": ["/traces/day.jsonl"]},
        {"source_type": "trace"},
    ],
    ids=["mapping", "sweeper-workload", "sweeper-workload-json", "trace-paths-only", "source-type-only"],
)
def test_trace_workload_lowers_without_request_lengths(workload):
    request = from_sweeper_candidate(_agg_candidate(), workload=workload, model_facts=ModelFacts(is_moe=False))

    assert (request.sla.isl, request.sla.osl) == (None, None)
    assert not [key for key in request.overrides.raw if key.startswith("SlaConfig.")]


@pytest.mark.parametrize(
    ("workload", "overrides", "supplied"),
    [
        ({**_TRACE_WORKLOAD, "isl": 1, "osl": 1}, None, "workload.isl, workload.osl"),
        (_TRACE_WORKLOAD, {"SlaConfig": {"isl": 4000, "osl": 1000}}, "SlaConfig.isl, SlaConfig.osl"),
        (_TRACE_WORKLOAD, {"SlaConfig.osl": 1000}, "SlaConfig.osl"),
    ],
)
def test_trace_workload_rejects_supplied_request_lengths(workload, overrides, supplied):
    with pytest.raises(SweeperCandidateError, match=re.escape(f"no fixed request length; remove {supplied}")):
        from_sweeper_candidate(
            _agg_candidate(),
            workload=workload,
            generator_overrides=overrides,
            model_facts=ModelFacts(is_moe=False),
        )


@pytest.mark.parametrize(
    ("workload", "context_length", "warned"),
    [(_TRACE_WORKLOAD, 10000, True), (_TRACE_WORKLOAD, 16384, False), ({"isl": 8192, "osl": 1024}, 10000, False)],
    ids=["trace-unaligned", "trace-aligned", "synthetic-unaligned"],
)
def test_trtllm_trace_warns_when_max_seq_len_is_off_the_block_grid(caplog, workload, context_length, warned):
    with caplog.at_level(logging.WARNING, logger="aisimulate.generator.request.sweeper"):
        from_sweeper_candidate(
            _disagg_candidate(context_length=context_length),
            workload=workload,
            model_facts=ModelFacts(is_moe=True, architecture="DeepseekV3ForCausalLM"),
        )

    messages = [record.getMessage() for record in caplog.records if "tokens_per_block" in record.getMessage()]
    if warned:
        # Both roles use 64-token blocks; 10000 = 156 * 64 + 16.
        assert [message.split(".")[0] for message in messages] == [
            f"TRT-LLM {role} max_seq_len 10000 is not a multiple of tokens_per_block 64"
            for role in ("prefill", "decode")
        ]
    else:
        assert messages == []


def test_trace_workload_is_rejected_for_a_slurm_bundle():
    with pytest.raises(SweeperCandidateError, match="a Slurm bundle needs a fixed request length"):
        from_sweeper_candidate(
            _agg_candidate(),
            workload=_TRACE_WORKLOAD,
            deployment_target="slurm",
            model_facts=ModelFacts(is_moe=False),
        )


def test_trace_candidate_benchmarks_only_with_explicit_benchmark_lengths():
    def render(overrides=None):
        return generate_from_request(
            from_sweeper_candidate(
                _agg_candidate(),
                workload=_TRACE_WORKLOAD,
                model_facts=ModelFacts(is_moe=False, architecture="Qwen3ForCausalLM"),
                generator_overrides=overrides,
            )
        )

    default = render()
    benchmarked = render({"BenchConfig": {"isl": 2048, "osl": 512}})

    assert not {"bench_run.sh", "k8s_bench.yaml"} & default.keys()
    assert "name: benchmark" not in default["sflow.yaml"]
    assert 'BENCH_ISL="${AICONFIGURATOR_BENCH_ISL:-2048}"' in benchmarked["bench_run.sh"]
    assert "--isl 2048" in benchmarked["k8s_bench.yaml"]
    assert "name: benchmark" in benchmarked["sflow.yaml"]


@pytest.mark.parametrize("mode", ["agg", "disagg"])
@pytest.mark.parametrize("block_size", [64, None])
@pytest.mark.parametrize(
    ("backend", "backend_version"),
    [("vllm", "0.20.1"), ("sglang", "0.5.14"), ("trtllm", "1.3.0rc20"), ("trtllm", "1.3.0rc23")],
)
def test_trace_candidate_renders_like_a_synthetic_candidate(mode, block_size, backend, backend_version):
    make_candidate = _agg_candidate if mode == "agg" else _disagg_candidate
    candidate = make_candidate(backend=backend, backend_version=backend_version, model_name="Qwen/Qwen3-32B-FP8")
    for role in ("agg",) if mode == "agg" else ("prefill", "decode"):
        candidate["config"][f"{role}_block_size"] = block_size
    # SGLang has no KVBM integration.
    candidate["config"].get("adapters", {}).pop("dynamo.kvbm", None)

    def render(workload):
        return generate_from_request(
            from_sweeper_candidate(
                candidate,
                workload=workload,
                model_facts=ModelFacts(is_moe=False, architecture="Qwen3ForCausalLM"),
                generator_overrides={"K8sConfig": {"k8s_image": f"example/{backend}:{backend_version}"}},
            )
        )

    synthetic = render({"isl": 4000, "osl": 1000})
    traced = render(_TRACE_WORKLOAD)

    # Benchmarks need one fixed ISL/OSL, including sflow.yaml's benchmark task.
    # Otherwise only TRT-LLM's P/D transfer buffer size is derived from the
    # lengths, and a trace leaves it unset.
    assert {"bench_run.sh", "k8s_bench.yaml"} <= synthetic.keys()
    assert synthetic.keys() - traced.keys() == {"bench_run.sh", "k8s_bench.yaml"}
    assert "name: benchmark" in synthetic["sflow.yaml"]
    assert "name: benchmark" not in traced["sflow.yaml"]
    assert "aiperf profile" not in traced["sflow.yaml"]
    sized = _TRTLLM_TRANSFER_BUFFER_LINE.findall("".join(synthetic.values()))
    assert bool(sized) == (backend == "trtllm" and mode == "disagg" and block_size is not None)
    for name, text in traced.items():
        if name == "k8s_deploy.yaml":
            # The DGD carries each engine config as an inline YAML string.
            assert yaml.safe_load(text) == _without_transfer_buffer_size(yaml.safe_load(synthetic[name]))
        elif name != "sflow.yaml":
            assert text == _without_transfer_buffer_size(synthetic[name]), name


def _without_transfer_buffer_size(value):
    if isinstance(value, str):
        return _TRTLLM_TRANSFER_BUFFER_LINE.sub("", value)
    if isinstance(value, dict):
        return {key: _without_transfer_buffer_size(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_without_transfer_buffer_size(item) for item in value]
    return value


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


@pytest.mark.parametrize("version", ["1.3.0rc14", "1.3.0rc29"])
def test_trtllm_candidate_enables_kv_transfer_without_changing_evaluated_limits(version):
    candidate = _disagg_candidate(backend_version=version)
    request = from_sweeper_candidate(
        candidate,
        workload={"isl": 8192, "osl": 1024},
        model_facts=ModelFacts(is_moe=True, architecture="DeepseekV3ForCausalLM"),
    )

    artifacts = generate_from_request(request)

    for role in ("prefill", "decode"):
        engine = yaml.safe_load(artifacts[f"extra_engine_args_{role}.yaml"])
        assert engine["cache_transceiver_config"]["backend"] == "DEFAULT"
        assert engine["cache_transceiver_config"]["max_tokens_in_buffer"] > 0
        assert engine["max_batch_size"] == candidate["config"][f"{role}_max_num_seqs"]
        assert engine["max_num_tokens"] == candidate["config"][f"{role}_max_num_batched_tokens"]
        assert engine["max_seq_len"] == candidate["config"]["context_length"]
        assert (
            engine["kv_cache_config"]["free_gpu_memory_fraction"]
            == candidate["config"][f"{role}_gpu_memory_utilization"]
        )
    assert artifacts["k8s_deploy.yaml"].count("cache_transceiver_config:") == 2


def _trtllm_engine_templates() -> dict[str, str]:
    template_dir = Path(generator_api.__file__).parent / "config" / "backend_templates" / "trtllm"
    templates = {}
    for path in sorted(template_dir.glob("extra_engine_args*.yaml.j2")):
        version = path.name.removeprefix("extra_engine_args").removesuffix(".yaml.j2").lstrip(".")
        # The unversioned template serves versions older than every versioned one.
        templates[version or "0.19.0"] = path.read_text()
    return templates


def test_only_pre_transceiver_trtllm_templates_omit_kv_transfer():
    without = {
        version for version, text in _trtllm_engine_templates().items() if "cache_transceiver_config" not in text
    }

    assert without == {"0.20.0", "1.0.0rc3"}


@pytest.mark.parametrize(
    "version",
    [version for version, text in _trtllm_engine_templates().items() if "cache_transceiver_config" in text],
)
def test_trtllm_trace_candidate_keeps_kv_transfer_without_a_guessed_buffer(version):
    candidate = _disagg_candidate(backend_version=version, model_name="Qwen/Qwen3-32B-FP8")

    def engine_args(workload):
        artifacts = generate_from_request(
            from_sweeper_candidate(
                candidate,
                workload=workload,
                model_facts=ModelFacts(is_moe=False, architecture="Qwen3ForCausalLM"),
            )
        )
        return {role: yaml.safe_load(artifacts[f"extra_engine_args_{role}.yaml"]) for role in ("prefill", "decode")}

    synthetic = engine_args({"isl": 8192, "osl": 1024})
    traced = engine_args(_TRACE_WORKLOAD)

    for role in ("prefill", "decode"):
        # (8192 + 1024 + 1500) rounded up to the 64-token block.
        assert synthetic[role]["cache_transceiver_config"] == {"backend": "DEFAULT", "max_tokens_in_buffer": 10752}
        assert traced[role]["cache_transceiver_config"] == {"backend": "DEFAULT"}


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
    ("backend", "backend_version", "block_size"),
    [
        ("vllm", "0.24.0", 64),
        ("vllm", "0.24.0", None),
        ("sglang", "0.5.14", 64),
        ("trtllm", "1.3.0rc20", 64),
        ("trtllm", "1.3.0rc23", 64),
    ],
)
def test_agg_candidate_preserves_fixed_kv_capacity_in_generated_artifacts(backend, backend_version, block_size):
    request = from_sweeper_candidate(
        _agg_candidate(
            backend=backend,
            backend_version=backend_version,
            agg_block_size=block_size,
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
    _assert_artifact_block_size(artifacts, backend, "agg", block_size)
    expected_max_tokens = 16384
    if backend == "vllm":
        assert _cli_flag_value(artifacts["cli_args_agg"], "--num-gpu-blocks-override") == "256"
        assert "--gpu-memory-utilization" not in shlex.split(artifacts["cli_args_agg"])
    elif backend == "sglang":
        assert extra["kv_cache_max_tokens"] == expected_max_tokens
        assert _cli_flag_value(artifacts["cli_args_agg"], "--max-total-tokens") == str(expected_max_tokens)
        assert "--mem-fraction-static" not in shlex.split(artifacts["cli_args_agg"])
    else:
        assert extra["kv_cache_max_tokens"] == expected_max_tokens
        engine_args = yaml.safe_load(artifacts["extra_engine_args_agg.yaml"])
        assert engine_args["kv_cache_config"]["max_tokens"] == expected_max_tokens
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


@pytest.mark.parametrize("backend", ["sglang", "trtllm"])
@pytest.mark.parametrize("role", ["agg", "prefill", "decode"])
@pytest.mark.parametrize("omit_field", [False, True], ids=["null", "missing"])
def test_fixed_block_capacity_requires_explicit_geometry(backend, role, omit_field):
    candidate = (_agg_candidate if role == "agg" else _disagg_candidate)(backend=backend)
    candidate["config"].update(
        {f"{role}_gpu_memory_utilization": None, f"{role}_num_gpu_blocks": 256, f"{role}_block_size": None}
    )
    if omit_field:
        candidate["config"].pop(f"{role}_block_size")

    with pytest.raises(SweeperCandidateError, match=rf"{role}_block_size is required to convert .*num_gpu_blocks"):
        from_sweeper_candidate(
            candidate,
            workload={"isl": 128, "osl": 1},
            model_facts=ModelFacts(is_moe=True, architecture="DeepseekV3ForCausalLM"),
        )


@pytest.mark.parametrize("rule", [None, "benchmark"])
def test_trtllm_unset_block_size_delegates_flashmla_buffer_alignment(rule, caplog):
    # FlashMLA selects 64-token blocks at runtime. Rounding ISL + OSL + 1500
    # to the simulation's 32-token default would emit the invalid size 1632.
    request = from_sweeper_candidate(
        _disagg_candidate(
            backend_version="1.3.0rc23",
            prefill_block_size=None,
            decode_block_size=None,
            context_length=10000,
            adapters={},
        ),
        workload={"isl": 128, "osl": 1},
        model_facts=ModelFacts(is_moe=True, architecture="DeepseekV3ForCausalLM"),
    )
    request.overrides.raw["rule"] = rule

    artifacts = generate_from_request(request)

    for role in ("prefill", "decode"):
        engine = yaml.safe_load(artifacts[f"extra_engine_args_{role}.yaml"])
        assert "tokens_per_block" not in engine["kv_cache_config"]
        assert engine["cache_transceiver_config"] == {"backend": "DEFAULT"}
        assert f"TRT-LLM {role} tokens_per_block is unset" in caplog.text
    assert "max_tokens_in_buffer" not in artifacts["k8s_deploy.yaml"]


def _cli_flag_value(cli_args: str, flag: str) -> str:
    tokens = shlex.split(cli_args)
    return tokens[tokens.index(flag) + 1]


def _assert_artifact_block_size(artifacts, backend, role, expected):
    if backend == "trtllm":
        engine_args = yaml.safe_load(artifacts[f"extra_engine_args_{role}.yaml"])
        if expected is None:
            assert "tokens_per_block" not in engine_args["kv_cache_config"]
        else:
            assert engine_args["kv_cache_config"]["tokens_per_block"] == expected
        if role in {"prefill", "decode"}:
            assert engine_args["cache_transceiver_config"]["backend"] == "DEFAULT"
            if expected is None:
                assert "max_tokens_in_buffer" not in engine_args["cache_transceiver_config"]
            else:
                assert engine_args["cache_transceiver_config"]["max_tokens_in_buffer"] > 0
    else:
        flag = "--block-size" if backend == "vllm" else "--page-size"
        if expected is None:
            assert flag not in shlex.split(artifacts[f"cli_args_{role}"])
        else:
            assert _cli_flag_value(artifacts[f"cli_args_{role}"], flag) == str(expected)
