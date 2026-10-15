# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from aisimulate.sdk.config_adapter import (
    AdapterOverrides,
    EstimateRequestV1,
    InferenceXSource,
    ResolvedInferenceXSource,
    adapt_config,
    to_cli_estimate_kwargs,
)

pytestmark = pytest.mark.unit


def _config(**overrides):
    value = {
        "config_id": 7,
        "hardware": "h200",
        "framework": "vllm",
        "silicon_model": "llama70b",
        "precision": "fp8",
        "spec_method": "none",
        "disagg": False,
        "prefill_tp": 2,
        "prefill_ep": 1,
        "prefill_dp_attention": False,
        "prefill_num_workers": 2,
        "decode_tp": 4,
        "decode_ep": 1,
        "decode_dp_attention": False,
        "decode_num_workers": 1,
        "num_prefill_gpu": 4,
        "num_decode_gpu": 4,
    }
    value.update(overrides)
    return value


def _benchmark(**overrides):
    value = {"id": "bench-9", "isl": 1024, "osl": 128, "conc": 16}
    value.update(overrides)
    return value


def test_dense_agg_alias_quantization_and_topology():
    report = adapt_config(InferenceXSource(_config(), _benchmark()))
    outcome = report.outcomes[0]

    assert outcome.status == "adapted"
    assert outcome.request is not None
    assert outcome.request.model.path == "meta-llama/Meta-Llama-3.1-70B"
    assert outcome.request.quantization.gemm == "fp8"
    assert outcome.request.quantization.moe is None
    assert outcome.request.provenance.assumptions == (
        "InferenceX does not expose pipeline parallelism; pp_size defaults to 1.",
        "Aggregated worker replicas are omitted during cli_estimate lowering because "
        "cli_estimate has no aggregated worker-count parameter.",
    )
    assert to_cli_estimate_kwargs(outcome.request)["batch_size"] == 16


def test_agg_zero_worker_sentinel_is_normalized_before_worker_validation():
    outcome = adapt_config(InferenceXSource(_config(decode_num_workers=0), _benchmark())).outcomes[0]

    assert outcome.request is not None
    assert outcome.request.topology.worker.replicas == 1
    assert (
        "InferenceX aggregated decode_num_workers=0 is an irrelevant sentinel; replicas default to 1."
        in outcome.request.provenance.assumptions
    )


@pytest.mark.parametrize("disagg", [False, True])
@pytest.mark.parametrize("ep", [1, 4])
def test_vllm_attention_dp_flag_applies_to_every_worker(disagg, ep):
    config = _config(
        silicon_model="dsr1",
        disagg=disagg,
        is_multinode=disagg,
        decode_ep=ep,
        decode_dp_attention=True,
        prefill_dp_attention=True,
    )
    request = adapt_config(InferenceXSource(config, _benchmark())).requests[0]
    request = EstimateRequestV1.model_validate_json(request.model_dump_json())
    workers = (request.topology.prefill, request.topology.decode) if disagg else (request.topology.worker,)
    for worker in workers:
        assert worker.tp_size == 1
        assert worker.attention_dp_size == worker.gpus_per_replica
        assert worker.moe_tp_size * worker.moe_ep_size == worker.gpus_per_replica
    kwargs = to_cli_estimate_kwargs(request)
    assert kwargs["decode_batch_size" if disagg else "batch_size"] == 4
    assert kwargs["decode_attention_dp_size" if disagg else "attention_dp_size"] == 4
    if disagg:
        assert kwargs["prefill_attention_dp_size"] == 2
        assert kwargs["prefill_batch_size"] == 1


@pytest.mark.parametrize(
    "model,precision,path",
    [
        ("minimaxm2.7", "bf16", "MiniMaxAI/MiniMax-M2.7"),
        ("minimaxm2.7", "fp4", "nvidia/MiniMax-M2.7-NVFP4"),
        ("kimik2.6", "fp4", "nvidia/Kimi-K2.6-NVFP4"),
        ("kimik3", "fp4", "moonshotai/Kimi-K3"),
    ],
)
def test_registered_model_aliases_preserve_native_quantization(model, precision, path):
    config = _config(silicon_model=model, precision=precision)
    request = adapt_config(InferenceXSource(config, _benchmark())).requests[0]
    kwargs = to_cli_estimate_kwargs(EstimateRequestV1.model_validate_json(request.model_dump_json()))
    assert kwargs["model_path"] == path
    assert kwargs.get("gemm_quant_mode") is None
    assert kwargs.get("moe_quant_mode") is None
    assert kwargs["moe_tp_size"] == 4
    assert kwargs["moe_ep_size"] == 1


def test_db_export_id_is_retained_in_provenance():
    config = _config()
    config["id"] = config.pop("config_id")
    request = adapt_config(InferenceXSource(config, _benchmark())).requests[0]
    assert request.provenance.source_ids == {"config_id": 7, "benchmark_id": "bench-9"}


def test_moe_disagg_backend_folding_and_worker_arithmetic():
    report = adapt_config(
        InferenceXSource(
            _config(
                framework="dynamo-sglang",
                silicon_model="minimaxm2.5",
                disagg=True,
                prefill_tp=2,
                prefill_ep=2,
                prefill_num_workers=2,
                num_prefill_gpu=4,
                decode_tp=4,
                decode_ep=2,
                decode_num_workers=2,
                num_decode_gpu=8,
            ),
            _benchmark(conc=16),
        )
    )
    request = report.requests[0]
    kwargs = to_cli_estimate_kwargs(request)

    assert request.backend.name == "sglang"
    assert request.quantization.moe == "fp8_block"
    assert kwargs["prefill_num_workers"] == 2
    assert kwargs["prefill_batch_size"] == 1
    assert kwargs["decode_num_workers"] == 2
    assert kwargs["decode_batch_size"] == 8
    assert kwargs["decode_moe_tp_size"] == 2
    assert kwargs["decode_moe_ep_size"] == 2


@pytest.mark.parametrize(
    ("config", "concurrency", "message"),
    [
        (_config(hardware="amd-mi300x"), 16, "hardware"),
        (_config(silicon_model="unknown"), 16, "model/precision"),
        (_config(decode_tp=2, num_decode_gpu=4), 3, "divisible"),
        (
            _config(framework="dynamo-trtllm", decode_tp=2, num_decode_gpu=4),
            4,
            "does not match GPUs per worker",
        ),
        (_config(decode_num_workers=-1), 16, "must be positive"),
        (_config(disagg=True, prefill_num_workers=0, decode_num_workers=1), 16, "must be positive"),
    ],
)
def test_invalid_sources_are_rejected(config, concurrency, message):
    outcome = adapt_config(InferenceXSource(config, _benchmark(conc=concurrency))).outcomes[0]

    assert outcome.status == "rejected"
    assert message in outcome.diagnostics[-1].message


def test_mtp_requires_acceptance_and_can_be_explicitly_adapted():
    source = InferenceXSource(_config(spec_method="mtp"), _benchmark())
    rejected = adapt_config(source).outcomes[0]
    accepted = adapt_config(source, AdapterOverrides(nextn=1, nextn_accepted=0.8)).outcomes[0]

    assert rejected.status == "rejected"
    assert "nextn_accepted" in rejected.diagnostics[-1].message
    assert accepted.request is not None
    assert accepted.request.model.nextn == 1
    assert accepted.request.model.nextn_accepted == 0.8


def test_unpinned_backend_version_is_warning_not_rejection():
    outcome = adapt_config(InferenceXSource(_config(), _benchmark())).outcomes[0]

    assert outcome.status == "adapted"
    assert [diagnostic.code for diagnostic in outcome.diagnostics] == ["backend_version_unpinned"]


@pytest.mark.parametrize("framework", ["vllm", "sglang", "dynamo-trtllm"])
@pytest.mark.parametrize("width", [4, 8])
@pytest.mark.parametrize("attention_dp", [False, True])
@pytest.mark.parametrize("workers", [0, 1])
def test_single_node_ep_uses_shared_gpus_through_cli_lowering(framework, width, attention_dp, workers):
    config = _config(
        framework=framework,
        silicon_model="dsr1",
        is_multinode=False,
        decode_tp=width,
        decode_ep=width,
        decode_dp_attention=attention_dp,
        decode_num_workers=workers,
        num_decode_gpu=width * width,
    )
    original = config.copy()
    outcome = adapt_config(InferenceXSource(config, _benchmark(conc=64))).outcomes[0]

    assert outcome.status == "adapted"
    request = EstimateRequestV1.model_validate_json(outcome.request.model_dump_json())
    worker = request.topology.worker
    assert worker.replicas == 1
    assert worker.gpus_per_replica == width
    kwargs = to_cli_estimate_kwargs(request)
    assert kwargs["tp_size"] == (1 if attention_dp else width)
    assert kwargs["attention_dp_size"] == (width if attention_dp else 1)
    assert kwargs["batch_size"] == (64 // width if attention_dp else 64)
    assert kwargs["moe_tp_size"] == 1
    assert kwargs["moe_ep_size"] == width
    correction = next(d for d in outcome.diagnostics if d.code == "inferencex_gpu_count_normalized")
    assert correction.severity == "warning"
    assert correction.path == "config.num_decode_gpu"
    assert f"num_decode_gpu={width * width}" in correction.message
    assert f"effective GPU count is {width}" in correction.message
    assert correction.message in request.provenance.assumptions
    assert config == original


@pytest.mark.parametrize("framework", ["vllm", "sglang", "trtllm"])
@pytest.mark.parametrize("attention_dp", [False, True])
def test_single_node_ep_already_physical_gpu_count(framework, attention_dp):
    outcome = adapt_config(
        InferenceXSource(
            _config(
                framework=framework,
                silicon_model="dsr1",
                is_multinode=False,
                decode_ep=4,
                decode_dp_attention=attention_dp,
            ),
            _benchmark(),
        )
    ).outcomes[0]

    assert outcome.status == "adapted"
    worker = outcome.request.topology.worker
    assert worker.gpus_per_replica == 4
    assert worker.tp_size == (1 if attention_dp else 4)
    assert worker.attention_dp_size == (4 if attention_dp else 1)
    assert worker.batch_size == (4 if attention_dp else 16)
    assert all(d.code != "inferencex_gpu_count_normalized" for d in outcome.diagnostics)


@pytest.mark.parametrize("framework", ["sglang", "trtllm"])
def test_single_node_partial_ep_preserves_moe_tensor_parallelism(framework):
    outcome = adapt_config(
        InferenceXSource(
            _config(framework=framework, silicon_model="dsr1", is_multinode=False, decode_ep=2, num_decode_gpu=8),
            _benchmark(),
        )
    ).outcomes[0]

    assert outcome.status == "adapted"
    worker = outcome.request.topology.worker
    assert worker.gpus_per_replica == 4
    assert worker.tp_size == 4
    assert worker.moe_tp_size == 2
    assert worker.moe_ep_size == 2


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_decode_gpu": 12}, "must equal TP"),
        ({"decode_ep": 3, "num_decode_gpu": 12}, "supported shared GPU group"),
        ({"framework": "vllm", "decode_ep": 2, "num_decode_gpu": 8}, "supported shared GPU group"),
        ({"is_multinode": None}, "ambiguous"),
        ({"is_multinode": "false"}, "ambiguous"),
    ],
)
def test_single_node_ep_rejects_ambiguous_counts(overrides, message):
    config = _config(framework="sglang", silicon_model="dsr1", is_multinode=False, decode_ep=4, num_decode_gpu=16)
    config.update(overrides)
    if config["is_multinode"] is None:
        del config["is_multinode"]
    outcome = adapt_config(InferenceXSource(config, _benchmark())).outcomes[0]

    assert outcome.status == "rejected"
    assert outcome.diagnostics[-1].code == "inferencex_mapping_failed"
    assert message in outcome.diagnostics[-1].message


@pytest.mark.parametrize(
    ("framework", "tp", "attention_dp", "moe_tp", "moe_ep"),
    [("vllm", 4, 4, 1, 16), ("sglang", 16, 1, 4, 4), ("trtllm", 16, 1, 4, 4)],
)
@pytest.mark.parametrize(
    "overrides",
    [
        {"is_multinode": True},
        {"disagg": True},
        {"decode_num_workers": 2, "num_decode_gpu": 32},
    ],
)
def test_other_topologies_keep_reported_gpu_count(framework, tp, attention_dp, moe_tp, moe_ep, overrides):
    config = _config(
        framework=framework,
        silicon_model="dsr1",
        is_multinode=False,
        decode_tp=tp,
        decode_ep=4,
        num_decode_gpu=16,
    )
    config.update(overrides)
    outcome = adapt_config(InferenceXSource(config, _benchmark())).outcomes[0]

    assert outcome.status == "adapted"
    topology = outcome.request.topology
    worker = topology.decode if config["disagg"] else topology.worker
    assert worker.gpus_per_replica == 16
    assert worker.tp_size == tp
    assert worker.attention_dp_size == attention_dp
    assert worker.moe_tp_size == moe_tp
    assert worker.moe_ep_size == moe_ep
    assert all(d.code != "inferencex_gpu_count_normalized" for d in outcome.diagnostics)


@pytest.mark.parametrize("backend_version", [None, "0.11.0"])
def test_resolved_source_public_export_and_backend_warning(backend_version):
    source = ResolvedInferenceXSource(
        deployment={
            "schema_version": "resolved-deployment/1",
            "backend": "vllm",
            "system": "h200_sxm",
            "model_path": "meta-llama/Meta-Llama-3.1-70B",
            "workload": {"isl": 1024, "osl": 128, "concurrency": 16},
            "roles": {
                "aggregated": {
                    "topology": {"tp": 4, "pp": 1, "attention_dp": 1, "moe_tp": 4, "moe_ep": 1, "workers": 1},
                    "args": {"gpu_memory_utilization": 0.9, "kv_cache_dtype": "bfloat16"},
                    "quantization": {"gemm": "fp8", "moe": "fp8"},
                }
            },
        },
        config={**_config(), "id": 7},
        benchmark=_benchmark(),
        source_reference="fixture",
    )
    report = adapt_config(source, AdapterOverrides(backend_version=backend_version))
    assert report.outcomes[0].status == "adapted"
    lowered = to_cli_estimate_kwargs(report.requests[0])
    assert lowered.get("backend_version") == backend_version
    assert [item.code for item in report.outcomes[0].diagnostics] == (
        ["backend_version_unpinned"] if backend_version is None else []
    )
