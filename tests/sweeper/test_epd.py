# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native EPD topology, multimodal context, and legacy-semantic parity."""

from dataclasses import asdict, replace

import pytest

import aisimulate.sweeper.epd as epd_module
import aisimulate.sweeper.search as search_module
from aiconfigurator.sdk.sweep import _overlay_encoder_stage
from aisimulate.sweeper.config import SmartSearchConfig, Workload
from aisimulate.sweeper.engine_request import EngineControlTemplate
from aisimulate.sweeper.epd import (
    EpdResolutionError,
    apply_epd_metrics,
    resolve_epd_catalog,
    visual_context_tokens,
)
from aisimulate.sweeper.parallel_enum import ParallelShape, ReplicaParallelConfig
from aisimulate.sweeper.replay import (
    BackendDeploymentSpec,
    EncoderWorkerSpec,
    EpdDeploymentSpec,
    EstimatorSpec,
    ReplaySpec,
    RunnerCapabilities,
)


def _estimator() -> EstimatorSpec:
    return EstimatorSpec(
        model_path="Qwen/Qwen3-VL-8B-Instruct",
        model_architecture="Qwen3VLForConditionalGeneration",
        system="h200_sxm",
        backend="vllm",
        backend_version="0.11.0",
        performance_data_version="0.11.0",
        database_mode="SILICON",
        transfer_policy=("xshape",),
        forward_model="op_level",
        engine_step_backend="rust",
        systems_paths=("/systems",),
        performance_data_root="/systems",
    )


def _encoder(*, backend_key: str = "vllm") -> EncoderWorkerSpec:
    return EncoderWorkerSpec(
        candidate_id="vllm|h200|vllm|0.11.0|tp2|bs4|w1",
        backend_key=backend_key,
        estimator=_estimator(),
        tp=2,
        batch_size=4,
        num_workers=1,
        latency_ms=50.0,
        throughput_rps_per_worker=80.0,
        memory_gib_per_worker=1.5,
        rate_degradation=0.75,
        power_w_per_worker=200.0,
        power_coverage=1.0,
    )


def test_visual_context_tokens_support_one_and_multiple_images():
    one = Workload(
        isl=128,
        osl=16,
        concurrency=1,
        num_request_ratio=1,
        image_height=448,
        image_width=448,
        num_images_per_request=1,
    )
    multiple = one.model_copy(update={"num_images_per_request": 3})

    one_tokens = visual_context_tokens(one, _estimator().model_path)

    assert one_tokens > 0
    assert visual_context_tokens(multiple, _estimator().model_path) == 3 * one_tokens


def test_epd_overlay_matches_approved_legacy_rate_and_ttft_semantics():
    epd = EpdDeploymentSpec(
        encoder=_encoder(),
        language_gpus=40,
        language_topology="disagg",
        ttft_scale=1.8,
    )
    metrics = {
        "request_throughput_rps": 100.0,
        "output_throughput_tok_s": 10_000.0,
        "mean_ttft_ms": 60.0,
        "mean_tpot_ms": 8.0,
        "mean_e2e_latency_ms": 852.0,
        "duration_ms": 1_000.0,
        "gpu_hours": 40.0 / 3_600.0,
    }

    actual, metadata = apply_epd_metrics(metrics, epd, goal={})
    legacy = _overlay_encoder_stage(
        {
            "seq/s": 100.0,
            "tokens/s": 10_000.0,
            "ttft": 60.0,
            "tpot": 8.0,
            "osl": 100,
            "request_latency": 852.0,
            "num_total_gpus": 40,
            "power_w": 0.0,
        },
        {
            "encoder_latency": 50.0,
            "seq/s": 80.0,
            "num_total_gpus": 2,
            "tp": 2,
            "bs": 4,
            "memory": 1.5,
            "power_w": 200.0,
            "power_coverage": 1.0,
        },
        1,
        ttft_scale=1.8,
        encoder_degradation=0.75,
    )

    assert actual["request_throughput_rps"] == pytest.approx(legacy["seq/s"])
    assert actual["output_throughput_tok_s"] == pytest.approx(legacy["tokens/s"])
    assert actual["mean_ttft_ms"] == pytest.approx(legacy["ttft"])
    assert actual["mean_e2e_latency_ms"] == pytest.approx(legacy["request_latency"])
    assert actual["gpu_hours"] == pytest.approx(42 / 3_600.0)
    assert metadata["topology"] == "E+P+D"
    assert metadata["total_gpus"] == 42
    assert metadata["artifact_generation_supported"] is False


def test_epd_overlay_recomputes_goodput_after_encoder_latency():
    metrics = {
        "request_throughput_rps": 100.0,
        "output_throughput_tok_s": 10_000.0,
        "goodput_request_throughput_rps": 90.0,
        "goodput_output_throughput_tok_s": 9_000.0,
        "mean_ttft_ms": 60.0,
        "mean_tpot_ms": 8.0,
        "mean_e2e_latency_ms": 852.0,
    }

    actual, _ = apply_epd_metrics(
        metrics,
        EpdDeploymentSpec(encoder=_encoder(), language_gpus=40, ttft_scale=1.8),
        goal={"sla": {"ttft_ms": 120.0, "itl_ms": 10.0}},
    )

    assert actual["mean_ttft_ms"] == 150.0
    assert actual["goodput_request_throughput_rps"] == 0.0
    assert actual["goodput_output_throughput_tok_s"] == 0.0


def test_runner_capability_gates_epd_before_execution():
    deployment = BackendDeploymentSpec(
        deployment_mode="agg",
        backend="vllm",
        backend_version="0.11.0",
        epd=EpdDeploymentSpec(encoder=_encoder(), language_gpus=2),
    )
    replay = ReplaySpec(
        backend_deployment=deployment,
        workload={},
        goal={},
    )

    with pytest.raises(ValueError, match="does not support EPD"):
        RunnerCapabilities(
            supported_backend_topologies=(("vllm", "agg"),),
        ).require_compatible(replay)

    RunnerCapabilities(
        supported_backend_topologies=(("vllm", "agg"),),
        supported_epd_backend_topologies=(("vllm", "agg"),),
    ).require_compatible(replay)


def test_encoder_contract_is_canonical_json_data():
    payload = asdict(_encoder())

    assert payload["tp"] == 2
    assert payload["estimator"]["model_path"] == "Qwen/Qwen3-VL-8B-Instruct"


def test_encoder_catalog_resolves_system_timing_memory_and_worker_counts(monkeypatch):
    base = _estimator()
    resolved_spaces = []

    def resolve(space):
        resolved_spaces.append(space)
        return {
            "vllm": replace(
                base,
                system=space.hardware_sku,
                backend_version="encoder-version",
                performance_data_version="encoder-version",
            )
        }

    monkeypatch.setattr(epd_module, "resolve_estimator_specs", resolve)
    monkeypatch.setattr(
        epd_module.perf_database,
        "get_database_view",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        epd_module,
        "_get_encoder_worker_candidates",
        lambda **kwargs: [
            {
                "encoder_latency": 12.5,
                "seq/s": 40.0,
                "num_total_gpus": 2,
                "tp": 2,
                "bs": 4,
                "memory": 3.5,
                "power_w": 250.0,
                "power_coverage": 0.75,
            }
        ],
    )
    config = SmartSearchConfig.model_validate(
        {
            "search_space": {
                "model_name": base.model_path,
                "hardware_sku": "h200_sxm",
                "backend": ["vllm"],
                "gpu_budget": 8,
                "enable_epd": True,
                "encoder_hardware_sku": "b200_sxm",
                "encoder_tp_candidates": [2],
                "encoder_batch_size_candidates": [4],
                "encoder_num_workers_candidates": [1, 2],
            },
            "workload": {
                "isl": 128,
                "osl": 16,
                "concurrency": 1,
                "num_request_ratio": 1,
                "num_image_tokens": 256,
                "num_images_per_request": 1,
            },
        }
    )

    catalog = resolve_epd_catalog(
        config,
        estimator_specs={"vllm": base},
        role_estimator_specs={},
    )

    assert resolved_spaces[0].hardware_sku == "b200_sxm"
    assert {candidate.num_workers for candidate in catalog.values()} == {1, 2}
    selected = max(catalog.values(), key=lambda candidate: candidate.num_workers)
    assert selected.estimator.system == "b200_sxm"
    assert selected.total_gpus == 4
    assert selected.memory_gib_per_worker == 3.5
    assert selected.power_coverage == 0.75


@pytest.mark.parametrize(
    "power_fields",
    [
        {},
        {"power_w": 0.0, "power_coverage": 1.0},
        {"power_w": 250.0, "power_coverage": 0.0},
    ],
)
def test_encoder_catalog_rejects_missing_power(monkeypatch, power_fields):
    base = _estimator()
    monkeypatch.setattr(
        epd_module,
        "resolve_estimator_specs",
        lambda _space: {"vllm": base},
    )
    monkeypatch.setattr(
        epd_module.perf_database,
        "get_database_view",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        epd_module,
        "_get_encoder_worker_candidates",
        lambda **kwargs: [
            {
                "encoder_latency": 12.5,
                "seq/s": 40.0,
                "num_total_gpus": 2,
                "tp": 2,
                "bs": 4,
                "memory": 3.5,
                **power_fields,
            }
        ],
    )
    config = SmartSearchConfig.model_validate(
        {
            "search_space": {
                "model_name": base.model_path,
                "hardware_sku": "h200_sxm",
                "backend": ["vllm"],
                "gpu_budget": 8,
                "enable_epd": True,
                "encoder_num_workers_candidates": [1],
            },
            "workload": {
                "isl": 128,
                "osl": 16,
                "concurrency": 1,
                "num_request_ratio": 1,
                "num_image_tokens": 256,
                "num_images_per_request": 1,
            },
        }
    )

    with pytest.raises(EpdResolutionError, match="power"):
        resolve_epd_catalog(
            config,
            estimator_specs={"vllm": base},
            role_estimator_specs={},
        )


def test_candidate_materialization_accounts_for_encoder_pool_and_fails_artifacts_closed():
    base = _estimator()
    encoder = _encoder()
    config = SmartSearchConfig.model_validate(
        {
            "search_space": {
                "model_name": base.model_path,
                "hardware_sku": base.system,
                "backend": ["vllm"],
                "deployment_mode": ["agg"],
                "gpu_budget": 8,
                "enable_epd": True,
            },
            "workload": {
                "isl": 128,
                "osl": 16,
                "concurrency": 1,
                "num_request_ratio": 1,
                "num_image_tokens": 256,
                "num_images_per_request": 1,
            },
        }
    )
    parallel = ReplicaParallelConfig(
        shape=ParallelShape(tp=2, dp=1, moe_tp=1, moe_ep=1),
        replicas=2,
    )
    selection = {
        "deployment_mode": "agg",
        "backend": "vllm",
        "agg_max_num_batched_tokens": 8192,
        "agg_max_num_seqs": 256,
        "encoder_candidate": encoder.candidate_id,
    }

    class Factory:
        def capabilities(self):
            return RunnerCapabilities(
                supported_backend_topologies=(("vllm", "agg"),),
                supported_epd_backend_topologies=(("vllm", "agg"),),
            )

    prepared, result = search_module._materialize_one(
        selection,
        parallel,
        config=config,
        goal=config.goal,
        providers={},
        provider_plans={},
        runner_factory=Factory(),
        estimator_specs={"vllm": base},
        engine_controls={
            "vllm": EngineControlTemplate(
                backend="vllm",
                max_seq_len=32_768,
                model_family="QWEN3VL",
                is_moe=False,
                memory_fraction_kind="of_total",
            )
        },
        role_estimator_specs={},
        role_engine_controls={},
        encoder_catalog={encoder.candidate_id: encoder},
    )

    assert result is None
    assert prepared is not None
    assert prepared.sample["language_gpus"] == 4
    assert prepared.sample["used_gpus"] == 6
    assert prepared.sample["deployment_artifact_generation_supported"] is False
    assert prepared.replay_spec.backend_deployment.epd is not None
    assert prepared.replay_spec.backend_deployment.epd.total_gpus == 6
