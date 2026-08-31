# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encoder-disaggregated (EPD) search and replay composition.

The native Sweeper owns topology selection and replay contracts. Encoder
worker timing/memory estimation deliberately reuses the imported AIC estimator
until that estimator moves into the lower-level core facade; it is resolved
once before sampling and never re-resolved by a runner.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any

from aiconfigurator.sdk.sweep import _get_encoder_worker_candidates
from aiconfigurator_core.sdk import perf_database
from aiconfigurator_core.sdk.backends.base_backend import BaseBackend
from aiconfigurator_core.sdk.config import RuntimeConfig

from .config import (
    DatabaseMode,
    ForwardModel,
    SearchSpace,
    SmartSearchConfig,
    TransferKind,
    Workload,
)
from .estimator import resolve_estimator_specs
from .heterogeneous import RoleEstimatorSpecs
from .replay import EncoderWorkerSpec, EpdDeploymentSpec, EstimatorSpec

if TYPE_CHECKING:
    from .search_space import BranchSpace


class EpdResolutionError(ValueError):
    """An encoder estimator or worker catalog cannot be resolved exactly."""


def image_runtime_config(workload: Workload) -> RuntimeConfig:
    """Translate the fixed image profile to AIC's encoder runtime contract."""

    return RuntimeConfig(
        isl=workload.isl or 1,
        osl=workload.osl or 1,
        image_height=workload.image_height,
        image_width=workload.image_width,
        num_images_per_request=workload.num_images_per_request,
        num_image_tokens=workload.num_image_tokens,
    )


def visual_context_tokens(workload: Workload, model_path: str) -> int:
    """Post-merge vision tokens injected into the language-model context."""

    if not workload.has_images:
        return 0
    runtime = image_runtime_config(workload)
    return BaseBackend.effective_prefill_isl(model_path, runtime) - int(runtime.isl)


def _encoder_space(
    search_space: SearchSpace,
    base: EstimatorSpec,
) -> SearchSpace:
    database_mode = search_space.encoder_database_mode or DatabaseMode(base.database_mode)
    transfer_policy = search_space.encoder_transfer_policy or [TransferKind(value) for value in base.transfer_policy]
    systems_paths = search_space.encoder_systems_paths or list(base.systems_paths)
    return search_space.model_copy(
        update={
            "model_name": base.model_path,
            "hardware_sku": search_space.encoder_hardware_sku or base.system,
            "backend": [base.backend],
            "backend_version": search_space.requested_encoder_backend_version(base.backend),
            "database_mode": database_mode,
            "transfer_policy": transfer_policy,
            "forward_model": ForwardModel.OP_LEVEL,
            "systems_paths": systems_paths,
        }
    )


def _encoder_base_estimators(
    estimator_specs: Mapping[str, EstimatorSpec],
    role_estimator_specs: Mapping[str, RoleEstimatorSpecs],
) -> dict[str, EstimatorSpec]:
    bases = dict(estimator_specs)
    bases.update({pair_label: role_specs.prefill for pair_label, role_specs in role_estimator_specs.items()})
    return bases


def resolve_epd_catalog(
    config: SmartSearchConfig,
    *,
    estimator_specs: Mapping[str, EstimatorSpec],
    role_estimator_specs: Mapping[str, RoleEstimatorSpecs],
) -> dict[str, EncoderWorkerSpec]:
    """Resolve every encoder TP/batch/worker-count point before sampling."""

    if not config.search_space.enable_epd:
        return {}
    search_space = config.search_space
    catalog: dict[str, EncoderWorkerSpec] = {}
    bases = _encoder_base_estimators(estimator_specs, role_estimator_specs)
    if not bases:
        raise EpdResolutionError("EPD has no aggregate/prefill estimator identity")

    for backend_key, base in bases.items():
        encoder_space = _encoder_space(search_space, base)
        try:
            estimator = resolve_estimator_specs(encoder_space)[base.backend]
            database = perf_database.get_database_view(
                estimator.system,
                estimator.backend,
                estimator.backend_version,
                systems_paths=list(estimator.systems_paths),
                allow_missing_data=estimator.database_mode in {"EMPIRICAL", "SOL"},
                database_mode=estimator.database_mode,
                transfer_policy=list(estimator.transfer_policy),
            )
            rows = _get_encoder_worker_candidates(
                model_path=estimator.model_path,
                tp_list=search_space.encoder_tp_candidates,
                b_list=search_space.encoder_batch_size_candidates,
                runtime_config=image_runtime_config(config.workload),
                database=database,
                backend_name=estimator.backend,
                latency_correction=search_space.encoder_latency_correction,
            )
        except Exception as exc:
            raise EpdResolutionError(
                "cannot resolve EPD encoder candidates for "
                f"{backend_key!r} on {encoder_space.hardware_sku}/"
                f"{base.backend}: {exc}"
            ) from exc

        worker_counts = search_space.encoder_num_workers_candidates or list(
            range(1, search_space.max_encoder_workers + 1)
        )
        for row in rows:
            for num_workers in worker_counts:
                total_gpus = int(row["num_total_gpus"]) * int(num_workers)
                if total_gpus >= search_space.gpu_budget:
                    continue
                candidate_id = (
                    f"{backend_key}|{estimator.system}|{estimator.backend}|"
                    f"{estimator.backend_version}|tp{int(row['tp'])}|"
                    f"bs{int(row['bs'])}|w{int(num_workers)}"
                )
                catalog[candidate_id] = EncoderWorkerSpec(
                    candidate_id=candidate_id,
                    backend_key=backend_key,
                    estimator=estimator,
                    tp=int(row["tp"]),
                    batch_size=int(row["bs"]),
                    num_workers=int(num_workers),
                    latency_ms=float(row["encoder_latency"]),
                    throughput_rps_per_worker=float(row["seq/s"]),
                    memory_gib_per_worker=float(row["memory"]),
                    rate_degradation=search_space.encoder_rate_degradation,
                    power_w_per_worker=float(row.get("power_w", 0.0) or 0.0),
                    power_coverage=float(row.get("power_coverage", 0.0) or 0.0),
                )
    if not catalog:
        raise EpdResolutionError("no EPD encoder candidate leaves at least one GPU for the language pool")
    return catalog


def add_epd_branch_choices(
    branches: list[BranchSpace],
    catalog: Mapping[str, EncoderWorkerSpec],
) -> list[BranchSpace]:
    """Add the concrete encoder pool as a searched categorical choice."""

    updated: list[BranchSpace] = []
    for branch in branches:
        viable_backends = set(branch.knob_choices.get("backend", []))
        candidates = [
            candidate_id for candidate_id, candidate in catalog.items() if candidate.backend_key in viable_backends
        ]
        if not candidates:
            raise EpdResolutionError(
                f"deployment_mode={branch.deployment_mode!r} has no encoder candidate for any viable backend"
            )
        choices = dict(branch.knob_choices)
        choices["encoder_candidate"] = candidates
        updated.append(replace(branch, knob_choices=choices))
    return updated


def materialize_epd_deployment(
    selection: Mapping[str, Any],
    *,
    catalog: Mapping[str, EncoderWorkerSpec],
    language_gpus: int,
    deployment_mode: str,
    ttft_scale: float,
) -> EpdDeploymentSpec:
    """Bind one sampled catalog entry and enforce backend/GPU identity."""

    candidate_id = selection.get("encoder_candidate")
    if not isinstance(candidate_id, str) or candidate_id not in catalog:
        raise EpdResolutionError(f"unknown encoder_candidate {candidate_id!r}")
    encoder = catalog[candidate_id]
    if encoder.backend_key != selection.get("backend"):
        raise EpdResolutionError(
            "encoder candidate belongs to a different backend identity: "
            f"{encoder.backend_key!r} != {selection.get('backend')!r}"
        )
    if deployment_mode not in {"agg", "disagg"}:
        raise EpdResolutionError(f"EPD supports deployment_mode 'agg' or 'disagg', got {deployment_mode!r}")
    return EpdDeploymentSpec(
        encoder=encoder,
        language_gpus=language_gpus,
        language_topology=deployment_mode,
        ttft_scale=ttft_scale,
    )


def apply_epd_metrics(
    metrics: Mapping[str, float],
    epd: EpdDeploymentSpec,
    *,
    goal: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Overlay legacy EPD rate/latency semantics on native replay metrics."""

    result = {name: float(value) for name, value in metrics.items()}
    request_rate = result.get("request_throughput_rps")
    if request_rate is None or not math.isfinite(request_rate) or request_rate <= 0.0:
        raise ValueError("EPD replay requires a positive finite request_throughput_rps metric")
    capacity = epd.encoder.degraded_capacity_rps
    throughput_scale = min(1.0, capacity / request_rate)
    for name in (
        "request_throughput_rps",
        "output_throughput_tok_s",
        "goodput_request_throughput_rps",
        "goodput_output_throughput_tok_s",
    ):
        if name in result:
            result[name] *= throughput_scale

    added_latency = epd.encoder.latency_ms * epd.ttft_scale
    if "mean_ttft_ms" in result:
        result["mean_ttft_ms"] += added_latency
    if "mean_e2e_latency_ms" in result:
        result["mean_e2e_latency_ms"] += added_latency

    sla = goal.get("sla")
    if isinstance(sla, Mapping):
        violates = any(
            result.get(metric, 0.0) > float(sla[limit])
            for limit, metric in (
                ("ttft_ms", "mean_ttft_ms"),
                ("itl_ms", "mean_tpot_ms"),
                ("e2e_ms", "mean_e2e_latency_ms"),
            )
            if sla.get(limit) is not None and metric in result
        )
        if violates:
            for name in (
                "goodput_request_throughput_rps",
                "goodput_output_throughput_tok_s",
            ):
                if name in result:
                    result[name] = 0.0

    duration_ms = result.get("duration_ms")
    if duration_ms is not None and duration_ms >= 0.0:
        result["gpu_hours"] = epd.total_gpus * duration_ms / 3_600_000.0

    result.update(
        encoder_latency_ms=epd.encoder.latency_ms,
        encoder_capacity_rps=capacity,
        encoder_memory_gib=epd.encoder.memory_gib_per_worker,
        encoder_power_w=epd.encoder.power_w_per_worker,
        encoder_power_coverage=epd.encoder.power_coverage,
        encoder_gpus=float(epd.encoder.total_gpus),
    )
    metadata = {
        "topology": "E+agg" if epd.language_topology == "agg" else "E+P+D",
        "artifact_generation_supported": epd.artifact_generation_supported,
        "language_gpus": epd.language_gpus,
        "total_gpus": epd.total_gpus,
        "encoder": asdict(epd.encoder),
        "throughput_scale": throughput_scale,
        "added_ttft_ms": added_latency,
        "power_provenance_available": epd.encoder.power_coverage >= 1.0,
    }
    return result, metadata
