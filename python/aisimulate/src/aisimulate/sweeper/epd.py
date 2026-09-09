# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sweeper integration with the in-tree AIC EPD estimator and composition.

The implementation of AIC PR #1340 is already shipped in this repository.
Import its candidate/overlay helpers instead of maintaining a second model.
This analytical overlay is not an event-level encoder queue simulation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, replace

from .config import EncoderSearch, SmartSearchConfig, Workload
from .kv_estimate import resolve_backend_version
from .replay import EncoderPoolSpec, ReplayReport, ReplaySpec


def resolve_encoder_catalog(config: SmartSearchConfig) -> dict[str, EncoderPoolSpec]:
    """Resolve silicon timing/memory once; each candidate pins its exact identity."""
    search = config.search_space
    if search.encoder is None:
        raise ValueError("encoder catalog requires an EPD image workload")
    return resolve_encoder_pools(
        model_name=search.model_name,
        hardware_sku=search.hardware_sku,
        backends=search.backend,
        backend_version=search.backend_version,
        context_length=search.context_length,
        encoder=search.encoder,
        workload=config.workload,
        gpu_budget=search.gpu_budget,
    )


def resolve_encoder_pools(
    *,
    model_name: str,
    hardware_sku: str,
    backends: list[str],
    backend_version: str | None,
    context_length: int | None,
    encoder: EncoderSearch,
    workload: Workload,
    gpu_budget: int | None = None,
) -> dict[str, EncoderPoolSpec]:
    """Shared AIC resolution for search domains and one concrete CLI prediction."""
    from aiconfigurator.sdk.sweep import _get_encoder_worker_candidates
    from aiconfigurator_core.sdk.backends.base_backend import BaseBackend
    from aiconfigurator_core.sdk.config import RuntimeConfig
    from aiconfigurator_core.sdk.perf_database import get_database_view

    images = workload.images
    if images is None:
        raise ValueError("encoder catalog requires an EPD image workload")
    runtime = RuntimeConfig(
        isl=workload.isl,
        osl=workload.osl,
        image_height=images.height,
        image_width=images.width,
        num_images_per_request=images.count,
    )
    visual_tokens = BaseBackend.effective_prefill_isl(model_name, runtime) - runtime.isl
    if context_length is not None and runtime.isl + visual_tokens + runtime.osl > context_length:
        raise ValueError("EPD text + visual + output tokens exceed context_length")
    catalog = {}
    system = encoder.hardware_sku or hardware_sku
    for backend in dict.fromkeys(backends):
        version = (
            encoder.backend_version
            or (backend_version if system == hardware_sku else None)
            or resolve_backend_version(system, backend)
        )
        database = get_database_view(system, backend, version, database_mode="SILICON")
        if database is None:
            raise ValueError(f"no encoder database for {system}/{backend}/{version}")
        version = database.version
        rows = _get_encoder_worker_candidates(
            model_path=model_name,
            tp_list=encoder.tp,
            b_list=encoder.batch_size,
            runtime_config=runtime,
            database=database,
            backend_name=backend,
            latency_correction=encoder.latency_correction,
        )
        for row in rows:
            coverage = float(row.get("power_coverage", 0.0))
            power = float(row.get("power_w", 0.0))
            if not math.isfinite(coverage) or not 0 <= coverage <= 1 or not math.isfinite(power) or power < 0:
                raise ValueError("invalid encoder power evidence")
            for workers in encoder.workers:
                point = EncoderPoolSpec(
                    model=model_name,
                    system=system,
                    backend=backend,
                    backend_version=version,
                    tp=int(row["tp"]),
                    batch_size=int(row["bs"]),
                    workers=workers,
                    latency_ms=float(row["encoder_latency"]),
                    throughput_rps=float(row["seq/s"]),
                    memory_gib=float(row["memory"]),
                    rate_degradation=encoder.rate_degradation,
                    latency_correction=encoder.latency_correction,
                    visual_tokens=visual_tokens,
                    image_height=images.height,
                    image_width=images.width,
                    image_count=images.count,
                    power_w=power if power > 0 and coverage > 0 else None,
                    power_coverage=coverage if power > 0 else 0.0,
                )
                if gpu_budget is None or point.total_gpus < gpu_budget:
                    catalog[f"{backend}|tp{point.tp}|bs{point.batch_size}|w{workers}"] = point
    if not catalog:
        raise ValueError("no feasible encoder pool for the requested shape and GPU budget")
    return catalog


def add_encoder_choices(branches, catalog):
    """Expose resolved pools as optimizer choices; backend mismatches fail before replay."""
    result = []
    for branch in branches:
        backends = branch.knob_choices["backend"]
        choices = {key: value for key, value in catalog.items() if value.backend in backends}
        if not choices:
            raise ValueError(f"no encoder candidates for {branch.deployment_mode}")
        result.append(replace(branch, knob_choices={**branch.knob_choices, "encoder_candidate": list(choices)}))
    return result


def apply_encoder_overlay(report: ReplayReport, spec: ReplaySpec) -> ReplayReport:
    """Reuse AIC rate/TTFT composition; never label this as per-request replay."""
    from aiconfigurator.sdk.sweep import _overlay_encoder_stage

    encoder = spec.backend_deployment.encoder
    if encoder is None:
        return report
    source = report.metrics
    required = (
        "completed_requests",
        "duration_ms",
        "output_throughput_tok_s",
        "mean_ttft_ms",
        "mean_tpot_ms",
        "mean_e2e_latency_ms",
    )
    if any(name not in source or not math.isfinite(source[name]) or source[name] < 0 for name in required):
        raise ValueError("EPD requires finite complete language replay metrics")
    if source["duration_ms"] <= 0 or source["completed_requests"] <= 0:
        raise ValueError("EPD requires a completed language replay")
    deployment = spec.backend_deployment
    parallel = deployment.parallel_config

    def role_gpus(prefix, workers):
        return (
            workers
            * int(parallel[prefix + "tp"])
            * int(parallel[prefix + "pp"])
            * int(parallel[prefix + "attention_dp"])
        )

    language_gpus = (
        role_gpus("", deployment.num_workers)
        if deployment.deployment_mode == "agg"
        else role_gpus("prefill_", deployment.num_prefill_workers) + role_gpus("decode_", deployment.num_decode_workers)
    )
    row = _overlay_encoder_stage(
        {
            "seq/s": source["completed_requests"] * 1000 / source["duration_ms"],
            "tokens/s": source["output_throughput_tok_s"],
            "ttft": source["mean_ttft_ms"],
            "tpot": source["mean_tpot_ms"],
            "request_latency": source["mean_e2e_latency_ms"],
            "osl": spec.workload["osl"],
            "num_total_gpus": language_gpus,
        },
        {
            "seq/s": encoder.throughput_rps,
            "encoder_latency": encoder.latency_ms,
            "num_total_gpus": encoder.tp,
            "tp": encoder.tp,
            "bs": encoder.batch_size,
            "memory": encoder.memory_gib,
        },
        encoder.workers,
        encoder_degradation=encoder.rate_degradation,
        # Replay already includes language queueing. Single-point AIC semantics
        # add the raw encoder batch latency once (no sweep queueing multiplier).
        ttft_scale=1.0,
    )
    # Keep only aggregate metrics with defined overlay semantics. In particular,
    # language-only goodput, percentiles, power and raw records are not EPD data.
    metrics = {
        name: source[name]
        for name in (
            "completed_requests",
            "num_ttft_samples",
            "num_tpot_samples",
            "num_e2e_latency_samples",
            "mean_output_token_throughput_per_user",
            "mean_tpot_ms",
        )
        if name in source
    }
    duration = source["completed_requests"] / row["seq/s"] * 1000
    metrics.update(
        output_throughput_tok_s=row["tokens/s"],
        mean_ttft_ms=row["ttft"],
        mean_e2e_latency_ms=row["request_latency"],
        duration_ms=duration,
        gpu_hours=row["num_total_gpus"] * duration / 3_600_000,
        encoder_latency_ms=encoder.latency_ms,
        encoder_gpus=float(encoder.total_gpus),
        encoder_memory_gib=encoder.memory_gib,
        encoder_power_coverage=encoder.power_coverage,
    )
    if encoder.power_w is not None:
        metrics["encoder_power_w"] = encoder.power_w
    return ReplayReport(
        metrics=metrics,
        metadata={
            "metric_semantics": "analytical_epd_overlay",
            "encoder": asdict(encoder),
            "language_gpus": language_gpus,
            "total_gpus": row["num_total_gpus"],
            "language_replay_duration_ms": source["duration_ms"],
            "deployment_artifact_generation_supported": False,
            "prediction_config_supported": True,
            "sla_semantics": "aggregate_means_only",
            "aggregate_sla_bounds": spec.goal.get("sla"),
        },
    )
