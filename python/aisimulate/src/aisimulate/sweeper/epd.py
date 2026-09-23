# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sweeper integration with the in-tree AIC EPD estimator and composition.

The implementation of AIC PR #1340 is already shipped in this repository.
Import its candidate/overlay helpers instead of maintaining a second model.
This analytical overlay is not an event-level encoder queue simulation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, replace

from .config import EncoderSearch, SmartSearchConfig, Workload
from .kv_estimate import NoPerfDatabase, resolve_backend_version
from .replay import EncoderPoolSpec, NativeEncoderTiming, ReplayReport, ReplaySpec

logger = logging.getLogger(__name__)


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
    from aisimulate.sdk.sweep import _get_encoder_worker_candidates
    from aisimulate_core.sdk.backends.base_backend import BaseBackend
    from aisimulate_core.sdk.config import RuntimeConfig
    from aisimulate_core.sdk.perf_database import get_database_view

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
    geometry = None
    if encoder.mode == "native":
        # The replay lays out, encodes and transfers the processor's geometry,
        # pixel budget included, as the language worker's tower does.
        from aisimulate_core.sdk.backends.base_backend import image_geometry

        geometry = image_geometry(
            model_name, images.height, images.width, min_pixels=images.min_pixels, max_pixels=images.max_pixels
        )
        visual_tokens = geometry.visual_tokens * images.count
    else:
        visual_tokens = BaseBackend.effective_prefill_isl(model_name, runtime) - runtime.isl
    if context_length is not None and runtime.isl + visual_tokens + runtime.osl > context_length:
        raise ValueError("EPD text + visual + output tokens exceed context_length")
    catalog = {}
    system = encoder.hardware_sku or hardware_sku
    for backend in dict.fromkeys(backends):
        version = encoder.backend_version or (backend_version if system == hardware_sku else None)
        if not version:
            try:
                version = resolve_backend_version(system, backend)
            except NoPerfDatabase:
                logger.warning("Skipping encoder backend: no encoder database for %s/%s", system, backend)
                continue
        database = get_database_view(system, backend, version, database_mode="SILICON")
        if database is None:
            logger.warning("Skipping encoder backend: no encoder database for %s/%s/%s", system, backend, version)
            continue
        version = database.version
        if encoder.mode == "native":
            rows = _native_encoder_rows(
                encoder,
                model_name=model_name,
                system=system,
                backend=backend,
                version=version,
                database=database,
                runtime=runtime,
                images=images,
                geometry=geometry,
            )
        else:
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
                    mode=encoder.mode,
                    native=row.get("native"),
                )
                if gpu_budget is None or point.total_gpus < gpu_budget:
                    catalog[f"{backend}|tp{point.tp}|bs{point.batch_size}|w{workers}"] = point
    if not catalog:
        raise ValueError("no feasible encoder pool for the requested shape and GPU budget")
    return catalog


def _native_encoder_rows(encoder, *, model_name, system, backend, version, database, runtime, images, geometry):
    """One row per encoder tp at the loop's batch cap, with the terms the replay prices batches from.

    The forward is priced at replay time by the canonical timing model at the pool's
    tensor width, the oracle a language rank uses for its vision tower; memory and
    power come from the analytical candidate helper at the cap. The CPU preprocessing
    extrapolates the tokenizer manager's measured `process` stage per image: the
    encoder servers run the same image processor over the whole batch.
    """
    from aisimulate.sdk.errors import InsufficientMemoryError, NoFeasibleConfigError
    from aisimulate.sdk.sweep import _get_encoder_worker_candidates

    from ..config.engine import HostProfileConfig
    from ..vl.table import resolve_frontend

    (cap,) = encoder.batch_size
    profile = HostProfileConfig.model_validate(encoder.host_profile)
    frontend, digest = resolve_frontend(profile, model=model_name, images=images.model_dump(mode="json"), tensor=1)
    rows = []
    for tp in encoder.tp:
        try:
            (point,) = _get_encoder_worker_candidates(
                model_path=model_name,
                tp_list=[tp],
                b_list=[cap],
                runtime_config=runtime,
                database=database,
                backend_name=backend,
                latency_correction=encoder.latency_correction,
            )
        except (ValueError, InsufficientMemoryError, NoFeasibleConfigError) as exc:
            logger.debug("native encoder: tp=%s rejected: %s", tp, exc)
            continue
        rows.append(
            {
                **point,
                "native": NativeEncoderTiming(
                    preprocess_ms_per_image=frontend.stages[0].service_ms / images.count,
                    preprocess_source="frontend_process_extrapolated",
                    shape={
                        "sequences": geometry.sequences,
                        "patch_tokens": geometry.patch_tokens,
                        "transformer_tokens": geometry.transformer_tokens,
                        "output_tokens": geometry.output_tokens,
                    },
                    transfer_bytes_per_image=geometry.embedding_bytes,
                    transfer_bandwidth_gb_s=encoder.transfer_bandwidth_gb_per_second,
                    timing_model=encoder_timing_payload(
                        model=model_name, system=system, backend=backend, backend_version=version, tp=tp
                    ),
                    host_profile_path=profile.path,
                    host_profile_frontend=profile.frontend,
                    host_profile_digest=digest,
                ),
            }
        )
    return rows


def encoder_timing_payload(*, model: str, system: str, backend: str, backend_version: str, tp: int) -> dict:
    """The canonical timing payload of an encoder server: a prefill-shaped rank at
    the pool's tensor width whose model compiles the vision tower."""
    from aisimulate_core.sdk import ForwardPassPerfModelConfig

    from ..config.common import omit_inactive_moe_controls
    from .forward_pass_estimator import resolve_systems_paths

    canonical = ForwardPassPerfModelConfig(
        model=model,
        system=system,
        backend=backend,
        backend_version=backend_version,
        worker_type="prefill",
        tp=tp,
        systems_paths=resolve_systems_paths(None),
        encoder_parallel="tp",
    )
    return {"type": "external", "provider": "aic", "config": omit_inactive_moe_controls(canonical.to_dict())}


def language_gpus(deployment) -> int:
    """GPUs of the language workers of a resolved deployment."""
    parallel = deployment.parallel_config

    def role_gpus(prefix, workers):
        return (
            workers
            * int(parallel[prefix + "tp"])
            * int(parallel[prefix + "pp"])
            * int(parallel[prefix + "attention_dp"])
        )

    if deployment.deployment_mode == "agg":
        return role_gpus("", deployment.num_workers)
    return role_gpus("prefill_", deployment.num_prefill_workers) + role_gpus("decode_", deployment.num_decode_workers)


def add_encoder_choices(branches, catalog):
    """Expose resolved pools as optimizer choices; backend mismatches fail before replay."""
    result = []
    for branch in branches:
        backends = branch.knob_choices["backend"]
        choices = {key: value for key, value in catalog.items() if value.backend in backends}
        if not choices:
            logger.warning("Skipping deployment mode: no encoder candidates for %s", branch.deployment_mode)
            continue
        available_backends = {point.backend for point in choices.values()}
        result.append(
            replace(
                branch,
                knob_choices={
                    **branch.knob_choices,
                    "backend": [backend for backend in backends if backend in available_backends],
                    "encoder_candidate": list(choices),
                },
            )
        )
    if not result:
        raise ValueError("no feasible encoder pool for the supported deployment modes")
    return result


def apply_encoder_overlay(report: ReplayReport, spec: ReplaySpec) -> ReplayReport:
    """Reuse AIC rate/TTFT composition; never label this as per-request replay."""
    from aisimulate.sdk.sweep import _overlay_encoder_stage

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
    language = language_gpus(deployment)
    row = _overlay_encoder_stage(
        {
            "seq/s": source["completed_requests"] * 1000 / source["duration_ms"],
            "tokens/s": source["output_throughput_tok_s"],
            "ttft": source["mean_ttft_ms"],
            "tpot": source["mean_tpot_ms"],
            "request_latency": source["mean_e2e_latency_ms"],
            "osl": spec.workload["osl"],
            "num_total_gpus": language,
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
    metrics.update(power_w=None, power_coverage=None)
    if encoder.power_w is not None:
        metrics["encoder_power_w"] = encoder.power_w
    return ReplayReport(
        metrics=metrics,
        metadata={
            "metric_semantics": "analytical_epd_overlay",
            "encoder": asdict(encoder),
            "language_gpus": language,
            "total_gpus": row["num_total_gpus"],
            "language_replay_duration_ms": source["duration_ms"],
            "deployment_artifact_generation_supported": False,
            "prediction_config_supported": True,
            "sla_semantics": "aggregate_means_only",
            "aggregate_sla_bounds": spec.goal.get("sla"),
        },
    )
