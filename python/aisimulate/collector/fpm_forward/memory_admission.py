# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conservative memory admission for FPM parallel topologies."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from aisimulate.sdk.memory import KVCacheEstimator, kv_cache_budget_bytes
from aisimulate_core.sdk.errors import PerfDataNotAvailableError
from aisimulate_core.sdk.fpm_profile import FpmModelProfile
from aisimulate_core.sdk.perf_database import load_system_spec

from .capabilities import ModelCapabilityProfile
from .types import ParallelTopology

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DTypeMemoryEstimate:
    kv_cache_dtype: str
    disposition: str
    estimated_non_kv_bytes: int | None
    gpu_capacity_bytes: int | None
    reason: str
    provenance: str | None = None
    gpu_memory_budget_bytes: int | None = None

    def to_dict(self) -> dict[str, object]:
        budget = self.gpu_capacity_bytes if self.gpu_memory_budget_bytes is None else self.gpu_memory_budget_bytes
        payload = {
            "kv_cache_dtype": self.kv_cache_dtype,
            "disposition": self.disposition,
            "estimated_non_kv_bytes": self.estimated_non_kv_bytes,
            "gpu_capacity_bytes": self.gpu_capacity_bytes,
            "headroom_bytes": (
                budget - self.estimated_non_kv_bytes
                if self.estimated_non_kv_bytes is not None and self.gpu_capacity_bytes is not None
                else None
            ),
            "reason": self.reason,
        }
        if self.provenance is not None:
            payload["provenance"] = self.provenance
        if self.gpu_memory_budget_bytes is not None:
            payload["gpu_memory_budget_bytes"] = self.gpu_memory_budget_bytes
        return payload


@dataclass(frozen=True, slots=True)
class TopologyMemoryDecision:
    topology: ParallelTopology
    disposition: str
    max_new_tokens: int
    estimates: tuple[DTypeMemoryEstimate, ...]
    reason: str
    source: str = "aic_native_configured_max_new_tokens"
    max_batch_size: int = 1
    profile_max_num_tokens: int | None = None

    def to_dict(self) -> dict[str, object]:
        envelope = {
            "max_new_tokens": self.max_new_tokens,
            "aic_max_num_tokens": self.max_new_tokens,
            "max_batch_size": self.max_batch_size,
        }
        if self.profile_max_num_tokens is not None:
            envelope = {
                "requested_prefill_tokens": self.max_new_tokens,
                "max_num_tokens": self.profile_max_num_tokens,
                "max_batch_size": self.max_batch_size,
                "scope": "rank_local",
                "cuda_graph_reservation": "runtime_profiled_separately",
            }
        return {
            "topology": self.topology.to_dict(),
            "disposition": self.disposition,
            "source": self.source,
            "activation_envelope": envelope,
            "estimates": [estimate.to_dict() for estimate in self.estimates],
            "reason": self.reason,
        }


def _estimate_dtype(
    *,
    backend: str,
    model_path: str,
    system: str,
    capability: ModelCapabilityProfile,
    topology: ParallelTopology,
    kv_cache_dtype: str,
    max_new_tokens: int,
    fpm_profile: FpmModelProfile | None = None,
    max_batch_size: int | None = None,
    gpu_memory_utilization: float | None = None,
) -> DTypeMemoryEstimate:
    # Planner-owned capability data fails closed: resolve_model_capability
    # guarantees an fmha mapping for every resolved KV dtype, so a missing
    # entry is a broken invariant, not an AIC estimate failure — it must
    # never be recorded as a fail-open "unknown" admission outcome.
    try:
        fmha_quant_mode = capability.dtype.fmha_by_kv_dtype[kv_cache_dtype]
    except KeyError as error:
        raise ValueError(
            f"capability profile invariant violated: kv dtype {kv_cache_dtype!r} has no fmha mapping"
        ) from error
    if fpm_profile is not None:
        deployment = fpm_profile.select(
            model=model_path,
            system=system,
            backend=backend,
            backend_version=capability.aic_database_version,
            tp_size=topology.tp,
            pp_size=topology.pp,
            attention_dp_size=topology.dp,
            moe_tp_size=topology.moe_tp,
            moe_ep_size=topology.moe_ep,
            cp_size=topology.cp,
        )
        resources = deployment.resources
        resources.validate_envelope(
            max_num_tokens=max_new_tokens,
            max_batch_size=max_batch_size or resources.max_batch_size,
        )
        capacity = math.floor(float(load_system_spec(system)["gpu"]["mem_capacity"]))
        if capacity <= 0:
            raise ValueError(f"invalid GPU memory capacity for {system!r}: {capacity}")
        budget = math.floor(
            kv_cache_budget_bytes(
                capacity=capacity,
                non_kv=0,
                fraction=gpu_memory_utilization if gpu_memory_utilization is not None else 1.0,
                of_free=False,
            )
        )
        if resources.memory_source != "declared":
            return DTypeMemoryEstimate(
                kv_cache_dtype=kv_cache_dtype,
                disposition="unknown",
                estimated_non_kv_bytes=None,
                gpu_capacity_bytes=capacity,
                reason="memory capacity is determined by this collection worker's runtime initialization; "
                "no non-KV byte declaration is required",
                provenance=resources.provenance,
                gpu_memory_budget_bytes=budget if gpu_memory_utilization is not None else None,
            )
        rejected = resources.non_kv_bytes >= budget
        capacity_label = "configured GPU memory budget" if gpu_memory_utilization is not None else "GPU capacity"
        return DTypeMemoryEstimate(
            kv_cache_dtype=kv_cache_dtype,
            disposition="rejected" if rejected else "admitted",
            estimated_non_kv_bytes=resources.non_kv_bytes,
            gpu_capacity_bytes=capacity,
            reason=(
                f"declared rank-local non-KV resource bound is not below {capacity_label}"
                if rejected
                else f"declared rank-local non-KV resource bound is below {capacity_label}; "
                "runtime profiling still determines KV capacity and CUDA-graph memory"
            ),
            provenance=resources.provenance,
            gpu_memory_budget_bytes=budget if gpu_memory_utilization is not None else None,
        )
    try:
        breakdown = KVCacheEstimator.from_request(
            model_path,
            system,
            backend,
            capability.aic_database_version,
            max_num_tokens=max_new_tokens,
            max_batch_size=1,
            tp_size=topology.tp,
            pp_size=topology.pp,
            attention_dp_size=topology.dp,
            moe_tp_size=topology.moe_tp,
            moe_ep_size=topology.moe_ep,
            gemm_quant_mode=capability.dtype.gemm_quant_mode,
            moe_quant_mode=capability.dtype.moe_quant_mode,
            kvcache_quant_mode=kv_cache_dtype,
            fmha_quant_mode=fmha_quant_mode,
            comm_quant_mode=capability.dtype.comm_quant_mode,
        ).breakdown
    except PerfDataNotAvailableError as error:
        # Coverage gap: collection may be exactly what fills it - stay runnable.
        return DTypeMemoryEstimate(
            kv_cache_dtype=kv_cache_dtype,
            disposition="unknown",
            estimated_non_kv_bytes=None,
            gpu_capacity_bytes=None,
            reason=f"AIC memory estimate unavailable: {type(error).__name__}: {error}",
        )
    except ValueError as error:
        # A model-layer validator cannot turn into a Collector skip. The live
        # runtime is authoritative for structural feasibility; only a concrete
        # size-vs-capacity estimate below may filter generation-time work.
        return DTypeMemoryEstimate(
            kv_cache_dtype=kv_cache_dtype,
            disposition="unknown",
            estimated_non_kv_bytes=None,
            gpu_capacity_bytes=None,
            reason=f"AIC memory estimate unavailable: {type(error).__name__}: {error}",
        )
    except Exception as error:
        return DTypeMemoryEstimate(
            kv_cache_dtype=kv_cache_dtype,
            disposition="unknown",
            estimated_non_kv_bytes=None,
            gpu_capacity_bytes=None,
            reason=f"AIC memory estimate unavailable: {type(error).__name__}: {error}",
        )

    try:
        non_kv = math.ceil(float(breakdown["non_kv_bytes"]))
        capacity = math.floor(float(breakdown["gpu_memory_capacity_bytes"]))
        if non_kv < 0 or capacity <= 0:
            raise ValueError(f"invalid AIC memory estimate: non_kv={non_kv}, capacity={capacity}")
    except Exception as error:
        return DTypeMemoryEstimate(
            kv_cache_dtype=kv_cache_dtype,
            disposition="unknown",
            estimated_non_kv_bytes=None,
            gpu_capacity_bytes=None,
            reason=f"AIC memory estimate unavailable: {type(error).__name__}: {error}",
        )

    budget = math.floor(
        kv_cache_budget_bytes(
            capacity=capacity,
            non_kv=0,
            fraction=gpu_memory_utilization if gpu_memory_utilization is not None else 1.0,
            of_free=False,
        )
    )
    rejected = non_kv >= budget
    capacity_label = "configured GPU memory budget" if gpu_memory_utilization is not None else "GPU capacity"
    return DTypeMemoryEstimate(
        kv_cache_dtype=kv_cache_dtype,
        disposition="rejected" if rejected else "admitted",
        estimated_non_kv_bytes=non_kv,
        gpu_capacity_bytes=capacity,
        reason=(
            f"AIC configured max-new-token non-KV memory is not below {capacity_label}"
            if rejected
            else f"AIC configured max-new-token non-KV memory is below {capacity_label}"
        ),
        gpu_memory_budget_bytes=budget if gpu_memory_utilization is not None else None,
    )


def filter_memory_infeasible_topologies(
    *,
    backend: str,
    model_path: str,
    system: str,
    capability: ModelCapabilityProfile,
    topologies: tuple[ParallelTopology, ...],
    max_new_tokens: int,
    fpm_profile: FpmModelProfile | None = None,
    max_batch_size: int | None = None,
    gpu_memory_utilization: float | None = None,
) -> tuple[tuple[ParallelTopology, ...], tuple[TopologyMemoryDecision, ...]]:
    """Drop topologies that cannot fit the configured max-new-token envelope.

    This is intentionally a one-sided generation-time filter. A topology is
    rejected only when every requested KV dtype has a successful estimate
    and all estimates exceed the declared share of rank-local capacity (physical
    capacity when no GPU memory fraction is declared). Without a profile,
    unknown AIC estimates remain runnable. Supplied profiles must explicitly
    cover the requested deployment and envelope; validation errors propagate.
    """

    if max_new_tokens < 1:
        raise ValueError("FPM topology memory admission requires positive max_new_tokens")

    decisions = []
    admitted = []
    rejected_capacity = []
    for topology in topologies:
        resources = (
            fpm_profile.select(
                model=model_path,
                system=system,
                backend=backend,
                backend_version=capability.aic_database_version,
                tp_size=topology.tp,
                pp_size=topology.pp,
                attention_dp_size=topology.dp,
                moe_tp_size=topology.moe_tp,
                moe_ep_size=topology.moe_ep,
                cp_size=topology.cp,
            ).resources
            if fpm_profile is not None
            else None
        )
        estimates = tuple(
            _estimate_dtype(
                backend=backend,
                model_path=model_path,
                system=system,
                capability=capability,
                topology=topology,
                kv_cache_dtype=kv_cache_dtype,
                max_new_tokens=max_new_tokens,
                fpm_profile=fpm_profile,
                max_batch_size=max_batch_size,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            for kv_cache_dtype in capability.dtype.kv_cache_dtypes
        )
        dispositions = {estimate.disposition for estimate in estimates}
        if "admitted" in dispositions:
            disposition = "admitted"
            reason = "at least one requested KV dtype fits the configured max-new-token envelope"
            admitted.append(topology)
        elif "unknown" in dispositions:
            disposition = "unknown"
            reason = "AIC could not prove the topology is impossible; runtime verification is required"
            admitted.append(topology)
        else:
            disposition = "rejected"
            reason = "all requested KV dtypes exceed rank-local GPU memory budget at configured max new tokens"
            rejected_capacity.append((topology, estimates))
        decisions.append(
            TopologyMemoryDecision(
                topology=topology,
                disposition=disposition,
                max_new_tokens=max_new_tokens,
                estimates=estimates,
                reason=reason,
                source=(
                    f"fpm_profile_{resources.memory_source}"
                    if resources is not None
                    else "aic_native_configured_max_new_tokens"
                ),
                max_batch_size=resources.max_batch_size if resources is not None else 1,
                profile_max_num_tokens=resources.max_num_tokens if resources is not None else None,
            )
        )

    if rejected_capacity:
        details = []
        for topology, estimates in rejected_capacity:
            best = min(
                estimates,
                key=lambda estimate: estimate.estimated_non_kv_bytes
                if estimate.estimated_non_kv_bytes is not None
                else math.inf,
            )
            if best.estimated_non_kv_bytes is None:
                details.append(f"{topology.to_dict()}={best.reason}")
            else:
                budget = (
                    best.gpu_capacity_bytes if best.gpu_memory_budget_bytes is None else best.gpu_memory_budget_bytes
                )
                details.append(
                    f"{topology.to_dict()}={best.estimated_non_kv_bytes / 2**30:.2f}/{budget / 2**30:.2f} GiB"
                )
        logger.warning(
            "fpm_forward: dropped %d/%d topologies (%s configured max-new-token non-KV memory "
            "exceeds GPU memory budget, system=%s, max_new_tokens=%d): %s",
            len(rejected_capacity),
            len(topologies),
            "FPM profile" if fpm_profile is not None else "AIC",
            system,
            max_new_tokens,
            "; ".join(details),
        )
    if not admitted:
        source = "FPM profile" if fpm_profile is not None else "AIC"
        raise ValueError(
            f"{source} max-new-token memory admission rejected every FPM topology; "
            f"model={model_path!r}, system={system!r}, max_new_tokens={max_new_tokens}"
        )
    return tuple(admitted), tuple(decisions)
