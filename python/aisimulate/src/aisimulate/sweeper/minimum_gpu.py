# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Topology-neutral minimum-GPU sizing over evaluated Sweeper candidates."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import Candidate, OptimizationGoal
from .score import meets_aggregate_sla

_RATE_CAPACITY_KEY = "request_throughput_rps"
_CONCURRENCY_CAPACITY_KEYS = (
    "supported_concurrency",
    "kv_load_concurrency_capacity",
    "concurrency",
)
_ROLE_NAMES = ("encoder", "prefill", "decode", "agg", "attention", "ffn")


class LoadTarget(BaseModel):
    """One load-sizing target and its optional deployment ceiling."""

    model_config = ConfigDict(extra="forbid")

    request_rate: float | None = Field(default=None, gt=0)
    concurrency: float | None = Field(default=None, gt=0)
    max_gpus: int | None = Field(default=None, gt=0)
    allow_partial: bool = False

    @field_validator("request_rate", "concurrency")
    @classmethod
    def _finite_load(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("load targets must be finite")
        return value

    @model_validator(mode="after")
    def _one_load_shape(self) -> LoadTarget:
        if (self.request_rate is None) == (self.concurrency is None):
            raise ValueError("exactly one of request_rate or concurrency must be provided")
        return self

    @property
    def kind(self) -> Literal["request_rate", "concurrency"]:
        return "request_rate" if self.request_rate is not None else "concurrency"

    @property
    def value(self) -> float:
        value = self.request_rate if self.request_rate is not None else self.concurrency
        assert value is not None
        return float(value)


class LoadRecommendation(BaseModel):
    """One candidate scaled to a load target.

    ``replicas_needed`` and ``total_gpus_needed`` always describe the true
    uncapped minimum. ``deployed_*`` and ``supported_load`` describe the
    optional GPU-capped deployment.
    """

    model_config = ConfigDict(extra="forbid")

    candidate: Candidate
    target: LoadTarget
    capacity_per_replica: float = Field(gt=0)
    replicas_needed: int = Field(ge=1)
    total_gpus_needed: int = Field(ge=1)
    deployed_replicas: int = Field(ge=1)
    deployed_gpus: int = Field(ge=1)
    supported_load: float = Field(gt=0)
    load_served_pct: float = Field(ge=0, le=100)
    limiting_role: str
    partial: bool

    @property
    def capacity_per_gpu(self) -> float:
        return self.capacity_per_replica / self.candidate.used_gpus


class UnsupportedCandidateCapacity(ValueError):
    """An evaluated candidate does not expose the requested capacity."""


class NoFeasibleLoadRecommendation(ValueError):
    """No candidate can be sized under the requested policy."""

    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = tuple(reasons)
        detail = "; ".join(self.reasons[:5]) or "no candidates were provided"
        if len(self.reasons) > 5:
            detail += f"; +{len(self.reasons) - 5} more"
        super().__init__(f"no feasible load recommendation: {detail}")


def _finite_positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _first_capacity(mappings: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> float | None:
    for mapping in mappings:
        for key in keys:
            capacity = _finite_positive(mapping.get(key))
            if capacity is not None:
                return capacity
    return None


def _role_capacities(candidate: Candidate, target: LoadTarget) -> dict[str, float]:
    capacities: dict[str, float] = {}
    mappings: tuple[Mapping[str, Any], ...] = (candidate.metrics, candidate.config)
    for role in _ROLE_NAMES:
        keys = (
            (f"{role}_request_throughput_rps",)
            if target.kind == "request_rate"
            else (
                f"{role}_supported_concurrency",
                f"{role}_concurrency_capacity",
            )
        )
        capacity = _first_capacity(mappings, keys)
        if capacity is not None:
            capacities[role] = capacity
    return capacities


def candidate_capacity(candidate: Candidate, target: LoadTarget) -> tuple[float, str]:
    """Resolve per-replica capacity and the limiting role.

    End-to-end replay capacity is authoritative when present. Per-role
    capacity keys are used to identify the bottleneck and are also a fallback
    for topology extensions that publish only role-level rates.
    """
    if candidate.used_gpus <= 0:
        raise UnsupportedCandidateCapacity("candidate used_gpus must be positive")

    mappings: tuple[Mapping[str, Any], ...] = (candidate.metrics, candidate.config)
    overall_keys = (_RATE_CAPACITY_KEY,) if target.kind == "request_rate" else _CONCURRENCY_CAPACITY_KEYS
    overall = _first_capacity(mappings, overall_keys)
    roles = _role_capacities(candidate, target)
    if overall is None and roles:
        overall = min(roles.values())
    if overall is None:
        names = ", ".join(overall_keys)
        raise UnsupportedCandidateCapacity(
            f"candidate does not expose positive finite {target.kind} capacity ({names})"
        )

    if roles:
        limiting_role = min(roles, key=lambda role: (roles[role], role))
    else:
        explicit_role = candidate.config.get("limiting_role")
        if isinstance(explicit_role, str) and explicit_role:
            limiting_role = explicit_role
        elif candidate.config.get("deployment_mode") == "agg":
            limiting_role = "agg"
        else:
            limiting_role = "rate_matched_deployment"
    return overall, limiting_role


def size_candidate(candidate: Candidate, target: LoadTarget) -> LoadRecommendation:
    """Scale one evaluated candidate to the target load."""
    capacity, limiting_role = candidate_capacity(candidate, target)
    replicas_needed = max(1, math.ceil(target.value / capacity))
    total_gpus_needed = replicas_needed * candidate.used_gpus
    deployed_replicas = replicas_needed

    if target.max_gpus is not None and total_gpus_needed > target.max_gpus:
        max_replicas = target.max_gpus // candidate.used_gpus
        if max_replicas < 1:
            raise UnsupportedCandidateCapacity(
                f"one replica needs {candidate.used_gpus} GPUs, above max_gpus={target.max_gpus}"
            )
        if not target.allow_partial:
            raise UnsupportedCandidateCapacity(
                f"target needs {total_gpus_needed} GPUs, above max_gpus={target.max_gpus}; "
                "set allow_partial=true to return capped service"
            )
        deployed_replicas = max_replicas

    deployed_gpus = deployed_replicas * candidate.used_gpus
    supported_load = capacity * deployed_replicas
    load_served_pct = min(100.0, supported_load / target.value * 100.0)
    return LoadRecommendation(
        candidate=candidate,
        target=target,
        capacity_per_replica=capacity,
        replicas_needed=replicas_needed,
        total_gpus_needed=total_gpus_needed,
        deployed_replicas=deployed_replicas,
        deployed_gpus=deployed_gpus,
        supported_load=supported_load,
        load_served_pct=load_served_pct,
        limiting_role=limiting_role,
        partial=load_served_pct < 100.0,
    )


def _config_key(config: Mapping[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)


def recommend_min_gpus(
    candidates: Sequence[Candidate],
    target: LoadTarget,
    *,
    goal: OptimizationGoal,
    top_n: int = 5,
) -> list[LoadRecommendation]:
    """Return candidate deployments ordered by minimum true GPU requirement.

    Strict aggregate SLA filtering uses the same inclusive policy as scalar
    and Pareto analysis. Full-service recommendations always rank ahead of
    partial capped deployments.
    """
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    recommendations: list[LoadRecommendation] = []
    reasons: list[str] = []
    for index, candidate in enumerate(candidates):
        if goal.strict_sla:
            assert goal.sla is not None
            if not meets_aggregate_sla(candidate.metrics, goal.sla):
                reasons.append(f"candidate {index}: strict aggregate SLA violation")
                continue
        try:
            recommendations.append(size_candidate(candidate, target))
        except UnsupportedCandidateCapacity as exc:
            reasons.append(f"candidate {index}: {exc}")

    if not recommendations:
        raise NoFeasibleLoadRecommendation(reasons)

    def _rank_key(recommendation: LoadRecommendation) -> tuple[Any, ...]:
        latency = _finite_positive(recommendation.candidate.metrics.get("mean_e2e_latency_ms"))
        return (
            recommendation.partial,
            (recommendation.total_gpus_needed if not recommendation.partial else math.inf),
            -recommendation.load_served_pct,
            -recommendation.capacity_per_gpu,
            latency if latency is not None else math.inf,
            _config_key(recommendation.candidate.config),
        )

    return sorted(recommendations, key=_rank_key)[:top_n]
