# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attention--FFN disaggregation contracts for the AISimulate Sweeper.

This module owns the backend-neutral part of AFD: topology validation and
enumeration, pipeline latency math, P/D companion rate matching, GPU accounting,
and capability gates.  Runtime serving and deployment generation stay behind
adapters.  The formulas and default candidate order intentionally follow the
imported legacy AIC implementation in ``Task.build_afd_parallel_lists`` and
``AFDInferenceSession``.

Generic search-domain integration is separate. Keeping these contracts
independent lets exhaustive callers and future integration code share one
fail-closed source of truth for AFD legality and accounting.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from .parallel_enum import ReplicaParallelConfig
from .replay import RunnerCapabilities

AFD_SCHEMA_VERSION = 1
AFD_PREFILL_DEGRADATION = 0.9
AFD_DECODE_DEGRADATION = 0.95
AFD_TTFT_CORRECTION_FACTOR = 1.8
AFD_DECODE_LATENCY_CORRECTION = 1.0
AFD_COMPANION_MAX_CANDIDATES = 256
_LEGACY_SOURCE = "aiconfigurator.sdk.task_v2.build_afd_parallel_lists"


class AFDPhase(str, Enum):
    """The inference phase covered by an AFD pool."""

    PREFILL = "prefill"
    DECODE = "decode"
    BOTH = "both"


class AFDPipelineModel(str, Enum):
    """How A/F compute and activation transfers overlap."""

    OPTIMISTIC = "optimistic"
    CONSERVATIVE = "conservative"
    SERIAL = "serial"


class AFDReasonCategory(str, Enum):
    """Stable, machine-readable AFD infeasibility categories."""

    INVALID_TOPOLOGY = "invalid_topology"
    GPU_BUDGET = "gpu_budget"
    EXPERT_DIVISIBILITY = "expert_divisibility"
    INCOMPATIBLE_PHASE = "incompatible_phase"
    CANDIDATE_LIMIT = "candidate_limit"
    NO_FEASIBLE_TOPOLOGY = "no_feasible_topology"
    INVALID_MEASUREMENT = "invalid_measurement"
    UNSUPPORTED_ADAPTER = "unsupported_adapter"
    NO_FEASIBLE_COMPANION = "no_feasible_companion"
    LATENCY_SLA = "latency_sla"


class AFDInfeasible(ValueError):
    """An actionable AFD failure with a stable category and provenance."""

    def __init__(
        self,
        category: AFDReasonCategory,
        detail: str,
        *,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        self.category = category
        self.detail = detail
        self.provenance = dict(provenance or {})
        super().__init__(f"{category.value}: {detail}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "detail": self.detail,
            "provenance": dict(self.provenance),
        }


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AFDInfeasible(
            AFDReasonCategory.INVALID_TOPOLOGY,
            f"{name} must be a positive integer, got {value!r}",
            provenance={"field": name, "value": value},
        )


def _positive_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        valid = False
    else:
        valid = math.isfinite(float(value)) and float(value) > 0.0
    if not valid:
        raise AFDInfeasible(
            AFDReasonCategory.INVALID_TOPOLOGY,
            f"{name} must be a positive finite number, got {value!r}",
            provenance={"field": name, "value": value},
        )


def _enum_value(enum_type, value, name: str):
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        choices = [item.value for item in enum_type]
        raise AFDInfeasible(
            AFDReasonCategory.INVALID_TOPOLOGY,
            f"{name} must be one of {choices}, got {value!r}",
            provenance={"field": name, "value": value},
        ) from exc


@dataclass(frozen=True)
class AFDTopology:
    """One concrete A/F topology.

    Phase-1 legacy semantics are retained: one F replica spans every F GPU, so
    ``ffn_tp == ffn_workers == n_f_nodes * gpus_per_node``.  A workers use TP
    ``tp_a`` and data-parallel replicas across the A pool.
    """

    n_a_nodes: int
    n_f_nodes: int
    gpus_per_node: int
    tp_a: int
    a_batch_size: int
    f_moe_ep_size: int = 1
    num_microbatches: int = 3
    pipeline_model: AFDPipelineModel | str = AFDPipelineModel.OPTIMISTIC
    phase: AFDPhase | str = AFDPhase.DECODE
    combined_with_pd: bool = True
    comm_overhead_factor: float = 1.0
    boundary_on_attn: bool = True
    is_moe: bool = False
    num_experts: int = 0

    def __post_init__(self) -> None:
        for name in (
            "n_a_nodes",
            "n_f_nodes",
            "gpus_per_node",
            "tp_a",
            "a_batch_size",
            "f_moe_ep_size",
            "num_microbatches",
        ):
            _positive_int(name, getattr(self, name))
        if isinstance(self.num_experts, bool) or not isinstance(self.num_experts, int):
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_TOPOLOGY,
                f"num_experts must be a non-negative integer, got {self.num_experts!r}",
            )
        if self.num_experts < 0:
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_TOPOLOGY,
                f"num_experts must be non-negative, got {self.num_experts}",
            )
        if type(self.combined_with_pd) is not bool or type(self.boundary_on_attn) is not bool:
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_TOPOLOGY,
                "combined_with_pd and boundary_on_attn must be booleans",
            )
        if type(self.is_moe) is not bool:
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_TOPOLOGY,
                f"is_moe must be a boolean, got {self.is_moe!r}",
            )
        _positive_finite("comm_overhead_factor", self.comm_overhead_factor)

        phase = _enum_value(AFDPhase, self.phase, "phase")
        pipeline = _enum_value(AFDPipelineModel, self.pipeline_model, "pipeline_model")
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "pipeline_model", pipeline)

        if self.gpus_per_node % self.tp_a:
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_TOPOLOGY,
                f"tp_a={self.tp_a} must divide gpus_per_node={self.gpus_per_node}",
                provenance={"tp_a": self.tp_a, "gpus_per_node": self.gpus_per_node},
            )
        if phase is AFDPhase.BOTH and self.combined_with_pd:
            raise AFDInfeasible(
                AFDReasonCategory.INCOMPATIBLE_PHASE,
                "combined_with_pd=True is incompatible with phase='both'; "
                "use a single AFD phase plus a P/D companion, or set combined_with_pd=False",
                provenance={"phase": phase.value, "combined_with_pd": True},
            )
        if not self.is_moe and self.f_moe_ep_size != 1:
            raise AFDInfeasible(
                AFDReasonCategory.EXPERT_DIVISIBILITY,
                "dense AFD topologies require f_moe_ep_size=1",
                provenance={"f_moe_ep_size": self.f_moe_ep_size},
            )
        if self.ffn_tp % self.f_moe_ep_size:
            raise AFDInfeasible(
                AFDReasonCategory.EXPERT_DIVISIBILITY,
                f"f_moe_ep_size={self.f_moe_ep_size} must divide ffn_tp={self.ffn_tp}",
                provenance={
                    "f_moe_ep_size": self.f_moe_ep_size,
                    "ffn_tp": self.ffn_tp,
                },
            )
        if self.num_experts and (self.f_moe_ep_size > self.num_experts or self.num_experts % self.f_moe_ep_size):
            raise AFDInfeasible(
                AFDReasonCategory.EXPERT_DIVISIBILITY,
                f"f_moe_ep_size={self.f_moe_ep_size} must divide num_experts={self.num_experts}",
                provenance={
                    "f_moe_ep_size": self.f_moe_ep_size,
                    "num_experts": self.num_experts,
                },
            )

    @property
    def attention_workers(self) -> int:
        return self.n_a_nodes * self.gpus_per_node // self.tp_a

    @property
    def ffn_workers(self) -> int:
        return self.n_f_nodes * self.gpus_per_node

    @property
    def ffn_tp(self) -> int:
        return self.ffn_workers

    @property
    def attention_gpus(self) -> int:
        return self.attention_workers * self.tp_a

    @property
    def ffn_gpus(self) -> int:
        return self.ffn_workers

    @property
    def total_gpus(self) -> int:
        return self.attention_gpus + self.ffn_gpus

    @property
    def total_batch_size(self) -> int:
        return self.attention_workers * self.a_batch_size

    @property
    def microbatch_size(self) -> int:
        return math.ceil(self.a_batch_size / self.num_microbatches)

    @property
    def total_microbatch_size(self) -> int:
        return self.attention_workers * self.microbatch_size

    @property
    def af_node_ratio(self) -> float:
        return self.n_a_nodes / self.n_f_nodes

    @property
    def adapter_topology(self) -> str:
        return "afd+pd" if self.combined_with_pd else "afd"

    def provenance(self) -> dict[str, Any]:
        return {
            "schema_version": AFD_SCHEMA_VERSION,
            "source": _LEGACY_SOURCE,
            "topology": {
                "n_a_nodes": self.n_a_nodes,
                "n_f_nodes": self.n_f_nodes,
                "gpus_per_node": self.gpus_per_node,
                "tp_a": self.tp_a,
                "ffn_tp": self.ffn_tp,
                "f_moe_ep_size": self.f_moe_ep_size,
                "a_batch_size": self.a_batch_size,
                "num_microbatches": self.num_microbatches,
                "pipeline_model": self.pipeline_model.value,
                "phase": self.phase.value,
                "combined_with_pd": self.combined_with_pd,
                "comm_overhead_factor": self.comm_overhead_factor,
                "boundary_on_attn": self.boundary_on_attn,
                "is_moe": self.is_moe,
                "num_experts": self.num_experts,
            },
            "gpu_accounting": {
                "attention_gpus": self.attention_gpus,
                "ffn_gpus": self.ffn_gpus,
                "afd_total_gpus": self.total_gpus,
            },
        }


@dataclass(frozen=True)
class AFDParallelConfig:
    """One Sweeper candidate for pure AFD or AFD plus a P/D companion."""

    topology: AFDTopology
    companion: ReplicaParallelConfig | None = None

    def __post_init__(self) -> None:
        if self.topology.combined_with_pd != (self.companion is not None):
            expected = "an opposite-phase P/D companion" if self.topology.combined_with_pd else "no companion"
            raise AFDInfeasible(
                AFDReasonCategory.INCOMPATIBLE_PHASE,
                f"topology combined_with_pd={self.topology.combined_with_pd} requires {expected}",
                provenance=self.topology.provenance(),
            )

    @property
    def companion_role(self) -> str | None:
        if self.companion is None:
            return None
        return "decode" if self.topology.phase is AFDPhase.PREFILL else "prefill"

    @property
    def total_gpus(self) -> int:
        companion_gpus = self.companion.total_gpus if self.companion is not None else 0
        return self.topology.total_gpus + companion_gpus

    def provenance(self) -> dict[str, Any]:
        companion_gpus = self.companion.total_gpus if self.companion is not None else 0
        return {
            "schema_version": AFD_SCHEMA_VERSION,
            "source": _LEGACY_SOURCE,
            "mode": self.topology.adapter_topology,
            "topology": self.topology.provenance(),
            "companion_role": self.companion_role,
            "companion": asdict(self.companion) if self.companion is not None else None,
            "gpu_accounting": {
                "attention_gpus": self.topology.attention_gpus,
                "ffn_gpus": self.topology.ffn_gpus,
                "companion_gpus": companion_gpus,
                "total_gpus": self.total_gpus,
            },
        }


@dataclass(frozen=True)
class AFDSearchConfig:
    """Finite pinned or generated AFD topology domain.

    ``pinned_topologies`` replaces generation when non-empty.  Otherwise each
    candidate list is searched.  An empty TP/EP list requests the legacy
    hardware-derived defaults.
    """

    total_gpus: int
    gpus_per_node: int
    is_moe: bool
    num_experts: int = 0
    pinned_topologies: tuple[AFDTopology, ...] = ()
    tp_a_candidates: tuple[int, ...] = ()
    a_batch_size_candidates: tuple[int, ...] = (128,)
    f_moe_ep_size_candidates: tuple[int | str, ...] = ()
    microbatch_candidates: tuple[int, ...] = (2, 3, 4)
    pipeline_model_candidates: tuple[AFDPipelineModel | str, ...] = (
        AFDPipelineModel.OPTIMISTIC,
        AFDPipelineModel.CONSERVATIVE,
    )
    phase: AFDPhase | str = AFDPhase.DECODE
    combined_with_pd: bool = True
    comm_overhead_factor: float = 1.0
    boundary_on_attn: bool = True
    min_gpu_budget: int | None = None
    max_af_ratio: float = 4.0
    max_candidates: int = 10_000

    def __post_init__(self) -> None:
        _positive_int("total_gpus", self.total_gpus)
        _positive_int("gpus_per_node", self.gpus_per_node)
        _positive_int("max_candidates", self.max_candidates)
        if type(self.is_moe) is not bool:
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_TOPOLOGY,
                f"is_moe must be a boolean, got {self.is_moe!r}",
            )
        if type(self.combined_with_pd) is not bool or type(self.boundary_on_attn) is not bool:
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_TOPOLOGY,
                "combined_with_pd and boundary_on_attn must be booleans",
            )
        if isinstance(self.num_experts, bool) or not isinstance(self.num_experts, int) or self.num_experts < 0:
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_TOPOLOGY,
                f"num_experts must be a non-negative integer, got {self.num_experts!r}",
            )
        if self.min_gpu_budget is not None:
            _positive_int("min_gpu_budget", self.min_gpu_budget)
            if self.min_gpu_budget > self.total_gpus:
                raise AFDInfeasible(
                    AFDReasonCategory.GPU_BUDGET,
                    f"min_gpu_budget={self.min_gpu_budget} exceeds total_gpus={self.total_gpus}",
                )
        _positive_finite("max_af_ratio", self.max_af_ratio)
        _positive_finite("comm_overhead_factor", self.comm_overhead_factor)
        phase = _enum_value(AFDPhase, self.phase, "phase")
        pipelines = tuple(
            _enum_value(AFDPipelineModel, item, "pipeline_model_candidates") for item in self.pipeline_model_candidates
        )
        if not pipelines:
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_TOPOLOGY,
                "pipeline_model_candidates must not be empty",
            )
        for name, values in (
            ("tp_a_candidates", self.tp_a_candidates),
            ("a_batch_size_candidates", self.a_batch_size_candidates),
            ("microbatch_candidates", self.microbatch_candidates),
        ):
            if name != "tp_a_candidates" and not values:
                raise AFDInfeasible(
                    AFDReasonCategory.INVALID_TOPOLOGY,
                    f"{name} must not be empty",
                )
            for value in values:
                _positive_int(name, value)
        for value in self.f_moe_ep_size_candidates:
            if type(value) is int:
                _positive_int("f_moe_ep_size_candidates", value)
            elif type(value) is not str or value not in {"n_f_nodes", "ffn_tp", "tp_f"}:
                raise AFDInfeasible(
                    AFDReasonCategory.EXPERT_DIVISIBILITY,
                    f"f_moe_ep_size_candidates accepts positive integers, 'n_f_nodes', or 'ffn_tp'; got {value!r}",
                )
        if any(not isinstance(topology, AFDTopology) for topology in self.pinned_topologies):
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_TOPOLOGY,
                "pinned_topologies must contain only AFDTopology objects",
            )
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "pipeline_model_candidates", pipelines)


@dataclass(frozen=True)
class AFDEnumeration:
    """An AFD finite domain plus search coverage provenance."""

    candidates: tuple[AFDTopology, ...]
    generated_count: int
    rejection_counts: Mapping[str, int]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rejection_counts", MappingProxyType(dict(self.rejection_counts)))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


def _valid_ep(ep: int, *, ffn_tp: int, num_experts: int) -> bool:
    return (
        ep >= 1
        and ep <= ffn_tp
        and ffn_tp % ep == 0
        and (num_experts <= 0 or (ep <= num_experts and num_experts % ep == 0))
    )


def _resolve_ep_candidates(config: AFDSearchConfig, *, n_f_nodes: int) -> tuple[int, ...]:
    if not config.is_moe:
        return (1,)
    ffn_tp = n_f_nodes * config.gpus_per_node
    raw = config.f_moe_ep_size_candidates or (1, 2, "n_f_nodes", "ffn_tp")
    resolved: set[int] = set()
    for value in raw:
        if value == "n_f_nodes":
            resolved.add(n_f_nodes)
        elif value in {"ffn_tp", "tp_f"}:
            resolved.add(ffn_tp)
        else:
            resolved.add(value)
    return tuple(ep for ep in sorted(resolved) if _valid_ep(ep, ffn_tp=ffn_tp, num_experts=config.num_experts))


def _validate_pinned(config: AFDSearchConfig) -> tuple[AFDTopology, ...]:
    candidates: list[AFDTopology] = []
    seen: set[AFDTopology] = set()
    for topology in config.pinned_topologies:
        if topology.gpus_per_node != config.gpus_per_node:
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_TOPOLOGY,
                "pinned topology gpus_per_node does not match its AFD search hardware",
                provenance={
                    "topology_gpus_per_node": topology.gpus_per_node,
                    "search_gpus_per_node": config.gpus_per_node,
                },
            )
        if topology.is_moe != config.is_moe or topology.num_experts != config.num_experts:
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_TOPOLOGY,
                "pinned topology model traits do not match its AFD search contract",
                provenance={
                    "topology_is_moe": topology.is_moe,
                    "search_is_moe": config.is_moe,
                    "topology_num_experts": topology.num_experts,
                    "search_num_experts": config.num_experts,
                },
            )
        if topology.total_gpus > config.total_gpus or (
            config.min_gpu_budget is not None and topology.total_gpus < config.min_gpu_budget
        ):
            raise AFDInfeasible(
                AFDReasonCategory.GPU_BUDGET,
                f"pinned AFD topology uses {topology.total_gpus} GPUs outside "
                f"[{config.min_gpu_budget or 1}, {config.total_gpus}]",
                provenance=topology.provenance(),
            )
        if topology in seen:
            continue
        seen.add(topology)
        candidates.append(topology)
    return tuple(candidates)


def enumerate_afd_topologies(config: AFDSearchConfig) -> AFDEnumeration:
    """Enumerate pinned or searched AFD shapes in legacy canonical order."""

    rejections = {
        AFDReasonCategory.GPU_BUDGET.value: 0,
        "af_ratio": 0,
        AFDReasonCategory.EXPERT_DIVISIBILITY.value: 0,
        "duplicate_pipeline_regime": 0,
    }
    if config.pinned_topologies:
        pinned = _validate_pinned(config)
        if not pinned:
            raise AFDInfeasible(
                AFDReasonCategory.NO_FEASIBLE_TOPOLOGY,
                "pinned_topologies did not contain a usable AFD topology",
            )
        return AFDEnumeration(
            candidates=pinned,
            generated_count=len(pinned),
            rejection_counts=rejections,
            provenance={
                "schema_version": AFD_SCHEMA_VERSION,
                "source": _LEGACY_SOURCE,
                "domain": "pinned",
                "complete": True,
            },
        )

    total_nodes = config.total_gpus // config.gpus_per_node
    if total_nodes < 2:
        raise AFDInfeasible(
            AFDReasonCategory.GPU_BUDGET,
            "node-granular AFD needs one full A node and one full F node; "
            f"need at least {2 * config.gpus_per_node} GPUs, got {config.total_gpus}",
            provenance={
                "total_gpus": config.total_gpus,
                "gpus_per_node": config.gpus_per_node,
            },
        )
    if config.tp_a_candidates:
        tp_candidates = tuple(sorted({value for value in config.tp_a_candidates if config.gpus_per_node % value == 0}))
    else:
        tp_candidates = tuple(
            sorted({value for value in (1, 2, 4, config.gpus_per_node) if config.gpus_per_node % value == 0})
        )
    if not tp_candidates:
        raise AFDInfeasible(
            AFDReasonCategory.INVALID_TOPOLOGY,
            "no tp_a candidate is a positive divisor of gpus_per_node",
            provenance={
                "tp_a_candidates": list(config.tp_a_candidates),
                "gpus_per_node": config.gpus_per_node,
            },
        )

    candidates: list[AFDTopology] = []
    for n_a_nodes in range(1, total_nodes):
        for n_f_nodes in range(1, total_nodes - n_a_nodes + 1):
            used_gpus = (n_a_nodes + n_f_nodes) * config.gpus_per_node
            if config.min_gpu_budget is not None and used_gpus < config.min_gpu_budget:
                rejections[AFDReasonCategory.GPU_BUDGET.value] += 1
                continue
            if n_a_nodes / n_f_nodes > config.max_af_ratio:
                rejections["af_ratio"] += 1
                continue
            ep_candidates = _resolve_ep_candidates(config, n_f_nodes=n_f_nodes)
            if not ep_candidates:
                rejections[AFDReasonCategory.EXPERT_DIVISIBILITY.value] += 1
                continue
            for tp_a in tp_candidates:
                for a_batch_size in config.a_batch_size_candidates:
                    for ep in ep_candidates:
                        for num_microbatches in config.microbatch_candidates:
                            for pipeline in config.pipeline_model_candidates:
                                if pipeline is AFDPipelineModel.OPTIMISTIC and num_microbatches < 3:
                                    rejections["duplicate_pipeline_regime"] += 1
                                    continue
                                candidates.append(
                                    AFDTopology(
                                        n_a_nodes=n_a_nodes,
                                        n_f_nodes=n_f_nodes,
                                        gpus_per_node=config.gpus_per_node,
                                        tp_a=tp_a,
                                        a_batch_size=a_batch_size,
                                        f_moe_ep_size=ep,
                                        num_microbatches=num_microbatches,
                                        pipeline_model=pipeline,
                                        phase=config.phase,
                                        combined_with_pd=config.combined_with_pd,
                                        comm_overhead_factor=config.comm_overhead_factor,
                                        boundary_on_attn=config.boundary_on_attn,
                                        is_moe=config.is_moe,
                                        num_experts=config.num_experts,
                                    )
                                )
                                if len(candidates) > config.max_candidates:
                                    raise AFDInfeasible(
                                        AFDReasonCategory.CANDIDATE_LIMIT,
                                        "AFD search produced at least "
                                        f"{len(candidates)} candidates, exceeding "
                                        f"max_candidates={config.max_candidates}; narrow "
                                        "the domain or increase the limit so the complete "
                                        "domain can be evaluated",
                                        provenance={
                                            "generated_count": len(candidates),
                                            "count_is_lower_bound": True,
                                            "max_candidates": config.max_candidates,
                                        },
                                    )

    if not candidates:
        raise AFDInfeasible(
            AFDReasonCategory.NO_FEASIBLE_TOPOLOGY,
            "AFD search produced no valid topology; check GPU budget, TP/EP divisibility, "
            "A:F ratio, microbatch, and pipeline filters",
            provenance={"rejection_counts": dict(rejections)},
        )
    generated_count = len(candidates)

    return AFDEnumeration(
        candidates=tuple(candidates),
        generated_count=generated_count,
        rejection_counts=rejections,
        provenance={
            "schema_version": AFD_SCHEMA_VERSION,
            "source": _LEGACY_SOURCE,
            "domain": "searched",
            "complete": True,
            "candidate_order": [
                "n_a_nodes",
                "n_f_nodes",
                "tp_a",
                "a_batch_size",
                "f_moe_ep_size",
                "num_microbatches",
                "pipeline_model",
            ],
        },
    )


@dataclass(frozen=True)
class AFDLayerTimes:
    """Adapter-supplied per-layer A/F compute and communication times."""

    phase: AFDPhase | str
    attention_ms: float
    ffn_ms: float
    a_to_f_ms: float
    f_to_a_ms: float
    num_layers: int
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        phase = _enum_value(AFDPhase, self.phase, "phase")
        if phase is AFDPhase.BOTH:
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_MEASUREMENT,
                "AFDLayerTimes must describe one phase, not 'both'",
            )
        object.__setattr__(self, "phase", phase)
        _positive_int("num_layers", self.num_layers)
        for name in ("attention_ms", "ffn_ms", "a_to_f_ms", "f_to_a_ms"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise AFDInfeasible(
                    AFDReasonCategory.INVALID_MEASUREMENT,
                    f"{name} must be a finite non-negative number, got {value!r}",
                    provenance={"field": name, "value": value},
                )
        if max(self.attention_ms, self.ffn_ms, self.a_to_f_ms, self.f_to_a_ms) <= 0:
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_MEASUREMENT,
                "at least one AFD layer time must be positive",
            )
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


@dataclass(frozen=True)
class AFDPhaseEvaluation:
    """One phase's legacy-equivalent AFD pipeline evaluation."""

    phase: AFDPhase
    step_latency_ms: float
    sequence_rate: float
    tokens_per_second: float
    communication_hidden: bool
    balance_ratio: float
    cycle_ms: float
    pipeline_fill_ms: float
    requested_pipeline_model: AFDPipelineModel
    effective_pipeline_model: AFDPipelineModel
    total_gpus: int
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


def evaluate_afd_phase(
    topology: AFDTopology,
    times: AFDLayerTimes,
    *,
    input_length: int,
    output_length: int,
    latency_correction: float = 1.0,
) -> AFDPhaseEvaluation:
    """Evaluate one AFD phase with the legacy K=3/K=2/serial formulas."""

    _positive_int("input_length", input_length)
    _positive_int("output_length", output_length)
    _positive_finite("latency_correction", latency_correction)
    if topology.phase is not AFDPhase.BOTH and topology.phase is not times.phase:
        raise AFDInfeasible(
            AFDReasonCategory.INCOMPATIBLE_PHASE,
            f"topology phase={topology.phase.value!r} cannot evaluate measurement phase={times.phase.value!r}",
            provenance=topology.provenance(),
        )

    t_a = float(times.attention_ms)
    t_f = float(times.ffn_ms)
    t_a2f = float(times.a_to_f_ms) * topology.comm_overhead_factor
    t_f2a = float(times.f_to_a_ms) * topology.comm_overhead_factor
    t_c = t_a2f + t_f2a
    requested = topology.pipeline_model
    effective = requested
    hidden = False
    if requested is AFDPipelineModel.SERIAL:
        cycle = t_a + t_a2f + t_f + t_f2a
    elif requested is AFDPipelineModel.CONSERVATIVE:
        cycle = max(t_a + t_a2f, t_f + t_f2a)
    else:
        min_microbatches = 2.0 + t_c / max(t_a, t_f, 1e-9)
        if topology.num_microbatches < min_microbatches:
            effective = AFDPipelineModel.CONSERVATIVE
            cycle = max(t_a + t_a2f, t_f + t_f2a)
        else:
            cycle = max(t_a, t_f, t_c)
            hidden = t_c <= max(t_a, t_f)
    fill = t_a + t_f + t_a2f + t_f2a
    step = (fill + cycle * max(topology.num_microbatches * times.num_layers - 1, 0)) * float(latency_correction)
    if step <= 0:
        raise AFDInfeasible(
            AFDReasonCategory.INVALID_MEASUREMENT,
            "AFD pipeline evaluation produced a non-positive step latency",
        )
    if times.phase is AFDPhase.DECODE:
        tokens_per_second = topology.total_batch_size / (step / 1000.0)
        sequence_rate = tokens_per_second / output_length
    else:
        sequence_rate = topology.total_batch_size / (step / 1000.0)
        tokens_per_second = sequence_rate * input_length
    return AFDPhaseEvaluation(
        phase=times.phase,
        step_latency_ms=step,
        sequence_rate=sequence_rate,
        tokens_per_second=tokens_per_second,
        communication_hidden=hidden,
        balance_ratio=min(t_a, t_f) / max(t_a, t_f, 1e-9),
        cycle_ms=cycle,
        pipeline_fill_ms=fill,
        requested_pipeline_model=requested,
        effective_pipeline_model=effective,
        total_gpus=topology.total_gpus,
        provenance={
            "schema_version": AFD_SCHEMA_VERSION,
            "formula": "AFDInferenceSession._pipeline_global_step_latency",
            "layer_times": {
                "attention_ms": t_a,
                "ffn_ms": t_f,
                "a_to_f_ms": t_a2f,
                "f_to_a_ms": t_f2a,
                "num_layers": times.num_layers,
            },
            "measurement": dict(times.provenance),
            "topology": topology.provenance(),
            "latency_correction": latency_correction,
        },
    )


@dataclass(frozen=True)
class AFDCompanionOption:
    """One static P/D worker shape eligible to rate-match an AFD pool."""

    phase: AFDPhase | str
    sequence_rate_per_worker: float
    latency_ms: float
    gpus_per_worker: int
    parallel_config: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        phase = _enum_value(AFDPhase, self.phase, "phase")
        if phase is AFDPhase.BOTH:
            raise AFDInfeasible(
                AFDReasonCategory.INCOMPATIBLE_PHASE,
                "a P/D companion option must cover prefill or decode, not both",
            )
        object.__setattr__(self, "phase", phase)
        _positive_finite("sequence_rate_per_worker", self.sequence_rate_per_worker)
        _positive_finite("latency_ms", self.latency_ms)
        _positive_int("gpus_per_worker", self.gpus_per_worker)
        object.__setattr__(self, "parallel_config", MappingProxyType(dict(self.parallel_config)))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


@dataclass(frozen=True)
class AFDDeploymentEvaluation:
    """End-to-end AFD or AFD+P/D metrics with lossless GPU accounting."""

    sequence_rate: float
    tokens_per_second: float
    tokens_per_second_per_gpu: float
    ttft_ms: float
    tpot_ms: float
    request_latency_ms: float
    total_gpus: int
    afd_gpus: int
    companion_gpus: int
    companion_workers: int
    phase_evaluations: Mapping[str, AFDPhaseEvaluation]
    companion: AFDCompanionOption | None
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase_evaluations", MappingProxyType(dict(self.phase_evaluations)))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence_rate": self.sequence_rate,
            "tokens_per_second": self.tokens_per_second,
            "tokens_per_second_per_gpu": self.tokens_per_second_per_gpu,
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "request_latency_ms": self.request_latency_ms,
            "total_gpus": self.total_gpus,
            "afd_gpus": self.afd_gpus,
            "companion_gpus": self.companion_gpus,
            "companion_workers": self.companion_workers,
            "phase_evaluations": {
                phase: {
                    "phase": evaluation.phase.value,
                    "step_latency_ms": evaluation.step_latency_ms,
                    "sequence_rate": evaluation.sequence_rate,
                    "tokens_per_second": evaluation.tokens_per_second,
                    "communication_hidden": evaluation.communication_hidden,
                    "balance_ratio": evaluation.balance_ratio,
                    "cycle_ms": evaluation.cycle_ms,
                    "pipeline_fill_ms": evaluation.pipeline_fill_ms,
                    "requested_pipeline_model": evaluation.requested_pipeline_model.value,
                    "effective_pipeline_model": evaluation.effective_pipeline_model.value,
                    "total_gpus": evaluation.total_gpus,
                    "provenance": dict(evaluation.provenance),
                }
                for phase, evaluation in self.phase_evaluations.items()
            },
            "companion": (
                None
                if self.companion is None
                else {
                    "phase": self.companion.phase.value,
                    "sequence_rate_per_worker": self.companion.sequence_rate_per_worker,
                    "latency_ms": self.companion.latency_ms,
                    "gpus_per_worker": self.companion.gpus_per_worker,
                    "parallel_config": dict(self.companion.parallel_config),
                    "provenance": dict(self.companion.provenance),
                }
            ),
            "provenance": dict(self.provenance),
        }


def evaluate_pure_afd(
    topology: AFDTopology,
    measurements: Sequence[AFDLayerTimes],
    *,
    input_length: int,
    output_length: int,
    prefill_degradation: float = AFD_PREFILL_DEGRADATION,
    decode_degradation: float = AFD_DECODE_DEGRADATION,
    ttft_correction_factor: float = AFD_TTFT_CORRECTION_FACTOR,
    decode_latency_correction: float = AFD_DECODE_LATENCY_CORRECTION,
) -> AFDDeploymentEvaluation:
    """Evaluate a pure AFD topology for its configured phase coverage."""

    if topology.combined_with_pd:
        raise AFDInfeasible(
            AFDReasonCategory.INCOMPATIBLE_PHASE,
            "evaluate_pure_afd requires combined_with_pd=False; use rate_match_afd_with_pd for a companion pool",
            provenance=topology.provenance(),
        )
    for name, factor in (
        ("prefill_degradation", prefill_degradation),
        ("decode_degradation", decode_degradation),
        ("ttft_correction_factor", ttft_correction_factor),
        ("decode_latency_correction", decode_latency_correction),
    ):
        _positive_finite(name, factor)
    by_phase = {item.phase: item for item in measurements}
    expected = (AFDPhase.PREFILL, AFDPhase.DECODE) if topology.phase is AFDPhase.BOTH else (topology.phase,)
    missing = [phase.value for phase in expected if phase not in by_phase]
    if missing:
        raise AFDInfeasible(
            AFDReasonCategory.INVALID_MEASUREMENT,
            f"missing AFD layer measurements for phase(s): {missing}",
            provenance=topology.provenance(),
        )
    evaluated: dict[str, AFDPhaseEvaluation] = {}
    effective_rates: list[float] = []
    for phase in expected:
        correction = decode_latency_correction if phase is AFDPhase.DECODE else 1.0
        result = evaluate_afd_phase(
            topology,
            by_phase[phase],
            input_length=input_length,
            output_length=output_length,
            latency_correction=correction,
        )
        evaluated[phase.value] = result
        degradation = prefill_degradation if phase is AFDPhase.PREFILL else decode_degradation
        effective_rates.append(result.sequence_rate * degradation)
    sequence_rate = min(effective_rates)
    ttft = (
        evaluated[AFDPhase.PREFILL.value].step_latency_ms * ttft_correction_factor
        if AFDPhase.PREFILL.value in evaluated
        else 0.0
    )
    tpot = evaluated[AFDPhase.DECODE.value].step_latency_ms if AFDPhase.DECODE.value in evaluated else 0.0
    request_latency = ttft + tpot * max(output_length - 1, 0)
    # A prefill-only pool has request capacity but cannot produce output tokens
    # without a decode phase. Keep its phase rate in ``phase_evaluations`` and
    # avoid presenting it as end-to-end generation throughput.
    tokens_per_second = 0.0 if topology.phase is AFDPhase.PREFILL else sequence_rate * output_length
    return AFDDeploymentEvaluation(
        sequence_rate=sequence_rate,
        tokens_per_second=tokens_per_second,
        tokens_per_second_per_gpu=tokens_per_second / topology.total_gpus,
        ttft_ms=ttft,
        tpot_ms=tpot,
        request_latency_ms=request_latency,
        total_gpus=topology.total_gpus,
        afd_gpus=topology.total_gpus,
        companion_gpus=0,
        companion_workers=0,
        phase_evaluations=evaluated,
        companion=None,
        provenance={
            "schema_version": AFD_SCHEMA_VERSION,
            "mode": "pure_afd",
            "phase": topology.phase.value,
            "rate_match": "min(prefill_seq_s, decode_seq_s)",
            "degradation": {
                "prefill": prefill_degradation,
                "decode": decode_degradation,
            },
            "corrections": {
                "ttft": ttft_correction_factor,
                "decode_latency": decode_latency_correction,
            },
            "topology": topology.provenance(),
        },
    )


def rate_match_afd_with_pd(
    topology: AFDTopology,
    afd_measurement: AFDLayerTimes,
    companion_options: Sequence[AFDCompanionOption],
    *,
    input_length: int,
    output_length: int,
    total_gpu_budget: int | None = None,
    max_companion_gpus: int | None = None,
    max_companion_workers: int | None = None,
    prefill_degradation: float = AFD_PREFILL_DEGRADATION,
    decode_degradation: float = AFD_DECODE_DEGRADATION,
    ttft_correction_factor: float = AFD_TTFT_CORRECTION_FACTOR,
    decode_latency_correction: float = AFD_DECODE_LATENCY_CORRECTION,
    max_companion_candidates: int = AFD_COMPANION_MAX_CANDIDATES,
    target_ttft_ms: float | None = None,
    target_request_latency_ms: float | None = None,
) -> AFDDeploymentEvaluation:
    """Rate-match a single-phase AFD pool with a static P/D companion.

    Every worker count from one through the count required to keep pace is
    considered, preserving the legacy prefill-bound efficiency tradeoff.  The
    winning feasible combination maximizes output tokens/s/GPU, then uses fewer
    GPUs and lower corrected TTFT.
    """

    if not topology.combined_with_pd or topology.phase is AFDPhase.BOTH:
        raise AFDInfeasible(
            AFDReasonCategory.INCOMPATIBLE_PHASE,
            "rate_match_afd_with_pd requires a single-phase topology with combined_with_pd=True",
            provenance=topology.provenance(),
        )
    if afd_measurement.phase is not topology.phase:
        raise AFDInfeasible(
            AFDReasonCategory.INCOMPATIBLE_PHASE,
            f"AFD measurement phase={afd_measurement.phase.value!r} does not match "
            f"topology phase={topology.phase.value!r}",
        )
    for name, factor in (
        ("prefill_degradation", prefill_degradation),
        ("decode_degradation", decode_degradation),
        ("ttft_correction_factor", ttft_correction_factor),
        ("decode_latency_correction", decode_latency_correction),
    ):
        _positive_finite(name, factor)
    _positive_int("max_companion_candidates", max_companion_candidates)
    generated_companion_options = len(companion_options)
    for name, value in (
        ("target_ttft_ms", target_ttft_ms),
        ("target_request_latency_ms", target_request_latency_ms),
    ):
        if value is not None:
            _positive_finite(name, value)
    for name, value in (
        ("total_gpu_budget", total_gpu_budget),
        ("max_companion_gpus", max_companion_gpus),
        ("max_companion_workers", max_companion_workers),
    ):
        if value is not None:
            _positive_int(name, value)
    if total_gpu_budget is not None and topology.total_gpus >= total_gpu_budget:
        raise AFDInfeasible(
            AFDReasonCategory.GPU_BUDGET,
            f"AFD pool already uses {topology.total_gpus}/{total_gpu_budget} GPUs, "
            "leaving no capacity for its P/D companion",
            provenance=topology.provenance(),
        )

    afd_latency_correction = decode_latency_correction if topology.phase is AFDPhase.DECODE else 1.0
    afd_eval = evaluate_afd_phase(
        topology,
        afd_measurement,
        input_length=input_length,
        output_length=output_length,
        latency_correction=afd_latency_correction,
    )
    afd_degradation = prefill_degradation if topology.phase is AFDPhase.PREFILL else decode_degradation
    afd_rate = afd_eval.sequence_rate * afd_degradation
    companion_phase = AFDPhase.DECODE if topology.phase is AFDPhase.PREFILL else AFDPhase.PREFILL
    rejection_counts = {
        "phase": 0,
        AFDReasonCategory.GPU_BUDGET.value: 0,
        AFDReasonCategory.LATENCY_SLA.value: 0,
    }
    evaluated_companion_candidates = 0
    best: tuple[tuple[float, int, int, float], AFDCompanionOption, int, float, float, float] | None = None

    for option in companion_options:
        if option.phase is not companion_phase:
            rejection_counts["phase"] += 1
            continue
        if option.phase is AFDPhase.PREFILL:
            companion_rate_per_worker = option.sequence_rate_per_worker * prefill_degradation
            companion_latency = option.latency_ms * ttft_correction_factor
        else:
            companion_rate_per_worker = option.sequence_rate_per_worker * decode_degradation
            companion_latency = option.latency_ms * decode_latency_correction
        rate_ratio = afd_rate / companion_rate_per_worker
        required_workers = None if not math.isfinite(rate_ratio) else max(1, math.ceil(rate_ratio))
        configured_worker_limit: int | None = None
        if max_companion_workers is not None:
            configured_worker_limit = max_companion_workers
        if max_companion_gpus is not None:
            gpu_worker_limit = max_companion_gpus // option.gpus_per_worker
            configured_worker_limit = (
                gpu_worker_limit if configured_worker_limit is None else min(configured_worker_limit, gpu_worker_limit)
            )
        if total_gpu_budget is not None:
            available = total_gpu_budget - topology.total_gpus
            budget_worker_limit = available // option.gpus_per_worker
            configured_worker_limit = (
                budget_worker_limit
                if configured_worker_limit is None
                else min(configured_worker_limit, budget_worker_limit)
            )
        if required_workers is None:
            if configured_worker_limit is None:
                raise AFDInfeasible(
                    AFDReasonCategory.CANDIDATE_LIMIT,
                    "AFD companion rate matching requires more workers than its finite "
                    "candidate bound can represent; set a GPU/worker bound or use a "
                    "larger per-worker rate",
                    provenance={
                        "domain": "companion",
                        "max_companion_candidates": max_companion_candidates,
                        "sequence_rate_per_worker": option.sequence_rate_per_worker,
                    },
                )
            max_workers = configured_worker_limit
        else:
            max_workers = (
                required_workers if configured_worker_limit is None else min(required_workers, configured_worker_limit)
            )
        if max_workers < 1:
            rejection_counts[AFDReasonCategory.GPU_BUDGET.value] += 1
            continue
        if evaluated_companion_candidates + max_workers > max_companion_candidates:
            raise AFDInfeasible(
                AFDReasonCategory.CANDIDATE_LIMIT,
                "AFD companion search would evaluate "
                f"{evaluated_companion_candidates + max_workers} (option, worker-count) "
                f"candidates, exceeding max_companion_candidates={max_companion_candidates}; "
                "narrow the companion domain or increase the limit",
                provenance={
                    "domain": "companion",
                    "generated_options": generated_companion_options,
                    "evaluated_candidates_before_option": evaluated_companion_candidates,
                    "option_candidates": max_workers,
                    "max_companion_candidates": max_companion_candidates,
                },
            )
        evaluated_companion_candidates += max_workers

        for workers in range(1, max_workers + 1):
            companion_gpus = workers * option.gpus_per_worker
            total_gpus = topology.total_gpus + companion_gpus
            sequence_rate = min(afd_rate, workers * companion_rate_per_worker)
            if topology.phase is AFDPhase.DECODE:
                ttft = companion_latency
                tpot = afd_eval.step_latency_ms
            else:
                ttft = afd_eval.step_latency_ms * ttft_correction_factor
                tpot = companion_latency
            request_latency = ttft + tpot * max(output_length - 1, 0)
            if target_ttft_ms is not None and ttft > target_ttft_ms:
                rejection_counts[AFDReasonCategory.LATENCY_SLA.value] += 1
                continue
            if target_request_latency_ms is not None and request_latency > target_request_latency_ms:
                rejection_counts[AFDReasonCategory.LATENCY_SLA.value] += 1
                continue
            tokens_per_second = sequence_rate * output_length
            per_gpu = tokens_per_second / total_gpus
            key = (per_gpu, -total_gpus, -companion_gpus, -ttft)
            if best is None or key > best[0]:
                best = (
                    key,
                    option,
                    workers,
                    sequence_rate,
                    ttft,
                    tpot,
                )

    if best is None:
        raise AFDInfeasible(
            AFDReasonCategory.NO_FEASIBLE_COMPANION,
            f"no {companion_phase.value} companion satisfied phase, GPU, and latency "
            "constraints; inspect rejection_counts and companion provenance",
            provenance={
                "rejection_counts": rejection_counts,
                "afd_topology": topology.provenance(),
                "companion_options": generated_companion_options,
            },
        )

    _, companion, workers, sequence_rate, ttft, tpot = best
    companion_gpus = workers * companion.gpus_per_worker
    total_gpus = topology.total_gpus + companion_gpus
    tokens_per_second = sequence_rate * output_length
    return AFDDeploymentEvaluation(
        sequence_rate=sequence_rate,
        tokens_per_second=tokens_per_second,
        tokens_per_second_per_gpu=tokens_per_second / total_gpus,
        ttft_ms=ttft,
        tpot_ms=tpot,
        request_latency_ms=ttft + tpot * max(output_length - 1, 0),
        total_gpus=total_gpus,
        afd_gpus=topology.total_gpus,
        companion_gpus=companion_gpus,
        companion_workers=workers,
        phase_evaluations={topology.phase.value: afd_eval},
        companion=companion,
        provenance={
            "schema_version": AFD_SCHEMA_VERSION,
            "mode": "afd+pd",
            "rate_match": "min(degraded_afd_seq_s, workers * degraded_companion_seq_s)",
            "degradation": {
                "prefill": prefill_degradation,
                "decode": decode_degradation,
            },
            "corrections": {
                "ttft": ttft_correction_factor,
                "decode_latency": decode_latency_correction,
            },
            "gpu_accounting": {
                "attention_gpus": topology.attention_gpus,
                "ffn_gpus": topology.ffn_gpus,
                "companion_gpus": companion_gpus,
                "total_gpus": total_gpus,
            },
            "rejection_counts": rejection_counts,
            "companion_domain": {
                "generated_options": generated_companion_options,
                "evaluated_candidates": evaluated_companion_candidates,
                "max_candidates": max_companion_candidates,
                "complete": True,
            },
            "topology": topology.provenance(),
            "companion": dict(companion.provenance),
        },
    )


def require_afd_adapter_support(
    adapter_name: str,
    supported_topologies: Collection[str],
    topology: AFDTopology,
) -> None:
    """Fail closed unless an adapter explicitly advertises this AFD mode."""

    required = topology.adapter_topology
    if required not in supported_topologies and "*" not in supported_topologies:
        raise AFDInfeasible(
            AFDReasonCategory.UNSUPPORTED_ADAPTER,
            f"{adapter_name} does not advertise topology {required!r}; supported: "
            f"{sorted(supported_topologies)}. Runtime AFD serving and generation require "
            "an explicit adapter capability.",
            provenance={"adapter": adapter_name, "required_topology": required},
        )


def require_afd_runner_support(
    capabilities: RunnerCapabilities,
    backend: str,
    topology: AFDTopology,
) -> None:
    """Fail closed unless a replay runner advertises backend/AFD support."""

    required = topology.adapter_topology
    if not capabilities.supports_backend_topology(backend, required):
        advertised = [
            f"{candidate_backend}/{candidate_topology}"
            for candidate_backend, candidate_topology in capabilities.supported_backend_topologies
        ]
        raise AFDInfeasible(
            AFDReasonCategory.UNSUPPORTED_ADAPTER,
            f"runner does not advertise backend/topology {backend!r}/{required!r}; advertised: {advertised}",
            provenance={"backend": backend, "required_topology": required},
        )


__all__ = [
    "AFD_COMPANION_MAX_CANDIDATES",
    "AFD_DECODE_DEGRADATION",
    "AFD_DECODE_LATENCY_CORRECTION",
    "AFD_PREFILL_DEGRADATION",
    "AFD_SCHEMA_VERSION",
    "AFD_TTFT_CORRECTION_FACTOR",
    "AFDCompanionOption",
    "AFDDeploymentEvaluation",
    "AFDEnumeration",
    "AFDInfeasible",
    "AFDLayerTimes",
    "AFDParallelConfig",
    "AFDPhase",
    "AFDPhaseEvaluation",
    "AFDPipelineModel",
    "AFDReasonCategory",
    "AFDSearchConfig",
    "AFDTopology",
    "enumerate_afd_topologies",
    "evaluate_afd_phase",
    "evaluate_pure_afd",
    "rate_match_afd_with_pd",
    "require_afd_adapter_support",
    "require_afd_runner_support",
]
