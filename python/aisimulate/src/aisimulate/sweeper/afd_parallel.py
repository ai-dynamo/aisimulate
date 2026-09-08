# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attention--FFN topology and enumeration contracts for the AISimulate Sweeper.

This module owns only backend-neutral A/F parallel shapes, their validation,
and finite candidate enumeration. Performance measurements, foreground
execution, replay, and deployment generation live in their owning layers.
The default candidate order follows the imported legacy AIC implementation in
``Task.build_afd_parallel_lists``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

AFD_SCHEMA_VERSION = 1
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
        contract_mismatches = {
            name: {"topology": actual, "search": expected}
            for name, actual, expected in (
                ("phase", topology.phase.value, config.phase.value),
                ("combined_with_pd", topology.combined_with_pd, config.combined_with_pd),
                (
                    "comm_overhead_factor",
                    topology.comm_overhead_factor,
                    config.comm_overhead_factor,
                ),
                ("boundary_on_attn", topology.boundary_on_attn, config.boundary_on_attn),
            )
            if actual != expected
        }
        if contract_mismatches:
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_TOPOLOGY,
                "pinned topology contradicts its AFD search phase, mode, or communication contract",
                provenance={"mismatches": contract_mismatches},
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


__all__ = [
    "AFD_SCHEMA_VERSION",
    "AFDEnumeration",
    "AFDInfeasible",
    "AFDPhase",
    "AFDPipelineModel",
    "AFDReasonCategory",
    "AFDSearchConfig",
    "AFDTopology",
    "enumerate_afd_topologies",
]
