# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified native port; original paths and license: root THIRD_PARTY_NOTICES.md.
# Source: https://github.com/LPhgh/aisimulate/tree/af56328c4ba7ffa1b8c42c8375244568c2e7bb8c

"""Attention--FFN parallel topology contracts for the AISimulate Sweeper.

This module owns only the backend-neutral A/F parallel shape, validation, GPU
accounting, and complete finite enumeration. Performance measurement, staged
evaluation, search integration, replay, and deployment generation live in their
own layers. Candidates are enumerated in deterministic node, batch, and
microbatch order.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

from aisimulate_core._native import (
    AFD_DEFAULT_ROUTING_LOCATION,
    resolve_afd_parallelism,
    validate_afd_attention_tp,
    validate_afd_model_shape,
    validate_afd_routing_location,
)

AFD_SCHEMA_VERSION = 2
_SOURCE = "aisimulate.sweeper.afd_parallel.enumerate_afd_topologies"


def validate_afd_model(model: str) -> dict[str, Any]:
    """Validate cached source metadata before preparing performance databases."""
    from aisimulate_core.sdk.models.afd import layer_layout
    from aisimulate_core.sdk.models.helpers import _get_model_info

    info = _get_model_info(model)
    required = ("layers", "num_experts", "topk", "num_shared_experts", "num_moe_layers")
    missing = [name for name in required if name not in info]
    if missing:
        raise ValueError(f"AFD source metadata for {model!r} is missing {missing}")
    validate_afd_model_shape(
        info["layers"],
        info.get("num_experts") or 0,
        info.get("topk") or 0,
        info["num_shared_experts"],
        info["num_moe_layers"],
    )
    layer_layout(info)
    return info


class AFDReasonCategory(str, Enum):
    """Stable, machine-readable AFD infeasibility categories."""

    INVALID_TOPOLOGY = "invalid_topology"
    GPU_BUDGET = "gpu_budget"
    EXPERT_DIVISIBILITY = "expert_divisibility"
    CANDIDATE_LIMIT = "candidate_limit"
    NO_FEASIBLE_TOPOLOGY = "no_feasible_topology"
    INVALID_MEASUREMENT = "invalid_measurement"
    OUT_OF_MEMORY = "out_of_memory"


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


@dataclass(frozen=True)
class AFDTopology:
    """One concrete A/F topology.

    A replicas own independent token batches. F ranks are physical GPU ranks
    sharing one expert pool. A supports attention TP; F uses pure expert parallelism.
    """

    n_a_nodes: int
    n_f_nodes: int
    gpus_per_node: int
    a_tp_size: int
    batch_size_per_a_replica: int
    f_moe_ep_size: int | None = None
    num_microbatches: int = 3
    pipeline_layout: str = "four_stage"
    comm_overhead_factor: float = 1.0
    routing_location: str = AFD_DEFAULT_ROUTING_LOCATION
    is_moe: bool = False
    num_experts: int = 0
    # Per-pool perf-DB systems (hetero support). ``None`` inherits
    # the deployment's top-level hardware. Each width is validated against
    # its own pool's selected perf-DB system spec.
    a_system: str | None = None
    f_system: str | None = None
    f_gpus_per_node: int | None = None

    def __post_init__(self) -> None:
        if self.f_gpus_per_node is None:
            object.__setattr__(self, "f_gpus_per_node", self.gpus_per_node)
        _positive_int("f_gpus_per_node", self.f_gpus_per_node)
        if self.routing_location not in ("a", "f"):
            raise AFDInfeasible(AFDReasonCategory.INVALID_TOPOLOGY, "routing_location must be a or f")
        if self.pipeline_layout not in ("four_stage", "two_stage"):
            raise AFDInfeasible(AFDReasonCategory.INVALID_TOPOLOGY, "pipeline_layout must be four_stage or two_stage")
        for name in (
            "n_a_nodes",
            "n_f_nodes",
            "gpus_per_node",
            "a_tp_size",
            "batch_size_per_a_replica",
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
        if type(self.is_moe) is not bool:
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_TOPOLOGY,
                f"is_moe must be a boolean, got {self.is_moe!r}",
            )
        _positive_finite("comm_overhead_factor", self.comm_overhead_factor)

        if self.gpus_per_node % self.a_tp_size:
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_TOPOLOGY,
                f"a_tp_size={self.a_tp_size} must divide gpus_per_node={self.gpus_per_node}",
                provenance={"a_tp_size": self.a_tp_size, "gpus_per_node": self.gpus_per_node},
            )
        if self.f_moe_ep_size is not None:
            _positive_int("f_moe_ep_size", self.f_moe_ep_size)
        resolved = json.loads(
            resolve_afd_parallelism(
                self.a_tp_size,
                self.n_a_nodes,
                self.n_f_nodes,
                self.gpus_per_node,
                self.f_moe_ep_size,
                self.num_experts if self.is_moe else 0,
                f_gpus_per_node=self.f_gpus_per_node,
            )
        )
        object.__setattr__(self, "f_moe_ep_size", resolved["f_moe_ep_size"])

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> AFDTopology:
        values = dict(value)
        derived = {
            name: values.pop(name)
            for name in ("a_replica_count", "f_rank_count", "a_gpu_count", "f_gpu_count", "f_moe_tp_size")
            if name in values
        }
        topology = cls(**values)
        for name, count in derived.items():
            if count != getattr(topology, name):
                raise ValueError(f"AFD {name} disagrees with the pool geometry")
        return topology

    @property
    def a_replica_count(self) -> int:
        return self.n_a_nodes * self.gpus_per_node // self.a_tp_size

    @property
    def f_rank_count(self) -> int:
        return self.n_f_nodes * self.f_gpus_per_node

    @property
    def a_gpu_count(self) -> int:
        return self.n_a_nodes * self.gpus_per_node

    @property
    def f_gpu_count(self) -> int:
        return self.n_f_nodes * self.f_gpus_per_node

    @property
    def f_moe_tp_size(self) -> int:
        return self.f_gpu_count // self.f_moe_ep_size

    @property
    def total_gpus(self) -> int:
        return self.a_gpu_count + self.f_gpu_count

    @property
    def total_batch_size(self) -> int:
        return self.a_replica_count * self.batch_size_per_a_replica

    @property
    def microbatch_size(self) -> int:
        return math.ceil(self.batch_size_per_a_replica / self.num_microbatches)

    @property
    def total_microbatch_size(self) -> int:
        return self.a_replica_count * self.microbatch_size

    @property
    def af_node_ratio(self) -> float:
        return self.n_a_nodes / self.n_f_nodes

    def provenance(self) -> dict[str, Any]:
        return {
            "schema_version": AFD_SCHEMA_VERSION,
            "topology": {
                "n_a_nodes": self.n_a_nodes,
                "n_f_nodes": self.n_f_nodes,
                "gpus_per_node": self.gpus_per_node,
                "f_gpus_per_node": self.f_gpus_per_node,
                "a_tp_size": self.a_tp_size,
                "a_replica_count": self.a_replica_count,
                "f_rank_count": self.f_rank_count,
                "a_gpu_count": self.a_gpu_count,
                "f_gpu_count": self.f_gpu_count,
                "f_moe_tp_size": self.f_moe_tp_size,
                "f_moe_ep_size": self.f_moe_ep_size,
                "batch_size_per_a_replica": self.batch_size_per_a_replica,
                "num_microbatches": self.num_microbatches,
                "pipeline_layout": self.pipeline_layout,
                "comm_overhead_factor": self.comm_overhead_factor,
                "routing_location": self.routing_location,
                "is_moe": self.is_moe,
                "num_experts": self.num_experts,
                "a_system": self.a_system,
                "f_system": self.f_system,
            },
            "gpu_accounting": {
                "a_gpu_count": self.a_gpu_count,
                "f_gpu_count": self.f_gpu_count,
                "afd_total_gpus": self.total_gpus,
            },
        }


@dataclass(frozen=True)
class AFDSearchConfig:
    """Finite pinned or generated AFD topology domain.

    ``pinned_topologies`` replaces generation when non-empty.  Otherwise each
    candidate list is searched.  An empty TP/EP list requests the hardware-derived defaults.
    """

    total_gpus: int
    gpus_per_node: int
    is_moe: bool
    num_experts: int = 0
    pinned_topologies: tuple[AFDTopology, ...] = ()
    a_tp_size_candidates: tuple[int, ...] = ()
    batch_size_per_a_replica_candidates: tuple[int, ...] = (128,)
    f_moe_ep_size_candidates: tuple[int | str, ...] = ()
    microbatch_candidates: tuple[int, ...] = (2, 3, 4)
    pipeline_layout: str = "four_stage"
    comm_overhead_factor: float = 1.0
    routing_location: str = AFD_DEFAULT_ROUTING_LOCATION
    min_gpu_budget: int | None = None
    max_af_ratio: float = 4.0
    max_candidates: int = 10_000
    f_gpus_per_node: int | None = None

    def __post_init__(self) -> None:
        if self.f_gpus_per_node is None:
            object.__setattr__(self, "f_gpus_per_node", self.gpus_per_node)
        _positive_int("f_gpus_per_node", self.f_gpus_per_node)
        if self.routing_location not in ("a", "f"):
            raise AFDInfeasible(AFDReasonCategory.INVALID_TOPOLOGY, "routing_location must be a or f")
        if self.pipeline_layout not in ("four_stage", "two_stage"):
            raise AFDInfeasible(AFDReasonCategory.INVALID_TOPOLOGY, "pipeline_layout must be four_stage or two_stage")
        _positive_int("total_gpus", self.total_gpus)
        _positive_int("gpus_per_node", self.gpus_per_node)
        _positive_int("max_candidates", self.max_candidates)
        if type(self.is_moe) is not bool:
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_TOPOLOGY,
                f"is_moe must be a boolean, got {self.is_moe!r}",
            )
        if isinstance(self.num_experts, bool) or not isinstance(self.num_experts, int) or self.num_experts < 0:
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_TOPOLOGY,
                f"num_experts must be a non-negative integer, got {self.num_experts!r}",
            )
        if not self.is_moe and self.num_experts:
            raise AFDInfeasible(
                AFDReasonCategory.EXPERT_DIVISIBILITY,
                "dense AFD searches require num_experts=0",
                provenance={"field": "num_experts", "value": self.num_experts},
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
        for name, values in (
            ("a_tp_size_candidates", self.a_tp_size_candidates),
            ("batch_size_per_a_replica_candidates", self.batch_size_per_a_replica_candidates),
            ("microbatch_candidates", self.microbatch_candidates),
        ):
            if name != "a_tp_size_candidates" and not values:
                raise AFDInfeasible(
                    AFDReasonCategory.INVALID_TOPOLOGY,
                    f"{name} must not be empty",
                )
            for value in values:
                _positive_int(name, value)
            if len(set(values)) != len(values):
                raise AFDInfeasible(
                    AFDReasonCategory.INVALID_TOPOLOGY,
                    f"{name} must not contain duplicates",
                    provenance={"field": name, "values": list(values)},
                )
        invalid_tp = [value for value in self.a_tp_size_candidates if self.gpus_per_node % value]
        if invalid_tp:
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_TOPOLOGY,
                "a_tp_size_candidates must divide gpus_per_node",
                provenance={
                    "gpus_per_node": self.gpus_per_node,
                    "invalid_candidates": invalid_tp,
                },
            )
        for value in self.f_moe_ep_size_candidates:
            if type(value) is int:
                _positive_int("f_moe_ep_size_candidates", value)
            elif type(value) is not str or value not in {"n_f_nodes", "f_gpu_count"}:
                raise AFDInfeasible(
                    AFDReasonCategory.EXPERT_DIVISIBILITY,
                    f"f_moe_ep_size_candidates accepts positive integers, 'n_f_nodes', or 'f_gpu_count'; got {value!r}",
                )
        if len(set(self.f_moe_ep_size_candidates)) != len(self.f_moe_ep_size_candidates):
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_TOPOLOGY,
                "f_moe_ep_size_candidates must not contain duplicates",
                provenance={
                    "field": "f_moe_ep_size_candidates",
                    "values": list(self.f_moe_ep_size_candidates),
                },
            )
        if not self.is_moe and any(value != 1 for value in self.f_moe_ep_size_candidates):
            raise AFDInfeasible(
                AFDReasonCategory.EXPERT_DIVISIBILITY,
                "dense AFD searches require f_moe_ep_size_candidates to be empty or (1,)",
                provenance={"field": "f_moe_ep_size_candidates", "value": self.f_moe_ep_size_candidates},
            )
        if any(not isinstance(topology, AFDTopology) for topology in self.pinned_topologies):
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_TOPOLOGY,
                "pinned_topologies must contain only AFDTopology objects",
            )
        if len(set(self.pinned_topologies)) != len(self.pinned_topologies):
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_TOPOLOGY,
                "pinned_topologies must not contain duplicates",
                provenance={
                    "field": "pinned_topologies",
                    "values": [topology.provenance() for topology in self.pinned_topologies],
                },
            )


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


def _valid_ep(ep: int, *, f_gpu_count: int, num_experts: int) -> bool:
    return (
        ep >= 1
        and ep <= f_gpu_count
        and f_gpu_count % ep == 0
        and (num_experts <= 0 or (ep <= num_experts and num_experts % ep == 0))
    )


def _resolve_ep_candidates(config: AFDSearchConfig, *, n_f_nodes: int) -> tuple[int, ...]:
    if not config.is_moe:
        raise NotImplementedError("Dense AFD is not implemented")
    f_gpu_count = n_f_nodes * config.f_gpus_per_node
    raw = config.f_moe_ep_size_candidates or ("f_gpu_count",)
    resolved: set[int] = set()
    for value in raw:
        if value == "n_f_nodes":
            resolved.add(n_f_nodes)
        elif value == "f_gpu_count":
            resolved.add(f_gpu_count)
        else:
            resolved.add(value)
    candidates = []
    for ep in sorted(resolved):
        if not _valid_ep(ep, f_gpu_count=f_gpu_count, num_experts=config.num_experts):
            continue
        try:
            resolve_afd_parallelism(
                1,
                1,
                n_f_nodes,
                config.gpus_per_node,
                ep,
                None,
                f_gpus_per_node=config.f_gpus_per_node,
            )
        except NotImplementedError:
            # This generated F width may require unsupported MoE TP while
            # another width can honor the same authored EP choice.
            continue
        candidates.append(ep)
    return tuple(candidates)


def _validate_pinned(config: AFDSearchConfig) -> tuple[AFDTopology, ...]:
    candidates: list[AFDTopology] = []
    for topology in config.pinned_topologies:
        if (topology.gpus_per_node, topology.f_gpus_per_node) != (config.gpus_per_node, config.f_gpus_per_node):
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_TOPOLOGY,
                "pinned topology gpus_per_node does not match its AFD search hardware",
                provenance={
                    "topology_gpus_per_node": topology.gpus_per_node,
                    "search_gpus_per_node": config.gpus_per_node,
                    "topology_f_gpus_per_node": topology.f_gpus_per_node,
                    "search_f_gpus_per_node": config.f_gpus_per_node,
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
                (
                    "comm_overhead_factor",
                    topology.comm_overhead_factor,
                    config.comm_overhead_factor,
                ),
                ("routing_location", topology.routing_location, config.routing_location),
                ("pipeline_layout", topology.pipeline_layout, config.pipeline_layout),
            )
            if actual != expected
        }
        if contract_mismatches:
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_TOPOLOGY,
                "pinned topology contradicts fixed AFD search settings: " + ", ".join(sorted(contract_mismatches)),
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
        candidates.append(topology)
    return tuple(candidates)


def enumerate_afd_topologies(config: AFDSearchConfig) -> AFDEnumeration:
    """Enumerate pinned or searched AFD shapes in deterministic order."""

    validate_afd_routing_location(config.routing_location)
    for a_tp_size in config.a_tp_size_candidates:
        validate_afd_attention_tp(a_tp_size)
    if not config.is_moe:
        raise NotImplementedError("Dense AFD is not implemented; only pure routed MoE is supported")
    for topology in config.pinned_topologies:
        validate_afd_routing_location(topology.routing_location)
    rejections = {
        AFDReasonCategory.GPU_BUDGET.value: 0,
        "af_ratio": 0,
        AFDReasonCategory.EXPERT_DIVISIBILITY.value: 0,
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
                "source": "AFDSearchConfig.pinned_topologies",
                "domain": "pinned",
                "complete": True,
            },
        )

    max_a_nodes = (config.total_gpus - config.f_gpus_per_node) // config.gpus_per_node
    if max_a_nodes < 1:
        raise AFDInfeasible(
            AFDReasonCategory.GPU_BUDGET,
            "node-granular AFD needs one full A node and one full F node; "
            f"need at least {config.gpus_per_node + config.f_gpus_per_node} GPUs, got {config.total_gpus}",
            provenance={
                "total_gpus": config.total_gpus,
                "gpus_per_node": config.gpus_per_node,
            },
        )
    if config.a_tp_size_candidates:
        tp_candidates = tuple(sorted(config.a_tp_size_candidates))
    else:
        tp_candidates = (1,)
    if not tp_candidates:
        raise AFDInfeasible(
            AFDReasonCategory.INVALID_TOPOLOGY,
            "no a_tp_size candidate is a positive divisor of gpus_per_node",
            provenance={
                "a_tp_size_candidates": list(config.a_tp_size_candidates),
                "gpus_per_node": config.gpus_per_node,
            },
        )

    candidates: list[AFDTopology] = []
    usable_gpus = 0
    for n_a_nodes in range(1, max_a_nodes + 1):
        max_f_nodes = (config.total_gpus - n_a_nodes * config.gpus_per_node) // config.f_gpus_per_node
        for n_f_nodes in range(1, max_f_nodes + 1):
            used_gpus = n_a_nodes * config.gpus_per_node + n_f_nodes * config.f_gpus_per_node
            usable_gpus = max(usable_gpus, used_gpus)
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
            for a_tp_size in tp_candidates:
                for batch_size_per_a_replica in config.batch_size_per_a_replica_candidates:
                    for ep in ep_candidates:
                        for num_microbatches in config.microbatch_candidates:
                            candidates.append(
                                AFDTopology(
                                    n_a_nodes=n_a_nodes,
                                    n_f_nodes=n_f_nodes,
                                    gpus_per_node=config.gpus_per_node,
                                    f_gpus_per_node=config.f_gpus_per_node,
                                    a_tp_size=a_tp_size,
                                    batch_size_per_a_replica=batch_size_per_a_replica,
                                    f_moe_ep_size=ep,
                                    num_microbatches=num_microbatches,
                                    pipeline_layout=config.pipeline_layout,
                                    comm_overhead_factor=config.comm_overhead_factor,
                                    routing_location=config.routing_location,
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
            "A:F ratio, and microbatch filters",
            provenance={"rejection_counts": dict(rejections)},
        )
    generated_count = len(candidates)

    return AFDEnumeration(
        candidates=tuple(candidates),
        generated_count=generated_count,
        rejection_counts=rejections,
        provenance={
            "schema_version": AFD_SCHEMA_VERSION,
            "source": _SOURCE,
            "domain": "searched",
            "complete": True,
            "gpu_budget": {
                "granularity": "node",
                "total_gpus": config.total_gpus,
                "gpus_per_node": config.gpus_per_node,
                "f_gpus_per_node": config.f_gpus_per_node,
                "usable_gpus": usable_gpus,
                "remainder_gpus": config.total_gpus - usable_gpus,
            },
            "candidate_order": [
                "n_a_nodes",
                "n_f_nodes",
                "a_tp_size",
                "batch_size_per_a_replica",
                "f_moe_ep_size",
                "num_microbatches",
            ],
        },
    )


__all__ = [
    "AFD_SCHEMA_VERSION",
    "AFDEnumeration",
    "AFDInfeasible",
    "AFDReasonCategory",
    "AFDSearchConfig",
    "AFDTopology",
    "enumerate_afd_topologies",
]
