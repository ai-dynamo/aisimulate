# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental, backend-neutral deployment simulation sweeper.

The package root eagerly exposes only schemas and ABI contracts. Search and
AIConfigurator-backed helpers are loaded lazily so an optional adapter can import
``aisimulate.sweeper.provider`` without importing a simulation backend.
"""

from __future__ import annotations

import importlib
from typing import Any

from .config import (
    AdapterSearchConfig,
    Candidate,
    OptimizationGoal,
    OptimizationTarget,
    SearchSpace,
    SLATarget,
    SmartSearchConfig,
    SweepConfig,
    Workload,
)
from .discovery import (
    SWEEP_CONFIG_PROVIDER_ENTRY_POINT_GROUP,
    ProviderResolutionError,
    resolve_providers,
)
from .provider import (
    API_VERSION,
    SEARCH_SPACE_FRAGMENT_API_VERSION,
    AdapterReplaySpec,
    AdapterSearchPlan,
    CandidateContext,
    ConditionalSearchSpace,
    InfeasibleCandidate,
    RuntimeHookSpec,
    SearchSpaceFragment,
    SweepConfigProvider,
    SweepContext,
)
from .replay import (
    REPLAY_SPEC_API_VERSION,
    BackendDeploymentSpec,
    HookCapability,
    ReplayOutputRequirements,
    ReplayReport,
    ReplaySpec,
    Runner,
    RunnerCapabilities,
    RunnerFactory,
    canonical_json,
)
from .result import (
    RESULT_SCHEMA_VERSION,
    CandidateProvenance,
    CandidateRecord,
    CandidateRetention,
    CandidateStatus,
    OperationProvenance,
    ReasonCategory,
    ResultViews,
    SearchStrategy,
    SweepCounts,
    SweepResult,
    SweepRunProvenance,
)

_LAZY_EXPORTS = {
    "AFD_SCHEMA_VERSION": (".afd_parallel", "AFD_SCHEMA_VERSION"),
    "AFDEnumeration": (".afd_parallel", "AFDEnumeration"),
    "AFDInfeasible": (".afd_parallel", "AFDInfeasible"),
    "AFDPhase": (".afd_parallel", "AFDPhase"),
    "AFDPipelineModel": (".afd_parallel", "AFDPipelineModel"),
    "AFDReasonCategory": (".afd_parallel", "AFDReasonCategory"),
    "AFDSearchConfig": (".afd_parallel", "AFDSearchConfig"),
    "AFDTopology": (".afd_parallel", "AFDTopology"),
    "enumerate_afd_topologies": (
        ".afd_parallel",
        "enumerate_afd_topologies",
    ),
    "build_backend_deployment": (".deploy", "build_backend_deployment"),
    "NoPerfDatabase": (".kv_estimate", "NoPerfDatabase"),
    "estimate_kv_tokens": (".kv_estimate", "estimate_kv_tokens"),
    "feasible_shape_tokens": (".kv_estimate", "feasible_shape_tokens"),
    "ModelHardware": (".model_hw", "ModelHardware"),
    "NoViableParallelConfig": (".model_hw", "NoViableParallelConfig"),
    "parallel_configs_for": (".model_hw", "parallel_configs_for"),
    "resolve_model_hardware": (".model_hw", "resolve_model_hardware"),
    "DisaggParallelConfig": (".parallel_enum", "DisaggParallelConfig"),
    "ParallelShape": (".parallel_enum", "ParallelShape"),
    "ReplicaParallelConfig": (".parallel_enum", "ReplicaParallelConfig"),
    "enumerate_disagg_configs": (".parallel_enum", "enumerate_disagg_configs"),
    "enumerate_parallel_configs": (".parallel_enum", "enumerate_parallel_configs"),
    "enumerate_worker_shapes": (".parallel_enum", "enumerate_worker_shapes"),
    "unroll_sample": (".sample", "unroll_sample"),
    "BranchSampler": (".sampler", "BranchSampler"),
    "Suggestion": (".sampler", "Suggestion"),
    "make_branch_sampler": (".sampler", "make_branch_sampler"),
    "objective_value": (".score", "objective_value"),
    "aggregate_sla_violations": (".score", "aggregate_sla_violations"),
    "analyze_candidates": (".score", "analyze_candidates"),
    "meets_aggregate_sla": (".score", "meets_aggregate_sla"),
    "score_report": (".score", "score_report"),
    "is_feasible": (".score", "is_feasible"),
    "make_candidate": (".score", "make_candidate"),
    "rank": (".score", "rank"),
    "Sweeper": (".search", "Sweeper"),
    "BranchSpace": (".search_space", "BranchSpace"),
    "enumerate_branches": (".search_space", "enumerate_branches"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(importlib.import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


__all__ = [
    "AFD_SCHEMA_VERSION",
    "API_VERSION",
    "REPLAY_SPEC_API_VERSION",
    "RESULT_SCHEMA_VERSION",
    "SEARCH_SPACE_FRAGMENT_API_VERSION",
    "SWEEP_CONFIG_PROVIDER_ENTRY_POINT_GROUP",
    "AFDEnumeration",
    "AFDInfeasible",
    "AFDPhase",
    "AFDPipelineModel",
    "AFDReasonCategory",
    "AFDSearchConfig",
    "AFDTopology",
    "AdapterReplaySpec",
    "AdapterSearchConfig",
    "AdapterSearchPlan",
    "BackendDeploymentSpec",
    "BranchSampler",
    "BranchSpace",
    "Candidate",
    "CandidateContext",
    "CandidateProvenance",
    "CandidateRecord",
    "CandidateRetention",
    "CandidateStatus",
    "ConditionalSearchSpace",
    "DisaggParallelConfig",
    "HookCapability",
    "InfeasibleCandidate",
    "ModelHardware",
    "NoPerfDatabase",
    "NoViableParallelConfig",
    "OperationProvenance",
    "OptimizationGoal",
    "OptimizationTarget",
    "ParallelShape",
    "ProviderResolutionError",
    "ReasonCategory",
    "ReplayOutputRequirements",
    "ReplayReport",
    "ReplaySpec",
    "ReplicaParallelConfig",
    "ResultViews",
    "Runner",
    "RunnerCapabilities",
    "RunnerFactory",
    "RuntimeHookSpec",
    "SLATarget",
    "SearchSpace",
    "SearchSpaceFragment",
    "SearchStrategy",
    "SmartSearchConfig",
    "Suggestion",
    "SweepConfig",
    "SweepConfigProvider",
    "SweepContext",
    "SweepCounts",
    "SweepResult",
    "SweepRunProvenance",
    "Sweeper",
    "Workload",
    "aggregate_sla_violations",
    "analyze_candidates",
    "build_backend_deployment",
    "canonical_json",
    "enumerate_afd_topologies",
    "enumerate_branches",
    "enumerate_disagg_configs",
    "enumerate_parallel_configs",
    "enumerate_worker_shapes",
    "estimate_kv_tokens",
    "feasible_shape_tokens",
    "is_feasible",
    "make_branch_sampler",
    "make_candidate",
    "meets_aggregate_sla",
    "objective_value",
    "parallel_configs_for",
    "rank",
    "resolve_model_hardware",
    "resolve_providers",
    "score_report",
    "unroll_sample",
]
