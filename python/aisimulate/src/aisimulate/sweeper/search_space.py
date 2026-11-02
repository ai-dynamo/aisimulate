# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified native port; original paths and license: root THIRD_PARTY_NOTICES.md.
# Source: https://github.com/LPhgh/aisimulate/tree/af56328c4ba7ffa1b8c42c8375244568c2e7bb8c

"""Build the per-branch candidate space the sampler searches over.

A *branch* is one **deployment_mode** (agg / disagg) — one Vizier study
each, since the modes have structurally different parallel configs. ``backend`` is NOT a
branch: it is a searched categorical knob within the study. For each mode we take the
**union** of every configured backend's KV-feasible parallel configs
(:func:`aisimulate.sweeper.model_hw.parallel_configs_for`) as the valid projection pool, recording per
config which backends support it. The sampler projects structured latent features onto this
pool. Backends with no perf DB, no viable config, or no support from the injected replay
runner are dropped from the backend knob.
"""

from __future__ import annotations

import warnings
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field, replace
from heapq import merge
from typing import Any

from aisimulate_core._native import validate_afd_attention_tp, validate_afd_routing_location

from .afd_parallel import (
    AFDInfeasible,
    AFDReasonCategory,
    AFDSearchConfig,
    AFDTopology,
    enumerate_afd_topologies,
    validate_afd_model,
)
from .config import ENGINE_MODEL_CONTROL_FIELDS, SmartSearchConfig
from .kv_estimate import NoPerfDatabase, resolve_backend_version
from .model_hw import (
    NoViableParallelConfig,
    _profile_parallel_configs,
    parallel_configs_for,
    resolve_model_hardware,
)
from .parallel_enum import DisaggParallelConfig, ParallelShape, ReplicaParallelConfig
from .replay import RunnerCapabilities


class RunnerIncompatibleError(NoViableParallelConfig):
    """No configured backend/topology pair is supported by the Replay runner."""


_ParallelConfig = ReplicaParallelConfig | DisaggParallelConfig | AFDTopology

_AGG_ENGINE = ("agg_max_num_batched_tokens", "agg_max_num_seqs")
_PREFILL_ENGINE = (
    "prefill_max_num_batched_tokens",
    "prefill_max_num_seqs",
)
_DECODE_ENGINE = (
    "decode_max_num_batched_tokens",
    "decode_max_num_seqs",
)
_DISAGG_ENGINE = _PREFILL_ENGINE + _DECODE_ENGINE
_ROLE_OPTIONAL_ENGINE = ("block_size", "gpu_memory_utilization")


@dataclass(frozen=True)
class ConditionalDimensionSpace:
    """Namespaced selector and its conditionally active child dimensions."""

    selector: str
    values: tuple[Any, ...]
    knob_choices: dict[str, list[Any]] = field(default_factory=dict)
    float_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    log_float_ranges: frozenset[str] = frozenset()
    log_discrete_choices: frozenset[str] = frozenset()


@dataclass(frozen=True)
class BranchSpace:
    """One ``deployment_mode`` branch of the search (backend is a searched knob)."""

    deployment_mode: str
    # Union of every searched backend's KV-feasible parallel configs.
    parallel_configs: tuple[_ParallelConfig, ...]
    # parallel config -> the backends for which it is legal+KV-feasible. Projection
    # hard-filters on this map; the search loop keeps a defensive gate.
    supported_backends: dict[_ParallelConfig, frozenset[str]]
    # Searchable atomic knob -> its configured choice list (incl. "backend").
    knob_choices: dict[str, list[Any]]
    # Pinned branch budget; optional only for lightweight unit-test fixtures.
    gpu_budget: int | None = None
    # Continuous workload dimensions. Currently only Pareto ``kv_load_ratio`` uses
    # this; list-valued component knobs remain discrete choices above.
    float_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    integer_ranges: dict[str, tuple[int, int]] = field(default_factory=dict)
    log_float_ranges: frozenset[str] = frozenset()
    log_integer_ranges: frozenset[str] = frozenset()
    log_discrete_choices: frozenset[str] = frozenset()
    flat_parallel_choices: bool = False
    parallel_independent_choices: dict[str, tuple[int, ...]] = field(default_factory=dict)
    parallel_independent_log_ranges: dict[str, tuple[int, int]] = field(default_factory=dict)
    parallel_custom_choices: dict[str, tuple[ReplicaParallelConfig, ...]] = field(default_factory=dict)
    conditional_dimensions: tuple[ConditionalDimensionSpace, ...] = ()
    domain_provenance: dict[str, Any] = field(default_factory=dict)
    prefer_smallest: bool = False


def _parallel_leaf_values(config: _ParallelConfig) -> dict[str, int]:
    def role_values(prefix: str, role: ReplicaParallelConfig) -> dict[str, int]:
        return {
            f"{prefix}replicas": role.replicas,
            f"{prefix}tp": role.shape.tp,
            f"{prefix}pp": role.shape.pp,
            f"{prefix}attention_dp": role.shape.dp,
            f"{prefix}moe_tp": role.shape.moe_tp,
            f"{prefix}moe_ep": role.shape.moe_ep,
        }

    if isinstance(config, ReplicaParallelConfig):
        return role_values("", config)
    if isinstance(config, AFDTopology):
        raise TypeError("AFD branches use one finite flat topology domain")
    return {
        **role_values("prefill_", config.prefill),
        **role_values("decode_", config.decode),
    }


def matches_parallel_domains(config: _ParallelConfig, choices, log_ranges) -> bool:
    """Filter the legal joint pool by user-owned independent leaf domains."""
    if not choices and not log_ranges:
        return True
    values = _parallel_leaf_values(config)
    return all(
        log_ranges[name][0] <= values[name] <= log_ranges[name][1]
        if name in log_ranges
        else choices.get(name) is None or values[name] in choices[name]
        for name in choices.keys() | log_ranges.keys()
    )


def _parallel_role(config: _ParallelConfig, role: str) -> ReplicaParallelConfig:
    if isinstance(config, ReplicaParallelConfig):
        if role != "agg":
            raise ValueError(f"aggregated parallel config has no {role!r} role")
        return config
    if isinstance(config, AFDTopology):
        raise ValueError("AFD topology has no ordinary parallel role")
    return config.prefill if role == "prefill" else config.decode


def _engine_knobs(deployment_mode: str) -> tuple[str, ...]:
    if deployment_mode == "agg":
        return _AGG_ENGINE
    if deployment_mode == "disagg":
        return _DISAGG_ENGINE
    return ()


def _shape_from_dict(d: dict[str, Any]) -> ParallelShape:
    """A per-worker :class:`ParallelShape` from a pinned shape dict. Omitted dims
    default to 1 (so dense models can write just ``{tp: N}``); ``pp`` defaults to 1."""
    if "tp" not in d:
        raise ValueError(f"a parallel_configs shape needs a 'tp' field, got {d}")
    return ParallelShape(
        tp=int(d["tp"]),
        dp=int(d.get("attention_dp", 1)),
        moe_tp=int(d.get("moe_tp", 1)),
        moe_ep=int(d.get("moe_ep", 1)),
        pp=int(d.get("pp", 1)),
    )


def _replica_from_dict(d: dict[str, Any]) -> ReplicaParallelConfig:
    return ReplicaParallelConfig(shape=_shape_from_dict(d), replicas=int(d.get("replicas", 1)))


def _parse_parallel_entry(entry: dict[str, Any], deployment_mode: str):
    """Parse one pinned ``parallel_configs`` entry into the config object: a flat
    shape dict for agg, or a ``{prefill, decode}`` pair for disagg."""
    if deployment_mode == "agg":
        return _replica_from_dict(entry)
    return DisaggParallelConfig(
        prefill=_replica_from_dict(entry["prefill"]),
        decode=_replica_from_dict(entry["decode"]),
    )


def _fpm_parallel_configs(search_space, deployment_mode: str):
    """Apply declared topology domains before grouped compatibility or resource checks."""
    from aisimulate.fpm_profile import load_fpm_profile

    profile = load_fpm_profile(search_space.fpm_profile)
    configs = _profile_parallel_configs(
        profile,
        hardware_sku=search_space.hardware_sku_for("agg" if deployment_mode == "agg" else "prefill"),
        decode_hardware_sku=(search_space.hardware_sku_for("decode") if deployment_mode == "disagg" else None),
        backend="vllm",
        backend_version=search_space.requested_backend_version("vllm"),
        gpu_budget=search_space.gpu_budget,
        min_gpu_budget=search_space.min_gpu_budget,
        deployment_mode=deployment_mode,
    )
    raw_pinned = search_space.parallel_configs_by_mode.get(deployment_mode, search_space.parallel_configs)
    if raw_pinned:
        pinned = {_parse_parallel_entry(entry, deployment_mode) for entry in raw_pinned}
        configs = [config for config in configs if config in pinned]
    for role, entries in search_space.parallel_custom_configs_by_mode.get(deployment_mode, {}).items():
        choices = {_replica_from_dict(entry) for entry in entries}
        configs = [config for config in configs if _parallel_role(config, role) in choices]
    domains = search_space.parallel_independent_by_mode.get(deployment_mode, {})
    bounds = search_space.parallel_independent_log_ranges_by_mode.get(deployment_mode, {})
    return [
        config
        for config in configs
        if all(
            bounds[name][0] <= value <= bounds[name][1]
            if name in bounds
            else domains.get(name) is None or value in domains[name]
            for name, value in _parallel_leaf_values(config).items()
        )
    ]


def branch_knob_choices(search_space, deployment_mode: str) -> dict[str, list[Any]]:
    """Backend-owned atomic knobs for one deployment branch."""
    names = _engine_knobs(deployment_mode)
    names = tuple(
        name
        for name in names
        if ("aggregated" if name.startswith("agg_") else name.split("_", 1)[0]) not in search_space.afd_roles
    )
    choices = {name: list(getattr(search_space, name)) for name in names}
    if deployment_mode == "agg":
        roles = ("agg",)
    elif deployment_mode == "disagg":
        roles = ("prefill", "decode")
    else:
        roles = ()
    for role in roles:
        if ("aggregated" if role == "agg" else role) in search_space.afd_roles:
            continue
        for suffix in _ROLE_OPTIONAL_ENGINE:
            name = f"{role}_{suffix}"
            value = getattr(search_space, name)
            if isinstance(value, list):
                choices[name] = list(value)
    return choices


def _runner_supports_parallel_config(
    capabilities: RunnerCapabilities | None,
    deployment_mode: str,
    config: _ParallelConfig,
) -> bool:
    """Apply runner topology limits before a config enters the sampler domain."""

    if capabilities is None:
        return True
    if deployment_mode != "disagg":
        return True
    if not isinstance(config, DisaggParallelConfig):
        return False
    return capabilities.supports_attention_dp(
        deployment_mode,
        config.prefill.shape.dp if isinstance(config.prefill, ReplicaParallelConfig) else 1,
        config.decode.shape.dp if isinstance(config.decode, ReplicaParallelConfig) else 1,
    )


def _engine_memory_kwargs(search_space):
    model_controls = {
        name: getattr(search_space, name)
        for name in ENGINE_MODEL_CONTROL_FIELDS
        if getattr(search_space, name) not in (None, False)
    }
    return {
        **({"model_controls": model_controls} if model_controls else {}),
        **({"nextn": search_space.aic_nextn} if search_space.aic_nextn else {}),
    }


def _estimator_root_kwargs(search_space, role):
    paths = search_space.systems_paths_for(role)
    from .forward_pass_estimator import resolve_systems_paths

    return {"systems_paths": list(resolve_systems_paths(paths))}


def _role_runtime(search_space, backend: str, role: str) -> tuple[int, int, float, int | None]:
    token_name = f"{role}_max_num_batched_tokens"
    sequence_name = f"{role}_max_num_seqs"
    tokens = getattr(search_space, token_name)
    sequences = getattr(search_space, sequence_name)
    memory = getattr(search_space, f"{role}_gpu_memory_utilization")
    block_size = getattr(search_space, f"{role}_block_size")
    blocks = getattr(search_space, f"{role}_num_gpu_blocks")
    memory_values = memory if isinstance(memory, list) else [memory]
    concrete_memory = [float(value) for value in memory_values if value is not None]
    default_memory = 0.88 if backend == "sglang" else 0.9
    block_values = block_size if isinstance(block_size, list) else [block_size]
    concrete_blocks = [int(value) for value in block_values if value is not None]
    fixed_tokens = int(blocks) * min(concrete_blocks) if blocks is not None and concrete_blocks else None
    return (
        int(search_space.engine_integer_log_ranges.get(token_name, [0, max(tokens)])[1]),
        int(search_space.engine_integer_log_ranges.get(sequence_name, [0, max(sequences)])[1]),
        min(concrete_memory) if concrete_memory else default_memory,
        fixed_tokens,
    )


def _runtime_by_role(search_space, backend: str, mode: str) -> dict[str, tuple[int, int, float, int | None]]:
    roles = ("agg",) if mode == "agg" else ("prefill", "decode")
    return {role: _role_runtime(search_space, backend, role) for role in roles}


def _context_by_role(search_space, mode: str, max_seq_len: int | None) -> dict[str, int] | None:
    if mode == "agg":
        return None
    result = {}
    for role in ("prefill", "decode"):
        value = getattr(search_space, f"{role}_context_length")
        if value is not None:
            result[role] = value
        elif max_seq_len is not None:
            result[role] = max_seq_len
    return result or None


def _heterogeneous_disagg_configs(
    search_space,
    *,
    backend: str,
    max_seq_len: int | None,
    candidate_configs: list[DisaggParallelConfig] | None = None,
) -> list[DisaggParallelConfig]:
    """Enumerate each P/D role against its own hardware, then apply the shared budget."""
    role_hardware = {role: search_space.hardware_sku_for(role) for role in ("prefill", "decode")}
    backend_version = search_space.requested_backend_version(backend)
    if backend_version is None:
        role_versions = {
            role: resolve_backend_version(hardware, backend, **_estimator_root_kwargs(search_space, role))
            for role, hardware in role_hardware.items()
        }
        if len(set(role_versions.values())) != 1:
            raise NoPerfDatabase(
                "heterogeneous P/D hardware requires one common backend_version; "
                f"latest versions for backend={backend!r} are {role_versions}. "
                "Set search_space.backend_version to a version supported by both SKUs."
            )
        backend_version = role_versions["prefill"]

    per_role: dict[str, list[ReplicaParallelConfig]] = {}
    for role, hardware in role_hardware.items():
        try:
            configs = parallel_configs_for(
                search_space.model_name,
                hardware,
                gpu_budget=search_space.gpu_budget,
                deployment_mode="agg",
                worker_type=role,
                backend=backend,
                backend_version=backend_version,
                min_gpu_budget=None,
                max_seq_len=max_seq_len,
                role_runtime={"agg": _role_runtime(search_space, backend, role)},
                role_max_seq_len=(
                    {"agg": getattr(search_space, f"{role}_context_length") or max_seq_len}
                    if getattr(search_space, f"{role}_context_length") is not None or max_seq_len is not None
                    else None
                ),
                **_engine_memory_kwargs(search_space),
                **_estimator_root_kwargs(search_space, role),
                **({"fpm_profile": search_space.fpm_profile} if search_space.fpm_profile is not None else {}),
                **(
                    {"candidate_configs": list(dict.fromkeys(getattr(config, role) for config in candidate_configs))}
                    if candidate_configs is not None
                    else {}
                ),
            )
        except (NoPerfDatabase, NoViableParallelConfig) as exc:
            raise type(exc)(f"{role} hardware_sku={hardware!r}: {exc}") from exc
        per_role[role] = configs

    configs = [
        DisaggParallelConfig(prefill=prefill, decode=decode)
        for prefill in per_role["prefill"]
        for decode in per_role["decode"]
        if prefill.total_gpus + decode.total_gpus <= search_space.gpu_budget
        and (
            search_space.min_gpu_budget is None or prefill.total_gpus + decode.total_gpus >= search_space.min_gpu_budget
        )
    ]
    if not configs:
        raise NoViableParallelConfig(
            f"heterogeneous P/D roles have no legal pair within gpu_budget={search_space.gpu_budget}"
        )
    return configs


def _afd_roles_branch(config, mode, max_seq_len, runner_capabilities):
    """Compose independently enumerated roles without materializing a Cartesian product."""
    from ..recommend import _choices

    ss = config.search_space
    backend = "sglang"
    if runner_capabilities is not None and not runner_capabilities.supports_backend_topology(backend, mode):
        raise RunnerIncompatibleError(f"runner does not support {backend}/{mode}")
    if runner_capabilities is not None and "afd" not in runner_capabilities.supported_worker_types:
        raise RunnerIncompatibleError("runner does not support AFD worker type")
    roles = ("aggregated",) if mode == "agg" else ("prefill", "decode")
    domains, provenance = {}, {}
    facts_cache = {}
    for role in roles:
        if role in ss.afd_roles:
            afd = ss.afd_roles[role]["afd"]
            validate_afd_routing_location(afd["routing_location"])
            for tp in _choices(afd.get("tp_a"), default=[1]):
                validate_afd_attention_tp(tp)
    validate_afd_model(ss.model_name)
    for role in roles:
        legacy_role = "agg" if role == "aggregated" else role
        if role in ss.afd_roles:
            afd = ss.afd_roles[role]["afd"]
            systems = (afd.get("a_system") or ss.hardware_sku, afd.get("f_system") or ss.hardware_sku)
            for system in (ss.hardware_sku, *systems):
                if system not in facts_cache:
                    facts_cache[system] = resolve_model_hardware(
                        ss.model_name, system, backend=backend, systems_paths=ss.systems_paths
                    )
            facts = facts_cache[systems[0]]
            domain = enumerate_afd_topologies(
                AFDSearchConfig(
                    total_gpus=ss.gpu_budget,
                    gpus_per_node=facts.gpus_per_node,
                    f_gpus_per_node=facts_cache[systems[1]].gpus_per_node,
                    is_moe=facts.is_moe,
                    num_experts=facts.num_experts,
                    a_tp_size_candidates=tuple(_choices(afd.get("tp_a"), default=[])),
                    batch_size_per_a_replica_candidates=tuple(_choices(afd["a_batch_size"], default=[])),
                    f_moe_ep_size_candidates=tuple(_choices(afd.get("f_moe_ep_size"), default=[])),
                    microbatch_candidates=tuple(_choices(afd["num_microbatches"], default=[2, 3, 4])),
                    pipeline_layout=afd["pipeline_layout"],
                    comm_overhead_factor=afd["comm_overhead_factor"],
                    routing_location=afd["routing_location"],
                    max_af_ratio=afd["max_af_ratio"],
                    max_candidates=ss.max_topology_candidates,
                    min_gpu_budget=ss.min_gpu_budget if mode == "agg" else None,
                )
            )
            domains[role] = tuple(replace(t, a_system=systems[0], f_system=systems[1]) for t in domain.candidates)
            provenance[role] = dict(domain.provenance)
        else:
            entries = ss.parallel_custom_configs_by_mode.get(mode, {}).get(legacy_role, [])
            pinned = [_replica_from_dict(entry) for entry in entries] if entries else None
            domain = parallel_configs_for(
                ss.model_name,
                ss.hardware_sku_for(legacy_role),
                backend=backend,
                backend_version=ss.requested_backend_version(backend),
                deployment_mode="agg",
                worker_type=legacy_role,
                gpu_budget=ss.gpu_budget,
                min_gpu_budget=None,
                max_seq_len=max_seq_len,
                role_runtime={"agg": _role_runtime(ss, backend, legacy_role)},
                role_max_seq_len={"agg": getattr(ss, f"{legacy_role}_context_length") or max_seq_len}
                if legacy_role != "agg"
                else None,
                **_engine_memory_kwargs(ss),
                **_estimator_root_kwargs(ss, legacy_role),
                **({"candidate_configs": pinned} if pinned else {}),
            )
            if pinned:
                pinned_set = set(pinned)
                domain = [c for c in domain if c in pinned_set]
            choices = ss.parallel_independent_by_mode.get(mode, {})
            bounds = ss.parallel_independent_log_ranges_by_mode.get(mode, {})
            prefix = "" if mode == "agg" else f"{role}_"
            domains[role] = tuple(
                c
                for c in domain
                if all(
                    bounds[name][0] <= value <= bounds[name][1]
                    if name in bounds
                    else choices.get(name) is None or value in choices[name]
                    for name, value in ((prefix + k, v) for k, v in _parallel_leaf_values(c).items())
                )
            )
            if pinned and set(pinned) - set(domains[role]):
                raise NoViableParallelConfig(f"{role} pinned shape is infeasible")
    support = {}

    def add(candidate):
        if _runner_supports_parallel_config(runner_capabilities, mode, candidate):
            support[candidate] = frozenset({backend})
            if len(support) > ss.max_topology_candidates:
                raise AFDInfeasible(
                    AFDReasonCategory.CANDIDATE_LIMIT,
                    f"combined topology domain exceeds max_topology_candidates={ss.max_topology_candidates}",
                )

    if mode == "agg":
        for candidate in domains["aggregated"]:
            add(candidate)
    else:
        buckets = {}
        for index, candidate in enumerate(domains["decode"]):
            buckets.setdefault(candidate.total_gpus, []).append(index)
        gpu_counts = sorted(buckets)
        for prefill in domains["prefill"]:
            start = bisect_left(gpu_counts, max(0, (ss.min_gpu_budget or 0) - prefill.total_gpus))
            stop = bisect_right(gpu_counts, ss.gpu_budget - prefill.total_gpus)
            # Merge original domain ordinals to preserve stable P-major / D-minor ordering.
            for index in merge(*(buckets[g] for g in gpu_counts[start:stop])):
                add(DisaggParallelConfig(prefill=prefill, decode=domains["decode"][index]))
    if not support:
        raise NoViableParallelConfig("no viable AFD role composition within GPU budget")
    knobs = branch_knob_choices(ss, mode)
    knobs["backend"] = [backend]
    if config.workload.load_choices is not None:
        knobs["traffic_load"] = list(config.workload.load_choices)
    ordinary_roles = {"agg" if role == "aggregated" else role for role in roles if role not in ss.afd_roles}
    # Continuous domains are stored separately from discrete knob choices.
    floats = {k: tuple(v) for k, v in ss.engine_float_ranges.items() if k.split("_", 1)[0] in ordinary_roles}
    integers = {k: tuple(v) for k, v in ss.engine_integer_log_ranges.items() if k in knobs}
    for name in integers:
        knobs.pop(name, None)
    if config.workload.load_range is not None:
        floats["traffic_load"] = tuple(config.workload.load_range)
    return BranchSpace(
        deployment_mode=mode,
        parallel_configs=tuple(support),
        supported_backends=support,
        knob_choices=knobs,
        gpu_budget=ss.gpu_budget,
        flat_parallel_choices=True,
        float_ranges=floats,
        integer_ranges=integers,
        log_float_ranges=frozenset(k for k in ss.engine_log_ranges if k in floats)
        | (frozenset({"traffic_load"}) if config.workload.load_log_scale and "traffic_load" in floats else frozenset()),
        log_integer_ranges=frozenset(integers),
        log_discrete_choices=frozenset(k for k in ss.engine_log_discrete if k in knobs),
        domain_provenance={"roles": provenance, "candidate_count": len(support), "complete": True},
    )


def enumerate_branches(
    config: SmartSearchConfig,
    *,
    max_seq_len: int | None = None,
    runner_capabilities: RunnerCapabilities | None = None,
) -> list[BranchSpace]:
    """One :class:`BranchSpace` per ``deployment_mode``. Within each, ``backend`` is a
    searched knob: the parallel-config domain is the **union** of every configured
    backend's KV-feasible configs, tagged with which backends support each.

    A backend with no perf DB / no viable config for a mode is dropped (skipped). A mode
    for which *no* backend is viable is skipped with a warning (so a viable mode still
    runs); only if **no** mode is viable does it raise :class:`NoViableParallelConfig`. A
    *pinned* config that is legal for no backend is a hard error (fail fast — the pin is
    wrong). ``max_seq_len`` is forwarded to :func:`parallel_configs_for` (``None`` -> the
    model's max context length).
    """
    ss = config.search_space
    branches: list[BranchSpace] = []
    skipped: list[str] = []  # modes dropped because no backend was viable
    skip_warnings: list[str] = []
    deployment_modes = tuple(dict.fromkeys(ss.deployment_mode))
    unique_backends = tuple(dict.fromkeys(ss.backend))
    runner_incompatibilities = {
        mode: tuple(
            backend
            for backend in unique_backends
            if runner_capabilities is not None and not runner_capabilities.supports_backend_topology(backend, mode)
        )
        for mode in deployment_modes
    }
    all_runner_incompatible = runner_capabilities is not None and all(
        runner_incompatibilities[mode] == unique_backends for mode in deployment_modes
    )
    runner_incompatibility_details = "; ".join(
        f"deployment_mode={mode!r}: runner-incompatible backends={list(backends)}"
        for mode, backends in runner_incompatibilities.items()
        if backends
    )
    for deployment_mode in deployment_modes:
        active = ("aggregated",) if deployment_mode == "agg" else ("prefill", "decode")
        if any(role in ss.afd_roles for role in active):
            branches.append(_afd_roles_branch(config, deployment_mode, max_seq_len, runner_capabilities))
            continue
        # Pinned configs (if any) are parsed once, then validated per backend; otherwise
        # each backend contributes its full enumerated menu.
        raw_pinned = ss.parallel_configs_by_mode.get(deployment_mode, ss.parallel_configs)
        pinned = [_parse_parallel_entry(e, deployment_mode) for e in raw_pinned] if raw_pinned else None
        for candidate in pinned or ():
            if candidate.total_gpus > ss.gpu_budget:
                raise NoViableParallelConfig(
                    f"pinned parallel config {candidate}: used_gpus={candidate.total_gpus} exceeds "
                    f"gpu_budget={ss.gpu_budget} (optimization.constraints.max_candidate_gpus in recommend); "
                    "remove this preset or increase the GPU budget"
                )
        raw_custom = ss.parallel_custom_configs_by_mode.get(deployment_mode, {})
        custom_by_role = {
            role: tuple(_replica_from_dict(entry) for entry in entries) for role, entries in raw_custom.items()
        }
        profile_configs = _fpm_parallel_configs(ss, deployment_mode) if ss.fpm_profile is not None else None
        candidate_configs = profile_configs if profile_configs is not None else pinned

        def matches_custom(config: _ParallelConfig) -> bool:
            return all(_parallel_role(config, role) in choices for role, choices in custom_by_role.items())

        support: dict[_ParallelConfig, set[str]] = {}
        runner_incompatible = list(runner_incompatibilities[deployment_mode])
        for backend in unique_backends:
            if runner_capabilities is not None and not runner_capabilities.supports_backend_topology(
                backend, deployment_mode
            ):
                continue
            try:
                if deployment_mode == "disagg" and (
                    ss.hardware_sku_for("prefill") != ss.hardware_sku_for("decode")
                    or ss.systems_paths_for("prefill") != ss.systems_paths_for("decode")
                ):
                    legal = _heterogeneous_disagg_configs(
                        ss,
                        backend=backend,
                        max_seq_len=max_seq_len,
                        candidate_configs=candidate_configs,
                    )
                else:
                    legal = parallel_configs_for(
                        ss.model_name,
                        ss.hardware_sku_for("agg" if deployment_mode == "agg" else "prefill"),
                        gpu_budget=ss.gpu_budget,
                        deployment_mode=deployment_mode,
                        backend=backend,
                        backend_version=ss.requested_backend_version(backend),
                        min_gpu_budget=ss.min_gpu_budget,
                        max_seq_len=max_seq_len,
                        role_runtime=_runtime_by_role(ss, backend, deployment_mode),
                        role_max_seq_len=_context_by_role(ss, deployment_mode, max_seq_len),
                        **_engine_memory_kwargs(ss),
                        **_estimator_root_kwargs(ss, "agg" if deployment_mode == "agg" else "prefill"),
                        **({"fpm_profile": ss.fpm_profile} if ss.fpm_profile is not None else {}),
                        **({"candidate_configs": candidate_configs} if candidate_configs is not None else {}),
                    )
            except (NoPerfDatabase, NoViableParallelConfig):
                continue  # backend unusable for this mode -> drop it from the search
            legal = [
                cfg for cfg in legal if _runner_supports_parallel_config(runner_capabilities, deployment_mode, cfg)
            ]
            if custom_by_role:
                legal = [cfg for cfg in legal if matches_custom(cfg)]
            legal = [
                cfg
                for cfg in legal
                if matches_parallel_domains(
                    cfg,
                    ss.parallel_independent_by_mode.get(deployment_mode, {}),
                    ss.parallel_independent_log_ranges_by_mode.get(deployment_mode, {}),
                )
            ]
            legal_set = set(legal)
            for cfg in pinned if pinned is not None else legal:
                if cfg in legal_set:
                    support.setdefault(cfg, set()).add(backend)

        if not support:
            if (pinned is not None or custom_by_role) and all_runner_incompatible:
                raise RunnerIncompatibleError(
                    "no configured backend/topology is supported by the Replay runner; "
                    f"{runner_incompatibility_details}"
                )
            if pinned is not None or custom_by_role:
                # an explicit pin that no backend can run is a user error -> fail fast
                raise NoViableParallelConfig(
                    f"deployment_mode={deployment_mode!r}: no configured backend can run the pinned "
                    f"parallel_configs (illegal shape, replay-incompatible backend, or no perf DB)"
                    + (f"; runner-incompatible backends={runner_incompatible}" if runner_incompatible else "")
                )
            # natural infeasibility for this mode -> skip it, keep any viable modes
            skip_warnings.append(
                f"smart-sweep: deployment_mode={deployment_mode!r} skipped — no configured backend "
                f"has a viable parallel config within gpu_budget={ss.gpu_budget}"
                + (f"; runner-incompatible backends={runner_incompatible}" if runner_incompatible else "")
            )
            skipped.append(deployment_mode)
            continue
        if pinned is not None:
            illegal = [c for c in pinned if c not in support]
            if illegal:
                raise NoViableParallelConfig(
                    f"pinned parallel_configs are legal/KV-feasible for no configured backend: {illegal}"
                    + (
                        f"; deployment_mode={deployment_mode!r}: runner-incompatible backends={runner_incompatible}"
                        if runner_incompatible
                        else ""
                    )
                )

        knob_choices = branch_knob_choices(ss, deployment_mode)
        viable_backends = set().union(*support.values())
        knob_choices["backend"] = [backend for backend in dict.fromkeys(ss.backend) if backend in viable_backends]
        float_ranges: dict[str, tuple[float, float]] = {}
        kv_load_range = config.workload.kv_load_ratio_range
        generic_kv_load = config.workload.load_search_field == "kv_load_ratio"
        if kv_load_range is not None and not generic_kv_load:
            float_ranges["kv_load_ratio"] = kv_load_range
        elif config.workload.kv_load_ratio is not None and not generic_kv_load:
            # A scalar KV load is pinned for both scalar and Pareto goals. Keep it in
            # the constant path so every decoded selection carries the requested ratio.
            knob_choices["kv_load_ratio"] = [float(config.workload.kv_load_ratio)]
        log_float_ranges: set[str] = set()
        integer_ranges: dict[str, tuple[int, int]] = {}
        log_integer_ranges: set[str] = set()
        log_discrete_choices = {name for name in ss.engine_log_discrete if name in knob_choices}
        active_roles = {"agg"} if deployment_mode == "agg" else {"prefill", "decode"}
        for name, bounds in ss.engine_float_ranges.items():
            if name.split("_", 1)[0] not in active_roles:
                continue
            float_ranges[name] = (float(bounds[0]), float(bounds[1]))
            if name in ss.engine_log_ranges:
                log_float_ranges.add(name)
        for name, bounds in ss.engine_integer_log_ranges.items():
            if name.split("_", 1)[0] not in active_roles:
                continue
            knob_choices.pop(name, None)
            integer_ranges[name] = (int(bounds[0]), int(bounds[1]))
            log_integer_ranges.add(name)
        if config.workload.load_choices is not None:
            knob_choices["traffic_load"] = list(config.workload.load_choices)
        elif config.workload.load_range is not None:
            float_ranges["traffic_load"] = (
                float(config.workload.load_range[0]),
                float(config.workload.load_range[1]),
            )
            if config.workload.load_log_scale:
                log_float_ranges.add("traffic_load")
        branches.append(
            # Omitted ranges are derived from the legal pool; requested leaf
            # domains have already been intersected with joint feasibility.
            BranchSpace(
                deployment_mode=deployment_mode,
                parallel_configs=tuple(support),
                supported_backends={cfg: frozenset(bs) for cfg, bs in support.items()},
                knob_choices=knob_choices,
                gpu_budget=ss.gpu_budget,
                float_ranges=float_ranges,
                integer_ranges=integer_ranges,
                log_float_ranges=frozenset(log_float_ranges),
                log_integer_ranges=frozenset(log_integer_ranges),
                log_discrete_choices=frozenset(log_discrete_choices),
                flat_parallel_choices=deployment_mode in ss.flat_parallel_modes,
                parallel_independent_choices={
                    name: tuple(
                        sorted(
                            set(values)
                            if values is not None
                            else {_parallel_leaf_values(config)[name] for config in support}
                        )
                    )
                    for name, values in ss.parallel_independent_by_mode.get(deployment_mode, {}).items()
                },
                parallel_independent_log_ranges={
                    name: (int(bounds[0]), int(bounds[1]))
                    for name, bounds in ss.parallel_independent_log_ranges_by_mode.get(deployment_mode, {}).items()
                },
                parallel_custom_choices=custom_by_role,
            )
        )

    if not branches:
        if skipped and all_runner_incompatible:
            raise RunnerIncompatibleError(
                f"no configured backend/topology is supported by the Replay runner; {runner_incompatibility_details}"
            )
        raise NoViableParallelConfig(
            f"no deployment_mode has a viable parallel config (skipped {skipped}); check "
            f"backends / model / hardware / gpu_budget={ss.gpu_budget}"
            + (f"; {runner_incompatibility_details}" if runner_incompatibility_details else "")
        )
    for message in skip_warnings:
        warnings.warn(message, stacklevel=2)
    return branches
