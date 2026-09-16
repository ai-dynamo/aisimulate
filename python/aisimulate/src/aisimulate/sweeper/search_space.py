# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the per-branch candidate space the sampler searches over.

A *branch* is one **deployment_mode** (agg / disagg / afd / afd+pd) — one Vizier study
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
from dataclasses import dataclass, field
from typing import Any

from .afd_parallel import (
    AFDInfeasible,
    AFDParallelConfig,
    AFDReasonCategory,
    AFDSearchConfig,
    AFDTopology,
    enumerate_afd_topologies,
)
from .config import SmartSearchConfig
from .kv_estimate import NoPerfDatabase, resolve_backend_version
from .model_hw import ModelHardware, NoViableParallelConfig, parallel_configs_for, resolve_model_hardware
from .parallel_enum import DisaggParallelConfig, ParallelShape, ReplicaParallelConfig
from .replay import RunnerCapabilities


class RunnerIncompatibleError(NoViableParallelConfig):
    """No configured backend/topology pair is supported by the Replay runner."""


_ParallelConfig = ReplicaParallelConfig | DisaggParallelConfig | AFDParallelConfig

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
    if isinstance(config, AFDParallelConfig):
        raise TypeError("AFD branches use one finite flat topology domain")
    return {
        **role_values("prefill_", config.prefill),
        **role_values("decode_", config.decode),
    }


def _parallel_role(config: _ParallelConfig, role: str) -> ReplicaParallelConfig:
    if isinstance(config, ReplicaParallelConfig):
        if role != "agg":
            raise ValueError(f"aggregated parallel config has no {role!r} role")
        return config
    if isinstance(config, AFDParallelConfig):
        if config.companion is None or config.companion_role != role:
            raise ValueError(f"AFD parallel config has no {role!r} companion")
        return config.companion
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


def branch_knob_choices(search_space, deployment_mode: str) -> dict[str, list[Any]]:
    """Backend-owned atomic knobs for one deployment branch."""
    names = _engine_knobs(deployment_mode)
    if deployment_mode == "afd+pd":
        names = _DECODE_ENGINE if search_space.afd_phase == "prefill" else _PREFILL_ENGINE
    choices = {name: list(getattr(search_space, name)) for name in names}
    if deployment_mode == "agg":
        roles = ("agg",)
    elif deployment_mode == "disagg":
        roles = ("prefill", "decode")
    elif deployment_mode == "afd+pd":
        roles = ("decode",) if search_space.afd_phase == "prefill" else ("prefill",)
    else:
        roles = ()
    for role in roles:
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
    if isinstance(config, AFDParallelConfig):
        return config.companion is None or capabilities.supports_attention_dp("disagg", config.companion.shape.dp)
    if deployment_mode != "disagg":
        return True
    if not isinstance(config, DisaggParallelConfig):
        return False
    return capabilities.supports_attention_dp(
        deployment_mode,
        config.prefill.shape.dp,
        config.decode.shape.dp,
    )


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


def _heterogeneous_disagg_configs(
    search_space,
    *,
    backend: str,
    max_seq_len: int | None,
) -> list[DisaggParallelConfig]:
    """Enumerate each P/D role against its own hardware, then apply the shared budget."""
    role_hardware = {role: search_space.hardware_sku_for(role) for role in ("prefill", "decode")}
    backend_version = search_space.backend_version
    if backend_version is None:
        role_versions = {
            role: resolve_backend_version(hardware, backend, systems_path=search_space.systems_path)
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
                backend=backend,
                backend_version=backend_version,
                min_gpu_budget=None,
                max_seq_len=max_seq_len,
                role_runtime={"agg": _role_runtime(search_space, backend, role)},
                systems_path=search_space.systems_path,
                **({"fpm_profile": search_space.fpm_profile} if search_space.fpm_profile is not None else {}),
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


def _pinned_afd_topology(
    raw: dict[str, Any],
    *,
    search_space,
    model_hardware: ModelHardware,
    combined_with_pd: bool,
) -> AFDTopology:
    values = dict(raw)
    values.setdefault("f_moe_ep_size", 1)
    values.setdefault("num_microbatches", 3)
    values.setdefault("pipeline_model", search_space.afd_pipeline_model_candidates[0])
    values.setdefault("comm_overhead_factor", search_space.afd_comm_overhead_factor)
    values.setdefault("boundary_on_attn", search_space.afd_boundary_on_attn)
    values.update(
        {
            "gpus_per_node": model_hardware.gpus_per_node,
            "phase": search_space.afd_phase,
            "combined_with_pd": combined_with_pd,
            "is_moe": model_hardware.is_moe,
            "num_experts": model_hardware.num_experts,
        }
    )
    return AFDTopology(**values)


def _afd_branch(
    config: SmartSearchConfig,
    deployment_mode: str,
    *,
    max_seq_len: int | None,
    runner_capabilities: RunnerCapabilities | None,
    skip_warnings: list[str],
) -> BranchSpace | None:
    """Build one complete, finite AFD topology domain for the generic sampler."""

    ss = config.search_space
    if config.workload.kv_load_ratio is not None or config.workload.load_search_field == "kv_load_ratio":
        raise AFDInfeasible(
            AFDReasonCategory.INVALID_TOPOLOGY,
            "AFD search does not yet expose scheduler-visible KV capacity; use an absolute traffic load",
        )

    model_hardware = resolve_model_hardware(ss.model_name, ss.hardware_sku, backend=ss.backend[0])
    combined_with_pd = deployment_mode == "afd+pd"
    pinned = tuple(
        _pinned_afd_topology(
            raw,
            search_space=ss,
            model_hardware=model_hardware,
            combined_with_pd=combined_with_pd,
        )
        for raw in ss.afd_pinned_topologies
    )
    topology_domain = enumerate_afd_topologies(
        AFDSearchConfig(
            total_gpus=ss.gpu_budget,
            min_gpu_budget=None if combined_with_pd else ss.min_gpu_budget,
            gpus_per_node=model_hardware.gpus_per_node,
            is_moe=model_hardware.is_moe,
            num_experts=model_hardware.num_experts,
            pinned_topologies=pinned,
            tp_a_candidates=tuple(ss.afd_tp_a_candidates or ()),
            a_batch_size_candidates=tuple(ss.afd_batch_size_candidates or (128,)),
            f_moe_ep_size_candidates=tuple(ss.afd_f_moe_ep_size_candidates or ()),
            microbatch_candidates=tuple(ss.afd_microbatch_candidates),
            pipeline_model_candidates=tuple(ss.afd_pipeline_model_candidates),
            phase=ss.afd_phase,
            combined_with_pd=combined_with_pd,
            comm_overhead_factor=ss.afd_comm_overhead_factor,
            boundary_on_attn=ss.afd_boundary_on_attn,
            max_af_ratio=ss.afd_max_af_ratio,
            max_candidates=ss.afd_max_candidates,
        )
    )

    pinned_companions = tuple(_replica_from_dict(raw) for raw in ss.afd_companion_parallel_configs)
    support: dict[AFDParallelConfig, set[str]] = {}
    runner_incompatible: list[str] = []
    companion_role = "decode" if ss.afd_phase == "prefill" else "prefill"
    for backend in ss.backend:
        if runner_capabilities is not None and not runner_capabilities.supports_backend_topology(
            backend, deployment_mode
        ):
            runner_incompatible.append(backend)
            continue
        if not combined_with_pd:
            candidates = (AFDParallelConfig(topology=topology) for topology in topology_domain.candidates)
        else:
            try:
                legal_companions = parallel_configs_for(
                    ss.model_name,
                    ss.hardware_sku,
                    gpu_budget=ss.gpu_budget,
                    deployment_mode="agg",
                    backend=backend,
                    backend_version=ss.backend_version,
                    min_gpu_budget=None,
                    max_seq_len=max_seq_len,
                    role_runtime={"agg": _role_runtime(ss, backend, companion_role)},
                )
            except (NoPerfDatabase, NoViableParallelConfig):
                continue
            legal_set = set(legal_companions)
            companion_domain = pinned_companions or tuple(legal_companions)
            candidates = (
                AFDParallelConfig(topology=topology, companion=companion)
                for topology in topology_domain.candidates
                for companion in companion_domain
                if companion in legal_set
                and topology.total_gpus + companion.total_gpus <= ss.gpu_budget
                and (ss.min_gpu_budget is None or topology.total_gpus + companion.total_gpus >= ss.min_gpu_budget)
            )
        for candidate in candidates:
            if _runner_supports_parallel_config(runner_capabilities, deployment_mode, candidate):
                support.setdefault(candidate, set()).add(backend)
                if len(support) > ss.afd_max_candidates:
                    raise AFDInfeasible(
                        AFDReasonCategory.CANDIDATE_LIMIT,
                        f"AFD combined domain exceeds afd_max_candidates={ss.afd_max_candidates}; "
                        "narrow the topology or companion domain",
                        provenance={"generated_count": len(support), "count_is_lower_bound": True},
                    )

    unsupported_topologies = [
        topology for topology in pinned if not any(candidate.topology == topology for candidate in support)
    ]
    unsupported_companions = [
        companion
        for companion in pinned_companions
        if not any(candidate.companion == companion for candidate in support)
    ]
    if unsupported_topologies or unsupported_companions:
        raise NoViableParallelConfig(
            f"deployment_mode={deployment_mode!r}: pinned AFD topology or companion has no legal, "
            "runner-compatible candidate within the GPU budget; "
            f"unsupported_topologies={unsupported_topologies}, unsupported_companions={unsupported_companions}"
        )
    if not support:
        skip_warnings.append(
            f"smart-sweep: deployment_mode={deployment_mode!r} skipped — no configured backend has a viable "
            f"AFD candidate within gpu_budget={ss.gpu_budget}"
            + (f"; runner-incompatible backends={runner_incompatible}" if runner_incompatible else ""),
        )
        return None

    knob_choices = branch_knob_choices(ss, deployment_mode)
    viable_backends = set().union(*support.values())
    knob_choices["backend"] = [backend for backend in dict.fromkeys(ss.backend) if backend in viable_backends]
    if config.workload.load_choices is not None:
        knob_choices["traffic_load"] = list(config.workload.load_choices)
    float_ranges: dict[str, tuple[float, float]] = {}
    log_float_ranges: set[str] = set()
    integer_ranges: dict[str, tuple[int, int]] = {}
    log_integer_ranges: set[str] = set()
    active_roles = {companion_role} if combined_with_pd else set()
    for name, bounds in ss.engine_float_ranges.items():
        if name.split("_", 1)[0] in active_roles:
            float_ranges[name] = (float(bounds[0]), float(bounds[1]))
            if name in ss.engine_log_ranges:
                log_float_ranges.add(name)
    for name, bounds in ss.engine_integer_log_ranges.items():
        if name.split("_", 1)[0] in active_roles:
            knob_choices.pop(name, None)
            integer_ranges[name] = (int(bounds[0]), int(bounds[1]))
            log_integer_ranges.add(name)
    if config.workload.load_range is not None:
        float_ranges["traffic_load"] = (
            float(config.workload.load_range[0]),
            float(config.workload.load_range[1]),
        )
        if config.workload.load_log_scale:
            log_float_ranges.add("traffic_load")

    return BranchSpace(
        deployment_mode=deployment_mode,
        parallel_configs=tuple(support),
        supported_backends={candidate: frozenset(backends) for candidate, backends in support.items()},
        knob_choices=knob_choices,
        gpu_budget=ss.gpu_budget,
        float_ranges=float_ranges,
        integer_ranges=integer_ranges,
        log_float_ranges=frozenset(log_float_ranges),
        log_integer_ranges=frozenset(log_integer_ranges),
        log_discrete_choices=frozenset(name for name in ss.engine_log_discrete if name in knob_choices),
        flat_parallel_choices=True,
        domain_provenance={
            "afd": dict(topology_domain.provenance),
            "generated_topologies": topology_domain.generated_count,
            "candidate_count": len(support),
            "complete": True,
        },
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

    # Dedupe modes (preserving order): a repeated deployment_mode would yield duplicate
    # branches and hence colliding Vizier study_ids (one study per mode).
    for deployment_mode in deployment_modes:
        if deployment_mode in {"afd", "afd+pd"}:
            if runner_incompatibilities[deployment_mode] == unique_backends:
                if all_runner_incompatible:
                    raise RunnerIncompatibleError(
                        "no configured backend/topology is supported by the Replay runner; "
                        f"{runner_incompatibility_details}"
                    )
                message = (
                    f"deployment_mode={deployment_mode!r}: no runner-compatible AFD candidate; "
                    f"runner-incompatible backends={list(runner_incompatibilities[deployment_mode])}"
                )
                if ss.afd_pinned_topologies or ss.afd_companion_parallel_configs:
                    raise NoViableParallelConfig(message)
                skipped.append(deployment_mode)
                skip_warnings.append(f"smart-sweep: {message}")
                continue
            branch = _afd_branch(
                config,
                deployment_mode,
                max_seq_len=max_seq_len,
                runner_capabilities=runner_capabilities,
                skip_warnings=skip_warnings,
            )
            if branch is None:
                skipped.append(deployment_mode)
            else:
                branches.append(branch)
            continue
        # Pinned configs (if any) are parsed once, then validated per backend; otherwise
        # each backend contributes its full enumerated menu.
        raw_pinned = ss.parallel_configs_by_mode.get(deployment_mode, ss.parallel_configs)
        pinned = [_parse_parallel_entry(e, deployment_mode) for e in raw_pinned] if raw_pinned else None
        raw_custom = ss.parallel_custom_configs_by_mode.get(deployment_mode, {})
        custom_by_role = {
            role: tuple(_replica_from_dict(entry) for entry in entries) for role, entries in raw_custom.items()
        }

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
                if deployment_mode == "disagg" and (ss.hardware_sku_for("prefill") != ss.hardware_sku_for("decode")):
                    legal = _heterogeneous_disagg_configs(
                        ss,
                        backend=backend,
                        max_seq_len=max_seq_len,
                    )
                else:
                    legal = parallel_configs_for(
                        ss.model_name,
                        ss.hardware_sku_for("agg" if deployment_mode == "agg" else "prefill"),
                        gpu_budget=ss.gpu_budget,
                        deployment_mode=deployment_mode,
                        backend=backend,
                        backend_version=ss.backend_version,
                        min_gpu_budget=ss.min_gpu_budget,
                        max_seq_len=max_seq_len,
                        role_runtime=_runtime_by_role(ss, backend, deployment_mode),
                        systems_path=ss.systems_path,
                        **({"fpm_profile": ss.fpm_profile} if ss.fpm_profile is not None else {}),
                    )
            except (NoPerfDatabase, NoViableParallelConfig):
                continue  # backend unusable for this mode -> drop it from the search
            legal = [
                cfg for cfg in legal if _runner_supports_parallel_config(runner_capabilities, deployment_mode, cfg)
            ]
            if custom_by_role:
                legal = [cfg for cfg in legal if matches_custom(cfg)]
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
            # Independent mode exposes each YAML leaf as an optimizer dimension.
            # Omitted ranges are derived from the legal pool; explicit ranges may
            # still form infeasible Cartesian combinations, which the main loop gates.
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
