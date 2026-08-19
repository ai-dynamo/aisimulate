# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the per-branch candidate space the sampler searches over.

A *branch* is one **deployment_mode** (agg / disagg) — one Vizier study each, since
agg and disagg have structurally different parallel configs. ``backend`` is NOT a
branch: it is a searched categorical knob within the study. For each mode we take the
**union** of every configured backend's KV-feasible parallel configs
(:func:`aisimulate.sweeper.model_hw.parallel_configs_for`) as the valid projection pool, recording per
config which backends support it. The sampler projects structured latent features onto this
pool. Backends with no perf DB, no viable config, or no support from the injected replay
runner are dropped from the backend knob.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import Any

from .afd import (
    AFDParallelConfig,
    AFDSearchConfig,
    AFDTopology,
    enumerate_afd_topologies,
)
from .config import SmartSearchConfig
from .kv_estimate import NoPerfDatabase
from .model_hw import (
    ModelHardware,
    NoViableParallelConfig,
    parallel_configs_for,
    resolve_model_hardware,
)
from .parallel_enum import (
    DisaggParallelConfig,
    ParallelShape,
    ReplicaParallelConfig,
    RoleParallelCandidates,
)
from .replay import RunnerCapabilities

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
    # Search-domain generation and pruning evidence. Candidate-level topology
    # provenance is carried separately through materialization.
    domain_provenance: dict[str, Any] = field(default_factory=dict)


def _engine_knobs(deployment_mode: str) -> tuple[str, ...]:
    if deployment_mode == "agg":
        return _AGG_ENGINE
    if deployment_mode == "disagg":
        return _DISAGG_ENGINE
    return ()


def _shape_from_dict(d: dict[str, Any]) -> ParallelShape:
    """A per-worker :class:`ParallelShape` from a pinned shape dict. Omitted dims
    default to 1 (so dense models can write just ``{tp: N}``); ``pp`` and ``cp``
    default to 1."""
    if "tp" not in d:
        raise ValueError(f"a parallel_configs shape needs a 'tp' field, got {d}")
    return ParallelShape(
        tp=int(d["tp"]),
        dp=int(d.get("attention_dp", 1)),
        moe_tp=int(d.get("moe_tp", 1)),
        moe_ep=int(d.get("moe_ep", 1)),
        pp=int(d.get("pp", 1)),
        cp=int(d.get("cp", 1)),
    )


def _replica_from_dict(d: dict[str, Any]) -> ReplicaParallelConfig:
    return ReplicaParallelConfig(
        shape=_shape_from_dict(d), replicas=int(d.get("replicas", 1))
    )


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
        batch_candidates = getattr(search_space, f"{role}_batch_size_candidates")
        if batch_candidates is not None:
            choices.pop(f"{role}_max_num_seqs")
            choices[f"{role}_batch_size"] = list(batch_candidates)
        context_candidates = getattr(
            search_space, f"{role}_context_tokens_candidates"
        )
        if context_candidates is not None:
            choices.pop(f"{role}_max_num_batched_tokens")
            choices[f"{role}_context_tokens"] = list(context_candidates)
    return choices


def _role_parallel_candidates(search_space, role: str) -> RoleParallelCandidates | None:
    """Translate configured role lists; empty tuples retain capability defaults."""

    names = {
        "gpus_per_worker": "num_gpu_candidates",
        "tp": "tp_candidates",
        "pp": "pp_candidates",
        "attention_dp": "dp_candidates",
        "moe_tp": "moe_tp_candidates",
        "moe_ep": "moe_ep_candidates",
        "cp": "cp_candidates",
    }
    configured = {
        target: getattr(search_space, f"{role}_{source}")
        for target, source in names.items()
    }
    workers = getattr(search_space, f"{role}_num_workers_candidates")
    if all(value is None for value in configured.values()) and workers is None:
        return None
    return RoleParallelCandidates(
        **{
            target: tuple(value) if value is not None else ()
            for target, value in configured.items()
        },
        workers=tuple(workers) if workers is not None else None,
    )


def _runner_supports_parallel_config(
    capabilities: RunnerCapabilities | None,
    deployment_mode: str,
    config: _ParallelConfig,
) -> bool:
    """Apply runner topology limits before a config enters the sampler domain."""

    if capabilities is None:
        return True
    if isinstance(config, AFDParallelConfig):
        if config.companion is None:
            return True
        return capabilities.supports_attention_dp("disagg", config.companion.shape.dp)
    if deployment_mode != "disagg":
        return True
    if not isinstance(config, DisaggParallelConfig):
        return False
    return capabilities.supports_attention_dp(
        deployment_mode,
        config.prefill.shape.dp,
        config.decode.shape.dp,
    )


def _pinned_afd_topology(
    entry: dict[str, Any],
    *,
    facts: ModelHardware,
    search_space,
    combined_with_pd: bool,
) -> AFDTopology:
    """Materialize a user pin while keeping hardware/model/mode facts authoritative."""

    controlled = {
        "gpus_per_node": facts.gpus_per_node,
        "phase": search_space.afd_phase,
        "combined_with_pd": combined_with_pd,
        "comm_overhead_factor": search_space.afd_comm_overhead_factor,
        "boundary_on_attn": search_space.afd_boundary_on_attn,
        "is_moe": facts.is_moe,
        "num_experts": facts.num_experts,
    }
    conflicts = {
        name: (entry[name], value) for name, value in controlled.items() if name in entry and entry[name] != value
    }
    if conflicts:
        raise ValueError(f"pinned AFD topology conflicts with model/hardware/mode facts: {conflicts}")
    return AFDTopology(**{**entry, **controlled})


def _decode_companion_candidates(
    candidates: RoleParallelCandidates | None,
) -> RoleParallelCandidates:
    """Force the generic aggregate enumerator to retain decode's CP=1 default."""

    if candidates is None:
        return RoleParallelCandidates(
            gpus_per_worker=(),
            tp=(),
            pp=(),
            attention_dp=(),
            moe_tp=(),
            moe_ep=(),
            cp=(1,),
        )
    return replace(candidates, cp=candidates.cp or (1,))


def _enumerate_afd_branch(
    config: SmartSearchConfig,
    deployment_mode: str,
    *,
    max_seq_len: int | None,
    runner_capabilities: RunnerCapabilities | None,
) -> BranchSpace | None:
    """Build one backend-tagged AFD pool from topology and companion domains."""

    ss = config.search_space
    combined = deployment_mode == "afd+pd"
    if config.workload.kv_load_ratio is not None:
        raise ValueError(
            "AFD search does not yet expose a scheduler-visible KV capacity; use "
            "fixed concurrency/request_rate or a trace instead of kv_load_ratio"
        )
    supported_requested = [
        backend
        for backend in dict.fromkeys(ss.backend)
        if runner_capabilities is None or runner_capabilities.supports_backend_topology(backend, deployment_mode)
    ]
    if not supported_requested:
        return None

    facts = resolve_model_hardware(ss.model_name, ss.hardware_sku, backend=supported_requested[0])
    pinned = tuple(
        _pinned_afd_topology(
            entry,
            facts=facts,
            search_space=ss,
            combined_with_pd=combined,
        )
        for entry in ss.afd_pinned_topologies
    )
    batch_candidates = tuple(ss.afd_batch_size_candidates or ())
    enumeration = enumerate_afd_topologies(
        AFDSearchConfig(
            total_gpus=ss.gpu_budget,
            gpus_per_node=facts.gpus_per_node,
            is_moe=facts.is_moe,
            num_experts=facts.num_experts,
            pinned_topologies=pinned,
            tp_a_candidates=tuple(ss.afd_tp_a_candidates or ()),
            a_batch_size_candidates=batch_candidates,
            f_moe_ep_size_candidates=tuple(ss.afd_f_moe_ep_size_candidates or ()),
            microbatch_candidates=tuple(ss.afd_microbatch_candidates),
            pipeline_model_candidates=tuple(ss.afd_pipeline_model_candidates),
            phase=ss.afd_phase,
            combined_with_pd=combined,
            comm_overhead_factor=ss.afd_comm_overhead_factor,
            boundary_on_attn=ss.afd_boundary_on_attn,
            min_gpu_budget=None if combined else ss.min_gpu_budget,
            max_af_ratio=ss.afd_max_af_ratio,
            max_candidates=ss.afd_max_candidates,
            candidate_overflow=ss.afd_candidate_overflow,
        )
    )

    support: dict[_ParallelConfig, set[str]] = {}
    pruning = {
        "companion_worker_ceiling": 0,
        "gpu_budget": 0,
        "minimum_gpu_budget": 0,
        "runner_attention_dp": 0,
    }
    companion_role = "decode" if ss.afd_phase == "prefill" else "prefill"
    for backend in supported_requested:
        if not combined:
            for topology in enumeration.candidates:
                support.setdefault(AFDParallelConfig(topology), set()).add(backend)
            continue

        role_candidates = _role_parallel_candidates(ss, companion_role)
        if companion_role == "decode":
            role_candidates = _decode_companion_candidates(role_candidates)
        try:
            companions = parallel_configs_for(
                ss.model_name,
                ss.hardware_sku,
                gpu_budget=ss.gpu_budget,
                deployment_mode="agg",
                backend=backend,
                min_gpu_budget=None,
                max_seq_len=max_seq_len,
                agg_candidates=role_candidates,
            )
        except (NoPerfDatabase, NoViableParallelConfig):
            continue
        assert all(isinstance(item, ReplicaParallelConfig) for item in companions)
        max_workers = ss.max_decode_workers if companion_role == "decode" else ss.max_prefill_workers
        for topology in enumeration.candidates:
            for companion in companions:
                assert isinstance(companion, ReplicaParallelConfig)
                if companion.replicas > max_workers:
                    pruning["companion_worker_ceiling"] += 1
                    continue
                candidate = AFDParallelConfig(topology, companion)
                if candidate.total_gpus > ss.gpu_budget:
                    pruning["gpu_budget"] += 1
                    continue
                if ss.min_gpu_budget is not None and candidate.total_gpus < ss.min_gpu_budget:
                    pruning["minimum_gpu_budget"] += 1
                    continue
                if not _runner_supports_parallel_config(runner_capabilities, deployment_mode, candidate):
                    pruning["runner_attention_dp"] += 1
                    continue
                support.setdefault(candidate, set()).add(backend)

    if not support:
        return None
    knob_choices = branch_knob_choices(ss, deployment_mode)
    viable_backends = set().union(*support.values())
    knob_choices["backend"] = [backend for backend in dict.fromkeys(ss.backend) if backend in viable_backends]
    return BranchSpace(
        deployment_mode=deployment_mode,
        parallel_configs=tuple(support),
        supported_backends={cfg: frozenset(backends) for cfg, backends in support.items()},
        knob_choices=knob_choices,
        gpu_budget=ss.gpu_budget,
        domain_provenance={
            "afd_enumeration": {
                **dict(enumeration.provenance),
                "generated_count": enumeration.generated_count,
                "accepted_topologies": len(enumeration.candidates),
                "rejection_counts": dict(enumeration.rejection_counts),
                "truncated": enumeration.truncated,
            },
            "combined_pruning": pruning,
            "accepted_candidates": len(support),
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
    # Dedupe modes (preserving order): a repeated deployment_mode would yield duplicate
    # branches and hence colliding Vizier study_ids (one study per mode).
    for deployment_mode in dict.fromkeys(ss.deployment_mode):
        if deployment_mode in {"afd", "afd+pd"}:
            afd_branch = _enumerate_afd_branch(
                config,
                deployment_mode,
                max_seq_len=max_seq_len,
                runner_capabilities=runner_capabilities,
            )
            if afd_branch is None:
                warnings.warn(
                    f"smart-sweep: deployment_mode={deployment_mode!r} skipped — "
                    "no configured backend has a replay-capable AFD topology and "
                    "companion within the GPU budget",
                    stacklevel=2,
                )
                skipped.append(deployment_mode)
            else:
                branches.append(afd_branch)
            continue
        # Pinned configs (if any) are parsed once, then validated per backend; otherwise
        # each backend contributes its full enumerated menu.
        pinned = (
            [_parse_parallel_entry(e, deployment_mode) for e in ss.parallel_configs]
            if ss.parallel_configs
            else None
        )
        support: dict[_ParallelConfig, set[str]] = {}
        runner_incompatible = [
            backend
            for backend in ss.backend
            if runner_capabilities is not None
            and not runner_capabilities.supports_backend_topology(
                backend, deployment_mode
            )
        ]
        for backend in ss.backend:
            if (
                runner_capabilities is not None
                and not runner_capabilities.supports_backend_topology(
                    backend, deployment_mode
                )
            ):
                continue
            try:
                domain_kwargs: dict[str, Any] = {}
                for role in ("agg", "prefill", "decode"):
                    candidates = _role_parallel_candidates(ss, role)
                    if candidates is not None:
                        domain_kwargs[f"{role}_candidates"] = candidates
                for field_name in (
                    "num_gpu_per_replica",
                    "max_gpu_per_replica",
                    "max_prefill_workers",
                    "max_decode_workers",
                ):
                    if field_name in ss.model_fields_set:
                        value = getattr(ss, field_name)
                        domain_kwargs[field_name] = (
                            tuple(value)
                            if field_name == "num_gpu_per_replica"
                            else value
                        )
                legal = parallel_configs_for(
                    ss.model_name,
                    ss.hardware_sku,
                    gpu_budget=ss.gpu_budget,
                    deployment_mode=deployment_mode,
                    backend=backend,
                    min_gpu_budget=ss.min_gpu_budget,
                    max_seq_len=max_seq_len,
                    **domain_kwargs,
                )
            except (NoPerfDatabase, NoViableParallelConfig):
                continue  # backend unusable for this mode -> drop it from the search
            legal = [
                cfg
                for cfg in legal
                if _runner_supports_parallel_config(
                    runner_capabilities, deployment_mode, cfg
                )
            ]
            legal_set = set(legal)
            for cfg in pinned if pinned is not None else legal:
                if cfg in legal_set:
                    support.setdefault(cfg, set()).add(backend)

        if not support:
            if pinned is not None:
                # an explicit pin that no backend can run is a user error -> fail fast
                raise NoViableParallelConfig(
                    f"deployment_mode={deployment_mode!r}: no configured backend can run the pinned "
                    f"parallel_configs (illegal shape, replay-incompatible backend, or no perf DB)"
                )
            # natural infeasibility for this mode -> skip it, keep any viable modes
            warnings.warn(
                f"smart-sweep: deployment_mode={deployment_mode!r} skipped — no configured backend "
                f"has a viable parallel config within gpu_budget={ss.gpu_budget}"
                + (
                    f"; runner-incompatible backends={runner_incompatible}"
                    if runner_incompatible
                    else ""
                ),
                stacklevel=2,
            )
            skipped.append(deployment_mode)
            continue
        if pinned is not None:
            illegal = [c for c in pinned if c not in support]
            if illegal:
                raise NoViableParallelConfig(
                    f"pinned parallel_configs are legal/KV-feasible for no configured backend: {illegal}"
                )

        knob_choices = branch_knob_choices(ss, deployment_mode)
        viable_backends = set().union(*support.values())
        knob_choices["backend"] = [
            backend
            for backend in dict.fromkeys(ss.backend)
            if backend in viable_backends
        ]
        float_ranges: dict[str, tuple[float, float]] = {}
        kv_load_range = config.workload.kv_load_ratio_range
        if kv_load_range is not None:
            float_ranges["kv_load_ratio"] = kv_load_range
        elif config.workload.kv_load_ratio is not None:
            # A scalar KV load is pinned for both scalar and Pareto goals. Keep it in
            # the constant path so every decoded selection carries the requested ratio.
            knob_choices["kv_load_ratio"] = [float(config.workload.kv_load_ratio)]
        branches.append(
            BranchSpace(
                deployment_mode=deployment_mode,
                parallel_configs=tuple(support),
                supported_backends={cfg: frozenset(bs) for cfg, bs in support.items()},
                knob_choices=knob_choices,
                gpu_budget=ss.gpu_budget,
                float_ranges=float_ranges,
            )
        )

    if not branches:
        raise NoViableParallelConfig(
            f"no deployment_mode has a viable parallel config (skipped {skipped}); check "
            f"backends / model / hardware / gpu_budget={ss.gpu_budget}"
        )
    return branches
