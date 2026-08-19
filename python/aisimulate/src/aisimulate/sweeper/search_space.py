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
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import product
from typing import Any

from .config import SmartSearchConfig
from .engine_request import EngineControlTemplate
from .heterogeneous import (
    DisaggRole,
    RoleEstimatorSpecs,
    RoleFailureCategory,
)
from .kv_estimate import NoPerfDatabase
from .model_hw import NoViableParallelConfig, parallel_configs_for
from .parallel_enum import (
    DisaggParallelConfig,
    ParallelShape,
    ReplicaParallelConfig,
    RoleParallelCandidates,
)
from .replay import EstimatorSpec, RunnerCapabilities

_ParallelConfig = ReplicaParallelConfig | DisaggParallelConfig
_SchedulerPoint = tuple[int, ...]

_AGG_ENGINE = ("agg_max_num_batched_tokens", "agg_max_num_seqs")
_DISAGG_ENGINE = (
    "prefill_max_num_batched_tokens",
    "prefill_max_num_seqs",
    "decode_max_num_batched_tokens",
    "decode_max_num_seqs",
)


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
    # Stable, pre-sampling accounting for configured domains that were rejected.
    # This remains attached to the viable branch even when only some backend pairs
    # were pruned, instead of losing those failures in the enumeration step.
    pruning_diagnostics: tuple[BranchPruningDiagnostic, ...] = ()
    enumeration_counts: dict[str, int] = field(default_factory=dict)
    # Appended to preserve the positional constructor slots above. Exact scheduler
    # limits under which each topology/backend pair is KV-feasible; values follow
    # ``scheduler_knob_names`` order. Empty metadata preserves the legacy
    # unconditional relation used by lightweight fixtures.
    scheduler_knob_names: tuple[str, ...] = ()
    scheduler_support: dict[
        _ParallelConfig, dict[str, frozenset[_SchedulerPoint]]
    ] = field(default_factory=dict)

    def supports_selection(
        self, config: _ParallelConfig, selection: Mapping[str, Any]
    ) -> bool:
        """Whether backend, topology, and concrete scheduler limits are legal."""

        backend = selection.get("backend")
        if backend not in self.supported_backends.get(config, frozenset()):
            return False
        if not self.scheduler_knob_names:
            return True
        points = self.scheduler_support.get(config, {}).get(str(backend))
        if not points:
            return False
        try:
            point = tuple(int(selection[name]) for name in self.scheduler_knob_names)
        except (KeyError, TypeError, ValueError):
            return False
        return point in points


@dataclass(frozen=True)
class BranchPruningDiagnostic:
    """One deterministic pre-search domain-pruning reason."""

    backend: str
    category: RoleFailureCategory
    detail: str
    role: DisaggRole | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "backend": self.backend,
            "role": self.role.value if self.role is not None else None,
            "category": self.category.value,
            "detail": self.detail,
        }


def _engine_knobs(deployment_mode: str) -> tuple[str, ...]:
    return _AGG_ENGINE if deployment_mode == "agg" else _DISAGG_ENGINE


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
    choices = {name: list(getattr(search_space, name)) for name in names}
    roles = ("agg",) if deployment_mode == "agg" else ("prefill", "decode")
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


def _scheduler_domain(
    search_space, deployment_mode: str
) -> tuple[tuple[str, ...], tuple[dict[str, int], ...]]:
    """Concrete finite scheduler points in deterministic knob order."""

    choices = branch_knob_choices(search_space, deployment_mode)
    names = tuple(sorted(choices))
    points = tuple(
        dict(zip(names, (int(value) for value in values), strict=True))
        for values in product(*(choices[name] for name in names))
    )
    return names, points


def _role_scheduler_limits(
    selection: Mapping[str, int], role: str
) -> tuple[int, int]:
    """Return ``(max_num_tokens, max_batch_size)`` for one concrete role."""

    context_name = (
        f"{role}_context_tokens"
        if f"{role}_context_tokens" in selection
        else f"{role}_max_num_batched_tokens"
    )
    batch_name = (
        f"{role}_batch_size"
        if f"{role}_batch_size" in selection
        else f"{role}_max_num_seqs"
    )
    return int(selection[context_name]), int(selection[batch_name])


def _scheduler_kwargs(
    selection: Mapping[str, int], deployment_mode: str
) -> dict[str, int]:
    """Capacity-estimator kwargs for an exact scheduler point."""

    if deployment_mode == "agg":
        max_num_tokens, max_batch_size = _role_scheduler_limits(selection, "agg")
        return {
            "max_num_tokens": max_num_tokens,
            "max_batch_size": max_batch_size,
        }
    prefill_tokens, prefill_batch = _role_scheduler_limits(selection, "prefill")
    decode_tokens, decode_batch = _role_scheduler_limits(selection, "decode")
    return {
        "prefill_max_num_tokens": prefill_tokens,
        "prefill_max_batch_size": prefill_batch,
        "decode_max_num_tokens": decode_tokens,
        "decode_max_batch_size": decode_batch,
    }


def _scheduler_point(
    names: tuple[str, ...], selection: Mapping[str, int]
) -> _SchedulerPoint:
    return tuple(int(selection[name]) for name in names)


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

    if capabilities is None or deployment_mode != "disagg":
        return True
    if not isinstance(config, DisaggParallelConfig):
        return False
    return capabilities.supports_attention_dp(
        deployment_mode,
        config.prefill.shape.dp,
        config.decode.shape.dp,
    )


def _engine_kwargs(search_space, role: str | None = None) -> dict[str, Any]:
    """KV/shape controls for a shared or independently resolved role."""

    get = (
        (lambda name: getattr(search_space, name))
        if role is None
        else (lambda name: search_space.role_value(role, name))
    )
    kwargs: dict[str, Any] = {}
    if get("enable_wideep"):
        kwargs["enable_wideep"] = True
    if (moe_backend := get("moe_backend")) is not None:
        kwargs["moe_backend"] = moe_backend
    for name in (
        "gemm_quant_mode",
        "moe_quant_mode",
        "kvcache_quant_mode",
        "fmha_quant_mode",
        "comm_quant_mode",
    ):
        if (value := get(name)) is not None:
            kwargs[name] = value
    if search_space.aic_nextn is not None:
        kwargs["nextn"] = search_space.aic_nextn
    if (memory_fraction := get("free_gpu_memory_fraction")) is not None:
        kwargs["memory_fraction"] = memory_fraction
    return kwargs


def _heterogeneous_support(
    config: SmartSearchConfig,
    *,
    pinned: list[_ParallelConfig] | None,
    runner_capabilities: RunnerCapabilities | None,
    role_estimator_specs: Mapping[str, RoleEstimatorSpecs],
    role_engine_controls: Mapping[str, Mapping[str, EngineControlTemplate]],
    scheduler_knob_names: tuple[str, ...],
    scheduler_selections: tuple[dict[str, int], ...],
) -> tuple[
    dict[_ParallelConfig, set[str]],
    dict[_ParallelConfig, dict[str, set[_SchedulerPoint]]],
    tuple[BranchPruningDiagnostic, ...],
]:
    """Enumerate independent role shapes, then pair them under one GPU budget."""

    ss = config.search_space
    support: dict[_ParallelConfig, set[str]] = {}
    scheduler_support: dict[
        _ParallelConfig, dict[str, set[_SchedulerPoint]]
    ] = {}
    diagnostics: list[BranchPruningDiagnostic] = []
    for pair_label, estimators in role_estimator_specs.items():
        pair = estimators.pair
        if runner_capabilities is not None:
            unsupported_role = next(
                (
                    role
                    for role in (DisaggRole.PREFILL, DisaggRole.DECODE)
                    if not runner_capabilities.supports_backend_topology(
                        pair.backend_for(role), "disagg"
                    )
                ),
                None,
            )
            pair_supported = runner_capabilities.supports_disaggregated_backend_pair(
                pair.prefill, pair.decode
            )
            if unsupported_role is not None or not pair_supported:
                diagnostics.append(
                    BranchPruningDiagnostic(
                        backend=pair_label,
                        role=unsupported_role,
                        category=RoleFailureCategory.UNSUPPORTED_BACKEND,
                        detail=(
                            "runner does not support the role backend/topology"
                            if unsupported_role is not None
                            else "runner does not explicitly support the heterogeneous backend pair"
                        ),
                    )
                )
                continue
        per_role: dict[
            str, dict[tuple[int, int], list[ReplicaParallelConfig]]
        ] = {}
        pair_failure: BranchPruningDiagnostic | None = None
        for role in (DisaggRole.PREFILL, DisaggRole.DECODE):
            name = role.value
            estimator = estimators.estimator_for(role)
            template = role_engine_controls[pair_label][name]
            role_candidates = _role_parallel_candidates(ss, name)
            max_workers = (
                ss.max_prefill_workers
                if role is DisaggRole.PREFILL
                else ss.max_decode_workers
            )
            limits = tuple(
                dict.fromkeys(
                    _role_scheduler_limits(selection, name)
                    for selection in scheduler_selections
                )
            )
            legal_by_limits: dict[
                tuple[int, int], list[ReplicaParallelConfig]
            ] = {}
            first_failure: BranchPruningDiagnostic | None = None
            for max_num_tokens, max_batch_size in limits:
                try:
                    legal = parallel_configs_for(
                        estimator.model_path,
                        estimator.system,
                        gpu_budget=ss.gpu_budget,
                        deployment_mode="agg",
                        backend=estimator.backend,
                        min_gpu_budget=None,
                        max_seq_len=template.max_seq_len,
                        max_num_tokens=max_num_tokens,
                        max_batch_size=max_batch_size,
                        backend_version=estimator.backend_version,
                        systems_paths=list(estimator.systems_paths),
                        agg_candidates=role_candidates,
                        **_engine_kwargs(ss, name),
                    )
                except NoPerfDatabase as exc:
                    first_failure = first_failure or BranchPruningDiagnostic(
                        backend=pair_label,
                        role=role,
                        category=RoleFailureCategory.KV_CAPACITY,
                        detail=str(exc),
                    )
                    legal_by_limits[(max_num_tokens, max_batch_size)] = []
                    continue
                except NoViableParallelConfig as exc:
                    first_failure = first_failure or BranchPruningDiagnostic(
                        backend=pair_label,
                        role=role,
                        category=RoleFailureCategory.NO_PARALLEL_CONFIG,
                        detail=str(exc),
                    )
                    legal_by_limits[(max_num_tokens, max_batch_size)] = []
                    continue
                legal_by_limits[(max_num_tokens, max_batch_size)] = [
                    value
                    for value in legal
                    if isinstance(value, ReplicaParallelConfig)
                    and value.replicas <= max_workers
                ]
            if not any(legal_by_limits.values()):
                pair_failure = first_failure or BranchPruningDiagnostic(
                    backend=pair_label,
                    role=role,
                    category=RoleFailureCategory.NO_PARALLEL_CONFIG,
                    detail="no role topology is legal for any configured scheduler limit",
                )
                break
            per_role[name] = legal_by_limits
        if pair_failure is not None:
            diagnostics.append(pair_failure)
            continue
        pair_accepted = False
        has_budget_pair = False
        for selection in scheduler_selections:
            legal_pairs = [
                DisaggParallelConfig(prefill=prefill, decode=decode)
                for prefill in per_role["prefill"][
                    _role_scheduler_limits(selection, "prefill")
                ]
                for decode in per_role["decode"][
                    _role_scheduler_limits(selection, "decode")
                ]
                if prefill.total_gpus + decode.total_gpus <= ss.gpu_budget
                and prefill.total_gpus + decode.total_gpus <= ss.max_gpu_per_replica
                and prefill.total_gpus + decode.total_gpus in ss.num_gpu_per_replica
                and (
                    ss.min_gpu_budget is None
                    or prefill.total_gpus + decode.total_gpus >= ss.min_gpu_budget
                )
            ]
            has_budget_pair = has_budget_pair or bool(legal_pairs)
            legal_set = set(legal_pairs)
            for candidate in pinned if pinned is not None else legal_pairs:
                if candidate in legal_set and _runner_supports_parallel_config(
                    runner_capabilities, "disagg", candidate
                ):
                    support.setdefault(candidate, set()).add(pair_label)
                    scheduler_support.setdefault(candidate, {}).setdefault(
                        pair_label, set()
                    ).add(_scheduler_point(scheduler_knob_names, selection))
                    pair_accepted = True
        if not pair_accepted:
            diagnostics.append(
                BranchPruningDiagnostic(
                    backend=pair_label,
                    category=(
                        RoleFailureCategory.NO_PARALLEL_CONFIG
                        if has_budget_pair
                        else RoleFailureCategory.GPU_BUDGET
                    ),
                    detail=(
                        "no legal parallel config remains after pinned-domain and runner-capability filtering"
                        if has_budget_pair
                        else "no prefill/decode parallel-config pair fits the shared "
                        f"gpu range [{ss.min_gpu_budget or 1}, {ss.gpu_budget}]"
                    ),
                )
            )
    return support, scheduler_support, tuple(diagnostics)


def enumerate_branches(
    config: SmartSearchConfig,
    *,
    max_seq_len: int | None = None,
    runner_capabilities: RunnerCapabilities | None = None,
    estimator_specs: Mapping[str, EstimatorSpec] | None = None,
    role_estimator_specs: Mapping[str, RoleEstimatorSpecs] | None = None,
    role_engine_controls: Mapping[str, Mapping[str, EngineControlTemplate]]
    | None = None,
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
    skipped_enumeration_reports: list[dict[str, Any]] = []
    # Dedupe modes (preserving order): a repeated deployment_mode would yield duplicate
    # branches and hence colliding Vizier study_ids (one study per mode).
    for deployment_mode in dict.fromkeys(ss.deployment_mode):
        # Pinned configs (if any) are parsed once, then validated per backend; otherwise
        # each backend contributes its full enumerated menu.
        pinned = (
            [_parse_parallel_entry(e, deployment_mode) for e in ss.parallel_configs]
            if ss.parallel_configs
            else None
        )
        heterogeneous = deployment_mode == "disagg" and ss.has_role_overrides
        scheduler_knob_names, scheduler_selections = _scheduler_domain(
            ss, deployment_mode
        )
        runner_incompatible: list[str] = []
        pruning_diagnostics: tuple[BranchPruningDiagnostic, ...] = ()
        enumeration_counts: dict[str, int] = {}
        scheduler_support: dict[
            _ParallelConfig, dict[str, set[_SchedulerPoint]]
        ] = {}
        if heterogeneous:
            if role_estimator_specs is None or role_engine_controls is None:
                raise ValueError(
                    "heterogeneous disagg enumeration requires role estimator and engine-control catalogs"
                )
            support, scheduler_support, pruning_diagnostics = _heterogeneous_support(
                config,
                pinned=pinned,
                runner_capabilities=runner_capabilities,
                role_estimator_specs=role_estimator_specs,
                role_engine_controls=role_engine_controls,
                scheduler_knob_names=scheduler_knob_names,
                scheduler_selections=scheduler_selections,
            )
            accepted_pairs = set().union(*support.values()) if support else set()
            enumeration_counts = {
                "considered": len(role_estimator_specs),
                "accepted": len(accepted_pairs),
                "pruned": len(role_estimator_specs) - len(accepted_pairs),
            }
            for diagnostic in pruning_diagnostics:
                warnings.warn(
                    "smart-sweep: heterogeneous backend pair pruned — "
                    f"backend={diagnostic.backend!r}, "
                    f"role={diagnostic.role.value if diagnostic.role else 'pair'!r}, "
                    f"category={diagnostic.category.value!r}: {diagnostic.detail}",
                    stacklevel=2,
                )
            if runner_capabilities is not None:
                runner_incompatible = [
                    label
                    for label, estimators in role_estimator_specs.items()
                    if not all(
                        runner_capabilities.supports_backend_topology(
                            estimators.pair.backend_for(role), "disagg"
                        )
                        for role in (DisaggRole.PREFILL, DisaggRole.DECODE)
                    )
                    or not runner_capabilities.supports_disaggregated_backend_pair(
                        estimators.pair.prefill, estimators.pair.decode
                    )
                ]
        else:
            support = {}
            enumeration_counts = {"considered": 0, "accepted": 0, "pruned": 0}
            scheduler_failures: dict[
                tuple[str, RoleFailureCategory],
                tuple[int, dict[str, int], str],
            ] = {}

            def record_scheduler_failure(
                backend: str,
                category: RoleFailureCategory,
                selection: dict[str, int],
                detail: str,
            ) -> None:
                key = (backend, category)
                current = scheduler_failures.get(key)
                if current is None:
                    scheduler_failures[key] = (1, dict(selection), detail)
                else:
                    scheduler_failures[key] = (
                        current[0] + 1,
                        current[1],
                        current[2],
                    )

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
                estimator_kwargs: dict[str, Any] = {}
                if estimator_specs is not None:
                    estimator = estimator_specs[backend]
                    estimator_kwargs.update(
                        backend_version=estimator.backend_version,
                        systems_paths=list(estimator.systems_paths),
                    )
                else:
                    requested_version = ss.requested_backend_version(backend)
                    if requested_version is not None:
                        estimator_kwargs["backend_version"] = requested_version
                    if ss.systems_paths != ["default"]:
                        estimator_kwargs["systems_paths"] = ss.systems_paths
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
                for scheduler_selection in scheduler_selections:
                    enumeration_counts["considered"] += 1
                    try:
                        legal = parallel_configs_for(
                            ss.model_name,
                            ss.hardware_sku,
                            gpu_budget=ss.gpu_budget,
                            deployment_mode=deployment_mode,
                            backend=backend,
                            min_gpu_budget=ss.min_gpu_budget,
                            max_seq_len=max_seq_len,
                            **_scheduler_kwargs(
                                scheduler_selection, deployment_mode
                            ),
                            **domain_kwargs,
                            **estimator_kwargs,
                            **_engine_kwargs(ss),
                        )
                    except NoPerfDatabase as exc:
                        enumeration_counts["pruned"] += 1
                        record_scheduler_failure(
                            backend,
                            RoleFailureCategory.KV_CAPACITY,
                            scheduler_selection,
                            str(exc),
                        )
                        continue
                    except NoViableParallelConfig as exc:
                        enumeration_counts["pruned"] += 1
                        record_scheduler_failure(
                            backend,
                            RoleFailureCategory.NO_PARALLEL_CONFIG,
                            scheduler_selection,
                            str(exc),
                        )
                        continue
                    enumerated_count = len(legal)
                    legal = [
                        cfg
                        for cfg in legal
                        if _runner_supports_parallel_config(
                            runner_capabilities, deployment_mode, cfg
                        )
                    ]
                    legal_set = set(legal)
                    accepted = [
                        cfg
                        for cfg in (pinned if pinned is not None else legal)
                        if cfg in legal_set
                    ]
                    if not accepted:
                        enumeration_counts["pruned"] += 1
                        record_scheduler_failure(
                            backend,
                            RoleFailureCategory.NO_PARALLEL_CONFIG,
                            scheduler_selection,
                            (
                                "no pinned parallel config is legal under this scheduler point"
                                if pinned is not None
                                else "runner capability filtering removed every parallel config"
                                if enumerated_count
                                else "parallel enumeration returned no config"
                            ),
                        )
                        continue
                    enumeration_counts["accepted"] += 1
                    for cfg in accepted:
                        support.setdefault(cfg, set()).add(backend)
                        scheduler_support.setdefault(cfg, {}).setdefault(
                            backend, set()
                        ).add(
                            _scheduler_point(
                                scheduler_knob_names, scheduler_selection
                            )
                        )
            pruning_diagnostics = tuple(
                BranchPruningDiagnostic(
                    backend=backend,
                    category=category,
                    detail=(
                        f"{count} scheduler point(s) pruned; first "
                        f"{first_selection}: {first_detail}"
                    ),
                )
                for (backend, category), (
                    count,
                    first_selection,
                    first_detail,
                ) in scheduler_failures.items()
            )

        if not support:
            enumeration_report = (
                {
                    "deployment_mode": deployment_mode,
                    "counts": {
                        name: enumeration_counts[name]
                        for name in ("considered", "accepted", "pruned")
                    },
                    "pruning_diagnostics": [
                        diagnostic.as_dict()
                        for diagnostic in pruning_diagnostics
                    ],
                }
                if pruning_diagnostics or enumeration_counts
                else None
            )
            if pinned is not None:
                # an explicit pin that no backend can run is a user error -> fail fast
                raise NoViableParallelConfig(
                    f"deployment_mode={deployment_mode!r}: no configured backend can run the pinned "
                    f"parallel_configs (illegal shape, replay-incompatible backend, or no perf DB)",
                    enumeration_reports=(
                        (enumeration_report,)
                        if enumeration_report is not None
                        else ()
                    ),
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
            if enumeration_report is not None:
                skipped_enumeration_reports.append(enumeration_report)
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
        if heterogeneous:
            knob_choices["backend"] = [
                label
                for label in role_estimator_specs or {}
                if label in viable_backends
            ]
        else:
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
                scheduler_knob_names=scheduler_knob_names,
                scheduler_support={
                    cfg: {
                        backend: frozenset(points)
                        for backend, points in by_backend.items()
                    }
                    for cfg, by_backend in scheduler_support.items()
                },
                gpu_budget=ss.gpu_budget,
                float_ranges=float_ranges,
                pruning_diagnostics=pruning_diagnostics,
                enumeration_counts=enumeration_counts,
            )
        )

    if not branches:
        raise NoViableParallelConfig(
            f"no deployment_mode has a viable parallel config (skipped {skipped}); check "
            f"backends / model / hardware / gpu_budget={ss.gpu_budget}",
            enumeration_reports=tuple(skipped_enumeration_reports),
        )
    return branches
