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
from typing import Any

from .config import SmartSearchConfig
from .engine_request import EngineControlTemplate
from .heterogeneous import DisaggRole, RoleEstimatorSpecs
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
) -> dict[_ParallelConfig, set[str]]:
    """Enumerate independent role shapes, then pair them under one GPU budget."""

    ss = config.search_space
    support: dict[_ParallelConfig, set[str]] = {}
    for pair_label, estimators in role_estimator_specs.items():
        pair = estimators.pair
        if runner_capabilities is not None and any(
            not runner_capabilities.supports_backend_topology(
                pair.backend_for(role), "disagg"
            )
            for role in (DisaggRole.PREFILL, DisaggRole.DECODE)
        ):
            continue
        per_role: dict[str, list[ReplicaParallelConfig]] = {}
        failed = False
        for role in (DisaggRole.PREFILL, DisaggRole.DECODE):
            name = role.value
            estimator = estimators.estimator_for(role)
            template = role_engine_controls[pair_label][name]
            role_candidates = _role_parallel_candidates(ss, name)
            try:
                legal = parallel_configs_for(
                    estimator.model_path,
                    estimator.system,
                    gpu_budget=ss.gpu_budget,
                    deployment_mode="agg",
                    backend=estimator.backend,
                    min_gpu_budget=None,
                    max_seq_len=template.max_seq_len,
                    backend_version=estimator.backend_version,
                    systems_paths=list(estimator.systems_paths),
                    agg_candidates=role_candidates,
                    **_engine_kwargs(ss, name),
                )
            except (NoPerfDatabase, NoViableParallelConfig):
                failed = True
                break
            max_workers = (
                ss.max_prefill_workers
                if role is DisaggRole.PREFILL
                else ss.max_decode_workers
            )
            per_role[name] = [
                value
                for value in legal
                if isinstance(value, ReplicaParallelConfig)
                and value.replicas <= max_workers
            ]
        if failed:
            continue
        legal_pairs = [
            DisaggParallelConfig(prefill=prefill, decode=decode)
            for prefill in per_role["prefill"]
            for decode in per_role["decode"]
            if prefill.total_gpus + decode.total_gpus <= ss.gpu_budget
            and prefill.total_gpus + decode.total_gpus <= ss.max_gpu_per_replica
            and prefill.total_gpus + decode.total_gpus in ss.num_gpu_per_replica
            and (
                ss.min_gpu_budget is None
                or prefill.total_gpus + decode.total_gpus >= ss.min_gpu_budget
            )
        ]
        legal_set = set(legal_pairs)
        for candidate in pinned if pinned is not None else legal_pairs:
            if candidate in legal_set and _runner_supports_parallel_config(
                runner_capabilities, "disagg", candidate
            ):
                support.setdefault(candidate, set()).add(pair_label)
    return support


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
        runner_incompatible: list[str] = []
        if heterogeneous:
            if role_estimator_specs is None or role_engine_controls is None:
                raise ValueError(
                    "heterogeneous disagg enumeration requires role estimator and engine-control catalogs"
                )
            support = _heterogeneous_support(
                config,
                pinned=pinned,
                runner_capabilities=runner_capabilities,
                role_estimator_specs=role_estimator_specs,
                role_engine_controls=role_engine_controls,
            )
            if runner_capabilities is not None:
                runner_incompatible = [
                    label
                    for label, estimators in role_estimator_specs.items()
                    if any(
                        not runner_capabilities.supports_backend_topology(
                            estimators.pair.backend_for(role), "disagg"
                        )
                        for role in (DisaggRole.PREFILL, DisaggRole.DECODE)
                    )
                ]
        else:
            support = {}
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
                    legal = parallel_configs_for(
                        ss.model_name,
                        ss.hardware_sku,
                        gpu_budget=ss.gpu_budget,
                        deployment_mode=deployment_mode,
                        backend=backend,
                        min_gpu_budget=ss.min_gpu_budget,
                        max_seq_len=max_seq_len,
                        **domain_kwargs,
                        **estimator_kwargs,
                        **_engine_kwargs(ss),
                    )
                except (NoPerfDatabase, NoViableParallelConfig):
                    continue  # unusable for this mode -> drop it from the search
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
