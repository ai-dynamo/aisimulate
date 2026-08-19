# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The smart sweep: SearchSpace -> ranked candidates (best-first).

One Vizier study per ``deployment_mode`` branch searches the parallel-config + knob
space (backend is one of the knobs); each suggestion is unrolled, translated to a
deployment, evaluated by replay, scored, and fed back to the optimizer; feasible
candidates are ranked across branches.

Each round is a **barrier**: the study suggests trials until ``per_round`` unique full
samples complete successfully (ask), they are evaluated **in parallel across worker
processes** (``SweepConfig.parallel_evals``; ``<= 1`` runs sequentially), then their
scores are fed back (tell). Exact duplicates use a run-local result cache and trigger
replacement asks. Vizier ask/tell stay on the main process — workers run only the pure
unroll->materialize ReplaySpec->runner->score path and never touch the study (the
Vizier trial handle never crosses the process boundary).

The replay implementation is always injected as a :class:`RunnerFactory`. Optional
feature adapters are resolved explicitly or through the ``aisimulate.sweep_config_providers``
entry-point group.
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import time
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass, replace
from enum import Enum
from multiprocessing.util import Finalize
from numbers import Real
from typing import Any

from tqdm import tqdm

from .config import Candidate, OptimizationGoal, OptimizationTarget, SmartSearchConfig
from .deploy import build_backend_deployment
from .discovery import resolve_providers
from .kv_estimate import resolve_backend_version
from .kv_load import InfeasibleKVCapacity, resolve_kv_load
from .provider import (
    AdapterReplaySpec,
    AdapterSearchPlan,
    CandidateContext,
    RuntimeHookSpec,
    SearchSpaceFragment,
    SweepConfigProvider,
    SweepContext,
)
from .replay import (
    REPLAY_SPEC_API_VERSION,
    ReplayReport,
    ReplaySpec,
    Runner,
    RunnerFactory,
    canonical_json,
    validate_json_value,
)
from .result import (
    CandidateRecord,
    CandidateRetention,
    CandidateStatus,
    ReasonCategory,
    ResultViews,
    SearchStrategy,
    SweepCounts,
    SweepResult,
    make_candidate_provenance,
    make_run_provenance,
    retain_candidate_records,
)
from .sample import unroll_sample
from .sampler import BranchSampler, Suggestion, make_branch_sampler
from .score import (
    aggregate_sla_violations,
    analyze_candidates,
    is_feasible,
    make_candidate,
)
from .search_space import BranchSpace, enumerate_branches

logger = logging.getLogger(__name__)


# Result of evaluating one suggestion (no Vizier here). ``observe_metrics`` is
# fed to sampler.observe. Both failed and infeasible results are reported with
# observe_infeasible so invalid trials never steer the sampler as high scores.
@dataclass(frozen=True)
class _EvalResult:
    candidate: Candidate | None
    observe_metrics: dict[str, float] | None
    outcome: str
    reason: str
    reason_category: ReasonCategory | None
    runner_metadata: dict[str, Any]
    report_metrics: dict[str, float] | None = None


@dataclass(frozen=True)
class _ReplayEvaluation:
    metrics: dict[str, float] | None
    metadata: dict[str, Any]
    outcome: str
    reason: str
    reason_category: ReasonCategory | None


@dataclass(frozen=True)
class _PreparedCandidate:
    sample: dict[str, Any]
    replay_spec: ReplaySpec


_ADAPTER_PARAM_PREFIX = "adapter::"
_ADAPTER_PARAM_SEPARATOR = "::"


def _adapter_param(adapter_name: str, local_name: str) -> str:
    return (
        f"{_ADAPTER_PARAM_PREFIX}{adapter_name}{_ADAPTER_PARAM_SEPARATOR}{local_name}"
    )


def _adapter_selection(
    selection: Mapping[str, Any], adapter_name: str
) -> dict[str, Any]:
    prefix = _adapter_param(adapter_name, "")
    return {
        key.removeprefix(prefix): deepcopy(value)
        for key, value in selection.items()
        if key.startswith(prefix)
    }


def _prepare_providers(
    config: SmartSearchConfig,
    *,
    injected: Mapping[str, SweepConfigProvider] | None,
    show_progress: bool,
) -> tuple[dict[str, SweepConfigProvider], dict[str, AdapterSearchPlan]]:
    invalid_names = [
        name for name in config.adapters if _ADAPTER_PARAM_SEPARATOR in name
    ]
    if invalid_names:
        raise ValueError(
            f"adapter names cannot contain reserved separator "
            f"{_ADAPTER_PARAM_SEPARATOR!r}: {invalid_names}"
        )
    providers = resolve_providers(config.adapters, injected=injected)
    base_context = SweepContext(
        core_search_space=config.search_space.model_dump(mode="json"),
        workload=config.workload.model_dump(mode="json"),
        goal=config.goal.model_dump(mode="json"),
        show_progress=show_progress,
    )
    plans: dict[str, AdapterSearchPlan] = {}
    for name, provider in providers.items():
        context = SweepContext(
            core_search_space=deepcopy(base_context.core_search_space),
            workload=deepcopy(base_context.workload),
            goal=deepcopy(base_context.goal),
            show_progress=base_context.show_progress,
        )
        plan = provider.generate_search_space(
            deepcopy(config.adapters[name].search_space), context
        )
        _validate_search_plan(name, plan)
        # The adapter owns the object it returned and may reuse internal buffers
        # later. Take a complete core-owned snapshot at the ABI boundary.
        plans[name] = deepcopy(plan)
    configured_modes = set(config.search_space.deployment_mode)
    for name, plan in plans.items():
        unknown = (
            set(plan.fragment.choices_by_branch)
            | set(plan.fragment.float_ranges_by_branch)
        ) - configured_modes
        if unknown:
            raise ValueError(
                f"adapter {name!r} returned unknown deployment branch(es): "
                f"{sorted(unknown)}"
            )
    return providers, plans


def _validate_search_plan(name: str, plan: Any) -> None:
    if not isinstance(plan, AdapterSearchPlan):
        raise TypeError(
            f"adapter {name!r} generate_search_space must return AdapterSearchPlan"
        )
    if not isinstance(plan.fragment, SearchSpaceFragment):
        raise TypeError(f"adapter {name!r} returned an invalid SearchSpaceFragment")
    try:
        if type(plan.diagnostics) is not dict:
            raise TypeError("search diagnostics must be a dictionary")
        if type(plan.potential_runtime_hooks) is not tuple:
            raise TypeError("potential_runtime_hooks must be a tuple")
        _validate_search_fragment(plan.fragment)
        validate_json_value(plan.state, path=f"adapter {name!r} search plan state")
        validate_json_value(
            plan.diagnostics, path=f"adapter {name!r} search diagnostics"
        )
        for index, hook in enumerate(plan.potential_runtime_hooks):
            _validate_runtime_hook(
                hook,
                path=f"adapter {name!r} potential hook {index}",
            )
        canonical_json(plan)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"adapter {name!r} returned an invalid/non-JSON search plan: {exc}"
        ) from exc


def _validate_search_fragment(fragment: SearchSpaceFragment) -> None:
    if type(fragment.choices_by_branch) is not dict:
        raise TypeError("choices_by_branch must be a dictionary")
    for branch, parameters in fragment.choices_by_branch.items():
        if type(branch) is not str or not branch:
            raise TypeError("categorical branch names must be non-empty strings")
        if type(parameters) is not dict:
            raise TypeError(f"categorical branch {branch!r} must be a dictionary")
        for parameter, values in parameters.items():
            if type(parameter) is not str or not parameter:
                raise TypeError("categorical parameter names must be non-empty strings")
            if type(values) is not list:
                raise TypeError(
                    f"categorical parameter {parameter!r} choices must be a list"
                )
            validate_json_value(
                values, path=f"categorical parameter {parameter!r} choices"
            )

    if type(fragment.float_ranges_by_branch) is not dict:
        raise TypeError("float_ranges_by_branch must be a dictionary")
    for branch, parameters in fragment.float_ranges_by_branch.items():
        if type(branch) is not str or not branch:
            raise TypeError("continuous branch names must be non-empty strings")
        if type(parameters) is not dict:
            raise TypeError(f"continuous branch {branch!r} must be a dictionary")
        for parameter, bounds in parameters.items():
            if type(parameter) is not str or not parameter:
                raise TypeError("continuous parameter names must be non-empty strings")
            if type(bounds) is not tuple or len(bounds) != 2:
                raise TypeError(
                    f"continuous parameter {parameter!r} bounds must be a pair"
                )
            if any(type(bound) not in (int, float) for bound in bounds) or not all(
                math.isfinite(float(bound)) for bound in bounds
            ):
                raise ValueError(
                    f"continuous parameter {parameter!r} bounds must be finite numbers"
                )


def _validate_runtime_hook(hook: Any, *, path: str) -> None:
    if not isinstance(hook, RuntimeHookSpec):
        raise TypeError(f"{path} must be a RuntimeHookSpec")
    if type(hook.provider) is not str or not hook.provider:
        raise TypeError(f"{path} provider must be a non-empty string")
    if type(hook.kind) is not str or not hook.kind:
        raise TypeError(f"{path} kind must be a non-empty string")
    if type(hook.api_version) is not int or hook.api_version < 1:
        raise TypeError(f"{path} api_version must be a positive integer")
    if type(hook.config) is not dict:
        raise TypeError(f"{path} config must be a dictionary")
    validate_json_value(hook.config, path=f"{path} config")


def _validate_provider_replay_spec(name: str, spec: Any) -> None:
    if not isinstance(spec, AdapterReplaySpec):
        raise TypeError(
            f"adapter {name!r} materialize_replay must return AdapterReplaySpec"
        )
    try:
        if type(spec.config) is not dict:
            raise TypeError("replay config must be a dictionary")
        if type(spec.runtime_hooks) is not tuple:
            raise TypeError("runtime_hooks must be a tuple")
        validate_json_value(spec.config, path=f"adapter {name!r} replay config")
        for index, hook in enumerate(spec.runtime_hooks):
            _validate_runtime_hook(
                hook,
                path=f"adapter {name!r} runtime hook {index}",
            )
        canonical_json(spec)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"adapter {name!r} returned an invalid/non-JSON replay spec: {exc}"
        ) from exc


def _merge_adapter_spaces(
    branches: list[BranchSpace],
    plans: Mapping[str, AdapterSearchPlan],
) -> list[BranchSpace]:
    """Namespace and merge every adapter fragment into each core branch."""
    merged: list[BranchSpace] = []
    for branch in branches:
        choices = dict(branch.knob_choices)
        float_ranges = dict(branch.float_ranges)
        for name, plan in plans.items():
            local_choices = plan.fragment.choices_by_branch.get(
                branch.deployment_mode, {}
            )
            local_ranges = plan.fragment.float_ranges_by_branch.get(
                branch.deployment_mode, {}
            )
            overlap = set(local_choices).intersection(local_ranges)
            if overlap:
                raise ValueError(
                    f"adapter {name!r} defined parameters as both categorical and "
                    f"continuous: {sorted(overlap)}"
                )
            for local_name, values in local_choices.items():
                if _ADAPTER_PARAM_SEPARATOR in local_name:
                    raise ValueError(
                        f"adapter {name!r} search parameter {local_name!r} contains "
                        f"reserved separator {_ADAPTER_PARAM_SEPARATOR!r}"
                    )
                if not values:
                    raise ValueError(
                        f"adapter {name!r} search parameter {local_name!r} "
                        f"has no choices in branch {branch.deployment_mode!r}"
                    )
                choices[_adapter_param(name, local_name)] = list(values)
            for local_name, bounds in local_ranges.items():
                if _ADAPTER_PARAM_SEPARATOR in local_name:
                    raise ValueError(
                        f"adapter {name!r} search parameter {local_name!r} contains "
                        f"reserved separator {_ADAPTER_PARAM_SEPARATOR!r}"
                    )
                low, high = bounds
                if low >= high:
                    raise ValueError(
                        f"adapter {name!r} search parameter {local_name!r} needs "
                        f"low < high, got {bounds!r}"
                    )
                float_ranges[_adapter_param(name, local_name)] = (low, high)
        merged.append(replace(branch, knob_choices=choices, float_ranges=float_ranges))
    return merged


def _freeze(value: Any) -> Any:
    """Convert a nested suggestion/context value into a stable hashable key."""
    if is_dataclass(value) and not isinstance(value, type):
        return ("dataclass", type(value).__qualname__, _freeze(asdict(value)))
    if isinstance(value, Enum):
        return ("enum", type(value).__qualname__, _freeze(value.value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                (_freeze(key), _freeze(item))
                for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
            ),
        )
    if isinstance(value, list):
        return ("list", tuple(_freeze(item) for item in value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_freeze(item) for item in value))
    if isinstance(value, set):
        return ("set", tuple(sorted((_freeze(item) for item in value), key=repr)))
    if isinstance(value, frozenset):
        return (
            "frozenset",
            tuple(sorted((_freeze(item) for item in value), key=repr)),
        )
    if value is None:
        return ("none",)
    if type(value) in (str, int, float, bool):
        return (type(value).__name__, value)
    try:
        hash(value)
    except TypeError:
        return ("repr", type(value).__qualname__, repr(value))
    return ("hashable", type(value).__qualname__, value)


def _suggestion_cache_key(suggestion: Suggestion, context: Any) -> Any:
    """A run-local full-sample key; parallel equality alone is not a cache hit."""
    return (context, _freeze(suggestion.selection), _freeze(suggestion.parallel_config))


def _suggestion_snapshot(
    suggestion: Suggestion, config: SmartSearchConfig
) -> dict[str, Any]:
    """Best-effort JSON snapshot for a candidate rejected before replay."""

    try:
        return unroll_sample(
            search_space=config.search_space,
            selection=suggestion.selection,
            parallel_config=suggestion.parallel_config,
        )
    except Exception:
        parallel_config = suggestion.parallel_config
        if is_dataclass(parallel_config) and not isinstance(parallel_config, type):
            parallel_payload: Any = asdict(parallel_config)
        else:
            parallel_payload = repr(parallel_config)
        return {
            **deepcopy(suggestion.selection),
            "parallel_config": parallel_payload,
        }


def _materialize_one(
    selection: dict[str, Any],
    parallel_config: Any,
    *,
    config: SmartSearchConfig,
    goal: OptimizationGoal,
    providers: Mapping[str, SweepConfigProvider],
    provider_plans: Mapping[str, AdapterSearchPlan],
    runner_factory: RunnerFactory,
) -> tuple[_PreparedCandidate | None, _EvalResult | None]:
    """Build a complete replay specification on the main process."""
    try:
        sample = unroll_sample(
            search_space=config.search_space,
            selection=selection,
            parallel_config=parallel_config,
        )
        backend_version = resolve_backend_version(
            config.search_space.hardware_sku, selection["backend"]
        )
        # The resolved perf-model version is part of the evaluated contract. Keep it
        # on the candidate so downstream artifact generation cannot independently
        # select a different backend version.
        sample["backend_version"] = backend_version
        concurrency = config.workload.concurrency
        if "kv_load_ratio" in selection:
            ratio = float(selection["kv_load_ratio"])
            resolution = resolve_kv_load(
                sample,
                workload=config.workload,
                parallel_config=parallel_config,
                ratio=ratio,
                backend_version=backend_version,
            )
            concurrency = resolution.concurrency
            sample["kv_load_ratio"] = resolution.ratio
            sample["kv_load_concurrency_capacity"] = resolution.concurrency_capacity
            load_role = "decode" if sample["deployment_mode"] == "disagg" else "agg"
            sample["kv_load_capacity_tokens"] = resolution.role_capacity_tokens[
                load_role
            ]
            for role, tokens in resolution.role_capacity_tokens.items():
                sample[f"{role}_kv_capacity_tokens"] = tokens
        if concurrency is not None:
            # Preserve the concrete load on every candidate, including a fixed absolute
            # concurrency and one derived from kv_load_ratio.
            sample["concurrency"] = concurrency
        backend_deployment = build_backend_deployment(
            sample, backend_version=backend_version
        )
        adapter_specs: dict[str, AdapterReplaySpec] = {}
        for name, provider in providers.items():
            candidate_context = CandidateContext(
                sample=deepcopy(sample),
                backend_deployment=deepcopy(backend_deployment),
                concurrency=concurrency,
            )
            adapter_spec = provider.materialize_replay(
                deepcopy(provider_plans[name]),
                _adapter_selection(selection, name),
                candidate_context,
            )
            _validate_provider_replay_spec(name, adapter_spec)
            # Frozen dataclasses do not freeze nested JSON containers. Snapshot
            # the return value so an adapter can safely reuse an output buffer
            # without mutating candidates already prepared in this round.
            adapter_specs[name] = deepcopy(adapter_spec)
        replay_spec = ReplaySpec(
            backend_deployment=backend_deployment,
            workload=config.workload.model_dump(mode="json"),
            goal=goal.model_dump(mode="json"),
            concurrency=concurrency,
            adapters=adapter_specs,
        )
        canonical_json(replay_spec)
        runner_factory.capabilities().require_compatible(replay_spec)
        if adapter_specs:
            sample["adapters"] = {
                name: deepcopy(adapter_spec.config)
                for name, adapter_spec in adapter_specs.items()
            }
    except InfeasibleKVCapacity as exc:
        return None, _EvalResult(
            candidate=None,
            observe_metrics=None,
            outcome="infeasible",
            reason=f"candidate KV capacity infeasible: {exc}",
            reason_category=ReasonCategory.KV_CAPACITY,
            runner_metadata={},
        )
    except Exception as exc:
        logger.exception("Sweeper candidate build failed")
        return None, _EvalResult(
            candidate=None,
            observe_metrics=None,
            outcome="failed",
            reason=f"candidate build failed: {type(exc).__name__}: {exc}",
            reason_category=ReasonCategory.CANDIDATE_MATERIALIZATION,
            runner_metadata={},
        )
    return _PreparedCandidate(sample=sample, replay_spec=replay_spec), None


def _run_replay_detailed(spec: ReplaySpec, runner: Runner) -> _ReplayEvaluation:
    """Run one replay while retaining validated runner provenance metadata."""

    try:
        try:
            report = runner.run(spec)
        except Exception as exc:
            logger.exception("Sweeper candidate replay failed")
            return _ReplayEvaluation(
                metrics=None,
                metadata={},
                outcome="failed",
                reason=f"replay failed: {type(exc).__name__}: {exc}",
                reason_category=ReasonCategory.REPLAY_RUNTIME,
            )
        if not isinstance(report, ReplayReport):
            return _ReplayEvaluation(
                metrics=None,
                metadata={},
                outcome="failed",
                reason=(
                    "replay failed: TypeError: runner.run must return ReplayReport, "
                    f"got {type(report).__name__}"
                ),
                reason_category=ReasonCategory.RUNNER_CONTRACT,
            )
        if type(report.metrics) is not dict:
            return _ReplayEvaluation(
                metrics=None,
                metadata={},
                outcome="failed",
                reason="replay failed: TypeError: runner report metrics must be a dictionary",
                reason_category=ReasonCategory.INVALID_METRICS,
            )
        if type(report.metadata) is not dict:
            return _ReplayEvaluation(
                metrics=None,
                metadata={},
                outcome="failed",
                reason="replay failed: TypeError: runner report metadata must be a dictionary",
                reason_category=ReasonCategory.RUNNER_CONTRACT,
            )
        metrics: dict[str, float] = {}
        for name, value in report.metrics.items():
            if type(name) is not str:
                return _ReplayEvaluation(
                    metrics=None,
                    metadata={},
                    outcome="failed",
                    reason=(
                        "replay failed: TypeError: runner metric names must be strings, "
                        f"got {name!r}"
                    ),
                    reason_category=ReasonCategory.INVALID_METRICS,
                )
            if isinstance(value, bool) or not isinstance(value, Real):
                return _ReplayEvaluation(
                    metrics=None,
                    metadata={},
                    outcome="failed",
                    reason=(
                        f"replay failed: TypeError: runner metric {name!r} "
                        "must be a real number"
                    ),
                    reason_category=ReasonCategory.INVALID_METRICS,
                )
            normalized = float(value)
            if not math.isfinite(normalized):
                return _ReplayEvaluation(
                    metrics=None,
                    metadata={},
                    outcome="failed",
                    reason=(
                        f"replay failed: ValueError: runner metric {name!r} must be finite"
                    ),
                    reason_category=ReasonCategory.INVALID_METRICS,
                )
            metrics[name] = normalized
        try:
            validate_json_value(report.metadata, path="runner report metadata")
        except (TypeError, ValueError) as exc:
            return _ReplayEvaluation(
                metrics=None,
                metadata={},
                outcome="failed",
                reason=f"replay failed: {type(exc).__name__}: {exc}",
                reason_category=ReasonCategory.RUNNER_CONTRACT,
            )
        return _ReplayEvaluation(
            metrics=metrics,
            metadata=deepcopy(report.metadata),
            outcome="replayed",
            reason="",
            reason_category=None,
        )
    except Exception as exc:  # fail closed if contract normalization itself regresses
        logger.exception("Sweeper candidate replay failed")
        return _ReplayEvaluation(
            metrics=None,
            metadata={},
            outcome="failed",
            reason=f"replay failed: {type(exc).__name__}: {exc}",
            reason_category=ReasonCategory.UNKNOWN,
        )


def _run_replay(
    spec: ReplaySpec, runner: Runner
) -> tuple[dict[str, float] | None, str, str]:
    """Compatibility projection of the detailed replay result used by older tests."""

    result = _run_replay_detailed(spec, runner)
    return result.metrics, result.outcome, result.reason


def _score_prepared(
    prepared: _PreparedCandidate,
    replay_result: _ReplayEvaluation,
    *,
    config: SmartSearchConfig,
    goal: OptimizationGoal,
) -> _EvalResult:
    """Score a runner result on the main process."""
    report = replay_result.metrics
    outcome = replay_result.outcome
    reason = replay_result.reason
    if outcome == "failed":
        return _EvalResult(
            candidate=None,
            observe_metrics=None,
            outcome=outcome,
            reason=reason,
            reason_category=replay_result.reason_category,
            runner_metadata=replay_result.metadata,
        )
    if report is None:
        return _EvalResult(
            candidate=None,
            observe_metrics=None,
            outcome="failed",
            reason="runner contract violation: a successful replay returned no metrics",
            reason_category=ReasonCategory.RUNNER_CONTRACT,
            runner_metadata=replay_result.metadata,
        )
    effective_targets = (
        set(goal.resolved_pareto_objectives) if goal.is_pareto else {goal.target}
    )
    if (
        effective_targets.intersection(
            {OptimizationTarget.GOODPUT, OptimizationTarget.GOODPUT_PER_GPU}
        )
        and "goodput_output_throughput_tok_s" not in report
    ):
        return _EvalResult(
            candidate=None,
            observe_metrics=None,
            outcome="failed",
            reason=(
                "runner contract violation: goodput objective requires "
                "goodput_output_throughput_tok_s; aggregate latency cannot be used "
                "as a fallback"
            ),
            reason_category=ReasonCategory.RUNNER_CONTRACT,
            runner_metadata=replay_result.metadata,
        )
    sample = prepared.sample
    if not is_feasible(int(sample["used_gpus"]), config.search_space.gpu_budget):
        # Over gpu_budget: report as infeasible to the optimizer (observe_infeasible, not
        # observe(metrics)) so a high score doesn't steer the sampler into the infeasible
        # region. The trial is gated, not ranked.
        return _EvalResult(
            candidate=None,
            observe_metrics=None,
            outcome="infeasible",
            reason=(
                f"over gpu_budget: used_gpus={int(sample['used_gpus'])} > "
                f"gpu_budget={config.search_space.gpu_budget}"
            ),
            reason_category=ReasonCategory.GPU_BUDGET,
            runner_metadata=replay_result.metadata,
        )
    if goal.strict_sla:
        assert goal.sla is not None  # OptimizationGoal validates this invariant.
        violations = aggregate_sla_violations(
            report,
            goal.sla,
            osl=config.workload.osl or 1,
        )
        if violations:
            return _EvalResult(
                candidate=None,
                observe_metrics=None,
                outcome="infeasible",
                reason=f"strict aggregate SLA violation: {'; '.join(violations)}",
                reason_category=ReasonCategory.STRICT_SLA,
                runner_metadata=replay_result.metadata,
                report_metrics=report,
            )
    if goal.is_pareto:
        candidate = make_candidate(
            sample,
            report,
            goal.target,
            pareto_objectives=goal.resolved_pareto_objectives,
        )
        # Pareto objectives are reported raw (each metric carries its own MAXIMIZE/MINIMIZE goal).
        observe_metrics: dict[str, float] = dict(candidate.objectives or {})
    else:
        candidate = make_candidate(sample, report, goal.target)
        observe_metrics = {
            "objective": candidate.score
        }  # single metric, pre-signed higher-is-better
    non_finite_objectives = [
        name
        for name, value in (candidate.objectives or {}).items()
        if not math.isfinite(value)
    ]
    if not math.isfinite(candidate.score) or non_finite_objectives:
        return _EvalResult(
            candidate=None,
            observe_metrics=None,
            outcome="failed",
            reason="runner contract violation: objective metrics must be present and finite",
            reason_category=ReasonCategory.INVALID_METRICS,
            runner_metadata=replay_result.metadata,
            report_metrics=report,
        )
    return _EvalResult(
        candidate=candidate,
        observe_metrics=observe_metrics,
        outcome="feasible",
        reason="",
        reason_category=None,
        runner_metadata=replay_result.metadata,
        report_metrics=report,
    )


# Worker-process plumbing: shared read-only state is sent once via the pool
# initializer; each process creates and reuses one runner.
_WORKER_CTX: dict[str, Any] = {}


def _init_worker(
    runner_factory: RunnerFactory,
) -> None:
    identity = getattr(mp.current_process(), "_identity", ())
    worker_id = int(identity[0]) if identity else 0
    runner = runner_factory.create(worker_id)
    # multiprocessing runs Finalize callbacks during a normal child-process
    # shutdown.  Unlike a plain atexit handler, this matches the ProcessPool
    # worker lifecycle and lets a runtime release worker-local resources.
    Finalize(None, runner.close, exitpriority=0)
    _WORKER_CTX.update(runner=runner)


def _worker_eval(spec: ReplaySpec) -> _ReplayEvaluation:
    return _run_replay_detailed(spec, _WORKER_CTX["runner"])


class Sweeper:
    """Compose and execute isolated backend-neutral configuration sweeps.

    The constructor owns stable dependencies. Every :meth:`run` call creates
    fresh studies, caches, runners, and worker pools, so mutable sweep state never
    leaks between runs.
    """

    def __init__(
        self,
        *,
        runner_factory: RunnerFactory,
        providers: Mapping[str, SweepConfigProvider] | None = None,
        sampler_factory: Callable[..., BranchSampler] = make_branch_sampler,
        show_progress: bool = True,
    ) -> None:
        self._runner_factory = runner_factory
        self._providers = dict(providers or {})
        self._sampler_factory = sampler_factory
        self._show_progress = show_progress

    def run(
        self,
        config: SmartSearchConfig,
        *,
        on_round: Callable[[int, list[Candidate]], None] | None = None,
    ) -> list[Candidate]:
        """Compatibility view returning the full ranked/frontier candidate list.

        New consumers should call :meth:`run_result` to retain status counts,
        rejection reasons, provenance, and stable top-N/Pareto views.
        """

        return self.run_result(
            config,
            top_n=None,
            candidate_retention=CandidateRetention.ALL,
            on_round=on_round,
        ).selected_candidates

    def run_result(
        self,
        config: SmartSearchConfig,
        *,
        top_n: int | None = 5,
        candidate_retention: CandidateRetention | str = CandidateRetention.ALL,
        on_round: Callable[[int, list[Candidate]], None] | None = None,
    ) -> SweepResult:
        """Run the sweep and return the canonical schema-versioned result.

        The configured runner factory is the only replay runtime injection point.
        Providers can be injected directly; otherwise configured providers are
        discovered from package entry points. Within a round, suggestions are evaluated
        across spawned worker processes when ``parallel_evals > 1``. Such callers must
        guard script entrypoints with ``if __name__ == "__main__":``.

        ``candidate_retention="all"`` preserves every unique feasible, infeasible,
        unsupported, timed-out, and failed candidate. ``"feasible"`` retains only
        feasible rows and ``"views"`` retains only the scalar top-N or Pareto front;
        run-wide counts always describe the complete run.
        """
        if top_n is not None and top_n < 1:
            raise ValueError(f"top_n must be positive or None, got {top_n}")
        retention = CandidateRetention(candidate_retention)
        runner_factory = self._runner_factory
        providers = self._providers
        sampler_factory = self._sampler_factory
        show_progress = self._show_progress

        goal = config.goal
        capabilities = runner_factory.capabilities()
        capabilities.require_replay_spec_version(REPLAY_SPEC_API_VERSION)

        # Preserve the legacy preflight order: reject an impossible backend/topology
        # search before adapters perform any potentially expensive preparation.
        branches = enumerate_branches(
            config,
            max_seq_len=config.search_space.context_length,
            runner_capabilities=capabilities,
        )
        resolved_providers, provider_plans = _prepare_providers(
            config, injected=providers, show_progress=show_progress
        )
        for name, plan in provider_plans.items():
            unsupported = [
                hook
                for hook in plan.potential_runtime_hooks
                if not capabilities.supports_hook(hook)
            ]
            if unsupported:
                labels = ", ".join(
                    f"{hook.provider}:{hook.kind}@{hook.api_version}"
                    for hook in unsupported
                )
                raise ValueError(
                    f"runner is incompatible with configured adapter {name!r}; "
                    f"unsupported runtime hook(s): {labels}"
                )

        branches = _merge_adapter_spaces(branches, provider_plans)

        sweep = config.sweep
        per_round = sweep.candidates_per_round or sweep.parallel_evals
        # Target number of successful unique replay configurations across all rounds.
        total = len(branches) * sweep.max_rounds * per_round
        candidates: list[Candidate] = []
        tally = {
            "feasible": 0,
            "infeasible": 0,
            "failed": 0,
            "unsupported": 0,
            "cache_hit": 0,
        }
        candidate_records: list[CandidateRecord] = []
        record_id_by_candidate_object: dict[int, str] = {}
        failure_reasons: dict[str, int] = {}
        # Unique per run: Vizier's datastore persists studies by id, so a fixed id would
        # make a later run inherit a stale study (and its old param space) -> decode crash.
        run_id = uuid.uuid4().hex
        run_nonce = run_id[:8]
        # Multi-objective (pareto) -> one Vizier metric per objective (each with its own
        # direction); single-objective -> the sampler's default single maximized "objective".
        sampler_objectives = (
            [(t.value, t.maximize) for t in goal.resolved_pareto_objectives]
            if goal.is_pareto
            else None
        )
        cache_context = _freeze(
            {
                "search_space": config.search_space.model_dump(mode="python"),
                "adapters": {
                    name: request.model_dump(mode="python")
                    for name, request in config.adapters.items()
                },
                "workload": config.workload.model_dump(mode="python"),
                "goal": goal.model_dump(mode="python"),
                "provider_plans": provider_plans,
            }
        )
        replay_cache: dict[Any, tuple[Candidate, dict[str, float]]] = {}

        def _best() -> float | None:
            return max((c.score for c in candidates), default=None)

        # Parallel across worker processes when parallel_evals > 1. Spawn keeps
        # runner runtimes isolated and lets each worker reuse one runner instance.
        use_pool = sweep.parallel_evals > 1 and per_round > 1
        max_eval_seconds = sweep.max_eval_seconds
        worker_count = min(sweep.parallel_evals, per_round)
        sequential_runner = None if use_pool else runner_factory.create(0)

        def _new_pool() -> ProcessPoolExecutor:
            return ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=mp.get_context("spawn"),
                initializer=_init_worker,
                initargs=(runner_factory,),
            )

        # One-element box so a runtime timeout can kill the hung pool and swap in a fresh one
        # (the closures below read/replace pool_box[0]).
        pool_box: list[Any] = [_new_pool() if use_pool else None]

        def _terminate_pool(pool: ProcessPoolExecutor | None) -> None:
            if pool is None:
                return
            for process in list((getattr(pool, "_processes", None) or {}).values()):
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
            pool.shutdown(wait=False, cancel_futures=True)

        def _replace_pool() -> None:
            _terminate_pool(pool_box[0])
            pool_box[0] = _new_pool()

        @contextmanager
        def _pool_lifecycle():
            try:
                yield
            except BaseException:
                # Do not wait forever for an unrelated hung replay when orchestration,
                # scoring, observation, or cancellation aborts the sweep.
                _terminate_pool(pool_box[0])
                pool_box[0] = None
                if sequential_runner is not None:
                    sequential_runner.close()
                raise
            else:
                # A completed sweep shuts workers down normally so their Runner
                # finalizers execute.  Only the timeout-recovery path above sends a
                # terminate signal, where cleanup is necessarily best-effort.
                if pool_box[0] is not None:
                    pool_box[0].shutdown(wait=True, cancel_futures=True)
                pool_box[0] = None
                if sequential_runner is not None:
                    sequential_runner.close()

        def _pool_error(detail: str) -> RuntimeError:
            return RuntimeError(
                f"Sweeper worker pool failed while {detail}. parallel_evals>1 uses "
                "spawned processes; guard a script entrypoint with `if __name__ == "
                '"__main__":`, or set sweep.parallel_evals=1 to evaluate sequentially.'
            )

        def _eval_batch(todo: list[tuple[Suggestion, _PreparedCandidate]]):
            """Yield ``(suggestion, _EvalResult)`` for each supported suggestion — across worker
            processes when a pool is set, else sequentially in-process. On the pool path it
            evaluates waves no larger than the worker count so queued work never consumes a
            candidate's ``max_eval_seconds`` budget. A replay that overruns is reported
            infeasible ("exceed runtime") and the wave's workers are force-killed (a shared
            pool can't cancel a running task), then a fresh pool handles later waves.
            """
            if pool_box[0] is None:
                assert sequential_runner is not None
                for suggestion, prepared in todo:
                    yield (
                        suggestion,
                        _score_prepared(
                            prepared,
                            _run_replay_detailed(
                                prepared.replay_spec, sequential_runner
                            ),
                            config=config,
                            goal=goal,
                        ),
                    )
                return

            for start in range(0, len(todo), worker_count):
                wave = todo[start : start + worker_count]
                pool = pool_box[0]
                assert pool is not None
                try:
                    # submit() can raise when an initializer or an earlier task killed
                    # the pool, so keep it inside the friendly-error wrapper.
                    futures = {
                        pool.submit(_worker_eval, prepared.replay_spec): (
                            suggestion,
                            prepared,
                        )
                        for suggestion, prepared in wave
                    }
                except BrokenProcessPool as exc:
                    raise _pool_error("submitting a candidate wave") from exc

                pending = set(futures)
                deadline = (
                    time.monotonic() + max_eval_seconds if max_eval_seconds else None
                )
                while pending:
                    remaining = (
                        None
                        if deadline is None
                        else max(0.0, deadline - time.monotonic())
                    )
                    done, pending = wait(
                        pending, timeout=remaining, return_when=FIRST_COMPLETED
                    )
                    if not done:
                        break
                    for future in done:
                        try:
                            replay_result = future.result()
                        except BrokenProcessPool as exc:
                            raise _pool_error("collecting a candidate result") from exc
                        except Exception as exc:
                            raise _pool_error(
                                f"collecting a candidate result ({type(exc).__name__}: {exc})"
                            ) from exc
                        suggestion, prepared = futures[future]
                        yield (
                            suggestion,
                            _score_prepared(
                                prepared,
                                replay_result,
                                config=config,
                                goal=goal,
                            ),
                        )

                if pending:
                    seconds = max_eval_seconds or 0.0
                    for future in pending:
                        suggestion, _prepared = futures[future]
                        yield (
                            suggestion,
                            _EvalResult(
                                candidate=None,
                                observe_metrics=None,
                                outcome="infeasible",
                                reason=f"exceed runtime: replay > {seconds:.0f}s",
                                reason_category=ReasonCategory.RUNTIME_TIMEOUT,
                                runner_metadata={},
                            ),
                        )
                    _replace_pool()

        with (
            _pool_lifecycle(),
            tqdm(
                total=total, desc="sweeper", unit="eval", disable=not show_progress
            ) as bar,
        ):

            def _record(
                outcome: str,
                candidate: Candidate | None,
                *,
                candidate_config: dict[str, Any] | None = None,
                reason: str = "",
                reason_category: ReasonCategory | None = None,
                runner_metadata: dict[str, Any] | None = None,
                provenance_metrics: dict[str, float] | None = None,
                retain_record: bool = True,
            ) -> None:
                tally[outcome] += 1
                if candidate is not None:
                    candidates.append(candidate)
                    bar.update(1)
                if retain_record:
                    if outcome == "feasible":
                        status = CandidateStatus.FEASIBLE
                    elif outcome == "unsupported":
                        status = CandidateStatus.UNSUPPORTED
                    elif reason_category is ReasonCategory.RUNTIME_TIMEOUT:
                        status = CandidateStatus.TIMED_OUT
                    elif outcome == "infeasible":
                        status = CandidateStatus.INFEASIBLE
                    else:
                        status = CandidateStatus.FAILED
                    snapshot = deepcopy(
                        candidate.config
                        if candidate is not None
                        else candidate_config or {}
                    )
                    record = CandidateRecord(
                        candidate_id=f"candidate-{len(candidate_records) + 1:06d}",
                        status=status,
                        config=snapshot,
                        used_gpus=(
                            candidate.used_gpus
                            if candidate is not None
                            else (
                                int(snapshot["used_gpus"])
                                if snapshot.get("used_gpus") is not None
                                else None
                            )
                        ),
                        score=candidate.score if candidate is not None else None,
                        metrics=(
                            deepcopy(candidate.metrics) if candidate is not None else {}
                        ),
                        objectives=(
                            deepcopy(candidate.objectives)
                            if candidate is not None
                            else None
                        ),
                        reason_category=(
                            None
                            if status is CandidateStatus.FEASIBLE
                            else reason_category or ReasonCategory.UNKNOWN
                        ),
                        reason=(None if status is CandidateStatus.FEASIBLE else reason),
                        provenance=make_candidate_provenance(
                            config,
                            snapshot,
                            metrics=(
                                provenance_metrics
                                if provenance_metrics is not None
                                else (
                                    candidate.metrics
                                    if candidate is not None
                                    else None
                                )
                            ),
                            runner_metadata=runner_metadata,
                        ),
                    )
                    candidate_records.append(record)
                    if candidate is not None:
                        record_id_by_candidate_object[id(candidate)] = (
                            record.candidate_id
                        )
                best = _best()
                bar.set_postfix(
                    feasible=tally["feasible"],
                    failed=tally["failed"],
                    best=("-" if best is None else f"{best:.4g}"),
                )

            round_no = 0
            for branch in branches:
                branch_stalled = False
                sampler = sampler_factory(
                    branch,
                    study_id=f"sweeper_{branch.deployment_mode}_{run_nonce}",
                    objectives=sampler_objectives,
                )
                bar.set_description(f"Sweeper {branch.deployment_mode}")
                for _ in range(sweep.max_rounds):
                    unique_this_round = 0
                    trial_attempts = 0
                    max_trial_attempts = (
                        per_round * 11
                    )  # requested batch + at most 10x replacement trials
                    while (
                        unique_this_round < per_round
                        and trial_attempts < max_trial_attempts
                    ):
                        ask_count = min(
                            per_round - unique_this_round,
                            max_trial_attempts - trial_attempts,
                        )
                        suggestions = sampler.suggest(
                            ask_count
                        )  # ask stays on the main process
                        if not suggestions:
                            break
                        trial_attempts += len(suggestions)

                        # Deduplicate against completed cache entries and within this ask batch.
                        # A duplicate trial still receives the cached measurement so f(z) remains
                        # deterministic, but only the first full sample reaches replay.
                        todo: list[tuple[Suggestion, _PreparedCandidate]] = []
                        prepared_by_key: dict[Any, _PreparedCandidate] = {}
                        primary_by_key: dict[Any, Suggestion] = {}
                        duplicates_by_key: dict[Any, list[Suggestion]] = {}
                        for suggestion in suggestions:
                            backend = suggestion.selection["backend"]
                            if backend not in branch.supported_backends.get(
                                suggestion.parallel_config, frozenset()
                            ):
                                reason = (
                                    f"backend {backend!r} does not support this parallel config"
                                )
                                sampler.observe_infeasible(
                                    suggestion,
                                    reason,
                                )
                                _record(
                                    "unsupported",
                                    None,
                                    candidate_config=_suggestion_snapshot(
                                        suggestion, config
                                    ),
                                    reason=reason,
                                    reason_category=ReasonCategory.BACKEND_TOPOLOGY,
                                )
                                continue

                            key = _suggestion_cache_key(suggestion, cache_context)
                            cached = replay_cache.get(key)
                            if cached is not None:
                                _, cached_metrics = cached
                                sampler.observe(suggestion, cached_metrics)
                                tally["cache_hit"] += 1
                                continue
                            if key in primary_by_key:
                                duplicates_by_key.setdefault(key, []).append(suggestion)
                                continue
                            primary_by_key[key] = suggestion

                        # Materialization stays on the main process: adapters see the
                        # resolved backend candidate and workers receive ReplaySpec only.
                        for key, suggestion in primary_by_key.items():
                            prepared, build_result = _materialize_one(
                                suggestion.selection,
                                suggestion.parallel_config,
                                config=config,
                                goal=goal,
                                providers=resolved_providers,
                                provider_plans=provider_plans,
                                runner_factory=runner_factory,
                            )
                            if build_result is not None:
                                candidate = build_result.candidate
                                observe_metrics = build_result.observe_metrics
                                outcome = build_result.outcome
                                reason = build_result.reason
                                assert candidate is None and observe_metrics is None
                                duplicates = duplicates_by_key.get(key, [])
                                sampler.observe_infeasible(suggestion, reason)
                                for duplicate in duplicates:
                                    sampler.observe_infeasible(duplicate, reason)
                                if outcome == "failed":
                                    failure_reasons[reason] = (
                                        failure_reasons.get(reason, 0)
                                        + 1
                                        + len(duplicates)
                                    )
                                _record(
                                    outcome,
                                    None,
                                    candidate_config=_suggestion_snapshot(
                                        suggestion, config
                                    ),
                                    reason=reason,
                                    reason_category=build_result.reason_category,
                                    runner_metadata=build_result.runner_metadata,
                                )
                                for _duplicate in duplicates:
                                    _record(
                                        outcome,
                                        None,
                                        retain_record=False,
                                    )
                                continue
                            assert prepared is not None
                            todo.append((suggestion, prepared))
                            prepared_by_key[key] = prepared

                        for suggestion, evaluation in _eval_batch(todo):
                            candidate = evaluation.candidate
                            observe_metrics = evaluation.observe_metrics
                            outcome = evaluation.outcome
                            reason = evaluation.reason
                            key = _suggestion_cache_key(suggestion, cache_context)
                            duplicates = duplicates_by_key.get(key, [])
                            if outcome in ("failed", "infeasible"):
                                sampler.observe_infeasible(suggestion, reason)
                                for duplicate in duplicates:
                                    sampler.observe_infeasible(duplicate, reason)
                                if outcome == "failed":
                                    failure_reasons[reason] = (
                                        failure_reasons.get(reason, 0)
                                        + 1
                                        + len(duplicates)
                                    )
                                _record(
                                    outcome,
                                    None,
                                    candidate_config=prepared_by_key[key].sample,
                                    reason=reason,
                                    reason_category=evaluation.reason_category,
                                    runner_metadata=evaluation.runner_metadata,
                                    provenance_metrics=evaluation.report_metrics,
                                )
                                for _duplicate in duplicates:
                                    _record(
                                        outcome,
                                        None,
                                        retain_record=False,
                                    )
                                continue

                            if candidate is None or observe_metrics is None:
                                raise RuntimeError(
                                    "Sweeper runner contract violation: a feasible outcome "
                                    "must include both a candidate and observation metrics"
                                )
                            sampler.observe(suggestion, observe_metrics)
                            replay_cache[key] = (candidate, dict(observe_metrics))
                            for duplicate in duplicates:
                                sampler.observe(duplicate, observe_metrics)
                                tally["cache_hit"] += 1
                            _record(
                                outcome,
                                candidate,
                                runner_metadata=evaluation.runner_metadata,
                                provenance_metrics=evaluation.report_metrics,
                            )
                            unique_this_round += 1
                    round_no += 1
                    if on_round is not None:
                        on_round(round_no, list(candidates))
                    if unique_this_round < per_round:
                        branch_stalled = True
                        if show_progress:
                            tqdm.write(
                                f"Sweeper {branch.deployment_mode} stopped early: projection stalled after "
                                f"{trial_attempts} Vizier trial(s), with {unique_this_round}/{per_round} "
                                "new replay configuration(s) in the round"
                            )
                        break
                if branch_stalled:
                    continue

        # Single-objective -> rank best-first by score; pareto -> the non-dominated front.
        selected_candidates = analyze_candidates(
            candidates,
            goal,
            osl=config.workload.osl or 1,
        )
        if not goal.is_pareto and top_n is not None:
            selected_candidates = selected_candidates[:top_n]
        if show_progress:
            replay_attempts = tally["feasible"] + tally["infeasible"] + tally["failed"]
            summary = (
                f"Sweeper done: {tally['feasible']}/{replay_attempts} replay attempt(s) feasible, "
                f"{tally['infeasible']} gated, {tally['unsupported']} backend-unsupported, "
                f"{tally['failed']} replay-failed, {tally['cache_hit']} cache hit(s)"
            )
            if not candidates:
                summary += " — NO feasible candidate (check backends / SLA / gpu_budget / replay errors)"
            elif goal.is_pareto:
                summary += (
                    f"; pareto front: {len(selected_candidates)} "
                    "non-dominated candidate(s)"
                )
            else:
                summary += f"; best {goal.target.value}={_best():.4g}"
            tqdm.write(summary)
            if failure_reasons:
                displayed = []
                for reason, count in list(failure_reasons.items())[:3]:
                    displayed.append(f"{reason} (x{count})" if count > 1 else reason)
                remaining = len(failure_reasons) - len(displayed)
                suffix = f" | +{remaining} more distinct reason(s)" if remaining else ""
                tqdm.write(
                    f"Sweeper failure reason(s): {' | '.join(displayed)}{suffix}"
                )
        selected_ids = [
            record_id_by_candidate_object[id(candidate)]
            for candidate in selected_candidates
        ]
        views = ResultViews(
            pareto_front=selected_ids if goal.is_pareto else [],
            top_n=[] if goal.is_pareto else selected_ids,
        )
        status_counts = dict.fromkeys(CandidateStatus, 0)
        for record in candidate_records:
            status_counts[record.status] += 1
        counts = SweepCounts(
            evaluated=(
                status_counts[CandidateStatus.FEASIBLE]
                + status_counts[CandidateStatus.INFEASIBLE]
                + status_counts[CandidateStatus.TIMED_OUT]
                + status_counts[CandidateStatus.FAILED]
            ),
            feasible=status_counts[CandidateStatus.FEASIBLE],
            infeasible=status_counts[CandidateStatus.INFEASIBLE],
            unsupported=status_counts[CandidateStatus.UNSUPPORTED],
            timed_out=status_counts[CandidateStatus.TIMED_OUT],
            failed=status_counts[CandidateStatus.FAILED],
            cache_hits=tally["cache_hit"],
        )
        return SweepResult(
            candidate_retention=retention,
            counts=counts,
            candidates=retain_candidate_records(
                candidate_records,
                retention=retention,
                views=views,
            ),
            views=views,
            provenance=make_run_provenance(
                config,
                search_strategy=SearchStrategy.OPTIMIZER_GUIDED,
                run_id=run_id,
            ),
        )
