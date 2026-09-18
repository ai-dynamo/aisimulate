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

from ..power import POWER_FIELDS, normalize_power_summary
from ..resources import ResourceLimitError
from .afd_perfmodel import AFDPerformanceModel, AICAFDPerformanceModel, attach_afd_measurements
from .config import Candidate, OptimizationGoal, OptimizationTarget, SmartSearchConfig
from .deploy import build_backend_deployment
from .discovery import resolve_providers
from .epd import add_encoder_choices, resolve_encoder_catalog
from .kv_estimate import resolve_backend_version
from .kv_load import InfeasibleKVCapacity, resolve_kv_load
from .provider import (
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
    validate_router_prefill_hardware,
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
from .score import aggregate_sla_violations, analyze_candidates, is_feasible, make_candidate, minimum_goodput_violations
from .search_space import BranchSpace, ConditionalDimensionSpace, enumerate_branches

logger = logging.getLogger(__name__)


# Result of evaluating one suggestion (no Vizier here). ``observe_metrics`` is
# fed to sampler.observe. Both failed and infeasible results are reported with
# observe_infeasible so invalid trials never steer the sampler as high scores.
@dataclass(frozen=True)
class _EvalResult:
    candidate: Candidate | None
    observe_metrics: dict[str, float | None] | None
    outcome: str
    reason: str
    reason_category: ReasonCategory | None
    runner_metadata: dict[str, Any]
    report_metrics: dict[str, float | None] | None = None
    config_snapshot: dict[str, Any] | None = None


@dataclass(frozen=True)
class _ReplayEvaluation:
    metrics: dict[str, float | None] | None
    metadata: dict[str, Any]
    outcome: str
    reason: str
    reason_category: ReasonCategory | None


@dataclass(frozen=True)
class _PreparedCandidate:
    sample: dict[str, Any]
    replay_spec: ReplaySpec
    prediction_config: dict[str, Any] | None = None


@dataclass
class _BranchSearchState:
    branch: BranchSpace
    sampler: BranchSampler
    budget: int | None
    attempts: int = 0
    stalled: bool = False


def _branch_sampler_seed(seed: int, deployment_mode: str) -> int:
    """Derive a stable seed from branch identity, independent of active-branch order."""

    offsets = {"agg": 0, "disagg": 1, "afd": 2, "afd+pd": 3}
    try:
        return seed + offsets[deployment_mode]
    except KeyError as exc:  # SearchSpace validation currently makes this unreachable.
        raise ValueError(f"unsupported deployment branch {deployment_mode!r}") from exc


_ADAPTER_PARAM_PREFIX = "adapter::"
_ADAPTER_PARAM_SEPARATOR = "::"


def _adapter_param(adapter_name: str, local_name: str) -> str:
    return f"{_ADAPTER_PARAM_PREFIX}{adapter_name}{_ADAPTER_PARAM_SEPARATOR}{local_name}"


def _adapter_selection(selection: Mapping[str, Any], adapter_name: str) -> dict[str, Any]:
    prefix = _adapter_param(adapter_name, "")
    return {key.removeprefix(prefix): deepcopy(value) for key, value in selection.items() if key.startswith(prefix)}


def _prepare_providers(
    config: SmartSearchConfig,
    *,
    injected: Mapping[str, SweepConfigProvider] | None,
    show_progress: bool,
) -> tuple[dict[str, SweepConfigProvider], dict[str, AdapterSearchPlan]]:
    invalid_names = [name for name in config.adapters if _ADAPTER_PARAM_SEPARATOR in name]
    if invalid_names:
        raise ValueError(
            f"adapter names cannot contain reserved separator {_ADAPTER_PARAM_SEPARATOR!r}: {invalid_names}"
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
        plan = provider.generate_search_space(deepcopy(config.adapters[name].search_space), context)
        _validate_search_plan(name, plan)
        # The adapter owns the object it returned and may reuse internal buffers
        # later. Take a complete core-owned snapshot at the ABI boundary.
        plans[name] = deepcopy(plan)
    configured_modes = set(config.search_space.deployment_mode)
    for name, plan in plans.items():
        unknown = (
            set(plan.fragment.choices_by_branch)
            | set(plan.fragment.float_ranges_by_branch)
            | set(plan.fragment.conditional_by_branch)
        ) - configured_modes
        if unknown:
            raise ValueError(f"adapter {name!r} returned unknown deployment branch(es): {sorted(unknown)}")
    return providers, plans


def _validate_search_plan(name: str, plan: Any) -> None:
    if not isinstance(plan, AdapterSearchPlan):
        raise TypeError(f"adapter {name!r} generate_search_space must return AdapterSearchPlan")
    if not isinstance(plan.fragment, SearchSpaceFragment):
        raise TypeError(f"adapter {name!r} returned an invalid SearchSpaceFragment")
    try:
        if type(plan.diagnostics) is not dict:
            raise TypeError("search diagnostics must be a dictionary")
        if type(plan.potential_runtime_hooks) is not tuple:
            raise TypeError("potential_runtime_hooks must be a tuple")
        _validate_search_fragment(plan.fragment)
        validate_json_value(plan.state, path=f"adapter {name!r} search plan state")
        validate_json_value(plan.diagnostics, path=f"adapter {name!r} search diagnostics")
        for index, hook in enumerate(plan.potential_runtime_hooks):
            _validate_runtime_hook(
                hook,
                path=f"adapter {name!r} potential hook {index}",
            )
        canonical_json(plan)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"adapter {name!r} returned an invalid/non-JSON search plan: {exc}") from exc


def _validate_search_fragment(fragment: SearchSpaceFragment) -> None:
    if type(fragment.api_version) is not int or fragment.api_version != SEARCH_SPACE_FRAGMENT_API_VERSION:
        raise ValueError(
            f"search-space fragment uses API version {fragment.api_version!r}; "
            f"aisimulate requires version {SEARCH_SPACE_FRAGMENT_API_VERSION}"
        )
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
                raise TypeError(f"categorical parameter {parameter!r} choices must be a list")
            validate_json_value(values, path=f"categorical parameter {parameter!r} choices")

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
                raise TypeError(f"continuous parameter {parameter!r} bounds must be a pair")
            if any(type(bound) not in (int, float) for bound in bounds) or not all(
                math.isfinite(float(bound)) for bound in bounds
            ):
                raise ValueError(f"continuous parameter {parameter!r} bounds must be finite numbers")
    if type(fragment.log_float_ranges_by_branch) is not dict:
        raise TypeError("log_float_ranges_by_branch must be a dictionary")
    for branch, parameters in fragment.log_float_ranges_by_branch.items():
        if type(branch) is not str or not branch:
            raise TypeError("log-range branch names must be non-empty strings")
        if type(parameters) is not list or any(type(parameter) is not str or not parameter for parameter in parameters):
            raise TypeError("log-range parameters must be non-empty string lists")
        unknown = set(parameters) - set(fragment.float_ranges_by_branch.get(branch, {}))
        if unknown:
            raise ValueError(f"log-range parameters need float bounds in branch {branch!r}: {sorted(unknown)}")
    if type(fragment.log_discrete_choices_by_branch) is not dict:
        raise TypeError("log_discrete_choices_by_branch must be a dictionary")
    for branch, parameters in fragment.log_discrete_choices_by_branch.items():
        if type(branch) is not str or not branch:
            raise TypeError("log-discrete branch names must be non-empty strings")
        if type(parameters) is not list or any(type(parameter) is not str or not parameter for parameter in parameters):
            raise TypeError("log-discrete parameters must be non-empty string lists")
        unknown = set(parameters) - set(fragment.choices_by_branch.get(branch, {}))
        if unknown:
            raise ValueError(f"log-discrete parameters need choices in branch {branch!r}: {sorted(unknown)}")

    if type(fragment.conditional_by_branch) is not dict:
        raise TypeError("conditional_by_branch must be a dictionary")
    for branch, conditions in fragment.conditional_by_branch.items():
        if type(branch) is not str or not branch:
            raise TypeError("conditional branch names must be non-empty strings")
        if type(conditions) is not list:
            raise TypeError(f"conditional branch {branch!r} must be a list")
        root_choices = fragment.choices_by_branch.get(branch, {})
        child_names: set[str] = set()
        for index, condition in enumerate(conditions):
            path = f"conditional branch {branch!r} entry {index}"
            if not isinstance(condition, ConditionalSearchSpace):
                raise TypeError(f"{path} must be a ConditionalSearchSpace")
            if type(condition.selector) is not str or not condition.selector:
                raise TypeError(f"{path} selector must be a non-empty string")
            if condition.selector not in root_choices:
                raise ValueError(f"{path} selector {condition.selector!r} is not a root categorical parameter")
            if type(condition.values) is not list or not condition.values:
                raise TypeError(f"{path} values must be a non-empty list")
            validate_json_value(condition.values, path=f"{path} values")
            unknown_values = [value for value in condition.values if value not in root_choices[condition.selector]]
            if unknown_values:
                raise ValueError(
                    f"{path} values are outside selector {condition.selector!r} choices: {unknown_values!r}"
                )
            if type(condition.choices) is not dict:
                raise TypeError(f"{path} choices must be a dictionary")
            if type(condition.float_ranges) is not dict:
                raise TypeError(f"{path} float_ranges must be a dictionary")
            overlap = set(condition.choices).intersection(condition.float_ranges)
            if overlap:
                raise ValueError(f"{path} children are both categorical and continuous: {sorted(overlap)}")
            names = set(condition.choices) | set(condition.float_ranges)
            root_overlap = names.intersection(root_choices)
            if root_overlap:
                raise ValueError(f"{path} children collide with root parameters: {sorted(root_overlap)}")
            duplicate_children = names.intersection(child_names)
            if duplicate_children:
                raise ValueError(f"{path} repeats conditional children: {sorted(duplicate_children)}")
            child_names.update(names)
            for parameter, values in condition.choices.items():
                if type(parameter) is not str or not parameter:
                    raise TypeError(f"{path} categorical child names must be non-empty strings")
                if type(values) is not list or not values:
                    raise TypeError(f"{path} categorical child {parameter!r} choices must be a non-empty list")
                validate_json_value(values, path=f"{path} categorical child {parameter!r} choices")
            for parameter, bounds in condition.float_ranges.items():
                if type(parameter) is not str or not parameter:
                    raise TypeError(f"{path} continuous child names must be non-empty strings")
                if type(bounds) is not tuple or len(bounds) != 2:
                    raise TypeError(f"{path} continuous child {parameter!r} bounds must be a pair")
                if any(type(bound) not in (int, float) for bound in bounds) or not all(
                    math.isfinite(float(bound)) for bound in bounds
                ):
                    raise ValueError(f"{path} continuous child {parameter!r} bounds must be finite numbers")
                if bounds[0] >= bounds[1]:
                    raise ValueError(f"{path} continuous child {parameter!r} needs low < high")
            for field_name, parameters, available in (
                ("log_float_ranges", condition.log_float_ranges, condition.float_ranges),
                ("log_discrete_choices", condition.log_discrete_choices, condition.choices),
            ):
                if type(parameters) is not list or any(
                    type(parameter) is not str or not parameter for parameter in parameters
                ):
                    raise TypeError(f"{path} {field_name} must be a non-empty string list")
                unknown = set(parameters) - set(available)
                if unknown:
                    raise ValueError(f"{path} {field_name} references unknown children: {sorted(unknown)}")


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
        raise TypeError(f"adapter {name!r} materialize_replay must return AdapterReplaySpec")
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
        raise TypeError(f"adapter {name!r} returned an invalid/non-JSON replay spec: {exc}") from exc


def _merge_adapter_spaces(
    branches: list[BranchSpace],
    plans: Mapping[str, AdapterSearchPlan],
) -> list[BranchSpace]:
    """Namespace and merge every adapter fragment into each core branch."""
    merged: list[BranchSpace] = []
    for branch in branches:
        choices = dict(branch.knob_choices)
        float_ranges = dict(branch.float_ranges)
        integer_ranges = dict(branch.integer_ranges)
        log_float_ranges = set(branch.log_float_ranges)
        log_integer_ranges = set(branch.log_integer_ranges)
        log_discrete_choices = set(branch.log_discrete_choices)
        conditional_dimensions = list(branch.conditional_dimensions)
        for name, plan in plans.items():
            local_choices = plan.fragment.choices_by_branch.get(branch.deployment_mode, {})
            local_ranges = plan.fragment.float_ranges_by_branch.get(branch.deployment_mode, {})
            local_log_ranges = set(plan.fragment.log_float_ranges_by_branch.get(branch.deployment_mode, []))
            local_log_discrete = set(plan.fragment.log_discrete_choices_by_branch.get(branch.deployment_mode, []))
            overlap = set(local_choices).intersection(local_ranges)
            if overlap:
                raise ValueError(
                    f"adapter {name!r} defined parameters as both categorical and continuous: {sorted(overlap)}"
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
                if local_name in local_log_discrete:
                    log_discrete_choices.add(_adapter_param(name, local_name))
            for local_name, bounds in local_ranges.items():
                if _ADAPTER_PARAM_SEPARATOR in local_name:
                    raise ValueError(
                        f"adapter {name!r} search parameter {local_name!r} contains "
                        f"reserved separator {_ADAPTER_PARAM_SEPARATOR!r}"
                    )
                low, high = bounds
                if low >= high:
                    raise ValueError(
                        f"adapter {name!r} search parameter {local_name!r} needs low < high, got {bounds!r}"
                    )
                float_ranges[_adapter_param(name, local_name)] = (low, high)
                if local_name in local_log_ranges:
                    log_float_ranges.add(_adapter_param(name, local_name))
            for condition in plan.fragment.conditional_by_branch.get(branch.deployment_mode, []):
                local_names = {condition.selector} | set(condition.choices) | set(condition.float_ranges)
                invalid_local_names = sorted(
                    local_name for local_name in local_names if _ADAPTER_PARAM_SEPARATOR in local_name
                )
                if invalid_local_names:
                    raise ValueError(
                        f"adapter {name!r} conditional search parameters contain reserved separator "
                        f"{_ADAPTER_PARAM_SEPARATOR!r}: {invalid_local_names}"
                    )
                selector = _adapter_param(name, condition.selector)
                if selector not in choices:
                    raise ValueError(
                        f"adapter {name!r} conditional selector {condition.selector!r} "
                        f"is not present in branch {branch.deployment_mode!r}"
                    )
                conditional_dimensions.append(
                    ConditionalDimensionSpace(
                        selector=selector,
                        values=tuple(deepcopy(condition.values)),
                        knob_choices={
                            _adapter_param(name, local_name): list(values)
                            for local_name, values in condition.choices.items()
                        },
                        float_ranges={
                            _adapter_param(name, local_name): bounds
                            for local_name, bounds in condition.float_ranges.items()
                        },
                        log_float_ranges=frozenset(
                            _adapter_param(name, local_name) for local_name in condition.log_float_ranges
                        ),
                        log_discrete_choices=frozenset(
                            _adapter_param(name, local_name) for local_name in condition.log_discrete_choices
                        ),
                    )
                )
        merged.append(
            replace(
                branch,
                knob_choices=choices,
                float_ranges=float_ranges,
                integer_ranges=integer_ranges,
                log_float_ranges=frozenset(log_float_ranges),
                log_integer_ranges=frozenset(log_integer_ranges),
                log_discrete_choices=frozenset(log_discrete_choices),
                conditional_dimensions=tuple(conditional_dimensions),
            )
        )
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
            tuple((_freeze(key), _freeze(item)) for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))),
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


def _with_encoder_snapshot(sample, selection, encoder_catalog):
    """Retain selected encoder evidence, without inventing an unresolved language shape."""
    sample = deepcopy(sample)
    key = selection.get("encoder_candidate")
    sample["encoder_candidate"] = deepcopy(key)
    sample["deployment_artifact_generation_supported"] = False
    sample["prediction_config_supported"] = False
    encoder = (encoder_catalog or {}).get(key) if isinstance(key, str) else None
    if encoder is not None:
        sample["encoder"] = asdict(encoder)
        language_gpus = sample.get("language_gpus", sample.get("used_gpus"))
        sample["language_gpus"] = language_gpus
        sample["used_gpus"] = language_gpus + encoder.total_gpus if type(language_gpus) is int else None
    return sample


def _suggestion_snapshot(suggestion: Suggestion, config: SmartSearchConfig, *, encoder_catalog=None) -> dict[str, Any]:
    """Best-effort JSON snapshot for a candidate rejected before replay."""

    try:
        sample = unroll_sample(
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
        sample = {
            **deepcopy(suggestion.selection),
            "parallel_config": parallel_payload,
        }
    if config.search_space.encoder is not None:
        sample = _with_encoder_snapshot(sample, suggestion.selection, encoder_catalog)
    return sample


def _materialize_one(
    selection: dict[str, Any],
    parallel_config: Any,
    *,
    config: SmartSearchConfig,
    goal: OptimizationGoal,
    providers: Mapping[str, SweepConfigProvider],
    provider_plans: Mapping[str, AdapterSearchPlan],
    runner_factory: RunnerFactory,
    afd_performance_model: AFDPerformanceModel | None = None,
    prediction_config_factory: Callable[[dict[str, Any], ReplaySpec], dict[str, Any]] | None = None,
    encoder_catalog: Mapping[str, Any] | None = None,
) -> tuple[_PreparedCandidate | None, _EvalResult | None]:
    """Build a complete replay specification on the main process."""
    epd_snapshot = None
    try:
        sample = unroll_sample(
            search_space=config.search_space,
            selection=selection,
            parallel_config=parallel_config,
        )
        encoder = None
        if config.search_space.encoder is not None:
            sample = _with_encoder_snapshot(sample, selection, encoder_catalog)
            epd_snapshot = deepcopy(sample)
            if "encoder" not in sample:
                raise ValueError("unknown encoder_candidate")
            encoder = encoder_catalog[selection["encoder_candidate"]]
            if encoder.backend != sample["backend"] or encoder.model != sample["model_name"]:
                raise ValueError("encoder candidate does not match language model/backend")
            if sample["used_gpus"] > config.search_space.gpu_budget:
                return None, _EvalResult(
                    candidate=None,
                    observe_metrics=None,
                    outcome="infeasible",
                    reason="language plus encoder pool exceeds gpu_budget",
                    reason_category=ReasonCategory.GPU_BUDGET,
                    runner_metadata={},
                    config_snapshot=epd_snapshot,
                )
        backend_version = config.search_space.backend_version
        if backend_version is None:
            if sample["deployment_mode"] == "disagg":
                role_hardware = {role: sample[f"{role}_hardware_sku"] for role in ("prefill", "decode")}
                if len(set(role_hardware.values())) == 1:
                    backend_version = resolve_backend_version(role_hardware["prefill"], selection["backend"])
                else:
                    role_versions = {
                        role: resolve_backend_version(hardware, selection["backend"])
                        for role, hardware in role_hardware.items()
                    }
                    if len(set(role_versions.values())) != 1:
                        raise ValueError(
                            "heterogeneous P/D hardware requires one common backend_version; "
                            f"latest versions for backend={selection['backend']!r} are {role_versions}. "
                            "Set search_space.backend_version to a version supported by both SKUs."
                        )
                    backend_version = role_versions["prefill"]
            else:
                backend_version = resolve_backend_version(config.search_space.hardware_sku, selection["backend"])
        # The resolved perf-model version is part of the evaluated contract. Keep it
        # on the candidate so downstream artifact generation cannot independently
        # select a different backend version.
        sample["backend_version"] = backend_version
        concurrency = config.workload.concurrency
        workload_payload = config.workload.model_dump(mode="json")
        if "traffic_load" in selection:
            load_value: int | float
            if config.workload.load_integer:
                load_value = max(1, round(float(selection["traffic_load"])))
            else:
                load_value = float(selection["traffic_load"])
            field = config.workload.load_search_field
            if field is None:
                raise ValueError("traffic_load selection has no load_search_field")
            workload_payload[field] = load_value
            sample["traffic_load"] = load_value
            sample[field] = load_value
            if field in {"concurrency", "replay_concurrency"}:
                concurrency = int(load_value)
        ratio_value = selection.get("kv_load_ratio")
        if ratio_value is None and config.workload.load_search_field == "kv_load_ratio":
            ratio_value = sample.get("kv_load_ratio")
        if ratio_value is not None:
            ratio = float(ratio_value)
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
            sample["kv_load_capacity_tokens"] = resolution.role_capacity_tokens[load_role]
            for role, tokens in resolution.role_capacity_tokens.items():
                sample[f"{role}_kv_capacity_tokens"] = tokens
        if concurrency is not None:
            # Preserve the concrete load on every candidate, including a fixed absolute
            # concurrency and one derived from kv_load_ratio.
            sample["concurrency"] = concurrency
        if encoder is not None:
            epd_snapshot = deepcopy(sample)
        backend_deployment = build_backend_deployment(sample, backend_version=backend_version, encoder=encoder)
        if sample["deployment_mode"] in {"afd", "afd+pd"}:
            backend_deployment = attach_afd_measurements(
                backend_deployment,
                sample=sample,
                workload=workload_payload,
                performance_model=afd_performance_model or AICAFDPerformanceModel(),
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
            if sample["deployment_mode"] == "disagg":
                validate_router_prefill_hardware(adapter_spec, sample["prefill_hardware_sku"])
            # Frozen dataclasses do not freeze nested JSON containers. Snapshot
            # the return value so an adapter can safely reuse an output buffer
            # without mutating candidates already prepared in this round.
            adapter_specs[name] = deepcopy(adapter_spec)
        replay_spec = ReplaySpec(
            backend_deployment=backend_deployment,
            workload=workload_payload,
            goal=goal.model_dump(mode="json"),
            concurrency=concurrency,
            adapters=adapter_specs,
        )
        canonical_json(replay_spec)
        runner_factory.capabilities().require_compatible(replay_spec)
        if adapter_specs:
            sample["adapters"] = {name: deepcopy(adapter_spec.config) for name, adapter_spec in adapter_specs.items()}
        prediction_config = (
            prediction_config_factory(deepcopy(sample), deepcopy(replay_spec))
            if prediction_config_factory is not None
            else None
        )
        if encoder is not None and prediction_config_factory is not None:
            from ..config.epd import validate_epd_prediction_mapping

            validate_epd_prediction_mapping(prediction_config, replay_spec)
            sample["prediction_config_supported"] = True
    except InfeasibleKVCapacity as exc:
        return None, _EvalResult(
            candidate=None,
            observe_metrics=None,
            outcome="infeasible",
            reason=f"candidate KV capacity infeasible: {exc}",
            reason_category=ReasonCategory.KV_CAPACITY,
            runner_metadata={},
            config_snapshot=deepcopy(sample) if epd_snapshot is not None else None,
        )
    except InfeasibleCandidate as exc:
        return None, _EvalResult(
            candidate=None,
            observe_metrics=None,
            outcome="infeasible",
            reason=f"candidate adapter selection infeasible: {exc}",
            reason_category=ReasonCategory.ADAPTER_CONSTRAINT,
            runner_metadata={},
            config_snapshot=deepcopy(sample) if epd_snapshot is not None else None,
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
            config_snapshot=deepcopy(sample) if epd_snapshot is not None else None,
        )
    return _PreparedCandidate(
        sample=sample,
        replay_spec=replay_spec,
        prediction_config=prediction_config,
    ), None


def _run_replay_detailed(spec: ReplaySpec, runner: Runner) -> _ReplayEvaluation:
    """Run one replay while retaining validated runner provenance metadata."""

    try:
        try:
            report = runner.run(spec)
        except ResourceLimitError:
            raise
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
                reason=(f"replay failed: TypeError: runner.run must return ReplayReport, got {type(report).__name__}"),
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
        metrics: dict[str, float | None] = {}
        for name, value in report.metrics.items():
            if type(name) is not str:
                return _ReplayEvaluation(
                    metrics=None,
                    metadata={},
                    outcome="failed",
                    reason=(f"replay failed: TypeError: runner metric names must be strings, got {name!r}"),
                    reason_category=ReasonCategory.INVALID_METRICS,
                )
            if value is None and name in POWER_FIELDS:
                metrics[name] = None
                continue
            if isinstance(value, bool) or not isinstance(value, Real):
                return _ReplayEvaluation(
                    metrics=None,
                    metadata={},
                    outcome="failed",
                    reason=(f"replay failed: TypeError: runner metric {name!r} must be a real number"),
                    reason_category=ReasonCategory.INVALID_METRICS,
                )
            normalized = float(value)
            if not math.isfinite(normalized):
                return _ReplayEvaluation(
                    metrics=None,
                    metadata={},
                    outcome="failed",
                    reason=(f"replay failed: ValueError: runner metric {name!r} must be finite"),
                    reason_category=ReasonCategory.INVALID_METRICS,
                )
            metrics[name] = normalized
        try:
            metrics.update(normalize_power_summary(metrics))
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
    except ResourceLimitError:
        raise
    except Exception as exc:  # fail closed if contract normalization itself regresses
        logger.exception("Sweeper candidate replay failed")
        return _ReplayEvaluation(
            metrics=None,
            metadata={},
            outcome="failed",
            reason=f"replay failed: {type(exc).__name__}: {exc}",
            reason_category=ReasonCategory.UNKNOWN,
        )


def _run_replay(spec: ReplaySpec, runner: Runner) -> tuple[dict[str, float] | None, str, str]:
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
    effective_targets = set(goal.resolved_pareto_objectives) if goal.is_pareto else {goal.target}
    if (
        effective_targets.intersection({OptimizationTarget.GOODPUT, OptimizationTarget.GOODPUT_PER_GPU})
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
            report_metrics=report,
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
                f"over gpu_budget: used_gpus={int(sample['used_gpus'])} > gpu_budget={config.search_space.gpu_budget}"
            ),
            reason_category=ReasonCategory.GPU_BUDGET,
            runner_metadata=replay_result.metadata,
            report_metrics=report,
        )
    if goal.requires_aggregate_sla:
        assert goal.sla is not None  # OptimizationGoal validates this invariant.
        violations = aggregate_sla_violations(report, goal.sla)
        if violations:
            return _EvalResult(
                candidate=None,
                observe_metrics=None,
                outcome="infeasible",
                reason=f"strict aggregate SLA violation: {'; '.join(violations)}",
                reason_category=ReasonCategory.SLA_CONSTRAINT,
                runner_metadata=replay_result.metadata,
                report_metrics=report,
            )
    load_violations = minimum_goodput_violations(report, goal.min_goodput_rps)
    if load_violations:
        missing_metric = "goodput_request_throughput_rps" not in report
        return _EvalResult(
            candidate=None,
            observe_metrics=None,
            outcome="failed" if missing_metric else "infeasible",
            reason=f"minimum goodput constraint: {'; '.join(load_violations)}",
            reason_category=ReasonCategory.RUNNER_CONTRACT if missing_metric else ReasonCategory.LOAD_CONSTRAINT,
            runner_metadata=replay_result.metadata,
            report_metrics=report,
        )
    # A completed replay with no qualifying latency samples is a modeled
    # infeasible outcome, not a runner failure. This preserves the intent of the
    # former worst-rank sentinel without putting non-finite scores in SweepResult.
    sample_metrics = {
        OptimizationTarget.E2E_LATENCY: "num_e2e_latency_samples",
        OptimizationTarget.TTFT: "num_ttft_samples",
    }
    for target in effective_targets:
        sample_metric = sample_metrics.get(target)
        sample_count = report.get(sample_metric) if sample_metric is not None else None
        if sample_metric is not None and (sample_count is None or sample_count <= 0.0):
            # Keep the historical sampler feedback even though the canonical
            # result records this completed replay as typed infeasible. Before
            # SweepResult, missing latency samples produced the worst-ranked
            # non-finite objective and still used sampler.observe().
            observation_candidate = make_candidate(
                sample,
                report,
                goal.target,
                pareto_objectives=(goal.resolved_pareto_objectives if goal.is_pareto else None),
            )
            observation_metrics = (
                dict(observation_candidate.objectives or {})
                if goal.is_pareto
                else {"objective": observation_candidate.score}
            )
            sample_detail = "missing" if sample_count is None else f"{sample_count:g}"
            return _EvalResult(
                candidate=None,
                observe_metrics=observation_metrics,
                outcome="infeasible",
                reason=f"{target.value} objective has no qualifying samples ({sample_metric}={sample_detail})",
                reason_category=ReasonCategory.NO_SAMPLES,
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
        observe_metrics = {"objective": candidate.score}  # single metric, pre-signed higher-is-better
    if prepared.prediction_config is not None:
        candidate = candidate.model_copy(update={"prediction_config": deepcopy(prepared.prediction_config)})
    non_finite_objectives = [name for name, value in (candidate.objectives or {}).items() if not math.isfinite(value)]
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
        prediction_config_factory: Callable[[dict[str, Any], ReplaySpec], dict[str, Any]] | None = None,
        afd_performance_model: AFDPerformanceModel | None = None,
    ) -> None:
        self._runner_factory = runner_factory
        self._providers = dict(providers or {})
        self._sampler_factory = sampler_factory
        self._show_progress = show_progress
        self._prediction_config_factory = prediction_config_factory
        self._afd_performance_model = afd_performance_model or AICAFDPerformanceModel()

    def run(
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
        prediction_config_factory = self._prediction_config_factory
        afd_performance_model = self._afd_performance_model

        goal = config.goal
        capabilities = runner_factory.capabilities()
        capabilities.require_replay_spec_version(REPLAY_SPEC_API_VERSION)
        encoder_catalog = None
        if config.search_space.encoder is not None:
            if not capabilities.supports_analytical_epd:
                raise ValueError("runner does not support analytical EPD")
            encoder_catalog = resolve_encoder_catalog(config)

        # Preserve the legacy preflight order: reject an impossible backend/topology
        # search before adapters perform any potentially expensive preparation.
        branches = enumerate_branches(
            config,
            max_seq_len=config.search_space.context_length,
            runner_capabilities=capabilities,
        )
        if encoder_catalog is not None:
            branches = add_encoder_choices(branches, encoder_catalog)
        resolved_providers, provider_plans = _prepare_providers(config, injected=providers, show_progress=show_progress)
        for name, plan in provider_plans.items():
            unsupported = [hook for hook in plan.potential_runtime_hooks if not capabilities.supports_hook(hook)]
            if unsupported:
                labels = ", ".join(f"{hook.provider}:{hook.kind}@{hook.api_version}" for hook in unsupported)
                raise ValueError(
                    f"runner is incompatible with configured adapter {name!r}; unsupported runtime hook(s): {labels}"
                )

        branches = _merge_adapter_spaces(branches, provider_plans)

        sweep = config.sweep
        per_round = sweep.candidates_per_round or sweep.parallel_evals
        if sweep.max_trials is not None and sweep.max_trials < len(branches):
            raise ValueError(
                f"optimizer.max_trials must be at least the number of active deployment branches ({len(branches)})"
            )
        # Legacy runs target successful unique evaluations; unified-CLI runs use
        # an exact global suggestion budget, including failures and cache hits.
        total = sweep.max_trials if sweep.max_trials is not None else len(branches) * sweep.max_rounds * per_round
        branch_budgets: list[int | None]
        if sweep.max_trials is None:
            branch_budgets = [None] * len(branches)
        else:
            base, remainder = divmod(sweep.max_trials, len(branches))
            branch_budgets = [base + (1 if index < remainder else 0) for index in range(len(branches))]
        candidates: list[Candidate] = []
        tally = {
            "resource_limited": 0,
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
            [(t.value, t.maximize) for t in goal.resolved_pareto_objectives] if goal.is_pareto else None
        )
        cache_context = _freeze(
            {
                "search_space": config.search_space.model_dump(mode="python"),
                "adapters": {name: request.model_dump(mode="python") for name, request in config.adapters.items()},
                "workload": config.workload.model_dump(mode="python"),
                "goal": goal.model_dump(mode="python"),
                "provider_plans": provider_plans,
            }
        )
        replay_cache: dict[Any, tuple[Candidate | None, dict[str, float]]] = {}

        def _best() -> float | None:
            return max((c.score for c in candidates), default=None)

        # Parallel across worker processes when parallel_evals > 1. Spawn keeps
        # runner runtimes isolated and lets each worker reuse one runner instance.
        resource_aware = callable(getattr(runner_factory, "admit_wave", None))
        use_pool = (
            resource_aware
            or (sweep.parallel_evals > 1 and per_round > 1)
            or (sweep.max_trials is not None and sweep.max_eval_seconds is not None)
        )
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
        pool_box: list[Any] = [_new_pool() if use_pool and not resource_aware else None]

        def _terminate_pool(pool: ProcessPoolExecutor | None) -> None:
            if pool is None:
                return
            from ..supervision import terminate_pool

            terminate_pool(pool)

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
                # Allow Runner finalizers to finish, then stop workers that exceed
                # the grace period instead of hanging the calling process.
                if pool_box[0] is not None:
                    from ..supervision import close_pool

                    close_pool(pool_box[0])
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
            if resource_aware:
                from ..resource_scheduler import InterruptedEvaluation, evaluate_waves

                for index, result in evaluate_waves(
                    [prepared.replay_spec for _, prepared in todo],
                    factory=runner_factory,
                    initializer=_init_worker,
                    evaluate=_worker_eval,
                    workers=worker_count,
                    timeout=max_eval_seconds,
                ):
                    suggestion, prepared = todo[index]
                    if isinstance(result, InterruptedEvaluation):
                        evaluation = _EvalResult(
                            candidate=None,
                            observe_metrics=None,
                            outcome="resource_limited" if result.resource_limited else "infeasible",
                            reason=result.reason,
                            reason_category=(
                                ReasonCategory.RESOURCE_LIMIT
                                if result.resource_limited
                                else ReasonCategory.RUNTIME_TIMEOUT
                            ),
                            runner_metadata=result.metadata,
                        )
                    else:
                        evaluation = _score_prepared(prepared, result, config=config, goal=goal)
                    yield suggestion, evaluation
                return

            if pool_box[0] is None:
                assert sequential_runner is not None
                for suggestion, prepared in todo:
                    yield (
                        suggestion,
                        _score_prepared(
                            prepared,
                            _run_replay_detailed(prepared.replay_spec, sequential_runner),
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
                deadline = time.monotonic() + max_eval_seconds if max_eval_seconds else None
                while pending:
                    remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                    done, pending = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
                    if not done:
                        break
                    for future in done:
                        try:
                            replay_result = future.result()
                        except ResourceLimitError:
                            raise
                        except BrokenProcessPool as exc:
                            raise _pool_error("collecting a candidate result") from exc
                        except Exception as exc:
                            raise _pool_error(f"collecting a candidate result ({type(exc).__name__}: {exc})") from exc
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
            tqdm(total=total, desc="sweeper", unit="eval", disable=not show_progress) as bar,
        ):

            def _record(
                outcome: str,
                candidate: Candidate | None,
                *,
                candidate_config: dict[str, Any] | None = None,
                prediction_config: dict[str, Any] | None = None,
                reason: str = "",
                reason_category: ReasonCategory | None = None,
                runner_metadata: dict[str, Any] | None = None,
                provenance_metrics: dict[str, float | None] | None = None,
                replay_spec: ReplaySpec | None = None,
            ) -> None:
                tally[outcome] += 1
                if candidate is not None:
                    candidates.append(candidate)
                    bar.update(1)
                if outcome == "feasible":
                    status = CandidateStatus.FEASIBLE
                elif outcome == "resource_limited":
                    status = CandidateStatus.RESOURCE_LIMITED
                elif outcome == "unsupported":
                    status = CandidateStatus.UNSUPPORTED
                elif reason_category is ReasonCategory.RUNTIME_TIMEOUT:
                    status = CandidateStatus.TIMED_OUT
                elif outcome == "infeasible":
                    status = CandidateStatus.INFEASIBLE
                else:
                    status = CandidateStatus.FAILED
                snapshot = deepcopy(candidate.config if candidate is not None else candidate_config or {})
                reported_metrics = (
                    provenance_metrics
                    if provenance_metrics is not None
                    else (candidate.metrics if candidate is not None else None)
                )
                record_metrics = deepcopy(reported_metrics) if reported_metrics is not None else {}
                record = CandidateRecord(
                    candidate_id=f"candidate-{len(candidate_records) + 1:06d}",
                    status=status,
                    config=snapshot,
                    prediction_config=deepcopy(
                        candidate.prediction_config if candidate is not None else prediction_config
                    ),
                    used_gpus=(
                        candidate.used_gpus
                        if candidate is not None
                        else (int(snapshot["used_gpus"]) if snapshot.get("used_gpus") is not None else None)
                    ),
                    score=candidate.score if candidate is not None else None,
                    metrics=record_metrics,
                    objectives=(deepcopy(candidate.objectives) if candidate is not None else None),
                    reason_category=(
                        None if status is CandidateStatus.FEASIBLE else reason_category or ReasonCategory.UNKNOWN
                    ),
                    reason=(None if status is CandidateStatus.FEASIBLE else reason),
                    provenance=make_candidate_provenance(
                        snapshot,
                        replay_spec=replay_spec,
                        metrics=reported_metrics,
                        runner_metadata=runner_metadata,
                    ),
                )
                candidate_records.append(record)
                if resource_aware:
                    from ..supervision import checkpoint

                    checkpoint("candidate_completed", record.model_dump(mode="json"))
                if candidate is not None:
                    record_id_by_candidate_object[id(candidate)] = record.candidate_id
                best = _best()
                bar.set_postfix(
                    feasible=tally["feasible"],
                    failed=tally["failed"],
                    best=("-" if best is None else f"{best:.4g}"),
                )

            branch_states: list[_BranchSearchState] = []
            for branch_index, branch in enumerate(branches):
                sampler_kwargs: dict[str, Any] = {
                    "study_id": f"sweeper_{branch.deployment_mode}_{run_nonce}",
                    "objectives": sampler_objectives,
                }
                if sweep.max_trials is not None:
                    sampler_kwargs.update(
                        algorithm=sweep.algorithm,
                        seed=_branch_sampler_seed(sweep.seed, branch.deployment_mode),
                    )
                branch_states.append(
                    _BranchSearchState(
                        branch=branch,
                        sampler=sampler_factory(branch, **sampler_kwargs),
                        budget=branch_budgets[branch_index],
                    )
                )

            round_no = 0

            def _run_branch_round(state: _BranchSearchState) -> None:
                nonlocal round_no

                branch = state.branch
                sampler = state.sampler
                bar.set_description(f"Sweeper {branch.deployment_mode}")
                remaining_budget = None if state.budget is None else state.budget - state.attempts
                round_target = per_round if remaining_budget is None else min(per_round, remaining_budget)
                if round_target <= 0:
                    return

                unique_this_round = 0
                trial_attempts = 0
                max_trial_attempts = per_round * 11  # requested batch + at most 10x replacement trials
                while unique_this_round < round_target and trial_attempts < max_trial_attempts:
                    ask_count = min(
                        round_target - unique_this_round,
                        max_trial_attempts - trial_attempts,
                    )
                    if state.budget is not None:
                        ask_count = min(ask_count, state.budget - state.attempts)
                    if ask_count <= 0:
                        break
                    suggestions = sampler.suggest(ask_count)  # ask stays on the main process
                    if not suggestions:
                        break
                    trial_attempts += len(suggestions)
                    state.attempts += len(suggestions)

                    # Deduplicate against completed cache entries and within this ask batch.
                    # A duplicate trial still receives the cached measurement so f(z) remains
                    # deterministic, but only the first full sample reaches replay.
                    todo: list[tuple[Suggestion, _PreparedCandidate]] = []
                    prepared_by_key: dict[Any, _PreparedCandidate] = {}
                    primary_by_key: dict[Any, Suggestion] = {}
                    duplicates_by_key: dict[Any, list[Suggestion]] = {}
                    for suggestion in suggestions:
                        if suggestion.infeasible_reason is not None:
                            reason = suggestion.infeasible_reason
                            sampler.observe_infeasible(suggestion, reason)
                            _record(
                                "infeasible",
                                None,
                                candidate_config=_suggestion_snapshot(
                                    suggestion, config, encoder_catalog=encoder_catalog
                                ),
                                reason=reason,
                                reason_category=ReasonCategory.PARALLEL_PROJECTION,
                            )
                            continue
                        backend = suggestion.selection["backend"]
                        if backend not in branch.supported_backends.get(suggestion.parallel_config, frozenset()):
                            reason = f"backend {backend!r} does not support this parallel config"
                            sampler.observe_infeasible(
                                suggestion,
                                reason,
                            )
                            _record(
                                "unsupported",
                                None,
                                candidate_config=_suggestion_snapshot(
                                    suggestion, config, encoder_catalog=encoder_catalog
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
                            afd_performance_model=afd_performance_model,
                            prediction_config_factory=prediction_config_factory,
                            encoder_catalog=encoder_catalog,
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
                                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                            _record(
                                outcome,
                                None,
                                candidate_config=build_result.config_snapshot
                                or _suggestion_snapshot(suggestion, config, encoder_catalog=encoder_catalog),
                                reason=reason,
                                reason_category=build_result.reason_category,
                                runner_metadata=build_result.runner_metadata,
                            )
                            tally["cache_hit"] += len(duplicates)
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
                        if outcome in ("failed", "infeasible", "resource_limited"):
                            if outcome == "resource_limited":
                                # A terminal host refusal consumes its trial without a fabricated score.
                                unique_this_round += 1
                            preserve_ranked_observation = (
                                evaluation.reason_category is ReasonCategory.NO_SAMPLES and observe_metrics is not None
                            )
                            if preserve_ranked_observation:
                                sampler.observe(suggestion, observe_metrics)
                                replay_cache[key] = (None, dict(observe_metrics))
                                for duplicate in duplicates:
                                    sampler.observe(duplicate, observe_metrics)
                                # A no-sample result still completed one unique
                                # replay, matching the previous round semantics.
                                unique_this_round += 1
                            else:
                                sampler.observe_infeasible(suggestion, reason)
                                for duplicate in duplicates:
                                    sampler.observe_infeasible(duplicate, reason)
                            if outcome == "failed":
                                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                            prepared = prepared_by_key[key]
                            _record(
                                outcome,
                                None,
                                candidate_config=prepared.sample,
                                prediction_config=prepared.prediction_config,
                                reason=reason,
                                reason_category=evaluation.reason_category,
                                runner_metadata=evaluation.runner_metadata,
                                provenance_metrics=evaluation.report_metrics,
                                replay_spec=prepared.replay_spec,
                            )
                            tally["cache_hit"] += len(duplicates)
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
                            replay_spec=prepared_by_key[key].replay_spec,
                        )
                        unique_this_round += 1

                round_no += 1
                if on_round is not None:
                    on_round(round_no, list(candidates))
                if unique_this_round < round_target:
                    state.stalled = True
                    if show_progress:
                        tqdm.write(
                            f"Sweeper {branch.deployment_mode} stopped early: projection stalled after "
                            f"{trial_attempts} Vizier trial(s), with {unique_this_round}/{round_target} "
                            "new replay configuration(s) in the round"
                        )

            if sweep.max_trials is None:
                # Preserve the legacy SDK's branch-major round ordering.
                for state in branch_states:
                    for _ in range(sweep.max_rounds):
                        if state.stalled:
                            break
                        _run_branch_round(state)
            else:
                # Unified CLI: give each active branch one batch per cycle. This
                # prevents either study from consuming its full allocation before
                # the other receives suggestions while retaining parallel fan-out.
                for _ in range(sweep.max_rounds):
                    progressed = False
                    for state in branch_states:
                        if state.stalled or (state.budget is not None and state.attempts >= state.budget):
                            continue
                        _run_branch_round(state)
                        progressed = True
                    if not progressed:
                        break

        # Strict filtering precedes scalar ranking or Pareto dominance.
        selected_candidates = analyze_candidates(candidates, goal)
        if not goal.is_pareto and top_n is not None:
            selected_candidates = selected_candidates[:top_n]
        if show_progress:
            replay_attempts = tally["feasible"] + tally["infeasible"] + tally["failed"]
            summary = (
                f"Sweeper done: {tally['feasible']}/{replay_attempts} replay attempt(s) feasible, "
                f"{tally['infeasible']} gated, {tally['unsupported']} backend-unsupported, "
                f"{tally['failed']} replay-failed, {tally['resource_limited']} resource-limited, "
                f"{tally['cache_hit']} cache hit(s)"
            )
            if not candidates:
                summary += " — NO feasible candidate (check backends / SLA / gpu_budget / replay errors)"
            elif goal.is_pareto:
                summary += f"; pareto front: {len(selected_candidates)} non-dominated candidate(s)"
            else:
                summary += f"; best {goal.target.value}={_best():.4g}"
            tqdm.write(summary)
            if failure_reasons:
                displayed = []
                for reason, count in list(failure_reasons.items())[:3]:
                    displayed.append(f"{reason} (x{count})" if count > 1 else reason)
                remaining = len(failure_reasons) - len(displayed)
                suffix = f" | +{remaining} more distinct reason(s)" if remaining else ""
                tqdm.write(f"Sweeper failure reason(s): {' | '.join(displayed)}{suffix}")
        selected_ids = [record_id_by_candidate_object[id(candidate)] for candidate in selected_candidates]
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
            resource_limited=status_counts[CandidateStatus.RESOURCE_LIMITED],
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
