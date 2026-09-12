# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vizier-backed sampler over a :class:`aisimulate.sweeper.search_space.BranchSpace`.

One Vizier study per branch (per the design). The study's parameters are:

- structured parallel features projected onto the branch's KV-feasible config
  pool.
- a continuous ``kv_load_ratio`` when a Pareto workload supplies a range.
- one parameter per multi-choice searchable knob (categorical for string choices
  and discrete for numeric choices). Adapter parameters are already namespaced
  when they reach the sampler. Single-choice knobs are injected as constants
  rather than Vizier parameters.

``suggest`` decodes each trial into a ``selection`` dict (the shape
:func:`aisimulate.sweeper.sample.unroll_sample` consumes) plus the chosen parallel-config
object; ``observe`` reports the (higher-is-better) score back to Vizier.

The sampler is swappable behind the :class:`BranchSampler` Protocol so another
optimizer can replace Vizier without touching orchestration.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from itertools import product
from typing import Any, Protocol

from ._quiet import configure_vizier_runtime
from .afd_parallel import AFDParallelConfig
from .parallel_enum import DisaggParallelConfig, ReplicaParallelConfig
from .parallel_projection import (
    InfeasibleParallelSelection,
    ParallelConfigProjector,
    ParallelProjection,
)
from .search_space import BranchSpace

_CONSTANT_PARAM = "_sweeper_constant"
_METRIC = "objective"
_SWEEPER_VIZIER_ALGO_ENV = "AISIMULATE_SWEEPER_VIZIER_ALGO"
_LEGACY_SPICA_VIZIER_ALGO_ENV = "SPICA_VIZIER_ALGO"


def _vizier_algorithm() -> str:
    """Resolve the experimental Vizier override with legacy compatibility."""
    if _SWEEPER_VIZIER_ALGO_ENV in os.environ:
        return os.environ[_SWEEPER_VIZIER_ALGO_ENV]
    return os.environ.get(_LEGACY_SPICA_VIZIER_ALGO_ENV, "DEFAULT")


@dataclass
class Suggestion:
    """One sampled candidate: the unroll selection + chosen parallel config,
    plus an opaque handle the sampler uses to report the score."""

    selection: dict[str, Any]
    parallel_config: ReplicaParallelConfig | DisaggParallelConfig | AFDParallelConfig
    handle: Any = field(repr=False)
    projection: ParallelProjection | None = field(default=None, repr=False)
    infeasible_reason: str | None = None


def _project_parallel(
    projector: ParallelConfigProjector,
    params: dict[str, Any],
    backend: str,
) -> tuple[
    ReplicaParallelConfig | DisaggParallelConfig | AFDParallelConfig,
    ParallelProjection | None,
    str | None,
]:
    """Project a suggestion while preserving infeasible samples for ask/tell."""

    try:
        projection = projector.project(params, backend)
        return projection.config, projection, None
    except InfeasibleParallelSelection as exc:
        fallback = next(
            (
                config
                for config in projector.branch.parallel_configs
                if backend in projector.branch.supported_backends.get(config, frozenset())
            ),
            None,
        )
        if fallback is None:
            raise
        return fallback, None, str(exc)


class BranchSampler(Protocol):
    """Stateful optimizer over one branch (swappable: Vizier, random, ...)."""

    branch: BranchSpace

    def suggest(self, count: int) -> list[Suggestion]: ...

    def observe(self, suggestion: Suggestion, metrics: dict[str, float]) -> None: ...

    def observe_infeasible(self, suggestion: Suggestion, reason: str) -> None: ...


def _decoder_for(choices: list[Any]) -> Callable[[Any], Any]:
    """How to turn a Vizier trial value back into the knob's native type."""
    if all(isinstance(c, str) for c in choices):
        return str  # categorical -> already a str
    if all(isinstance(c, int) and not isinstance(c, bool) for c in choices):
        return lambda v: round(float(v))  # discrete int (Vizier stores float)
    return float  # discrete float


def _index_decoder(choices: list[Any]) -> Callable[[Any], Any]:
    """Decode a categorical index back to an arbitrary JSON choice."""
    return lambda v: choices[round(float(v))]


def _vizier_modules() -> tuple[Any, Any]:
    """Import the optional Vizier runtime only after the caller opts into sampling.

    Keeping this dependency behind sampler construction lets configuration and package
    imports remain lightweight and free of Vizier/JAX process-wide initialization.
    """
    from vizier.service import clients
    from vizier.service import pyvizier as vz

    return clients, vz


class VizierBranchSampler:
    """A Vizier study over one :class:`BranchSpace`.

    ``objectives`` is the list of ``(metric_name, maximize)`` the study optimizes; the
    default is a single ``("objective", maximize=True)`` (the caller pre-signs the score).
    Pass >=2 objectives for a **multi-objective / Pareto** study: the ``DEFAULT`` algorithm
    (GP-UCB-PE) optimizes the Pareto tradeoff via hypervolume scalarization, and ``observe``
    reports every objective's raw value in one measurement.
    """

    def __init__(
        self,
        branch: BranchSpace,
        *,
        study_id: str,
        objectives: list[tuple[str, bool]] | None = None,
        algorithm: str = "bayesian",
        seed: int = 42,
    ):
        del seed  # Vizier's embedded DEFAULT designer owns its internal RNG.
        configure_vizier_runtime()
        clients, vz = _vizier_modules()

        # Vizier's embedded service defaults its SQLite database to the installed
        # package directory, which can be read-only. Sweeper studies are process-local
        # unless the caller explicitly configures Vizier storage, so use an in-memory
        # database for the default embedded-service path.
        if "database_url" not in clients.environment_variables.servicer_kwargs:
            clients.environment_variables.servicer_use_sql_ram()

        self.branch = branch
        self._objectives = objectives or [(_METRIC, True)]
        self._decoders: dict[str, Callable[[Any], Any]] = {}
        self._constants: dict[str, Any] = {}
        self._parallel_projector = ParallelConfigProjector(branch)
        self._parallel_pinned = len(branch.parallel_configs) == 1 and not branch.parallel_independent_choices

        problem = vz.ProblemStatement()
        root = problem.search_space.root
        if not self._parallel_pinned:
            for parameter in self._parallel_projector.parameters:
                if parameter.is_constant:
                    continue
                if parameter.kind == "float":
                    root.add_float_param(
                        parameter.name,
                        min_value=parameter.minimum,
                        max_value=parameter.maximum,
                        default_value=parameter.default,
                    )
                elif parameter.kind == "integer":
                    root.add_int_param(
                        parameter.name,
                        min_value=round(parameter.minimum),
                        max_value=round(parameter.maximum),
                        default_value=round(parameter.default),
                        scale_type=(vz.ScaleType.LOG if parameter.log_scale else vz.ScaleType.LINEAR),
                    )
                elif parameter.kind == "discrete":
                    root.add_discrete_param(
                        parameter.name,
                        feasible_values=parameter.values,
                        default_value=parameter.default,
                        scale_type=vz.ScaleType.LOG if parameter.log_scale else vz.ScaleType.LINEAR,
                    )
                else:
                    root.add_categorical_param(
                        parameter.name,
                        feasible_values=parameter.values,
                        default_value=parameter.default,
                    )
        for knob, (minimum, maximum) in branch.float_ranges.items():
            root.add_float_param(
                knob,
                min_value=minimum,
                max_value=maximum,
                default_value=(minimum + maximum) / 2.0,
                scale_type=(vz.ScaleType.LOG if knob in branch.log_float_ranges else vz.ScaleType.LINEAR),
            )
            self._decoders[knob] = float
        for knob, (minimum, maximum) in branch.integer_ranges.items():
            root.add_int_param(
                knob,
                min_value=minimum,
                max_value=maximum,
                default_value=minimum,
                scale_type=(vz.ScaleType.LOG if knob in branch.log_integer_ranges else vz.ScaleType.LINEAR),
            )
            self._decoders[knob] = lambda value: round(float(value))
        for knob, choices in branch.knob_choices.items():
            # Dedupe scalar choices before passing them to Vizier. Structured and
            # heterogeneous choices are encoded by index below, so duplicate decoded
            # values are harmless and do not need hashing.
            all_strings = all(isinstance(choice, str) for choice in choices)
            all_numeric = all(isinstance(choice, (int, float)) and not isinstance(choice, bool) for choice in choices)
            if all_strings or all_numeric:
                choices = list(dict.fromkeys(choices))
            if len(choices) <= 1:
                if choices:
                    self._constants[knob] = choices[0]  # fixed -> inject, not a param
                continue
            if all_strings:
                kwargs = {"default_value": choices[0]} if knob == "backend" else {}
                root.add_categorical_param(knob, list(choices), **kwargs)
                self._decoders[knob] = _decoder_for(choices)
            elif all_numeric:
                root.add_discrete_param(
                    knob,
                    sorted(float(c) for c in choices),
                    scale_type=(vz.ScaleType.LOG if knob in branch.log_discrete_choices else vz.ScaleType.LINEAR),
                )
                self._decoders[knob] = _decoder_for(choices)
            else:
                # Preserve arbitrary JSON choices (including booleans, null, lists,
                # dicts, and heterogeneous values) through a categorical index.
                root.add_categorical_param(knob, [str(i) for i in range(len(choices))])
                self._decoders[knob] = _index_decoder(choices)

        # GP designers reject a zero-dimensional study. A fully pinned request still
        # needs one internal constant so it can use the same ask/tell lifecycle.
        if problem.search_space.num_parameters() == 0:
            root.add_categorical_param(_CONSTANT_PARAM, ["0"], default_value="0")

        for name, maximize in self._objectives:
            goal = vz.ObjectiveMetricGoal.MAXIMIZE if maximize else vz.ObjectiveMetricGoal.MINIMIZE
            problem.metric_information.append(vz.MetricInformation(name=name, goal=goal))
        study_config = vz.StudyConfig.from_problem(problem)
        # EXPERIMENT (env-gated; default DEFAULT = GP-bandit). The multi-objective GP suggest
        # can spin/hang at low observation counts; RANDOM_SEARCH bypasses the GP (instant
        # suggest, uniform exploration) to cover the curve ends without that stall.
        study_config.algorithm = "RANDOM_SEARCH" if algorithm == "random" else _vizier_algorithm()
        self._study = clients.Study.from_study_config(study_config, owner="sweeper", study_id=study_id)

    def suggest(self, count: int) -> list[Suggestion]:
        suggestions: list[Suggestion] = []
        for trial in self._study.suggest(count=count):
            params = {name: getattr(value, "value", value) for name, value in dict(trial.parameters).items()}
            # backend is a searched knob now (in knob_choices) -> comes via _constants
            # (single backend) or _decoders (multiple), not a per-branch constant.
            selection: dict[str, Any] = {
                "deployment_mode": self.branch.deployment_mode,
                **self._constants,
            }
            for knob, decode in self._decoders.items():
                selection[knob] = decode(params[knob])
            if self._parallel_pinned:
                parallel_config = self.branch.parallel_configs[0]
                projection = None
                infeasible_reason = None
            else:
                parallel_config, projection, infeasible_reason = _project_parallel(
                    self._parallel_projector, params, selection["backend"]
                )
            suggestions.append(
                Suggestion(
                    selection=selection,
                    parallel_config=parallel_config,
                    handle=trial,
                    projection=projection,
                    infeasible_reason=infeasible_reason,
                )
            )
        return suggestions

    @staticmethod
    def _update_projection_metadata(suggestion: Suggestion) -> None:
        if suggestion.projection is None:
            return
        _, vz = _vizier_modules()

        metadata = vz.Metadata()
        metadata["sweeper_projection"] = json.dumps(suggestion.projection.metadata(), sort_keys=True)
        suggestion.handle.update_metadata(metadata)

    def observe(self, suggestion: Suggestion, metrics: dict[str, float]) -> None:
        _, vz = _vizier_modules()

        self._update_projection_metadata(suggestion)
        suggestion.handle.complete(vz.Measurement(metrics={k: float(v) for k, v in metrics.items()}))

    def observe_infeasible(self, suggestion: Suggestion, reason: str) -> None:
        """Mark a candidate that could not be evaluated (e.g. replay error) so the
        study still closes the trial and the optimizer moves on."""
        _, vz = _vizier_modules()

        self._update_projection_metadata(suggestion)
        suggestion.handle.complete(vz.Measurement(), infeasible_reason=reason)


class RandomBranchSampler:
    """Seeded random sampler over the same latent projection used by Vizier."""

    def __init__(self, branch: BranchSpace, *, seed: int) -> None:
        self.branch = branch
        self._rng = random.Random(seed)
        self._projector = ParallelConfigProjector(branch)
        self._parallel_pinned = len(branch.parallel_configs) == 1 and not branch.parallel_independent_choices

    def _parameter_value(self, parameter) -> Any:
        if parameter.kind == "float":
            if parameter.log_scale:
                return 2.0 ** self._rng.uniform(math.log2(parameter.minimum), math.log2(parameter.maximum))
            return self._rng.uniform(parameter.minimum, parameter.maximum)
        if parameter.kind == "integer":
            if parameter.log_scale:
                sampled = math.exp(self._rng.uniform(math.log(parameter.minimum), math.log(parameter.maximum)))
                return min(round(parameter.maximum), max(round(parameter.minimum), round(sampled)))
            return self._rng.randint(round(parameter.minimum), round(parameter.maximum))
        return self._rng.choice(parameter.values)

    def suggest(self, count: int) -> list[Suggestion]:
        suggestions: list[Suggestion] = []
        for _ in range(count):
            selection: dict[str, Any] = {"deployment_mode": self.branch.deployment_mode}
            for knob, choices in self.branch.knob_choices.items():
                if knob in self.branch.log_discrete_choices and len(choices) > 1:
                    positive = sorted(float(choice) for choice in choices)
                    sampled = math.exp(self._rng.uniform(math.log(positive[0]), math.log(positive[-1])))
                    selection[knob] = min(choices, key=lambda choice: abs(float(choice) - sampled))
                else:
                    selection[knob] = self._rng.choice(choices)
            for knob, (minimum, maximum) in self.branch.float_ranges.items():
                if knob in self.branch.log_float_ranges:
                    selection[knob] = math.exp(self._rng.uniform(math.log(minimum), math.log(maximum)))
                else:
                    selection[knob] = self._rng.uniform(minimum, maximum)
            for knob, (minimum, maximum) in self.branch.integer_ranges.items():
                if knob in self.branch.log_integer_ranges:
                    sampled = math.exp(self._rng.uniform(math.log(minimum), math.log(maximum)))
                    selection[knob] = min(maximum, max(minimum, round(sampled)))
                else:
                    selection[knob] = self._rng.randint(minimum, maximum)
            if self._parallel_pinned:
                parallel_config = self.branch.parallel_configs[0]
                projection = None
                infeasible_reason = None
            else:
                params = {
                    parameter.name: self._parameter_value(parameter)
                    for parameter in self._projector.parameters
                    if not parameter.is_constant
                }
                parallel_config, projection, infeasible_reason = _project_parallel(
                    self._projector, params, selection["backend"]
                )
            suggestions.append(
                Suggestion(
                    selection=selection,
                    parallel_config=parallel_config,
                    handle=None,
                    projection=projection,
                    infeasible_reason=infeasible_reason,
                )
            )
        return suggestions

    def observe(self, suggestion: Suggestion, metrics: dict[str, float]) -> None:
        del suggestion, metrics

    def observe_infeasible(self, suggestion: Suggestion, reason: str) -> None:
        del suggestion, reason


class SeededBayesianBranchSampler:
    """Local Vizier GP-UCB-PE designer with an explicit reproducible seed."""

    def __init__(
        self,
        branch: BranchSpace,
        *,
        objectives: list[tuple[str, bool]] | None,
        seed: int,
    ) -> None:
        configure_vizier_runtime()
        _, vz = _vizier_modules()
        import jax
        from vizier import algorithms as vza
        from vizier._src.algorithms.designers import gp_ucb_pe
        from vizier.pyvizier.converters import padding

        self.branch = branch
        self._vz = vz
        self._vza = vza
        self._decoders: dict[str, Callable[[Any], Any]] = {}
        self._constants: dict[str, Any] = {}
        self._parallel_projector = ParallelConfigProjector(branch)
        self._parallel_pinned = len(branch.parallel_configs) == 1 and not branch.parallel_independent_choices
        self._next_trial_id = 1
        self._active: dict[int, Any] = {}

        problem = vz.ProblemStatement()
        root = problem.search_space.root
        if not self._parallel_pinned:
            for parameter in self._parallel_projector.parameters:
                if parameter.is_constant:
                    continue
                if parameter.kind == "float":
                    root.add_float_param(
                        parameter.name,
                        min_value=parameter.minimum,
                        max_value=parameter.maximum,
                        default_value=parameter.default,
                    )
                elif parameter.kind == "integer":
                    root.add_int_param(
                        parameter.name,
                        min_value=round(parameter.minimum),
                        max_value=round(parameter.maximum),
                        default_value=round(parameter.default),
                        scale_type=(vz.ScaleType.LOG if parameter.log_scale else vz.ScaleType.LINEAR),
                    )
                elif parameter.kind == "discrete":
                    root.add_discrete_param(
                        parameter.name,
                        feasible_values=parameter.values,
                        default_value=parameter.default,
                        scale_type=(vz.ScaleType.LOG if parameter.log_scale else vz.ScaleType.LINEAR),
                    )
                else:
                    root.add_categorical_param(
                        parameter.name,
                        feasible_values=parameter.values,
                        default_value=parameter.default,
                    )
        for knob, (minimum, maximum) in branch.float_ranges.items():
            root.add_float_param(
                knob,
                min_value=minimum,
                max_value=maximum,
                default_value=(minimum + maximum) / 2.0,
                scale_type=(vz.ScaleType.LOG if knob in branch.log_float_ranges else vz.ScaleType.LINEAR),
            )
            self._decoders[knob] = float
        for knob, (minimum, maximum) in branch.integer_ranges.items():
            root.add_int_param(
                knob,
                min_value=minimum,
                max_value=maximum,
                default_value=minimum,
                scale_type=(vz.ScaleType.LOG if knob in branch.log_integer_ranges else vz.ScaleType.LINEAR),
            )
            self._decoders[knob] = lambda value: round(float(value))
        for knob, raw_choices in branch.knob_choices.items():
            choices = list(raw_choices)
            all_strings = all(isinstance(choice, str) for choice in choices)
            all_numeric = all(isinstance(choice, (int, float)) and not isinstance(choice, bool) for choice in choices)
            if all_strings or all_numeric:
                choices = list(dict.fromkeys(choices))
            if len(choices) <= 1:
                if choices:
                    self._constants[knob] = choices[0]
                continue
            if all_strings:
                root.add_categorical_param(knob, choices)
                self._decoders[knob] = _decoder_for(choices)
            elif all_numeric:
                root.add_discrete_param(
                    knob,
                    sorted(float(choice) for choice in choices),
                    scale_type=(vz.ScaleType.LOG if knob in branch.log_discrete_choices else vz.ScaleType.LINEAR),
                )
                self._decoders[knob] = _decoder_for(choices)
            else:
                root.add_categorical_param(knob, [str(i) for i in range(len(choices))])
                self._decoders[knob] = _index_decoder(choices)
        if problem.search_space.num_parameters() == 0:
            root.add_categorical_param(_CONSTANT_PARAM, ["0"], default_value="0")
        for name, maximize in objectives or [(_METRIC, True)]:
            problem.metric_information.append(
                vz.MetricInformation(
                    name=name,
                    goal=(vz.ObjectiveMetricGoal.MAXIMIZE if maximize else vz.ObjectiveMetricGoal.MINIMIZE),
                )
            )
        self._designer = gp_ucb_pe.VizierGPUCBPEBandit(
            problem,
            rng=jax.random.PRNGKey(seed),
            # Completed/active trial counts grow during every suggestion batch.
            # Masked trial-axis padding lets nearby counts reuse JAX executables
            # instead of compiling a new tensor shape for each count. Keep the
            # model, acquisition budget, seed, and feature/metric axes unchanged.
            padding_schedule=padding.PaddingSchedule(num_trials=padding.PaddingType.POWERS_OF_2),
        )

    def suggest(self, count: int) -> list[Suggestion]:
        suggestions: list[Suggestion] = []
        for raw in self._designer.suggest(count):
            trial = raw.to_trial(self._next_trial_id)
            self._next_trial_id += 1
            self._active[trial.id] = trial
            params = {name: getattr(value, "value", value) for name, value in dict(trial.parameters).items()}
            selection: dict[str, Any] = {
                "deployment_mode": self.branch.deployment_mode,
                **self._constants,
            }
            for knob, decode in self._decoders.items():
                selection[knob] = decode(params[knob])
            if self._parallel_pinned:
                parallel_config = self.branch.parallel_configs[0]
                projection = None
                infeasible_reason = None
            else:
                parallel_config, projection, infeasible_reason = _project_parallel(
                    self._parallel_projector, params, selection["backend"]
                )
            suggestions.append(
                Suggestion(
                    selection=selection,
                    parallel_config=parallel_config,
                    handle=trial,
                    projection=projection,
                    infeasible_reason=infeasible_reason,
                )
            )
        return suggestions

    def _complete(self, suggestion: Suggestion, measurement, reason=None) -> None:
        trial = suggestion.handle
        if reason is None:
            trial.complete(measurement)
        else:
            trial.complete(measurement, infeasibility_reason=reason)
        self._active.pop(trial.id, None)
        self._designer.update(
            completed=self._vza.CompletedTrials([trial]),
            all_active=self._vza.ActiveTrials(list(self._active.values())),
        )

    def observe(self, suggestion: Suggestion, metrics: dict[str, float]) -> None:
        self._complete(
            suggestion,
            self._vz.Measurement(metrics={name: float(value) for name, value in metrics.items()}),
        )

    def observe_infeasible(self, suggestion: Suggestion, reason: str) -> None:
        self._complete(suggestion, self._vz.Measurement(), reason)


@dataclass(frozen=True)
class _ConditionalSuggestionHandle:
    arm_index: int
    inner_suggestion: Suggestion = field(repr=False)


def _conditional_arm_id(assignments: dict[str, Any]) -> str:
    payload = json.dumps(assignments, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _expand_conditional_arms(branch: BranchSpace) -> list[tuple[str, BranchSpace]]:
    """Expand conditional selectors into flat sampler-compatible branch arms."""

    selectors = tuple(dict.fromkeys(condition.selector for condition in branch.conditional_dimensions))
    arms: list[tuple[str, BranchSpace]] = []
    for raw_values in product(*(branch.knob_choices[selector] for selector in selectors)):
        assignments = dict(zip(selectors, raw_values, strict=True))
        choices = {name: list(values) for name, values in branch.knob_choices.items()}
        for selector, value in assignments.items():
            choices[selector] = [value]
        float_ranges = dict(branch.float_ranges)
        log_float_ranges = set(branch.log_float_ranges)
        log_discrete_choices = set(branch.log_discrete_choices)
        for condition in branch.conditional_dimensions:
            if assignments[condition.selector] not in condition.values:
                continue
            choices.update({name: list(values) for name, values in condition.knob_choices.items()})
            float_ranges.update(condition.float_ranges)
            log_float_ranges.update(condition.log_float_ranges)
            log_discrete_choices.update(condition.log_discrete_choices)
        arms.append(
            (
                _conditional_arm_id(assignments),
                replace(
                    branch,
                    knob_choices=choices,
                    float_ranges=float_ranges,
                    log_float_ranges=frozenset(log_float_ranges),
                    log_discrete_choices=frozenset(log_discrete_choices),
                    conditional_dimensions=(),
                ),
            )
        )
    return arms


class ConditionalBranchSampler:
    """Stable round-robin facade over flat conditional selector arms.

    The installed Vizier designers reject hierarchical search spaces.  Each
    selector assignment therefore owns a normal flat sampler study, while this
    facade preserves the external one-branch lifecycle and delegates observations
    to the arm that produced the suggestion.
    """

    def __init__(
        self,
        branch: BranchSpace,
        *,
        sampler_factory: Callable[[BranchSpace, str, int], BranchSampler],
        study_id: str,
        seed: int,
    ) -> None:
        self.branch = branch
        self._cursor = 0
        self._samplers: list[BranchSampler] = []
        for arm_id, arm in _expand_conditional_arms(branch):
            arm_seed = (seed + int(arm_id[:8], 16)) % (2**31 - 1)
            self._samplers.append(sampler_factory(arm, f"{study_id}_conditional_{arm_id}", arm_seed))
        if not self._samplers:
            raise ValueError("a conditional branch must produce at least one selector arm")

    def suggest(self, count: int) -> list[Suggestion]:
        if count <= 0:
            return []
        arm_order = [(self._cursor + index) % len(self._samplers) for index in range(count)]
        self._cursor = (self._cursor + count) % len(self._samplers)
        counts = [arm_order.count(index) for index in range(len(self._samplers))]
        batches = [iter(sampler.suggest(counts[index])) for index, sampler in enumerate(self._samplers)]
        suggestions: list[Suggestion] = []
        for arm_index in arm_order:
            inner = next(batches[arm_index])
            suggestions.append(
                Suggestion(
                    selection=inner.selection,
                    parallel_config=inner.parallel_config,
                    handle=_ConditionalSuggestionHandle(arm_index, inner),
                    projection=inner.projection,
                    infeasible_reason=inner.infeasible_reason,
                )
            )
        return suggestions

    @staticmethod
    def _inner(suggestion: Suggestion) -> _ConditionalSuggestionHandle:
        handle = suggestion.handle
        if not isinstance(handle, _ConditionalSuggestionHandle):
            raise TypeError("suggestion was not created by this conditional sampler")
        return handle

    def observe(self, suggestion: Suggestion, metrics: dict[str, float]) -> None:
        handle = self._inner(suggestion)
        self._samplers[handle.arm_index].observe(handle.inner_suggestion, metrics)

    def observe_infeasible(self, suggestion: Suggestion, reason: str) -> None:
        handle = self._inner(suggestion)
        self._samplers[handle.arm_index].observe_infeasible(handle.inner_suggestion, reason)


def make_branch_sampler(
    branch: BranchSpace,
    *,
    study_id: str,
    objectives: list[tuple[str, bool]] | None = None,
    algorithm: str | None = None,
    seed: int = 42,
) -> BranchSampler:
    """Construct the default (Vizier) sampler for a branch. ``objectives`` (name, maximize)
    pairs default to a single maximized ``"objective"``; pass >=2 for a Pareto study."""

    def construct(flat_branch: BranchSpace, flat_study_id: str, flat_seed: int) -> BranchSampler:
        if algorithm == "random":
            return RandomBranchSampler(flat_branch, seed=flat_seed)
        if algorithm == "bayesian":
            return SeededBayesianBranchSampler(
                flat_branch,
                objectives=objectives,
                seed=flat_seed,
            )
        return VizierBranchSampler(
            flat_branch,
            study_id=flat_study_id,
            objectives=objectives,
            algorithm="bayesian",
            seed=flat_seed,
        )

    if branch.conditional_dimensions:
        return ConditionalBranchSampler(
            branch,
            sampler_factory=construct,
            study_id=study_id,
            seed=seed,
        )
    return construct(branch, study_id, seed)
