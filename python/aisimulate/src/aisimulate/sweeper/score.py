# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score a replay ``trace_report`` against the optimization goal.

Three steps, mirroring the profiler replay optimizer in the ``ai-dynamo/dynamo`` repository at
``components/src/dynamo/profiler/utils/replay_optimize``, adapted to sweeper's ``Candidate`` /
``OptimizationGoal`` and the merged replay report keys:

1. **objective** — map the goal target to a number from the report. The
   ``goodput_per_gpu`` / ``throughput_per_gpu`` targets divide ``goodput`` / ``throughput``
   (already a tok/s rate) by the **time-averaged provisioned GPU count**
   ``avg_gpu = gpu_hours / e2e_hours`` — units tok/s/gpu, matching a benchmark's
   "throughput per GPU". For a static deployment ``avg_gpu`` is the fixed GPU count
   (``gpu_hours = gpu_count * e2e_hours``); for a planner-scaled run it is the integral
   of provisioned GPUs over the run divided by its duration. (Dividing by ``gpu_hours``
   directly would be wrong — the rate already has time divided out.)
2. **feasibility** — within the GPU budget. SLA is not gated by default: a
   ``goodput`` target already counts SLA-satisfying requests. The explicit
   ``strict_sla`` option additionally filters aggregate means before ranking.
   Over-budget and strict-SLA-violating candidates are dropped.
3. **rank** — feasible candidates best-first by score, ties broken toward fewer GPUs.

For a ``pareto`` goal the score is a *vector* instead: :func:`objective_vector` reads one
value per objective, :func:`make_candidate` stores them on the candidate, and
:func:`pareto_front` returns the non-dominated set (step 3 becomes Pareto dominance, not a
scalar rank). The default objectives are throughput-per-GPU vs per-user throughput — the
InferenceX tok/s/gpu vs tok/s/user frontier.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .config import (
    Candidate,
    OptimizationGoal,
    OptimizationTarget,
    SLATarget,
)

# trace_report keys the report always carries (goodput_* only when an SLA was
# supplied to the replay). Surfaced into Candidate.metrics for inspection.
_METRIC_KEYS = (
    "output_throughput_tok_s",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "mean_e2e_latency_ms",
    "mean_output_token_throughput_per_user",
    "goodput_output_throughput_tok_s",
    "gpu_hours",
    "duration_ms",
    "planner_total_ticks",
)


def _avg_gpu(report: dict[str, float]) -> float:
    """Time-averaged provisioned GPU count = ``gpu_hours / e2e_hours`` (the integral of
    provisioned GPUs over the run, divided by its duration). For a static deployment this
    equals the fixed GPU count; for a planner-scaled run it averages over startup + serve +
    drain. Returns 0.0 when gpu_hours / duration are unavailable (guards divide-by-zero).
    """
    gpu_hours = float(report.get("gpu_hours", 0.0))
    duration_ms = float(report.get("duration_ms", 0.0))
    if gpu_hours <= 0.0 or duration_ms <= 0.0:
        return 0.0
    return gpu_hours / (duration_ms / 3_600_000.0)


def objective_value(report: dict[str, float], target: OptimizationTarget) -> float:
    """The raw objective metric (NOT yet signed for direction)."""
    if target is OptimizationTarget.THROUGHPUT:
        return float(report.get("output_throughput_tok_s", 0.0))
    if target is OptimizationTarget.E2E_LATENCY:
        return float(report.get("mean_e2e_latency_ms", math.inf))
    if target is OptimizationTarget.GOODPUT:
        return float(report.get("goodput_output_throughput_tok_s", 0.0))
    if target is OptimizationTarget.GOODPUT_PER_GPU:
        avg_gpu = _avg_gpu(report)
        goodput = float(report.get("goodput_output_throughput_tok_s", 0.0))
        return goodput / avg_gpu if avg_gpu > 0.0 else 0.0
    if target is OptimizationTarget.THROUGHPUT_PER_GPU:
        avg_gpu = _avg_gpu(report)
        throughput = float(report.get("output_throughput_tok_s", 0.0))
        return throughput / avg_gpu if avg_gpu > 0.0 else 0.0
    if target is OptimizationTarget.THROUGHPUT_PER_USER:
        # per-user interactivity (tok/s/user): mean of per-token-gap 1000/itl. Already a
        # rate, so no GPU/time normalization — this is the InferenceX x-axis.
        return float(report.get("mean_output_token_throughput_per_user", 0.0))
    if target is OptimizationTarget.PARETO:
        raise ValueError(
            "'pareto' is multi-objective; use objective_vector / pareto_front, not objective_value"
        )
    raise ValueError(f"unknown optimization target: {target!r}")


def score_report(report: dict[str, float], target: OptimizationTarget) -> float:
    """Objective normalized so **higher is better** (minimized targets negated)."""
    value = objective_value(report, target)
    return value if target.maximize else -value


def is_feasible(used_gpus: int, gpu_budget: int) -> bool:
    """A candidate is feasible iff it fits the GPU budget.

    SLA is deliberately not a gate here: the goodput targets already bake the SLA into
    their metric (the bridge counts only SLA-satisfying requests per-request), so an
    unconditional aggregate mean-latency gate would double-count it. Explicit strict
    aggregate gating is implemented by :func:`analyze_candidates` and the search loop.
    """
    return used_gpus <= gpu_budget


def request_latency_ms(report: Mapping[str, float], *, osl: int) -> float:
    """Return the legacy aggregate request-latency metric.

    Legacy AIC combines mean TTFT with one mean TPOT for every output token
    after the first: ``ttft + tpot * (osl - 1)``. A missing/non-finite input
    returns ``math.inf`` so strict filtering fails closed.
    """
    if osl < 1:
        raise ValueError(f"osl must be >= 1, got {osl}")
    try:
        ttft = float(report["mean_ttft_ms"])
        tpot = float(report["mean_tpot_ms"])
    except (KeyError, TypeError, ValueError):
        return math.inf
    if not math.isfinite(ttft) or not math.isfinite(tpot):
        return math.inf
    return ttft + tpot * max(osl - 1, 0)


def enumerate_request_latency_constraints(
    *,
    osl: int,
    request_latency_ms: float,
    ttft_ms: float | None = None,
) -> list[tuple[float, float]]:
    """Enumerate legacy-compatible ``(TTFT, TPOT)`` constraint pairs.

    Every returned pair lies exactly on
    ``ttft + tpot * (osl - 1) == request_latency_ms``. The deterministic
    anchors match legacy AIC while keeping this standalone module independent
    of the ``aiconfigurator`` package.
    """
    if osl <= 1:
        raise ValueError("request-latency constraint enumeration requires osl > 1")
    if not math.isfinite(request_latency_ms) or request_latency_ms <= 0:
        raise ValueError("request_latency_ms must be finite and positive")
    if ttft_ms is not None and (
        not math.isfinite(ttft_ms) or ttft_ms <= 0
    ):
        raise ValueError("ttft_ms must be finite and positive when provided")

    preferred_ttft = (
        request_latency_ms * 0.95 if ttft_ms is None else float(ttft_ms)
    )
    base_values = [
        300.0,
        400.0,
        500.0,
        600.0,
        800.0,
        1000.0,
        1200.0,
        1400.0,
        1600.0,
        2000.0,
        3000.0,
        5000.0,
        8000.0,
    ]
    interval_values = [
        request_latency_ms * fraction
        for fraction in (0.1, 0.2, 0.3, 0.5, 0.7)
    ]
    supplemental = [
        value
        for value in interval_values
        if value < base_values[0] or value > base_values[-1]
    ]
    ttft_values = sorted(
        value
        for value in {*base_values, *supplemental, preferred_ttft}
        if value < request_latency_ms
    )
    return [
        (ttft, (request_latency_ms - ttft) / (osl - 1))
        for ttft in ttft_values
    ]


def aggregate_sla_violations(
    report: Mapping[str, float],
    sla: SLATarget,
    *,
    osl: int,
) -> tuple[str, ...]:
    """Describe strict aggregate SLA violations using inclusive bounds.

    A value equal to its bound is feasible. Missing and non-finite metrics
    fail closed instead of silently admitting an unqualified candidate.
    """
    checks = (
        ("ttft", "mean_ttft_ms", sla.ttft_ms),
        ("tpot", "mean_tpot_ms", sla.itl_ms),
        ("e2e", "mean_e2e_latency_ms", sla.e2e_ms),
    )
    violations: list[str] = []
    for label, metric, bound in checks:
        if bound is None:
            continue
        try:
            value = float(report[metric])
        except (KeyError, TypeError, ValueError):
            violations.append(f"{label} metric {metric} is missing")
            continue
        if not math.isfinite(value):
            violations.append(f"{label} metric {metric} is non-finite")
        elif value > bound:
            violations.append(f"{label} {value:g}ms > {bound:g}ms")

    if sla.request_latency_ms is not None:
        value = request_latency_ms(report, osl=osl)
        if not math.isfinite(value):
            violations.append(
                "request latency needs finite mean_ttft_ms and mean_tpot_ms"
            )
        elif value > sla.request_latency_ms:
            violations.append(
                f"request_latency {value:g}ms > {sla.request_latency_ms:g}ms"
            )
    return tuple(violations)


def meets_aggregate_sla(
    report: Mapping[str, float],
    sla: SLATarget,
    *,
    osl: int,
) -> bool:
    """Whether aggregate mean metrics satisfy every configured SLA bound."""
    return not aggregate_sla_violations(report, sla, osl=osl)


def objective_vector(
    report: dict[str, float], objectives: list[OptimizationTarget]
) -> dict[str, float]:
    """Raw value (natural units, NOT signed) for each Pareto objective, keyed by target
    value. Dominance uses each objective's own direction (``target.maximize``)."""
    return {t.value: objective_value(report, t) for t in objectives}


def _dominates(
    a: dict[str, float], b: dict[str, float], objectives: list[OptimizationTarget]
) -> bool:
    """True iff ``a`` Pareto-dominates ``b``: at least as good on every objective (in that
    objective's own direction) and strictly better on at least one."""
    strictly_better = False
    for t in objectives:
        av, bv = a[t.value], b[t.value]
        better = av > bv if t.maximize else av < bv
        worse = av < bv if t.maximize else av > bv
        if worse:
            return False
        if better:
            strictly_better = True
    return strictly_better


def _config_tie_key(config: Mapping[str, Any]) -> str:
    """Stable last-resort ordering without importing a deployment adapter."""
    return json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)


def _has_finite_objectives(
    candidate: Candidate, objectives: Sequence[OptimizationTarget]
) -> bool:
    values = candidate.objectives
    return values is not None and all(
        objective.value in values
        and math.isfinite(float(values[objective.value]))
        for objective in objectives
    )


def pareto_front(
    candidates: list[Candidate], objectives: list[OptimizationTarget]
) -> list[Candidate]:
    """The non-dominated subset of ``candidates`` over ``objectives`` (each carrying an
    ``objectives`` vector), sorted by the **last** objective ascending — the x-axis — so the
    returned list traces the frontier left-to-right (e.g. low->high per-user throughput).
    """
    pool = [c for c in candidates if _has_finite_objectives(c, objectives)]
    front = [
        c
        for c in pool
        if not any(
            _dominates(o.objectives, c.objectives, objectives)
            for o in pool
            if o is not c
        )
    ]
    x_axis = objectives[-1].value

    def _sort_key(candidate: Candidate) -> tuple[Any, ...]:
        assert candidate.objectives is not None
        secondary = tuple(
            -candidate.objectives[objective.value]
            if objective.maximize
            else candidate.objectives[objective.value]
            for objective in objectives[:-1]
        )
        return (
            candidate.objectives[x_axis],
            secondary,
            candidate.used_gpus,
            _config_tie_key(candidate.config),
        )

    return sorted(front, key=_sort_key)


def make_candidate(
    config: dict,
    report: dict[str, float],
    target: OptimizationTarget,
    *,
    pareto_objectives: list[OptimizationTarget] | None = None,
) -> Candidate:
    """Build a scored :class:`Candidate` from its config + replay report.

    Single-objective: ``score`` is the signed objective (higher=better). Under a
    ``pareto`` target, ``pareto_objectives`` must be given; the per-objective raw values are
    stored in ``Candidate.objectives`` (Pareto dominance reads these) and ``score`` carries
    the first objective's value as a headline number (it is not used for ranking)."""
    metrics = {key: float(report[key]) for key in _METRIC_KEYS if key in report}
    if target is OptimizationTarget.PARETO:
        if not pareto_objectives:
            raise ValueError("a pareto candidate needs pareto_objectives")
        objectives = objective_vector(report, pareto_objectives)
        return Candidate(
            config=config,
            used_gpus=int(config.get("used_gpus", 0)),
            score=objectives[pareto_objectives[0].value],
            metrics=metrics,
            objectives=objectives,
        )
    return Candidate(
        config=config,
        used_gpus=int(config.get("used_gpus", 0)),
        score=score_report(report, target),
        metrics=metrics,
    )


def rank(candidates: list[Candidate]) -> list[Candidate]:
    """Best-first with deterministic legacy-compatible tie breaking.

    Non-finite scores are not rankable. Finite ties prefer fewer GPUs, then
    the canonical configuration key so engine-only and adapter-backed inputs
    produce the same order regardless of evaluation completion order.
    """
    finite = [candidate for candidate in candidates if math.isfinite(candidate.score)]
    return sorted(
        finite,
        key=lambda candidate: (
            -candidate.score,
            candidate.used_gpus,
            _config_tie_key(candidate.config),
        ),
    )


def analyze_candidates(
    candidates: Sequence[Candidate],
    goal: OptimizationGoal,
    *,
    osl: int,
) -> list[Candidate]:
    """Apply one SLA/ranking contract to engine-only or adapter results.

    Default SLA semantics remain replay's per-request goodput calculation.
    With ``strict_sla``, aggregate means are filtered *before* scalar ranking
    or Pareto dominance, matching legacy ``--strict-sla`` behavior.
    """
    pool = list(candidates)
    if goal.strict_sla:
        assert goal.sla is not None  # OptimizationGoal validates this invariant.
        pool = [
            candidate
            for candidate in pool
            if meets_aggregate_sla(candidate.metrics, goal.sla, osl=osl)
        ]
    return (
        pareto_front(pool, goal.resolved_pareto_objectives)
        if goal.is_pareto
        else rank(pool)
    )
