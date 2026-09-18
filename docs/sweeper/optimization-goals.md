---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Sweeper Optimization Goals
subtitle: Objective metrics, SLA constraints, and Pareto-front scoring
---

> [!WARNING]
> **Experimental.** Sweeper is intended for evaluation and feedback, not production capacity
> planning. Its API, configuration schema, search results, and deployment output may change
> without a standard deprecation period. Sweeper provides no SLA, accuracy, or configuration
> optimality guarantees.

An `OptimizationGoal` (the `goal:` block of a `SmartSearchConfig`) declares **what
"better" means** plus the SLA constraint. It is pinned, never searched. It picks one
`OptimizationTarget` and — for `pareto` — the list of scalar objectives whose frontier to
trace. Optional providers receive the complete goal in `SweepContext`; any feature-specific
mapping belongs to the provider, not the Sweeper core.

The whole `goal:` block is **optional**: it defaults to a `throughput` goal with no SLA
(`OptimizationGoal()` → `target = throughput`). `OptimizationGoal` itself is
`extra="forbid"`, so unknown goal keys are rejected.

The goal drives two things in `score.py`: the **objective** read from each replay
`trace_report` (`objective_value`), and **step 3** of scoring — `rank` (scalar) or
`pareto_front` (multi-objective).

## Targets

Every `OptimizationTarget` and the exact report metric it reads (`score.objective_value`):

| `target` | direction | report metric | needs SLA? |
|---|---|---|---|
| `throughput` | maximize | `output_throughput_tok_s` | no |
| `throughput_per_gpu` | maximize | `output_throughput_tok_s / avg_gpu` (tok/s/gpu) | no |
| `throughput_per_user` | maximize | `mean_output_token_throughput_per_user` (tok/s/user) | no |
| `ttft` | **minimize** | `mean_ttft_ms` | no |
| `e2e_latency` | **minimize** | `mean_e2e_latency_ms` | no |
| `goodput` | maximize | `goodput_output_throughput_tok_s` | **yes** |
| `goodput_per_gpu` | maximize | `goodput_output_throughput_tok_s / avg_gpu` (tok/s/gpu) | **yes** |
| `min_gpus` | minimize | concrete candidate `used_gpus` (provisioned GPUs) | **yes** |
| `pareto` | per-objective | a vector — one value per `pareto_objectives` entry | iff an objective needs it |

`ttft`, `e2e_latency`, and `min_gpus` are minimized targets: `OptimizationTarget.maximize` returns
`False` for all three (and raises for `pareto`, which has no single direction). `score_report` negates
minimized targets so **higher is always better** internally; for a Pareto goal the raw
(unsigned) value is kept and `_dominates` applies each objective's own direction.
Missing-key defaults differ by direction: a maximized target reads `0.0` when its key is
absent, but `ttft` and `e2e_latency` default to `+inf` when their metric is missing or has
no qualifying samples. Such a latency report scores worst-possible (`-inf` after negation)
rather than best.

The `*_per_user` metric is already a rate (mean of per-token-gap `1000/itl`), so it gets
**no** GPU/time normalization — it is the InferenceX x-axis (tok/s/user).

### `avg_gpu` — the per-GPU divisor

The two `*_per_gpu` targets divide a tok/s rate by the **time-averaged provisioned GPU
count** (`score._avg_gpu`):

```
avg_gpu = gpu_hours / e2e_hours        # e2e_hours = duration_ms / 3_600_000
```

This is the integral of provisioned GPUs over the run divided by its duration:

- **static deployment** — `avg_gpu` collapses to the fixed GPU count
  (`gpu_hours = gpu_count * e2e_hours`).
- **runtime-scaled run** — `avg_gpu` averages provisioned GPUs over startup + serve +
  drain.

Dividing the rate by `gpu_hours` directly would be wrong: the rate already has time
divided out. `_avg_gpu` returns `0.0` when `gpu_hours` **or** `duration_ms` is `<= 0`
(missing report keys default to `0.0`, so the guard covers both missing and non-positive
values), and the `*_per_gpu` targets then return `0.0` (divide-by-zero guard).

### SLA requirement rule

The **goodput** targets need an SLA because their metric
(`goodput_output_throughput_tok_s`) counts only SLA-satisfying requests (the replay
bridge's per-request goodput SLA). `_SLA_TARGETS = {goodput, goodput_per_gpu}`.

`OptimizationGoal._validate_goal` computes the *effective* objective set — `{target}` for
a scalar goal, or the resolved `pareto_objectives` for a Pareto goal — and requires an SLA
iff that set intersects `_SLA_TARGETS`. So an SLA is mandatory when:

- `target` is `goodput` or `goodput_per_gpu`, **or**
- `target` is `pareto` **and** its objectives include one of those.

A satisfying per-request goodput SLA has at least one configured bound. TTFT and ITL are
independently optional; an unset field is unbounded. `e2e_ms` remains mutually exclusive
with either token-latency field.
By default SLA is *not* gated during aggregate feasibility (`is_feasible` checks only
the GPU budget) — it lives inside the goodput metric, so an unconditional aggregate
latency gate would double-count it. `strict_sla: true` is the explicit legacy-compatible
opt-in: it filters aggregate mean metrics before scalar ranking or Pareto dominance.

`SLATarget` shape (ms, each `> 0`, `extra="forbid"`):

| field | meaning |
|---|---|
| `ttft_ms` | independently optional time-to-first-token bound |
| `itl_ms` | independently optional inter-token-latency bound |
| `e2e_ms` | end-to-end bound — standalone alternative |

Strict aggregate comparisons are inclusive (`value <= bound`). A configured bound with
a missing/non-finite report metric or no qualifying latency samples rejects the candidate.

## Minimum GPUs

`goal.target: min_gpus` selects the smallest qualifying configuration found by the sweep.
It always enforces the configured aggregate-mean latency bounds, even when `strict_sla` is
false. For fixed synthetic request-rate traffic, also set a positive `goal.min_goodput_rps`:
the measured `goodput_request_throughput_rps` must meet that floor. This metric counts only
requests satisfying the per-request SLA, divided by the complete replay duration, including
startup and drain. Choose the measurement duration and offered traffic explicitly; offering
exactly the required rate can fall below the floor on a short run. The floor must not exceed
the offered rate. Missing rate evidence fails closed; raw throughput or offered rate is never
substituted. Fixed synthetic concurrency needs no rate floor, but may specify one.

The optimizer observes `-used_gpus` for qualifying candidates and infeasibility for latency or
load failures. Final selection applies the same constraints to the full candidate pool before
top-N truncation, then sorts by GPU count, higher SLA-compliant output throughput, lower mean
E2E latency, and a deterministic configuration key. GPU count is provisioned topology size,
not time-averaged usage. A finite trial budget establishes the smallest qualifying configuration
found, not a global optimum. No qualifying candidate produces an empty selection.

This target supports static engine pools with fixed synthetic request-rate or concurrency
traffic. It rejects adapters, traces, sessions, candidate-relative KV load, and searched load
domains. Analytical EPD supports concurrency plus aggregate latency only, without a rate floor.
`min_gpus` cannot be a Pareto objective, and `min_goodput_rps` is only accepted with this target.

In the CLI, use `optimization.target: min_gpus` and
`optimization.constraints.min_goodput_rps`; SLA remains under `evaluation.sla`. See the
[minimum-GPU example](../cli/migrate-from-aiconfigurator.md#minimum-gpu-sizing).

## Pareto

`pareto` is the one **multi-objective** target. Instead of a scalar score it optimizes the
tradeoff between the scalar targets in `pareto_objectives`.

- **objectives** — `OptimizationGoal.pareto_objectives` (`list[OptimizationTarget] | None`).
  `resolved_pareto_objectives` returns it, or the default pair
  `_DEFAULT_PARETO_OBJECTIVES = (throughput_per_gpu, throughput_per_user)` only when it is
  unset (`None`) — the **InferenceX tok/s/gpu (y) vs tok/s/user (x) frontier**. An
  explicit list is kept as-is so the validator can reject it: `>= 2` entries, distinct, no
  `pareto` among them, and `pareto_objectives` is only legal under a `pareto` target.

- **optimizer selection** — the historical SDK path (`sweep.max_trials: null`)
  creates round-based Vizier studies. Its default algorithm is `DEFAULT`, subject
  to the [algorithm override](configuration.md#sampler-algorithm-override).
  When `sweep.max_trials` is set, `sweep.algorithm` selects the seeded Bayesian
  or random sampler. The unified CLI always supplies this total-trial budget
  from `optimizer.max_trials`; `optimizer.algorithm` selects `bayesian` or
  `random`. Do not assume every CLI study uses Vizier's embedded `DEFAULT`
  designer. For Pareto, samplers receive every objective and its direction.

- **front** — `score.pareto_front` returns the **non-dominated** subset after optional
  strict aggregate SLA filtering.
  `_dominates(a, b)` is true iff `a` is at least as good as `b` on **every** objective (in
  that objective's own `maximize` direction) and strictly better on at least one. The
  front is **sorted by the last objective ascending** — the x-axis — so the list traces
  the frontier left-to-right (e.g. low→high per-user throughput). `Sweeper.run`
  returns this front for a Pareto goal, and `rank` (best score, ties → fewer GPUs) for
  every scalar goal except `min_gpus`, which uses the constrained GPU-count ordering above.

- **swept load dimension** — `workload.concurrency` is always one fixed in-flight cap.
  A Pareto workload may instead set `kv_load_ratio: [min, max]`, which Vizier models as a
  continuous parameter. Each ratio is converted to an absolute concurrency from that
  candidate's decode/agg KV capacity, so the model compares equivalent load pressure across
  different replica and parallel configurations. If a synthetic Pareto workload omits all
  load fields, the range defaults to `[0.0, 1.0]`; see [traffic.md](traffic.md).

Per-objective raw values are stored on `Candidate.objectives` (keyed by
`OptimizationTarget` value, e.g. `{"throughput_per_gpu": .., "throughput_per_user": ..}`)
by `make_candidate`; `Candidate.score` carries the first objective's value as a headline
number only (not used for Pareto ranking). `objectives` is `None` for a scalar goal.

## SDK and unified CLI fields

This page describes `SmartSearchConfig`. In the unified CLI, put `target` and
`strict_sla` under `optimization`, SLA bounds under `evaluation.sla`, and search
controls under `optimizer`. The CLI's Pareto objectives are fixed to
`throughput_per_gpu` and `throughput_per_user`; the SDK allows an explicit
`goal.pareto_objectives` list. Do not copy a complete SDK YAML into the CLI.

For a CLI recommendation that minimizes TTFT, start with the
[bounded recommendation example](../cli/user-guide.md#recommend-under-a-gpu-budget)
and replace its goal with:

```yaml
optimization:
  target: ttft
  constraints: {max_candidate_gpus: 4}
```

Its score is negative mean TTFT, so a larger score means lower latency. Use
`metrics.mean_ttft_ms` when displaying the latency itself. The
[prediction interpretation guide](../cli/understand-your-prediction.md)
explains latency populations and why strict aggregate SLA is not a p99 gate.
