---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Sweeper Search Policies
subtitle: Choose bounded optimization or complete finite enumeration
---

> [!WARNING]
> **Experimental.** Sweeper is intended for evaluation and feedback, not production capacity
> planning. Search behavior and output may change without a standard deprecation period.

Sweeper supports two explicit policies under `sweep.policy`:

| Policy | Candidate selection | Stop rule | Coverage claim |
|---|---|---|---|
| `rapid` (default) | seeded Vizier optimizer | configured round budget or projection stall | approximate |
| `thorough` | canonical Cartesian enumeration | finite runnable space exhausted | complete |

## Rapid

Rapid search is optimizer-guided and bounded. Its target is
`branches * max_rounds * candidates_per_round` successful unique candidates. A round may request
replacement suggestions for duplicate, unsupported, failed, or infeasible points, but it stops
after at most eleven times the per-round target. `SearchExecutionReport.candidate_budget` records
that hard suggestion ceiling. A successful legacy run reports `target_reached`; unified CLI runs
with an explicit `max_trials` report `candidate_budget_reached` when that exact suggestion budget
is consumed.

`sweep.seed` initializes the pinned default Vizier designer. The same configuration, seed, runner,
and dependency versions therefore begin from the same optimizer state. A custom sampler factory is
passed `seed` when its callable accepts that keyword and owns its own reproducibility contract.
Experimental algorithms selected through `AISIMULATE_SWEEPER_VIZIER_ALGO` may define different seed
semantics. Seeded rapid runs support the pinned `DEFAULT`/`GP_UCB_PE`, Gaussian-process bandit,
random, quasi-random, grid, and shuffled-grid designers; another override fails before evaluation
instead of silently claiming reproducibility.

Rapid results are always marked `approximate=true` and `complete=false`; a small run is not promoted
to complete merely because it happened to revisit every point.

## Thorough

Thorough search sorts parallel configurations by their canonical JSON form, sorts parameter names,
preserves the configured order of each parameter's distinct JSON values, and visits the Cartesian
product. Backend/topology pairs rejected by capability preflight are not part of the runnable
space. `candidates_per_round` is an evaluation/callback batch size; `max_rounds` does not truncate
the exhaustive run. `seed` is recorded for provenance but deliberately does not reorder candidates.

Every dimension must be finite and discrete. Thorough search rejects continuous ranges instead of
silently choosing a grid. In particular, pin `workload.kv_load_ratio` to one scalar or expose an
explicit provider-owned choice list before selecting `thorough`.

Thorough completion means every runnable configured point was attempted. Replay failures remain
visible in `failed_candidates`; completion does not claim that every attempt produced a feasible
measurement.

## Execution Report

After a successful run, `Sweeper.last_report` is an immutable `SearchExecutionReport` containing:

- policy, seed, configured budget, finite-space size, and stop reason;
- complete/approximate markers;
- suggested, evaluated, feasible, infeasible, failed, unsupported, and cached counts;
- elapsed wall time.

`SearchExecutionReport.as_dict()` is JSON-ready. AIC-1471 owns the canonical result-envelope
serialization; this execution record remains separate from per-candidate fields until that envelope
embeds it losslessly.

## Reproducible Comparison Harness

Run both policies against the same config and deterministic fixture runner:

```bash
uv run --project python/aisimulate \
  python python/aisimulate/tools/benchmark_sweeper_policies.py \
  --config examples/sweeper/benchmark-policies.yaml \
  --output /tmp/sweeper-policy-benchmark.json
```

The artifact records the config SHA-256, source revision/dirty-diff hash, Python/platform/package
versions, wall time, peak Python memory, execution report, selected configuration, and score for
each policy.
Optional regression gates make a benchmark job fail when rapid's best score is below a configured
fraction of thorough's score or when rapid exceeds a configured wall-time ratio:

```bash
uv run --project python/aisimulate \
  python python/aisimulate/tools/benchmark_sweeper_policies.py \
  --config examples/sweeper/benchmark-policies.yaml \
  --min-rapid-score-ratio 0.90 \
  --max-rapid-time-ratio 20.00
```

The small smoke fixture uses 90% selected-score parity. Its rapid path pays fixed Vizier/JAX startup
cost against only three exhaustive candidates, so the illustrated 20x time-ratio ceiling catches
large regressions rather than claiming rapid is faster at that scale. Projects should check in
representative model/system/mode configs and set tighter time thresholds from repeated measurements
rather than treating this deterministic fixture as production accuracy or performance evidence.
