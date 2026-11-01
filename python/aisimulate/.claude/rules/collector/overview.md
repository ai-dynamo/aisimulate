---
description: >
  The collector in five things (Pin, Plan, Record, Note, Rule), the failure
  principle, the rules and their checkers, and what to record when.
paths:
  - "collector/**"
  - "tests/unit/collector/**"
  - "tools/perf_database/**"
---

# Collector Overview

Read this page first. The other files here go deeper on one topic each and use
the words defined here; if one of them disagrees with this page, it is out of
date.

| File | Topic |
|---|---|
| `layer_permissions.md` | Which layer may hold which rule; leaving cases out; capacity; serving parity; deviations; version contract; module boundary |
| `failure_handling.md` | What happens to a failing case, and when to escalate |
| `case_authoring.md` | Case YAML, op plans, running subsets |

## Five things

| Thing | What it is | Where it lives |
|---|---|---|
| **Pin** | The framework release and image each (backend, op family) is collected with; the collector refuses any other installed version | `collector/framework_manifest.yaml` |
| **Plan** | Every case an op collects: one list for every backend | `collector/op_plans.py` expands the `collector/cases/` YAML; `collector/skips.py` leaves cases out (`case_authoring.md`) |
| **Record** | How a table came to be: collected or reused, the runtime, `collector_hash` (one hash over the collector source), `case_plan_hash` (the cases attempted) and how the run went | `collection_meta.yaml` (the collector writes it) or `reuse.yaml` (a person writes it) next to the parquet |
| **Note** | A deviation the rows cannot show: the table measures a different path than serving runs | A comment at the deviating code (`layer_permissions.md`, "Deviations are notes") |
| **Rule** | What every collector change must satisfy | The rules below |

Evidence (logs, probe results, before/after diffs) lives under
`benchmarks/evidence/<id>/`; a note or a record may cite it. A fact has one
home, and data comments and docs point to it. Do not create findings files,
waivers, exception registries or evidence policies.

## The failure principle

Generate the same cases for all three frameworks. A few boundary points may
fail: a failure is a recorded observation, not a defect to filter away.

- **Leave a case out** only through `collector/skips.py`, and only when
  running it would mismeasure, hang, fail as a whole block, waste significant
  time, or time a model the backend cannot serve (`layer_permissions.md`,
  "Leaving cases out").
- **Everything finer runs**: a head count, a shape boundary, one framework's
  own limit. If it fails, the failure is recorded and compared with the
  previous run of the same (backend, version).
- **Capacity** is refused at run time, before allocation. The plan does not
  depend on the device (`layer_permissions.md`, "Capacity").
- **Failures are counted per consumer key**: the dimensions the SDK matches
  exactly (dtype or quant, head count, head_dim, window, experts, top-k, TP,
  EP). A key bucket where (almost) every case fails is an unavailable key, a
  fact to report. Targets: scattered failures at most 5% of a table; a bucket
  below 20% success counts as unavailable.

## Collection cost

Answer each question with the cheapest check that settles it:

1. GPU-free first: a plan diff (`--plan-only`, `--plan-table`), a framework
   source check, a unit test.
2. Then a targeted run of only the affected cases (`--ops`, `--model-path`,
   `--case-filter`) on one system per SM generation.
3. At a pin bump, recollect only the families whose kernels changed and
   declare reuse for the rest.

## Rules

| # | Rule | Checked by |
|---|---|---|
| 1 | Each (backend, op) has one pin, and the collector refuses any other installed version | The collector at start-up; `test_framework_manifest.py`; R5 |
| 2 | Measure what serving runs: the same kernel, fusion, graph or eager mode and autotuning. When that is impossible, write a deviation note | Review; `backend_facts.py --check` for labels |
| 3 | One case list per op for every backend; each getter is the one-line `plan_cases` call | `test_op_plans.py` |
| 4 | Leave cases out only through `collector/skips.py`, each entry with its reason as a comment | `test_skips.py`, `test_op_plans.py`, review |
| 5 | Rows depend only on the pin and the plan, never on environment variables | Review |
| 6 | Every table has a record; failures stay within budget and unavailable keys are reported | R1-R7; budget counts by hand |
| 7 | Every `collector/skips.py` entry and deviation note in scope is re-checked at a pin bump | The upgrader, in review |

## What to record, and when

| When | What | Who |
|---|---|---|
| Every collection run | Runtime identity, `collector_hash`, `case_plan_hash`, result counts and the failed cases, execution mode (graph or eager, autotuned or not), run scope (full plan or a subset) | The collector (counts and execution mode by hand) |
| Every collector change | Affected ops; whether measured values change, with a before/after diff of the affected ops on one system per SM generation when they do; the `collector/skips.py` entries and notes added or changed | The author, in the PR |
| Every pin bump | Recollect the families whose kernels changed and declare reuse (`reuse.yaml` with `approved_by`) for the rest; re-check each `collector/skips.py` entry and note in scope by a source check or a targeted run, keeping or deleting it; compare the failure set with the previous run | The upgrader |

## Open gaps

| Piece | Today | Target |
|---|---|---|
| moe_a2a | Each backend's distributed collector builds its own population for one launched world | Stays: each backend measures a different all-to-all library |
| vLLM XPU (`REGISTRY_XPU`) | Its own getters | Op plans |
| Record | Provenance and reuse; failures only in `errors_*.json` | Result counts, unavailable keys, execution mode |
| Consumer signal | Extrapolated answers are silent | The engine tags extrapolation and reports grid coverage |
