---
description: >
  The collector in five things (Pin, Plan, Record, Note, Rule), the failure
  principle, the seven rules and their checkers, and what to record when.
paths:
  - "collector/**"
  - "tests/unit/collector/**"
  - "tools/perf_database/**"
---

# Collector Overview

Read this page first. The other files in this directory go deeper on one topic
each and use the words defined here. If one of them disagrees with this page,
that file is out of date.

## Five things

| Thing | What it is | Where it lives | Written by | Used by |
|---|---|---|---|---|
| **Pin** | The framework version and image each (backend, op family) is collected with | `collector/framework_manifest.yaml` | Owner, at an upgrade | Collector (refuses any other installed version), CI |
| **Plan** | Every case an op collects | `collector/cases/`: base grids, model shapes, `capabilities.yaml` floors and backend gaps (see `case_authoring.md`). `collector/op_plans.py` expands each migrated op (`OP_PLANS`) once for every backend, and `collect.py --plan-only --sm N` counts it | Owner or agent | Collector, reviewers, CI |
| **Record** | How a data table came to be: collected or reused, with which runtime, code and plan, and how the run went | `collection_meta.yaml` and `reuse.yaml` next to the parquet | Collector writes `collection_meta.yaml`; a person writes `reuse.yaml` | SDK loader, CI (R1-R7) |
| **Note** | A known gap the rows cannot show. A **deviation** (`D-` id): the table measures a different path than serving runs. A **guard** (`G-` id): collector code, or a `backend_gaps` entry in `capabilities.yaml`, refuses cases that would mismeasure, hang or waste significant time | `collector/notes/<backend>.yaml`; fields in `docs/perf-model/collector/data-format.md` | Owner or agent, accepted in review | Code cites guard ids; reviewers and consumers read deviations; CI (R8) |
| **Rule** | What every collector change must satisfy | The seven rules below | Owner | Agents, reviewers, CI |

Each word has one meaning:

- **Evidence** is raw material under `benchmarks/evidence/<id>/`: logs, probe
  results, before/after diffs. A Note or a Record may cite it.
- A conclusion drawn from evidence is a **Note**. Do not create findings
  files, waivers, exception registries or evidence policies; those mechanisms
  were removed.
- A fact has one home. Code, data comments and docs cite a note id instead of
  restating the note.

## The failure principle

Generate the same cases for all three frameworks. A few boundary points may
fail: a failure is a recorded observation, not a defect to filter away.

- **Whole regions that cannot run** (an SM without a dtype, an op that needs a
  newer SM, or a dtype one backend's collector has no kernel for on an SM) are
  removed when the plan is expanded: `capabilities.yaml` floors and
  `backend_gaps`, or a quant-axis `min_sm`. A backend gap cites a guard note.
  This is the only plan-time filter.
- **Everything finer** runs: a head count, a shape boundary, one framework's
  own limit. If it fails, the failure is recorded and compared with the
  previous run of the same (backend, version).
- **A code guard** exists only when running the case would mismeasure (succeed
  on the wrong kernel), hang or kill the worker, or waste significant time.
  Each guard raises in the worker and cites a `G-` note.
- **Failures are counted per consumer key**: the dimensions the SDK matches
  exactly (dtype or quant, head count, head_dim, window, experts, top-k, TP,
  EP). A key bucket where (almost) every case fails is an unavailable key, a
  fact to report, not a failure. Targets: scattered failures at most 5% of a
  table; a bucket below 20% success counts as unavailable.

## Collection cost

Avoid full-plan collections. Answer each question with the cheapest check
that settles it:

1. GPU-free first: a plan diff, a framework source check, a unit test.
2. Then a targeted run of only the affected cases (`--ops`, `--model-path`,
   `--case-filter`) on one system per SM generation.
3. At a pin bump, recollect only the families whose kernels changed and
   declare reuse for the rest.

## Seven rules

| # | Rule | Checked by | Status |
|---|---|---|---|
| 1 | Each (backend, op) has one pin, and the collector refuses any other installed version | Collector at start-up; `test_framework_manifest.py`; R5 | In force |
| 2 | Measure what serving runs: the same kernel, fusion, graph or eager mode and autotuning. When that is impossible, write a deviation note | Op-probe comparison and review; `backend_facts.py --check` for labels | In force, by review |
| 3 | Generate the same cases for every framework; never filter per framework inside a getter | Review; for ops with an op plan, `test_op_plans.py` | Ops in `OP_PLANS` migrated; the other getters migrate op by op |
| 4 | Filter only whole regions at plan expansion; guard only mismeasure, hang or expensive cases, each with a `G-` note | Review; `test_capabilities.py` checks that each backend gap cites a guard note | In force for new code; existing guards migrate |
| 5 | Rows depend only on the pin and the plan, never on environment variables | Collector refuses known switches | Partly in force |
| 6 | Every table has a record; failures stay within budget and unavailable keys are reported | R1-R7; budget counts by hand until the Record migrates | Partly in force |
| 7 | Notes are re-checked at every pin bump: `verified_on` is never behind the pin | R8 | In force |

## What to record, and when

| When | What | Who |
|---|---|---|
| Every collection run | Runtime identity, plan hash, result counts and the failed cases, execution mode (graph or eager, autotuned or not), run scope (full plan or the CLI subset) | The collector, automatically (counts and execution mode by hand until the Record migrates) |
| Every collector change | Affected ops (Full CI `changed-ops` report); one line saying whether measured values change, with a before/after diff of the affected ops on one system per SM generation when they do; the notes diff | The author, in the PR |
| Every pin bump | Recollect the families whose kernels changed and declare reuse (`reuse.yaml` with `approved_by`) for the rest; re-check every note in scope by a source check or a targeted run, then move `verified_on` or delete the note; compare the failure set with the previous run | The upgrader |

## Migration status

| Piece | Today | Target |
|---|---|---|
| Notes | In force (R8) | Done |
| Evidence policy, resolver, waivers | Removed | Done |
| Plan | Ops in `OP_PLANS` (`collector/op_plans.py`): one case list for every backend, floors and backend gaps applied at expansion, `--plan-only --sm N` counts and `--plan-table` rows. Other ops: YAML inputs plus Python getters | Every op on an op plan; model dimensions read from bundled HF configs |
| Record | Provenance and reuse; failures only in `errors_*.json` | Result counts, unavailable keys, execution mode |
| Kernel-limit guards | GEMM has none. Other ops: `FIXME(kernel-limit)` raises, classified `kernel_limit` | Only mismeasure, hang and expensive guards, each citing a `G-` note; other limits simply fail |
| Consumer signal | Extrapolated answers are silent | The engine tags extrapolation and reports grid coverage |
