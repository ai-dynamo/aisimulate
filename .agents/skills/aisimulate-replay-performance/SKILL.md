---
name: aisimulate-replay-performance
description: Assess host runtime cost when implementing or reviewing AISimulate engine-only replay changes, including features and correctness fixes. Use for scheduling, admission rollback, event queues, worker lifecycle, request lowering, KV ownership, timing evidence, and reports. Require matched evidence for likely added hot-path work. Excludes Dynamo Router, Planner, KVBM, and live event-plane behavior.
---

# AISimulate replay performance

Apply this guide to feature work, correctness fixes, refactors, and performance work in
AISimulate-owned engine replay. Assess host CPU and wall time separately from simulated
serving latency. Costs multiply across requests, tokens, workers, and backend events.

## Assess the cost before adding work

- Identify work done once, per request, per token or block, per event, per worker, or
  per same-timestamp phase. Estimate its frequency and input-size dependence.
- Identify copied state and its size: resident cache, page pool, waiting leases, queues,
  and evidence. A cheap-looking `clone()` can grow with the whole simulation state.
- Check whether a narrow backend or model feature adds cost to other configurations,
  successful admissions, disabled features, or requests that need no state change.
- Prefer existing incremental state, borrowed immutable data, and bounded work. Justify
  extra caches or complex ownership with reuse, memory cost, and integrated benefit.

## Inspect these recurring costs

- Whole-cache, page-pool, or lease snapshots for every admission attempt, including
  successful attempts. Delaying a snapshot does not bound the cost of each remaining copy.
- Repeated queue, worker, cache, or timestamp scans; sorting or temporary containers
  where incremental state already exists.
- Repeated prompt expansion, token traversal, hashing, or KV-accounting work.
- Broad same-timestamp restarts; shared locks, global barriers, or slowest-worker waits.
- Event, lifecycle, or timing-evidence bookkeeping with no consumer; backend-specific
  data added to every backend's path.
- Cost moved from a local helper into cleanup, accounting, or report generation.

Concrete example: [#321](https://github.com/ai-dynamo/aisimulate/pull/321) reduces admission
snapshot copies added by a correctness change. Check both snapshot frequency and copied
state size; preserve rollback before considering lazy snapshots or shared storage.

## Protect behavior

- Preserve request and token totals, completed lifecycles, canonical report parity,
  deterministic same-timestamp order, and stable ties. Zero errors do not prove that
  work was not dropped, duplicated, or left unfinished.
- Preserve cache ownership, capacity, eviction, preemption, retraction, and handoff.
- Keep admission transactional. Prefix matching can split, touch, and lock cache nodes:
  establish rollback state before the first mutation, not merely before allocation.
  Restore queue and lease state on failure. Buffer events until commit and preserve
  their order; failure must not publish events or consume state needed by a retry.
- Share immutable state only with defined mutation isolation. Retain validation at
  public boundaries and preserve evidence, diagnostics, and error behavior.
- When output or lifecycle behavior can change, read the existing
  [replay-parity guide](../../../python/aisimulate/.agents/skills/aisimulate-replay-parity/SKILL.md).
  Its semantic qualification is separate from performance measurement.

## Require matched native replay evidence

- For a cold or rare path, give a supported bound on frequency, input size, and total
  work. A bounded cold-path change does not require a benchmark by default.
- For likely added hot-path work, require repeated matched before/after timings before
  performance approval. Use native `aisimulate predict --stack engine` runs. A forward
  prediction benchmark does not cover scheduling, admission, evidence, or report costs.
- Match immutable revisions, release builds, dependencies, data, workload, and host
  settings. Verify loaded native binaries. Use repeated paired timings on a representative
  workload and a small control. For cache snapshots, include a workload with substantial
  resident state and reuse, not only a short or empty-cache run.
- Check equivalent work and behavior before interpreting speed differences. Missing
  data, skipped coverage, incomplete execution, or behavior changes do not establish a
  speed result. Explain intentional added work separately from avoidable overhead.
- Use separate profiles to locate cost; check capture boundaries and lost samples.
  Combine related symbol families when helpers change. Keep builds and profiles out of
  timing runs, and retain attempted samples and explained failures.
- Reject complexity without repeatable integrated benefit. A useful feature can add
  cost, but measure and justify it; this guide sets no universal slowdown threshold.

## State the timing scope and result

`wall_time_ms` measures native preparation, replay execution, and report aggregation.
It excludes Python startup, trace loading, and output serialization. It supports a native
replay comparison, not a loop-only or full CLI claim. Report those scopes separately if
measured; do not confuse host runtime with simulated serving latency.

Report the cost concern and scaling, matched evidence and timing scope, behavior checks,
and remaining uncertainty. Separate a source-level risk from a measured regression.
Missing measurements are an evidence gap and do not support performance approval. In a
source-only review, identify the needed comparison without starting builds, profiles, or
benchmarks that the task does not authorize.

Route Dynamo Router, Planner, KVBM, discovery, scaling, and live event-plane performance
work to the corresponding Dynamo workflow.
