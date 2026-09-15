---
name: aisimulate-replay-performance
description: Review AISimulate engine-only replay changes for performance risk. Use whenever modifying or reviewing replay scheduling, event queues, worker lifecycle, request lowering, hashing, KV ownership or accounting, backend event handling, report generation, or other replay-loop code, even when the task is not explicitly about performance. Excludes Dynamo Router, Planner, KVBM, and live event-plane behavior.
---

# AISimulate replay performance awareness

Use this checklist while implementing or reviewing AISimulate-owned engine replay.
Replay costs multiply across requests, tokens, workers, timestamps, and backend events.

## Locate the cost

- Determine whether work runs once, per request, per token or block, per event, per
  worker, or once for every phase at the same timestamp.
- Check whether the change adds work to a common path for only one backend's benefit.
- Inspect both the representative saturated topology and a small control; they expose
  different overheads.
- Separate input preparation, replay-loop work, and report finalization.

## Watch for code smells

- Per-request cloning, allocation, sorting, serialization, or temporary containers.
- Repeated prompt expansion, token traversal, hashing, or KV-accounting work.
- Full queue, worker, cache, or timestamp scans where incremental state already exists.
- Broad same-timestamp loop restarts after phases that cannot expose earlier work.
- Shared locks, global barriers, centralized merge work, or slowest-worker waits.
- Repeated lifecycle or backend-event bookkeeping with no consumer.
- Backend-specific data or branches placed on every backend's hot path.
- A local improvement that moves cost into cleanup, accounting, or report generation.

## Protect behavior

- Preserve request and token totals, completion of every lifecycle, and canonical report
  parity.
- Preserve deterministic same-timestamp order and stable tie behavior.
- Preserve cache ownership, capacity, eviction order, preemption, retraction, handoff,
  and backend event semantics.
- Treat zero errors as insufficient when work can be dropped, duplicated, or left
  unfinished.

## Choose proportional validation

- For a bounded cold path, explain why it cannot scale with requests, tokens, or events.
- For a plausible hot-path change, profile the exact release binary and check capture
  boundaries and lost samples.
- For a candidate optimization, compare equal work on a saturated case and a small
  control. Run `$aisimulate-replay-parity` when output or lifecycle behavior can change.
- Combine related symbol families when a refactor moves work between helpers.
- Stop when the result does not repeat, misses its stated gate, changes work, or does not
  justify the added complexity.

## Respect the timer boundary

Current `wall_time_ms` includes input preparation and report finalization. Use it only as
end-to-end replay timing. Do not claim a replay-loop pass, regression, or speedup until a
supported loop-only metric exists.

Route Dynamo Router, Planner, KVBM, discovery, scaling, and live event-plane performance
work to the corresponding Dynamo workflow.
