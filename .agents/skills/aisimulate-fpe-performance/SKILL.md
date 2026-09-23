---
name: aisimulate-fpe-performance
description: Assess host runtime cost when implementing or reviewing AISimulate FPE changes, including features and correctness fixes. Use for prediction calls, performance databases, interpolation, indexes, caches, model setup, timing evidence, and Rust-Python conversions in prediction or replay. Require matched evidence for likely added hot-path work, even when performance is not the stated task.
---

# AISimulate FPE performance

Apply this guide to feature work, correctness fixes, refactors, and performance work.
Assess simulator host CPU and wall time separately from predicted serving latency.
Small host costs multiply across operations, layers, phases, and simulated requests.

## Assess the cost before adding work

- Identify setup, first-use, cache-hit, and cache-miss paths. Estimate call frequency per
  engine step and replay, and how work and copied state scale with input size.
- Identify affected models, database modes, and callers. Check whether a narrow feature
  adds work to common paths or configurations that do not use it.
- Prefer existing prepared data, indexes, borrowed immutable state, and bounded work.
  Justify a new cache with expected reuse, complete keys, data lifetime, and memory cost.
  Justify complex ownership with measured benefit to the caller.

## Inspect these recurring costs

- Scanning stable tables or rebuilding interpolation grids, indexes, or model data for
  each query. A small-table scan can still be cheaper than an index; measure both sizes.
- Cloning cached evidence, strings, vectors, or source tags on a cache hit; validating
  the same immutable evidence again during construction, lookup, and accumulation.
- Converting native results through Python objects before converting them back to Rust.
- Rebuilding evidence layouts or temporary buffers for each operation; doing bookkeeping
  when the applicable evidence is empty or has no consumer.
- Repeated parsing, hashing, key formatting, locking, or shared-cache mutation.
- Moving cost into another helper, setup, destruction, or report generation.

Concrete examples: [#239](https://github.com/ai-dynamo/aisimulate/pull/239) indexes repeated
prepared-grid scans; [#295](https://github.com/ai-dynamo/aisimulate/pull/295) removes repeated
evidence copies, validation, and native-to-Python conversions.

## Protect behavior

- Preserve prediction values, floating-point operation order, deterministic ties,
  equality handling, and fallback selection.
- Preserve provenance, diagnostics, coverage, misses, typed errors, error precedence,
  atomic failure, and negative-cache behavior, including power and energy evidence.
- Keep validation at public boundaries. Internal reuse of validated state requires
  established immutability and layout invariants; do not remove checks from untrusted
  input paths. Compare an internal fast path with an independent checked reference.
- Keep cache keys complete and lifetimes within the data they describe. Check SILICON,
  HYBRID, and EMPIRICAL behavior separately where affected.

## Require evidence proportional to the cost

- For a cold or rare path, give a supported bound on frequency, input size, and total
  work. A bounded cold-path change does not require a benchmark by default.
- For likely added hot-path work, require repeated matched before/after timings before
  performance approval. A profile locates cost; it does not establish no regression.
- Use the existing [forward-prediction tools](../../../python/aisimulate/tools/forward_perf_gate/README.md)
  for affected prediction calls. State the actual setup, cold-query, and warm-query
  boundaries; different harnesses use different meanings of "cold".
- Also compare native `aisimulate predict --stack engine` runs when evidence processing,
  ownership, cache lifetime, or call structure can change integrated cost. The forward
  gate calls `run_static_latency_only`; it cannot clear replay or evidence-path costs.
- Match immutable revisions, release builds, dependencies, data, workload, and host
  settings. Verify the loaded native binaries. Use repeated paired timings on a
  representative workload and a small control; keep profiles separate from timed runs.
- Verify equivalent work and behavior first. Missing data, skipped cases, incomplete
  execution, or changed outputs leave a coverage or behavior gap, not a speed result.
- For replay, `wall_time_ms` includes native preparation, execution, and report
  aggregation. It excludes Python startup, trace loading, and output serialization;
  it is neither loop-only timing nor full CLI time.
- Reject complexity without repeatable integrated benefit. A useful feature can add
  cost, but measure and justify it; this guide sets no universal slowdown threshold.

## Report within the task's permissions

Report the cost concern and scaling, matched evidence and timing scope, behavior checks,
and remaining uncertainty. Separate a source-level risk from a measured regression.
Missing measurements are an evidence gap and do not support performance approval. In a
source-only review, identify the needed comparison without starting builds, profiles, or
benchmarks that the task does not authorize.
