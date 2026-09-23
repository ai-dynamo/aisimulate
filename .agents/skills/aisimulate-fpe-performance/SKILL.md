---
name: aisimulate-fpe-performance
description: Design, implement, and review efficient AISimulate numerical prediction and data lookup. Use for FPE features, correctness fixes, and refactors that affect prediction algorithms, preparation, data representation, caches, or caller cost. Resolve substantial avoidable overhead before completion, even when performance is not the stated task.
---

# AISimulate FPE performance

Design new and changed code for efficient execution. Functional correctness and
efficient execution are both completion requirements. Resolve substantial, avoidable
overhead in the same PR before marking the implementation ready for merge.
Assess simulator host cost separately from predicted serving latency.

## Design

- Identify the expected workload, query frequency, input sizes, and data lifetime.
  Include affected callers, models, and database modes. Small costs can multiply across
  operations, layers, phases, and requests; state how the work scales.
- Choose algorithms, data structures, and ownership for those conditions. Compare total
  cost: preparation, first query, repeated queries, retained memory, and cleanup. An
  index or precomputation must earn its setup and storage cost through expected use;
  a simple scan can be appropriate for small inputs.
- Examine data representation and movement as well as arithmetic. Identify allocations,
  copied state, conversions, and memory access patterns on the caller's full path.
  Check whether optional functionality adds cost to configurations that do not use it.
- Prefer existing prepared data and borrowed immutable state where their lifetime fits.
  Justify added caching, concurrency, or ownership complexity with a concrete need and
  expected benefit. For a cache, account for reuse, hit and miss costs, key completeness,
  invalidation, and retained memory; a high hit count alone is not a benefit.

## Implement

- Keep work proportional to the requested prediction or lookup. Avoid redundant
  computation, unnecessary data movement, and repeated preparation of stable inputs.
  Inspect helpers and callers for costs hidden behind convenient APIs. Moving work to
  setup, another language, or cleanup does not remove it from the total cost.
- Use the simplest design that meets the workload and correctness needs. Do not add
  elaborate machinery to avoid a small bounded operation. Reuse validated internal
  state only when its immutability and layout invariants are established; keep checks
  at public boundaries and on untrusted input paths.
- Preserve numerical results and floating-point operation order, deterministic ties,
  equality handling, and fallback selection. Preserve provenance, diagnostics, coverage,
  misses, error behavior, and atomic failure. Check affected prediction modes and
  compare optimized paths with an independent checked reference where applicable.

## Validate and review

- Trace the changed execution paths and confirm that the evidence exercises them under
  realistic sizes and reuse patterns. Include caller overhead and affected configurations.
  An unchanged kernel benchmark cannot qualify a change outside that kernel.
- Choose proportionate evidence. For a cold or small path, a supported bound on frequency,
  input size, total work, and retained memory can suffice. For likely added hot-path work
  or material cost changes, require repeated matched before/after measurements. Inspect
  scaling at relevant sizes; a tiny input alone does not establish large-input behavior.
- Use the existing [forward-prediction tools](../../../python/aisimulate/tools/forward_perf_gate/README.md)
  for affected prediction paths. State preparation, first-use, and warm-query boundaries.
  Require native `aisimulate predict --stack engine` comparisons when evidence processing,
  scheduling, ownership, or other integrated costs can change. Forward-only timings
  cannot clear those paths.
- Record exact baseline and candidate revisions. Match release builds, dependencies,
  data, workload, and host settings, and verify the loaded native binaries. Use repeated
  paired timings on a representative workload and a small control. Report variation;
  keep profiles separate from timing samples. Measure memory when retention can change.
- Check equivalent work, complete execution, and required behavior before interpreting
  speed differences. Missing data, skipped coverage, and incomplete execution are
  missing evidence. For affected replay semantics, read the existing
  [replay-parity guide](../../../python/aisimulate/.agents/skills/aisimulate-replay-parity/SKILL.md);
  semantic qualification and performance measurement are separate checks.
- For native replay, `wall_time_ms` includes native preparation, execution, and report
  aggregation. It excludes Python startup, trace loading, and output serialization.
  Distinguish it from loop-only timing, full CLI time, and simulated serving latency.
- Investigate material costs and correct avoidable overhead before completion. Passing
  tests or documenting a slowdown is not sufficient. Distinguish necessary feature
  cost from implementation waste; explain material necessary costs, measured impact,
  and considered alternatives. There is no universal slowdown limit. Accept an efficient
  implementation with sufficient evidence without demanding further optimization.

Report the cost assessment, evidence and timing scope, behavior checks, and remaining
uncertainty. Distinguish source-level risks, measured regressions, and inconclusive
results. Missing required measurements leave an evidence gap, not performance approval.
In a source-only review, identify the concern and needed comparison without starting
builds, profiles, or benchmarks outside the task's permissions.

## Examples

- [#239](https://github.com/ai-dynamo/aisimulate/pull/239): repeated searches over prepared
  data made query cost grow with the full dataset and increased replay CPU time.
- [#295](https://github.com/ai-dynamo/aisimulate/pull/295): auxiliary result processing
  added copying, validation, and conversion costs outside the numerical calculation.

These illustrate failure patterns and consequences; their solutions are not requirements.
