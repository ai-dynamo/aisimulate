---
name: aisimulate-fpe-performance
description: Review AISimulate Forward Pass Engine and performance-database changes for performance risk. Use whenever modifying or reviewing prediction calls, SILICON, HYBRID, or EMPIRICAL queries, interpolation, lookup indexes, caches, model or database setup, or other code under the FPE hot path, even when the task is not explicitly about performance.
---

# AISimulate FPE performance awareness

Use this checklist while implementing or reviewing FPE changes. Small costs can multiply
by every operation, layer, phase, and simulated request.

## Locate the cost

- Determine whether the code runs during setup, first use, or every prediction call.
- Estimate how often it runs in one engine step and in a representative replay.
- Check whether work already has a stable cache, index, or prepared representation.
- Distinguish local lookup cost from the end-to-end cost seen by its caller.

## Watch for code smells

- Linear scans over stable tables or grids on every query.
- Rebuilding immutable indexes, interpolation grids, model data, or query plans.
- Repeated string formatting, key construction, parsing, hashing, allocation, or cloning.
- Lock acquisition or shared-cache mutation on a read-heavy path.
- Whole-result caches added without evidence that exact requests repeat.
- Setup or validation work that moves into the warm prediction path.
- A refactor that moves work between helpers and makes one symbol look cheaper without
  reducing the combined cost.

## Protect behavior

- Preserve prediction values and the production floating-point operation order.
- Preserve deterministic tie ordering, equality handling, and fallback selection.
- Preserve data provenance, successful results, misses, typed errors, and negative-cache
  behavior.
- Keep cache keys complete and cache lifetimes no longer than the data they describe.
- Treat SILICON, HYBRID, and EMPIRICAL paths as different behaviors; do not assume an
  improvement or parity result transfers between them.

## Choose proportional validation

- For a clearly cold or rare path, explain why its runtime impact is bounded.
- For a plausible hot-path change, use a focused profile or the existing
  `crates/core/parity_tests/perfmodel/benchmark_engine_step.py` harness.
- For a cache, ownership, or call-structure change, also use a representative integrated
  replay so moved costs remain visible.
- Compare equal work and identical predictions. Separate setup, first-use cold cost, and
  steady warm cost.
- Verify the executed source and native binary before trusting a profile.
- Reject extra complexity when the repeatable integrated return does not justify it.
