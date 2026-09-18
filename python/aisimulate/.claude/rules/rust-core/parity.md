---
description: >
  Regression discipline for the compiled engine: the frozen golden fixtures
  are the engine-step spec, deliberate modeling changes carry their golden
  diff, and per-op performance values are computed ONLY in the Rust engine
  (single oracle — #1357 Phase 3).
paths:
  - "crates/core/src/perfmodel/**"
  - "crates/core/perfmodel/**"
  - "crates/core/parity_tests/perfmodel/**"
  - "python/aisimulate/src/aisimulate_core/sdk/operations/**"
  - "python/aisimulate/src/aisimulate_core/sdk/perf_database.py"
  - "python/aisimulate/src/aisimulate_core/sdk/engine.py"
  - "python/aisimulate/src/aisimulate_core/sdk/rust_engine_step.py"
  - "python/aisimulate/tests/unit/sdk/test_opspec_coverage.py"
  - "python/aisimulate/tests/cross_package/test_single_oracle_contract.py"
---

# Rust-Core Regression Discipline (single-oracle era)

The compiled engine (`crates/core/src/perfmodel`) is the ONLY
engine-step executor AND the only per-op performance oracle — op-level and
FPM models alike. The Python engine-step path and the Python per-call query
math are gone; final answers are frozen in
`crates/core/parity_tests/perfmodel/goldens/` (and, for the per-run
synthetic FPM fixture, inline `_FPM_*_FROZEN` tables) and the parity suites
assert live-Rust-vs-frozen.

## Rule 1 — engine-step changes carry their golden diff

Any PR that changes what the compiled engine computes (operators, loaders,
interpolation, selection rules, engine composition) MUST in the same PR:

1. Keep `test_engine_step_parity.py` / `test_compile_engine_parity.py`
   green — either the change is answer-preserving (goldens untouched), or it
   is a deliberate modeling change and the PR refreshes the affected records
   with `crates/core/parity_tests/perfmodel/pin_goldens.py --refresh <keys>`
   (or `--refresh-all`) and lets the GOLDEN DIFF carry the review: reviewers
   see exactly which numbers moved and by how much. Never refresh to silence
   an unexplained failure.
2. Anchor new behavior: an oracle test in the Rust `#[cfg(test)]` module
   (hand-derived or generated from the modeling spec — there is no live
   Python reference to generate against), and/or a parity case pinned via
   `pin_goldens.py` (append-only mode) when a new config class becomes
   reachable.
3. Run both parity suites and cite the exact commands and results. Only
   repository-root workflows are active in AISimulate; the migrated nested
   `rust-engine-step-parity` workflow is not hosted-CI evidence.

## Rule 2 — the single-oracle invariant (per-op values live in Rust ONLY)

Per-op performance VALUES — latency, energy, the SOL decomposition — are
computed only by the compiled engine. Python owns model/topology
composition, data loading, orchestration, and presentation; it never owns
estimation math. Concretely:

- Do NOT add Python-side interpolation, roofline/SOL formulas,
  empirical-utilization estimates, or per-call table lookups anywhere under
  `python/aisimulate/src/aisimulate_core/sdk/` (banned def shapes: the
  `_query_*` and `_lookup_*` prefixes, `get_sol`, `get_empirical`). The
  correct home is the Rust operator/table layer plus, if needed, a new
  engine FFI.
- The deprecated `PerfDatabase.query_*` and public `Operation.query` shims
  have been removed. Do not reintroduce them. New per-op access goes through
  `EngineHandle.evaluate_ops_json` / `evaluate_ops_sol_json`, the per-phase
  surface (`run_static_per_op`), or whole runs. `Operation.query` bodies are
  limited to the orchestration whitelist (AFD comm ops) and must compose
  ENGINE-evaluated twin ops rather than implement performance math.
- The deliberate-edit gates are
  `python/aisimulate/tests/cross_package/test_single_oracle_contract.py`
  (frozen surfaces, banned def names, def inventories, and whitelists). A PR
  that must grow an exemption or whitelist there needs an explicit
  justification in its description.
- Name-based guards cannot catch a determined rename; treat any Python code
  that turns shapes+tables into latency as a violation regardless of its
  name.

## Adding a new Operation

A new `Operation` subclass must get a `_to_opspec` branch in
`sdk/engine.py`, an `Op` variant in
`crates/core/src/perfmodel/operators/op.rs` (**append at the tail**
— bincode variant indices are positional; mid-enum insertion requires an
`ENGINE_SPEC_SCHEMA_VERSION` bump on BOTH sides), the `engine/spec.rs`
round-trip fixture, and a pinned parity case
(`crates/core/parity_tests/perfmodel/pin_goldens.py`).
`python/aisimulate/tests/unit/sdk/test_opspec_coverage.py` fails until the op
converts or carries a justified `EXEMPT` entry — an unconvertible op in a
shipped model is a HARD ERROR at estimation time, not a silent fallback. If
the op must be reachable through the internal `_engine_query` kwarg mapping,
give it an `_ENGINE_QUERY_SHAPE` (`tokens`, `context`, `generation`, or
`module`) — never a public Python `query` body.

Name the independent expected value, not only the code path. A unit test that
pins an exact latency, energy, or SOL value must state how that value was
derived (hand calculation, published spec, traced measurement, or an existing
named golden). Do not copy the production formula into the test and call it an
oracle.

## Selection rules are regression surface too

Table/slice/kernel selection changes move golden numbers exactly like
formula changes; the same golden-diff rule applies. Python dicts iterate in
file/insertion order; Rust `BTreeMap` iterates sorted — any "first
available" fallback needs a load-order record (see `quants_in_load_order` /
`first_distribution` in `perf_database/{moe,moe_expert_compute}.rs`).

## Known intentional splits (do not "fix" without the tracking issue)

- AFD and the VL encoder phase are Python-side ORCHESTRATION; their per-op
  values cross the engine FFIs (the AFD comm ops evaluate standard
  P2P/NCCL/ElementWise twins through the single-op plumbing).
- `SOL_FULL` is a per-call diagnostic (never a selectable default mode); it
  is served by `evaluate_ops_sol_json` and raises for op families whose SOL
  path does not export its decomposition.
- Estimate-only systems (a spec yaml with no collected data) load under the
  SOL view only (`load_with_sources_opts` tolerance); every other mode keeps
  the loud missing-directory gate.
- The Python-loaded tables are the RAW collected data plane (enumeration,
  charts, support matrix) — no load-time SOL clamp or grid pre-expansion;
  the engine clamps and interpolates its own load.
- Rust reads parquet only (no `.txt` legacy loading) — new data drops must
  ship parquet.
