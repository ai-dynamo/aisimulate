<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# PR #533 review fixes and validation

Implementation commit `1de653c9ef28788a877a660cb81cb3379edb8c6a` fixes default
configuration compatibility, accepts explicit disabled speculation, expires
online observations against the global admission sequence, rejects unsupported
learned callers and fallback policies before source loading, and makes Rust own
the exported deployment identity. Native type stubs now include the online API.

The cross-store regression reproduces three 30 ms prefill labels followed by
six decode labels with `update_every=2`, `recent_window=16`, maximum age 3, and
zero historical weight. Flushing removes expired prefill labels and restores the
10 ms anchor prediction. Additional tests cover eager/lazy fitting, fitted-only
and pending-only stores, inclusive age boundaries, rejected observations,
rollback and checkpoint continuation. The runtime checkpoint schema is unchanged.

[validation.json](validation.json) records 263 Rust FPM tests, 178 SDK/API tests,
369 configuration/caller tests, 519 Runner tests and 389 frozen-parity tests,
all passing against the implementation commit. Both frozen golden files are
unchanged. Python checks use a freshly built candidate release extension, whose
hash and build command are recorded. [Compatibility evidence](compatibility/README.md)
checks normalized configurations with the actual prior-main Rust deserializer.
These are local results; they do not assert GitHub CI status on this revision.

## Frozen anchor-prefix cache

The immutable historical anchors now cache the baseline plus frozen-tree
prediction prefix. Recent rows still traverse the frozen trees; mutable trees
continue to fit normally. The cache preserves baseline-first, tree-order
floating-point accumulation, is derived again after grouping or checkpoint load,
and is omitted from serialized checkpoints. No cache allocation is made when all
trees are mutable. Memory diagnostics include its retained capacity.

The baseline is PR head `45b2c98a0f040d90e83cd5c1ff51c108f0cd8223`; the candidate
is the implementation commit above. The paired benchmark uses 150 trees with at
most 15 initial leaves, one synthetic decode store, and 32 eager updates per
fresh process. Five paired rounds across three cases and separate timing and
allocation phases give 60 processes and 1,920 accepted updates. Arm and case
order reverse each round. No build or test runs overlapped the measurement;
the desktop host was not CPU-pinned. Thread-count environment variables were one.

Each timing cell is the median of five process-level quantiles. Ranges describe
the five paired p50 changes, not confidence intervals. Arrows are baseline to
candidate. Setup time is measured separately from the feedback call.

| History / mutable trees | Update p50, ms | Update p95, ms | Paired p50 change, median [range] | Setup median, ms |
| --- | ---: | ---: | ---: | ---: |
| 256 / 1 | 1.036 → 0.443 | 1.341 → 0.710 | −56.98% [−60.53%, −54.76%] | 61.46 → 62.89 |
| 10,000 / 1 | 14.797 → 1.656 | 15.417 → 2.269 | −88.73% [−88.93%, −88.42%] | 188.05 → 200.75 |
| 10,000 / all | 249.469 → 248.344 | 267.432 → 265.945 | −0.26% [−3.23%, +1.65%] | 187.44 → 186.96 |

The cache adds 2,048 retained bytes for 256 anchors and 75,640 bytes for the
9,455 grouped anchors in the larger fixture. The all-tree case has unchanged
retained bytes. Requested allocation per update and maximum additional live
requested bytes are unchanged in all three cases. The cache is allocated during
setup. The large-tail setup median increases by about 12.70 ms.

Every process passed exact equality checks for initial predictions, pre-feedback
prediction trajectories, final predictions, final exported artifacts and semantic
diagnostics. Each case also matches configuration, seed-manifest and query hashes
across arms, rounds and phases. All 32 observations were accepted, published and
fitted eagerly; no pending observations or update failures remained. Diagnostic
comparison excludes grouping time and capacity-based training bytes. Binary hashes
are checked before and after measurement. This supports a lower eager-update cost
for these restricted-tail cases without changed numerical results. It does not
measure serving accuracy, replay speed, lazy-update cost or long-run eviction cost.

[measurements.json](measurements.json) contains all process summaries and exact
equality hashes. [samples.json](samples.json) retains per-update timings and
allocation counters, with hashes of the original process results.
[provenance.json](provenance.json) records revisions, binary/source/input hashes,
hardware and toolchain. Allocation counting measures Rust requested bytes and
additional live requested storage, excluding allocator metadata and non-Rust
allocation; it does not measure CPU bytes copied. RSS was not collected on macOS.

## Reproduction

The checked-in [probe.rs](probe.rs) is the executed copy of the existing
[prefix probe](../fpm_frozen_prefix_v1/probe.rs), with only `memory_kib()` changed
to return null fields because this host has no Linux `/proc` RSS counters.
The [controller.py](controller.py) runs the paired experiment and equality gates.
Timing measures the complete synchronous native feedback call. Query execution,
diagnostic serialization and hashing occur outside that timer. Allocation counting
uses separate processes, with the same allocator wrapper in both arms.

Create isolated source roots with `git archive REV Cargo.toml Cargo.lock crates`
for the two revisions above. Copy `probe.rs` into each root as
`crates/core/examples/fpm_prefix_probe.rs`. Build each sequentially:

```sh
# In each source root; use separate Cargo targets or touch the Rust sources
# before rebuilding when reusing a target directory across archived checkouts.
cargo build --locked --offline --release -p aisimulate-core \
  --no-default-features --example fpm_prefix_probe
# Copy the two executables to WORK/baseline-probe and WORK/candidate-probe.
```

Use [the existing fixture generator](../fpm_online_publication_v1/benchmark.py)
with the candidate SDK and scikit-learn 1.9.1:

```sh
python benchmarks/evidence/fpm_online_publication_v1/benchmark.py \
  prepare WORK/fixture --history-rows 10000
python benchmarks/evidence/fpm_online_publication_v1/benchmark.py \
  prepare WORK/small-fixture --history-rows 256
python benchmarks/evidence/fpm_pr533_review_v1/controller.py WORK
```

Both arms must consume the same fixtures. The recorded fixtures were generated
before the final candidate extension build; the generating extension's hash is
preserved separately in provenance. Input hashes are the exact measured inputs.
Controls are `recent_window=64`, `update_every=1`, `history_weight=0.1`, history
weight ramp 32, grouped history, eager derivatives, split sampling 0.5, and rank
tolerance 0.01. The target is 1.3 times each query's initial prediction, with 256
queries used for evaluation. Full scratch outputs and logs were retained locally
under `/private/tmp/pr533-review-evidence`; this directory is temporary storage.
