<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# PR 585 review corrections

Baseline: `6b57265e03a3844503b6a9f77f29f1f2dd8eea02`.
Measured runtime source snapshot: `4849d6676280aafe963d795a757d087a24d8f8a7`.
The snapshot is an isolated Git commit, preserved in `source.bundle` alongside the generator snapshot; the user's
branch was not committed or rewritten. The baseline commit is the bundle prerequisite.
The snapshots include implementation and test sources before generated expectations,
so generation provenance does not refer circularly to its own output commit.

## Source and build identity

`build-manifest.json.gz` records the immutable source commit/tree, individual runtime
source hashes, dependency locks, data tree, toolchain, build command and loaded native
binary hash. Both native artifacts were built in release mode with the explicit
`extension-module` feature and identical macOS dynamic-lookup linker flags.
Python source and native Rust source are identified separately from measured outputs.
Neither branch names nor source modification times establish binary identity.

To inspect the snapshots without moving a working branch, import the bundle into a
separate checkout containing the baseline commit. Decompress the manifest into a
temporary file; when using a separate Git object store, set its `source_git_dir` there.
The checked-in manifest intentionally omits the measurement machine's temporary Git path.

## Modeling changes and independent expectations

- Per-replica then per-microbatch partitioning conserves sequences and tokens. In the
  8-replica/1-request boundary, F consumes one sequence. The 8-sequence/3-microbatch
  boundary uses 3+3+2 rather than padding to nine. Rank-to-F mapping is checked against
  an explicit population reference for irregular widths and idle replicas.
- A restoration shares the attention resource after return. A hand-authored two-job
  timeline `[10,1,1,1,10]` ms takes 40 ms in four-stage and 42 ms in two-stage mode.
  Resource interval checks forbid overlap; a two-layer unit-stage timeline verifies
  feedback and deterministic restoration ties (12/13 ms).
- Native and memory compilation share the Rust hardware precision policy. Hopper's
  `nvfp4` request maps to `nvfp4_wo`, Blackwell retains `nvfp4`; explicit weight-only
  requests remain weight-only. Unknown or unsupported execution capabilities reject.
  Requested precision remains in configuration, and effective compiled modes appear
  in pool provenance. Missing data is not a precision-conversion policy.
- Missing A/F communication tables and missing half/all-gather buckets reject before
  memory sizing or worker execution. Integration cases separately exercise nonzero
  combine, shared merge and token all-gather, plus their combination.

These assertions precede refreshing expectations. Golden comparison retains relative
`1e-6`, absolute `1e-9` tolerance and exact request/token/GPU accounting. Numeric deltas
are modeling corrections, not measured serving accuracy or a claim that a framework
can deploy every modeled checkpoint.

## Reproduction

After building the identified sources and producing/verifying a build manifest:

```bash
python/aisimulate/.venv/bin/python scripts/afd_parity.py refresh-e2e \
  --case all --build-manifest /path/to/build-manifest.json --reason 'explained modeling correction'
python/aisimulate/.venv/bin/python scripts/afd_parity.py refresh-layouts \
  --case all --build-manifest /path/to/build-manifest.json --reason 'explained modeling correction'
python/aisimulate/.venv/bin/python benchmarks/evidence/afd-pr-preparation/measure_forward.py \
  --baseline /path/to/baseline --candidate-revision 4849d6676280aafe963d795a757d087a24d8f8a7 \
  --output /path/to/forward-results
python/aisimulate/.venv/bin/python benchmarks/evidence/afd-review585/measure.py \
  --baseline /path/to/baseline --candidate-revision 4849d6676280aafe963d795a757d087a24d8f8a7 \
  --output /path/to/replay-results.json
```

Forward timings use the unchanged repository worker protocol and 64-case matrix,
with five alternating paired rounds, 10 warmups and 100 warm iterations. Cold means
first target prediction after off-matrix priming, not cold process/model construction.

Replay measurements reuse `afd-pr-preparation/measure_replay.py::worker`. They separate
compile/materialization, empty-native setup, first runner call, warm runner calls,
reported native wall time and supervised fresh CLI time. Warm runner calls still
reconstruct timing providers; they are not pure cached FPM query measurements. Native
wall time excludes timing-model construction and final report work. The full CLI and
runner timers include those costs. Only the three wall-clock-derived fields are
excluded when checking ordinary modeled output equality.

Each replay row has five alternating paired fresh processes, nine runner calls per
process (five retained warm samples), and an independent supervised CLI invocation.
Both paths verify complete request/input/output totals and the loaded binary hash.
The matrix includes ordinary controls, low concurrency, saturation, tail microbatches,
depth 16 and all A-restoration operations. AFD simulated timing is expected to change;
request/token work is held fixed, but the intended timing correction can change the
batch/pass mix. Host CPU time and simulated serving latency are reported separately.
macOS CPU affinity is unavailable; raw paired samples retain variation. No builds or
correctness campaigns run concurrently with these timing campaigns.

## Verified results
Generation source: `7392d688601c99c5048fba27c70547a586495c2c`. Measurement/runtime source: `4849d6676280aafe963d795a757d087a24d8f8a7`.
Both are in the bundle (`source` and `measured` refs). Their Rust/Python runtime
inputs and dependency manifests are byte-identical; only the generator and its
regression tests differ. Subsequent documentation/import formatting does not
change the pinned runtime inputs. The native binary hash remains unchanged.
Checks completed: 2,336 Rust library tests (one ignored), 20 public Rust API tests,
425 AFD/general parity cases, 788 affected Python tests plus the added public
precision-provenance case. Ruff check/format and Rust format checks pass. Pool
EngineSpecs regenerated byte-identically; EngineSpec/ReplaySpec schemas are unchanged.
The 64-case forward campaign has zero errors and no regression classification
(`blocking=false`): 31 improvement, 47 OK, 50 unstable cold/warm points. Predicted
values match. Unstable points are retained rather than treated as stable wins.
| Workload | Requests | Warm runner ratio | Full CLI ratio | Native wall ratio | Native delta (ms) |
|---|---:|---:|---:|---:|---:|
| ordinary_fixed | 16 | 1.040 | 0.981 | 1.023 | +0.007 |
| ordinary_op_level | 16 | 0.996 | 0.978 | 0.955 | -0.049 |
| ordinary_op_level | 1024 | 1.012 | 0.971 | 1.027 | +1.158 |
| afd_four_stage_low | 16 | 0.988 | 1.000 | 1.039 | +0.126 |
| afd_four_stage_busy | 1024 | 1.035 | 1.008 | 1.078 | +3.540 |
| afd_two_stage | 1024 | 1.028 | 0.989 | 1.053 | +1.479 |
| afd_ragged | 64 | 1.000 | 1.009 | 1.116 | +0.384 |
| afd_depth16 | 64 | 1.006 | 0.993 | 1.207 | +0.600 |
| afd_restore | 32 | 1.005 | 0.987 | 1.216 | +0.363 |

Ratios are medians of the five paired ratios (candidate/baseline); native delta
is the difference between arm medians. Full runner ratios are 0.988–1.040 and
CLI ratios 0.971–1.009. Ordinary modeled outputs match exactly. All runner and
CLI request/token totals were checked. Full distributions, setup/first-call
measurements and memory samples remain in `replay-performance.json.gz`.

Cost assessment: small depths use fixed stack queues; larger depths use bounded
priority queues. Exact populations require up to three distinct shapes, independent
of request/A-node count; prepared modulo mapping avoids per-query scans of A
replicas. Native wall time rises about 21% at depth 16 and with restoration,
corresponding to about 0.600/0.363 ms in these runs. Their complete warm runner
ratios are 1.006/1.005. The event/resource work enforces the new dependency model;
setup-dominated full-call timings must not hide the separate native cost. Frozen
decode pays the stride-sampling setup once per provider; this campaign does not
establish host-cost parity for every long trace/session workload. No serving-
accuracy or all-workload performance approval is inferred.
