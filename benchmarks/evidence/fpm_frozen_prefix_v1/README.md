<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Frozen-prefix allocation evidence

This matched native benchmark measures eager online updates before and after
preserving the training buffers of frozen trees. The baseline is
`4a710f144930a40e1cfee5010b386b35319457a7`; the candidate is
`247e02da9c56da398391ebe77dad724d0c5ae654`. Both use the canonical
`ForwardPassPerfModel::best_available` constructor and ordinary feedback API.
The change leaves frozen predictor arrays copied and still evaluates the frozen
prefix when fitting the mutable suffix. It does not make total update work
proportional only to the mutable tail.

All 80 processes passed exact numerical, semantic and provenance gates. The
restricted-tail update median fell by 90.1%, 39.2% and 44.1% across the three
cases. The unrestricted control had identical requested-allocation measurements
and a median paired timing change of +0.34% (range −0.21% to +0.73%). This supports
removing the avoidable frozen training-buffer clones without an observed material
unrestricted-update cost change in this run.

Each timing entry is the median across five process-level p50 or p95 values.
Ranges are the five paired candidate/baseline changes, not confidence intervals.
All arrows show baseline → candidate.

| Scenario | Update p50, ms | Update p95, ms | Paired p50 change, median [range] |
| --- | ---: | ---: | ---: |
| 150 trees / 256 rows / tail 1 | 9.582 → 0.956 | 10.066 → 1.375 | -90.05% [-90.07%, -89.91%] |
| 150 trees / 10k rows / tail 1 | 24.440 → 14.899 | 25.427 → 15.780 | -39.16% [-39.16%, -38.33%] |
| 300 trees / 10k rows / tail 1 | 44.153 → 24.715 | 45.624 → 25.476 | -44.09% [-44.25%, -43.79%] |
| 150 trees / 10k rows / all trees | 243.289 → 243.979 | 261.982 → 262.553 | +0.34% [-0.21%, +0.73%] |

Requested allocation is the median per-update value within each process, then
the median across five processes. Added-live peak is the maximum within each
64-update process, then the median across five processes. MiB means 2²⁰ bytes.

| Scenario | Requested MiB/update | Maximum additional-live MiB/update |
| --- | ---: | ---: |
| 150 trees / 256 rows / tail 1 | 104.42 → 2.27 | 103.79 → 1.64 |
| 150 trees / 10k rows / tail 1 | 120.08 → 3.71 | 118.44 → 2.06 |
| 300 trees / 10k rows / tail 1 | 237.33 → 3.79 | 235.73 → 2.20 |
| 150 trees / 10k rows / all trees | 861.58 → 861.58 | 124.71 → 124.71 |

Construction and process memory remain separate from temporary update allocation.
The RSS columns are process high-water snapshots before/after the update phase;
each arrow still compares baseline with candidate. Retained bytes below are the
capacity-based training-state estimate after 64 updates.

| Scenario | Setup p50, ms | RSS HWM before updates, MiB | RSS HWM after updates, MiB | Retained training MiB |
| --- | ---: | ---: | ---: | ---: |
| 150 trees / 256 rows / tail 1 | 87.84 → 88.14 | 119.11 → 119.15 | 223.00 → 120.53 | 102.90 → 111.45 |
| 150 trees / 10k rows / tail 1 | 218.77 → 218.47 | 135.80 → 135.74 | 251.76 → 135.74 | 120.25 → 126.58 |
| 300 trees / 10k rows / tail 1 | 395.95 → 397.18 | 259.89 → 259.87 | 493.01 → 259.87 | 237.39 → 250.10 |
| 150 trees / 10k rows / all trees | 219.55 → 219.71 | 135.85 → 135.74 | 264.66 → 264.57 | 125.52 → 125.52 |

The experiment runs five paired rounds in fresh processes for each of four
scenarios, with 64 accepted updates per process. Arm and scenario order reverse
on alternating rounds. Timing and allocation counting use separate processes,
for 80 processes and 5,120 admitted updates overall. The statistical unit is the
paired process; the 64 sequential updates are not independent replicates.
Timing measures the complete synchronous feedback call, with allocator counters
disabled. The allocator wrapper still performs its enabled-flag checks in both
arms. Setup time covers canonical model construction; query calls, diagnostic
serialization, and output hashing are outside the update timer.

Inputs are synthetic decode iterations with 17 aggregate features, 150 or 300
trees, at most 15 leaves per initial tree, and either 256 or 10,000 historical
rows. The feedback target is 1.3 times each query's initial prediction. Both arms
use identical labels and the same 256-query evaluation set. The larger history
has 9,455 distinct binned rows. Controls are `recent_window=64`, `update_every=1`,
`history_weight=0.1`, a 32-observation history-weight ramp, grouped historical
rows, `derivative_refresh=eager`, split sampling 0.5, and rank tolerance 0.01.
The 300-tree fixture is a newly fitted valid training checkpoint: all 10,000
feature/target rows were checked against the original 150-tree fixture before
training. No trees or checkpoint training states were duplicated artificially.

The Rust global allocator records successful requested allocation sizes, full
new requested sizes for successful reallocations, counts, and deallocations
during one update. Its additional-live peak is the maximum net requested-byte
increase from the start of that update, including deallocation of earlier
buffers. This excludes allocator metadata, rounding, internal realloc copies,
non-Rust allocation, and allocations outside the update. It measures allocation
traffic and temporary live storage, not exact bytes copied by the CPU.

Process RSS is a separate measurement. Before-update RSS and high-water RSS are
sampled after construction, initial predictions and initial diagnostics.
After-update high-water RSS includes setup and every update; it cannot be
interpreted as an isolated
update's temporary allocation. The reported retained training-state estimate
counts vector capacities. Preserving the original frozen buffers can retain
more spare capacity than the baseline's deep clones, which may shrink capacity
to length. Those capacity differences are reported separately and are excluded
from semantic diagnostic equality.

Every run must match the expected binary, input manifest and query hashes.
Across arms, rounds and phases, each scenario must have identical initial
predictions, pre-feedback prediction trajectories, final predictions, final
exported artifacts, configuration and semantic diagnostics. Only grouping time
and capacity-derived training bytes are removed from diagnostic comparisons.
All 64 observations must be accepted, published and fitted eagerly, with no
pending observations or failed updates. Source, binary and fixture hashes are
checked again after the run. Full source inventories are checked against exact
git archives, including the tracked symlink; `Cargo.lock` and the probe are
identical in both arms.

[measurements.json](measurements.json) contains provenance, exact equality gates,
paired process summaries and measurement definitions.
[samples.json](samples.json) retains each update's timings and allocation
counters, process memory snapshots and original result-file hashes.
[validation.json](validation.json) records the independent numerical, rollback,
buffer-identity and full FPM tests. The source archive, binary, fixture, execution
script and execution-log hashes are recorded in the measurements. The maintained
probe and controller are byte-identical to their executed versions.

Reproduction uses [probe.rs](probe.rs), [controller.py](controller.py),
[prepare_fixture.py](prepare_fixture.py) and [analyze.py](analyze.py). On a Linux
workstation, create isolated baseline and candidate directories from the exact
revisions with `git archive REV Cargo.toml Cargo.lock crates`. Copy `probe.rs`
to `crates/core/examples/fpm_prefix_probe.rs` in both. Build each sequentially:

```sh
# In each isolated source root. Touch sources when reusing a Cargo target cache:
find crates -type f -name '*.rs' -exec touch {} +
CARGO_TARGET_DIR=/absolute/path/to/shared-target cargo build \
  --locked --offline --release -p aisimulate-core \
  --no-default-features --example fpm_prefix_probe
# Copy the finished example to WORK/baseline-probe or WORK/candidate-probe.
```

Do not rely on different checkout paths or commit IDs to invalidate a shared
Cargo target cache. The first candidate build reused the baseline executable;
the identical-binary gate caught it before measurement. Touching the candidate
Rust sources forced an actual compile, and the final executable hashes differ.
The discarded cache hit and an initial symlink-inventory preflight failure did
not contribute measurement samples.

Use the existing
[synthetic fixture generator](../fpm_online_publication_v1/benchmark.py) with an
SDK environment that supports training-checkpoint export to create `fixture`
and `small-fixture` beside `WORK`, with `prepare PATH --history-rows 10000` and
`prepare PATH --history-rows 256`. Place the generator at their common parent's
`benchmark.py` so the 300-tree preparer can reconstruct and verify the original
rows. Then run `prepare_fixture.py FIXTURE WORK/fixture-300 --trees 300`, saving
stdout as `WORK/prepare-fixture.log`, and copy that script into `WORK`.
Copy `manifest.source_archives` from `measurements.json` to
`WORK/source-archives.json`; these entries bind the extracted production sources
to the exact archived revisions. Finally run:

```sh
python3 controller.py WORK --rounds 5 --updates 64 --cpu 2
python3 analyze.py WORK OUTPUT
```

The controller fixes CPU affinity to CPU 2 on an Intel Core Ultra 9 285K and sets
common numerical-library thread counts to one. The benchmark uses
`rustc 1.96.1 (31fca3adb 2026-06-26)` and fixtures fitted with scikit-learn 1.9.1.
Hardware, toolchain, fixture-library versions and exact workload
configuration are included in the manifest. Inputs and full execution outputs
remain in the authorized workstation scratch directory
`/home/yueqiz/work/fpm-model-publication-v1/frozen-prefix-v1`.

These measurements cover one synthetic decode store, eager structural fitting,
64 admitted observations, two history sizes and two forest sizes. They do not
establish serving-workload accuracy, performance for other query types, long-run
steady state after many window evictions, or engine replay performance. Lazy
fitting and its fallback have separate correctness tests; their performance is
not inferred from this eager benchmark.
