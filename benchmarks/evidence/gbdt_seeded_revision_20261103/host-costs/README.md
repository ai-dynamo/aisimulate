# Matched seeded GBDT host costs before dependency-receipt reuse

Forty fresh measured workers completed: five baseline/candidate pairs for each
of the small caller-managed first-fit control, two-store seed, four-store seed
and original-A mixed fallback. Prediction bits, exported forest hashes, admitted
work, input hashes and dependency versions match within every pair. Warm native
queries and ordinary updates show no material regression in these cases. Raw
construction and restore are slower; their boundary costs are reported separately
below. The source audit found repeated checkpoint/source reads and hashes after
the same payloads had already been verified. These measurements motivated the
construction-local verified-receipt optimization; they are retained as the
pre-optimization result and do not qualify the final optimized build.

The baseline is merged PR #658 at
`9f4819e2b77afb4798bc8133c42ccc1d1b045e5c`, native
`d88bc55746664e144057e57e17ee5357c08667ea52861627a7ad6a9949f74cd4`.
The assembled candidate uses native
`73bc10ea8bbe3f2e7936d56d0e7ec25be77f792e34c32cf59386e0adcf5d491c`.
Each original worker records the actual dirty-tree source inventory; its old
observed Git HEAD is not presented as a clean candidate source revision.

## Scope and method

The workers use the complete public Python SDK calls and unchanged Gym wrapper
where applicable. The small control uses Gym's existing caller-managed buffer
and first fit at 20 rows, then 50 updates. This is not native source-free initial
training. Seeded cases use the same raw checkpoint on both versions: 20 training
rows per store, 50 subsequent updates per store and 1,000 warm queries per store.
The four-store seed includes pure decode, pure prefill, locally mixed and
cross-rank aggregated stores. The original-A fallback case retains the existing
caller-managed fit of its missing mixed store.

Five paired trials alternate AB/BA order and case order. Each worker verifies
its native binary, Python package location and source inventory before and after
the run. BLAS/OpenMP/NumExpr thread counts are one; `PYTHONHASHSEED=0`. The host is
macOS arm64, 18 logical CPUs, Python 3.12.13. No affinity was available. Other
agent builds/tests were idle throughout the timing window; unrelated host
activity, thermal state and filesystem caching were not isolated. Eight prior
dry workers checked the harness; their results are retained and excluded from
timings.

Timers exclude interpreter startup, imports and source inventory. Offline seed
training is separately recorded for seeded cases; the reported seeded
construction timer starts with an existing checkpoint. Checkpoint and snapshot
timers include the complete save/construct call. RSS covers the process with its
trainer imports and live restored models; native retained training bytes are
separate. These synthetic measurements do not qualify serving accuracy or
establish universal performance thresholds.

## Repeated prediction and updates

Times below are per-operation medians. Ratios are the median of the five paired
candidate/baseline ratios, not a ratio of independent medians. The range shows
all five paired ratios.

| Case / operation | Baseline | Candidate | Paired ratio | Pair range |
| --- | ---: | ---: | ---: | ---: |
| Small, warm SDK query | 3.628 μs | 3.613 μs | 0.996 | 0.975–1.042 |
| Small, warm Gym query | 4.436 μs | 4.449 μs | 1.016 | 0.979–1.022 |
| Two-store, warm SDK query | 3.616 μs | 3.659 μs | 1.007 | 0.986–1.024 |
| Four-store, warm SDK query | 6.328 μs | 6.268 μs | 0.992 | 0.977–1.006 |
| Fallback, warm original-A query | 11.384 μs | 10.663 μs | 0.938 | 0.905–1.015 |
| Fallback, warm updated SDK query | 3.442 μs | 3.381 μs | 0.997 | 0.927–1.017 |
| Small, ordinary update | 257.53 μs | 261.29 μs | 1.011 | 0.984–1.030 |
| Two-store, ordinary update | 266.91 μs | 263.94 μs | 0.989 | 0.979–1.014 |
| Four-store, ordinary update | 280.94 μs | 283.26 μs | 1.003 | 0.999–1.021 |
| Fallback, ordinary update | 31.43 μs | 30.82 μs | 0.965 | 0.902–1.023 |

The small caller-managed initial fit is 233.37→235.45 ms, paired ratio 1.004
(0.991–1.038). The fallback mixed-store fit is 23.55→23.58 ms, ratio 1.003
(0.874–1.221). No numerical or forest-structure assertion failed.

## Construction and persistence

| Case / operation | Baseline ms | Candidate ms | Paired ratio | Pair range |
| --- | ---: | ---: | ---: | ---: |
| Two-store, construct raw seed | 2.180 | 2.554 | 1.177 | 1.136–1.183 |
| Four-store, construct raw seed | 3.917 | 4.591 | 1.182 | 1.131–1.244 |
| Small, bare checkpoint restore | 7.469 | 8.887 | 1.242 | 1.138–1.600 |
| Two-store, bare checkpoint restore | 9.232 | 10.465 | 1.153 | 0.691–1.190 |
| Four-store, bare checkpoint restore | 12.493 | 16.457 | 1.287 | 1.257–1.321 |
| Fallback, bare checkpoint restore | 9.107 | 10.820 | 1.156 | 0.695–1.764 |
| Small, complete snapshot save | 62.322 | 65.756 | 1.082 | 0.976–1.363 |
| Two-store, complete snapshot save | 66.491 | 68.720 | 1.079 | 0.952–1.221 |
| Four-store, complete snapshot save | 59.665 | 69.782 | 1.067 | 0.985–1.206 |
| Small, complete snapshot restore | 2.322 | 3.989 | 1.709 | 1.674–1.922 |
| Two-store, complete snapshot restore | 4.119 | 7.062 | 1.711 | 1.684–1.786 |
| Four-store, complete snapshot restore | 7.853 | 13.061 | 1.635 | 1.596–1.738 |

Bare checkpoint save paired medians are 0.995–1.035 across the four cases;
the full samples are in the unchanged summaries. The baseline rejects complete
mixed-fallback snapshots. The candidate's new supported operation takes median
97.521 ms to save and 26.356 ms to restore. Those are capability costs, not a
paired speed comparison with a baseline operation that did not exist.

## Retained state and disk closure

| Case | Training-state bytes, both | Bare checkpoint bytes, both | Baseline snapshot files / bytes | Candidate snapshot files / bytes |
| --- | ---: | ---: | ---: | ---: |
| Small | 437,450 | 425,392 | 6 / 433,223 | 9 / 473,237 |
| Two-store | 889,140 | 869,648 | 6 / 878,278 | 10 / 952,552 |
| Four-store | 2,006,368 | 1,980,810 | 6 / 1,989,400 | 12 / 2,133,919 |
| Fallback | 124,882 | 181,871 | Unsupported | 19 / 342,102 |

Training-state and bare checkpoint sizes are identical. Complete snapshots add
the original source closure and manifest records, increasing the three comparable
disk sizes by 40,014, 74,274 and 144,519 bytes. Median final process RSS is
210.55→211.05 MiB (small), 212.78→213.13 MiB (two-store),
218.05→218.50 MiB (four-store), and 231.67→232.88 MiB (fallback).
The last case additionally retains a restored complete fallback model that the
baseline cannot construct. RSS is not an isolated native allocation measurement.

## Source assessment and required follow-up

For raw learned construction, the new private `pin_legacy_source` pass opens and
parses the manifest and hashes the artifact and row/state dependencies after the
checkpoint loader has already validated them. Immutable construction-time source
closure is required, but repeating verified input work is avoidable. Reusing a
construction-local receipt of the verified path, digest and byte length can
preserve closure semantics without a process-wide cache or delayed pinning.

Complete snapshots add required original-source/prior evidence and verify both
the archived source and current runtime semantics. The current restore also
overlaps source verification with outer manifest hashing and checkpoint-loader
verification. The outer-manifest scan followed by a runtime checkpoint read was
already present in the merged baseline; newly archived source verification adds
another overlap. The entire measured restore increase therefore cannot be called
unavoidable, nor can all of it be attributed to newly redundant work. Snapshot
save already deduplicates source paths and combines copy
with hashing in a bounded streaming pass; copied closure bytes are required.
The final implementation must preserve every boundary/identity check and be
measured separately after the narrow receipt-reuse changes. These original
worker bytes and their native identity remain unchanged.

## Evidence and reproduction

[small/summary.json](small/summary.json) and
[expanded/summary.json](expanded/summary.json) retain the full timing samples,
standard deviations, paired ratios and original worker path/hash references.
[trial-metrics.json](trial-metrics.json) retains each worker's metrics and identity
digests. Only repeated full source/dependency inventories were moved into
[raw-workers.tar.xz](raw-workers.tar.xz), together with every original worker log.
[raw-archive.json](raw-archive.json) lists all 96 original members, their hashes
and a separate private backup. Every member and backup was verified byte-identical
before removing the redundant loose files. The dry-run and measured summaries
remain unchanged; extraction restores their original relative worker paths.

Actual commands and quiet-window timing are recorded in
[host-costs-commands.json](../host-costs-commands.json). Use the same commands with
new output directories and the selected build hashes for a new measurement.
The initial [planned commands](../host-costs-planned-commands.json) intentionally
retain their original prepared status. The three standalone harnesses and the
[compaction script](../compact_host_costs.py) are included; their origins are
recorded in [evidence-relocation.json](../evidence-relocation.json).
