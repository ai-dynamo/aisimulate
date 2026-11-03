# Final matched seeded GBDT host costs

Raw results and reproduction scripts are preserved in the
[immutable qualification checkout](https://github.com/ai-dynamo/aisimulate/tree/3630fed8e30ec24edee78f569b9d7276e60ddb4a/benchmarks/evidence/gbdt_seeded_revision_20261103).
Run the recorded reproduction commands from that checkout with their pinned
external inputs; this review copy retains the report and archive links.

The final runtime removes the redundant checkpoint/source scans identified in
the [first campaign](https://github.com/ai-dynamo/aisimulate/blob/3630fed8e30ec24edee78f569b9d7276e60ddb4a/benchmarks/evidence/gbdt_seeded_revision_20261103/host-costs/README.md). Five fresh baseline/candidate pairs
for each of four cases passed every input, work, prediction-bit and forest-hash
check. Raw construction is close to baseline. Complete snapshot restore adds
0.41–0.87 ms median in the comparable cases; complete source capture adds save
cost and disk bytes. Warm-query differences are small but are reported rather
than treated as exact performance equality.

Baseline: merged prerequisite `9f4819e2b77afb4798bc8133c42ccc1d1b045e5c`,
native `d88bc55746664e144057e57e17ee5357c08667ea52861627a7ad6a9949f74cd4`.
Candidate: working-tree source recorded in
[candidate-optimized-source.json](https://github.com/ai-dynamo/aisimulate/blob/3630fed8e30ec24edee78f569b9d7276e60ddb4a/benchmarks/evidence/gbdt_seeded_revision_20261103/candidate-optimized-source.json), native
`6a12ba8d1af12a1930d8a030a400b22aa2aacd439f458002488adb71a2f42d42`.
No baseline timing sample is reused from the first campaign. Its 73bc10ea results
remain pre-optimization evidence.

## Method and coverage

Each measured worker is a fresh process. Five paired AB/BA trials alternate build
order and, for the expanded cases, case order. All other agent builds/tests were
idle during the run. BLAS/OpenMP/NumExpr thread counts are one, Python hash seed
is zero, and both dependency freezes match. The host is macOS arm64 with 18 logical
CPUs and Python 3.12.13. CPU affinity and thermal/external host isolation were not
available. Eight dry workers passed first and are excluded from timings.

The small control uses Gym's existing caller-owned first fit at 20 observations
then 50 updates. It does not exercise native source-free initialization. The
two/four-store cases construct the same raw training checkpoint on both builds,
with 20 rows per store, 50 subsequent updates per store and 1,000 warm queries
per store. Four stores cover pure decode/prefill, locally mixed and cross-rank
aggregated workloads. The original-A fallback case retains its existing caller
fit for the initially absent mixed store.

Timers cover complete public SDK calls or the existing Gym wrapper where named.
Startup, imports and source inventories are excluded. Offline training is
recorded separately from seeded construction. Persistence timers include the
complete save/restore call. All numerical outcomes and admitted workloads match.
These are synthetic host costs, not serving-accuracy evidence or universal
performance limits.

## Results

Operation times are medians per operation. The ratio is the median of the five
paired candidate/baseline ratios; it is not the quotient of the two medians.
The range includes every pair.

| Case / operation | Baseline | Candidate | Paired ratio | Pair range |
| --- | ---: | ---: | ---: | ---: |
| Small, warm SDK query | 3.728 μs | 3.789 μs | 1.040 | 0.961–1.058 |
| Small, warm Gym query | 4.597 μs | 4.484 μs | 1.001 | 0.976–1.058 |
| Two-store, warm SDK query | 3.807 μs | 3.832 μs | 1.008 | 0.972–1.055 |
| Four-store, warm SDK query | 6.719 μs | 6.692 μs | 0.985 | 0.962–1.023 |
| Fallback, warm original-A query | 11.328 μs | 10.894 μs | 0.960 | 0.934–0.986 |
| Fallback, warm updated SDK query | 3.467 μs | 3.534 μs | 1.028 | 1.008–1.048 |
| Small, ordinary update | 278.45 μs | 274.44 μs | 0.991 | 0.971–1.035 |
| Two-store, ordinary update | 278.96 μs | 279.57 μs | 1.007 | 0.980–1.012 |
| Four-store, ordinary update | 300.75 μs | 298.54 μs | 0.995 | 0.969–1.008 |
| Fallback, ordinary update | 32.50 μs | 31.59 μs | 0.972 | 0.930–1.040 |
| Two-store, construct raw seed | 2.267 ms | 2.311 ms | 1.053 | 1.016–1.109 |
| Four-store, construct raw seed | 4.113 ms | 4.039 ms | 0.985 | 0.951–1.059 |
| Small, bare checkpoint restore | 7.491 ms | 7.727 ms | 1.026 | 0.801–1.342 |
| Two-store, bare checkpoint restore | 12.978 ms | 8.786 ms | 0.692 | 0.554–1.058 |
| Four-store, bare checkpoint restore | 12.728 ms | 11.895 ms | 0.841 | 0.695–0.974 |
| Fallback, bare checkpoint restore | 8.922 ms | 9.609 ms | 1.092 | 0.940–1.416 |
| Small, complete snapshot save | 61.072 ms | 63.943 ms | 1.047 | 0.937–1.240 |
| Two-store, complete snapshot save | 54.380 ms | 64.342 ms | 1.158 | 1.055–1.199 |
| Four-store, complete snapshot save | 58.534 ms | 72.566 ms | 1.156 | 1.059–1.330 |
| Small, complete snapshot restore | 2.311 ms | 2.720 ms | 1.186 | 1.067–1.222 |
| Two-store, complete snapshot restore | 4.037 ms | 4.639 ms | 1.147 | 0.959–1.176 |
| Four-store, complete snapshot restore | 7.458 ms | 8.324 ms | 1.125 | 1.016–1.157 |

The small caller-managed first fit is 254.55→254.23 ms (paired ratio 1.025,
range 0.952–1.040). The fallback mixed fit is 23.20→21.69 ms (0.920,
0.687–1.022). The first two-store query measures 16.73→17.96 μs per operation
(1.102, 0.800–1.222), an isolated first-call result with substantial spread;
it does not establish a general first-query improvement or regression. Bare
checkpoint save paired medians are 0.905–0.994; their I/O spread is retained in
the summaries. No speedup claim relies on a single favorable sample.

The baseline cannot save complete mixed-fallback snapshots. The candidate takes
median 95.499 ms to save and 23.435 ms to restore that new capability. This is
not a paired comparison with a baseline operation that did not exist.

## Cause, required work and limits

The [recorded source delta](https://github.com/ai-dynamo/aisimulate/blob/3630fed8e30ec24edee78f569b9d7276e60ddb4a/benchmarks/evidence/gbdt_seeded_revision_20261103/construction-receipts-native-delta.json) reuses
construction-local receipts of the path, digest and byte length of bytes already
read and checked by the native loader. New private dependency capture no longer
rescans those same raw checkpoint payloads. Schema-3 snapshot restoration shares
fresh source and actual runtime-decoder receipts where valid. Independent
identity/payload checks, mixed-prior checks and extra declared dependencies remain
verified. No serialized declaration or process-wide cache substitutes for reading
the input. The [429 native tests](../optimized-rust-checks.json) include payload,
outer-pin, length and unrelated-model regressions.

Source capture at construction remains necessary: deferring it until snapshot
save would lose the binding to the source actually used. Likewise, a complete
snapshot must retain the original seed separately from current adaptive state and
must own its qualified prior. Dropping those files or validations would change the
feature contract. Copying/hashing is bounded and streamed, source paths are
deduplicated, and save still validates its staged portable closure before atomic
publication. These additional files explain remaining persistence work; there is
no repeated query-time file scan. The pre-existing baseline outer-manifest/runtime
read overlap is documented separately from new source-capture work.

The original paired raw-construction ratios of 1.177/1.182 are now 1.053/0.985.
The original complete-restore ratios of 1.709/1.711/1.635 are now
1.186/1.147/1.125. These are two separately matched campaigns, not a direct
paired comparison between candidate binaries. After removing the identified
duplicate work, remaining restore overhead is 0.409/0.601/0.866 ms median.
Snapshot save adds 2.871/9.962/14.032 ms median in the final run. Its I/O variation
and required larger source closure limit extrapolation to other machines/data.

The source audit found no added prediction/update work. Nevertheless, warm SDK
paired medians range 0.985–1.040 and fallback warm queries are consistently about
2.8% slower in these five pairs. At this size that is a 67 ns difference between
independent medians, not evidence of a scaling problem. Reported measurements
support bounded host cost for the tested sizes, not exact equality or a guarantee
for arbitrarily large checkpoints. Applications should avoid reconstructing or
snapshotting models in a per-query loop.

## State and closure sizes

Native training-state and bare checkpoint bytes remain identical across builds.
The two-store and four-store training states are 889,140 and 2,006,368 bytes;
the small/fallback states are 437,450/124,882 bytes.

| Case | Bare checkpoint bytes, both | Baseline snapshot files / bytes | Candidate snapshot files / bytes | Final RSS MiB, baseline → candidate |
| --- | ---: | ---: | ---: | ---: |
| Small | 425,392 | 6 / 433,223 | 9 / 473,237 | 210.59 → 211.28 |
| Two-store | 869,648 | 6 / 878,278 | 10 / 952,552 | 212.77 → 213.20 |
| Four-store | 1,980,810 | 6 / 1,989,400 | 12 / 2,133,919 | 218.14 → 218.44 |
| Fallback | 181,871 | Unsupported | 19 / 342,102 | 231.61 → 233.14 |

Snapshots add 40,014/74,274/144,519 bytes in the comparable cases. These disk
sizes exactly match the pre-optimization candidate. Process RSS includes trainer
imports and retained restored instances; fallback additionally keeps a restored
complete model unsupported by the baseline. RSS is not isolated native memory.

## Reproducible evidence

[small/summary.json](https://github.com/ai-dynamo/aisimulate/blob/3630fed8e30ec24edee78f569b9d7276e60ddb4a/benchmarks/evidence/gbdt_seeded_revision_20261103/host-costs-optimized/small/summary.json) and
[expanded/summary.json](https://github.com/ai-dynamo/aisimulate/blob/3630fed8e30ec24edee78f569b9d7276e60ddb4a/benchmarks/evidence/gbdt_seeded_revision_20261103/host-costs-optimized/expanded/summary.json) retain unchanged full samples,
standard deviations, paired ratios and original worker path/hash references.
[trial-metrics.json](https://github.com/ai-dynamo/aisimulate/blob/3630fed8e30ec24edee78f569b9d7276e60ddb4a/benchmarks/evidence/gbdt_seeded_revision_20261103/host-costs-optimized/trial-metrics.json) retains metrics and identity digests.
All 96 original dry/measured worker JSON/log members are preserved byte-identically
in [raw-workers.tar.xz](https://github.com/ai-dynamo/aisimulate/blob/3630fed8e30ec24edee78f569b9d7276e60ddb4a/benchmarks/evidence/gbdt_seeded_revision_20261103/host-costs-optimized/raw-workers.tar.xz), with a separate private backup and
complete hash/member ledger in [raw-archive.json](https://github.com/ai-dynamo/aisimulate/blob/3630fed8e30ec24edee78f569b9d7276e60ddb4a/benchmarks/evidence/gbdt_seeded_revision_20261103/host-costs-optimized/raw-archive.json). Extract the
archive here to restore the summaries' original relative worker paths.

[host-costs-optimized-commands.json](https://github.com/ai-dynamo/aisimulate/blob/3630fed8e30ec24edee78f569b9d7276e60ddb4a/benchmarks/evidence/gbdt_seeded_revision_20261103/host-costs-optimized-commands.json) records
actual commands, environments, quiet-window confirmation and completion. For a
new run use new output destinations and the selected native hashes. The three
standalone harnesses are unchanged from the first campaign; their origins are in
[evidence-relocation.json](https://github.com/ai-dynamo/aisimulate/blob/3630fed8e30ec24edee78f569b9d7276e60ddb4a/benchmarks/evidence/gbdt_seeded_revision_20261103/evidence-relocation.json). The
[compaction script](https://github.com/ai-dynamo/aisimulate/blob/3630fed8e30ec24edee78f569b9d7276e60ddb4a/benchmarks/evidence/gbdt_seeded_revision_20261103/compact_host_costs.py) verifies every backup/archive member
before removing redundant loose worker copies.
