<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Learned-model publication cost evidence

This directory records CPU cost and numerical compatibility for standalone online
GBDT and the combined calibrated/online causal selector. The paired workstation
campaign and its 30 cost-process measurements are complete; those probes do not
measure serving-workload prediction accuracy. The separate
[historical accuracy summary](#historical-accuracy-comparison) below extracts
exploratory MAPE results from the earlier selector sweep.

The accompanying [validation receipt](validation.json) records the Rust,
Python, and earlier numerical parity checks, with their exact scope. Timing
measurements here do not replace those correctness checks.

## Measured costs and compatibility

All 30 process measurements passed the collector's input and binary hash checks
and finite-prediction checks. Online runs also met the expected 256-label and
256-publication counts. Full
per-process statistics and hashes are in [measurements.json](measurements.json).

The first paired comparison measured the standalone publication and root
histogram reuse:

| History | Arm | Setup ms | First query µs | Warm query µs | Update p50 ms | Update p95 ms | Peak RSS MiB | Online state MiB |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | PR493 hybrid | 4.075 | 27.571 | 4.257 | 0.005235 | 0.005851 | 45.83 | — |
| 10,000 | Canonical frozen hybrid | 0.651 | 24.211 | 4.254 | 0.005179 | 0.005790 | 45.70 | — |
| 10,000 | Online before reuse | 231.093 | 62.873 | 4.278 | 259.155 | 269.589 | 302.18 | 125.85 |
| 10,000 | Online after reuse | 223.082 | 65.491 | 4.223 | 249.006 | 259.682 | 301.46 | 125.85 |
| 256 | Online before reuse | 92.473 | 65.238 | 5.560 | 94.072 | 98.193 | 272.57 | 110.17 |
| 256 | Online after reuse | 92.728 | 60.240 | 5.506 | 93.830 | 98.178 | 271.95 | 110.17 |

The 10,000-row update p50 decreased by 3.9%; the 256-row difference was 0.26%.
These are modest descriptive differences under the timing caveats below. All
256 initial and 256 final predictions are bit-identical before/after reuse in
all three rounds of both fixtures. This verifies the observed initial and final
outputs, not every intermediate fitted state.

The subsequent paired comparison measured the combined selector against raw
online fitting, with no concurrent release compilation:

| History | Arm | Setup ms | First query µs | Warm query µs | Update p50 ms | Update p95 ms | Peak RSS MiB | Online state MiB |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | Raw online | 224.005 | 63.383 | 4.209 | 248.912 | 259.144 | 303.28 | 125.85 |
| 10,000 | Combined selector | 223.028 | 71.494 | 4.251 | 248.748 | 259.213 | 302.54 | 125.85 |
| 256 | Raw online | 94.408 | 62.597 | 5.475 | 94.037 | 99.325 | 272.97 | 110.17 |
| 256 | Combined selector | 92.341 | 69.115 | 5.528 | 93.469 | 97.704 | 273.41 | 110.17 |

Warm queries before feedback cost approximately 1% more for the combined model
in both fixtures. Update costs remain dominated by online forest fitting; the
small timing differences do not establish a selector speedup. The online-state
estimate is identical because it counts the online learner only; combined
frozen/calibration/selector storage is included in process RSS, not that estimate.

PR493 compatibility-adapter versus canonical frozen predictions differ by at
most two ULPs: 2.22e-16 ms initially and 4.44e-16 ms after calibration. The new
binary enables `serde_json` float-roundtrip parsing; this comparison does not
claim bitwise equality with the older parser. The standalone-online checkpoint
continuation is exact across the parent and combined-capable binary: exported
artifact and all 256 final predictions match after seven accepted labels.

The existing forward-path five-round smoke passed without blocking regressions.
For its single selected operator-level case, median cold time was 55.680→56.707
µs and median warm time was 7.142→7.074 µs; predictions were identical. This is
one case of the 64-case matrix, not a full gate run.

## Protocol and scope

The cost probes run in fresh processes on one Intel Core Ultra 9 285K CPU core
(`taskset -c 2`), with OMP, OpenBLAS, MKL, and Rayon thread counts fixed to one.
Each comparison has three paired rounds, with arm order reversed on the middle
round. Summary values are medians of the three process-level statistics; an
update p95 is the median of each process's empirical p95, not a pooled p95.
A release compilation may have overlapped the middle 10,000-row comparison in
the first phase. Treat the small optimization difference as descriptive; these
three rounds do not establish a statistically robust speedup. The later combined
comparison runs after that build finishes.

Two synthetic decode fixtures contain 10,000 and 256 historical training rows.
Both use the 17 `fpm5` aggregate features, 150 trees, at most 15 leaves per seed
tree, learning rate 0.2, and the same 256 subsequent query shapes. Each query's
feedback target is 1.3 times its initial seed prediction. Every feedback label
triggers an update: W64, `update_every=1`, history mass 0.1 with a 32-observation
ramp, historical bin grouping, lazy derivatives, eager interval 32, candidate
sampling 0.5, and split-rank tolerance 0.01. The selector uses alpha 0.025,
warmup 10, relative margin 0.025, and persistence 1; hybrid calibration retains
32 observations and uses a 2×2 grid.

Setup measures model construction. Cold query is the first query in that fresh
process. Warm query averages 1,024 calls before feedback, after one full pass over
all 256 shapes, repeated seven times. Update statistics cover 256 chronological feedback calls.
These timings include the Python SDK and JSON bridge. Peak RSS is process-wide;
reported training state is the learner's retained-storage estimate, which omits
some allocator/runtime overhead and fitting scratch space.

The 10,000-row fixture retains 9,455 distinct historical bin vectors (only a
1.058× compression ratio), and all 256 updates take the eager fallback. It is a
stress case for full historical refresh, not evidence that lazy updates are
usually cheap. The 256-row fixture varies historical size only; neither fixture
covers production workload diversity, concurrency, mixed/prefill stores, delayed
feedback, or hardware execution accuracy. No accuracy claim follows from them.

## Provenance and reproduction

The baseline is PR493 commit `56f47543a15ae1f7842ecc5bcf653409c138d496`.
The standalone candidate is commit
`470898a39f526aaa0af479ba89092a92b5883ea8`. The pre-optimization arm was an
uncommitted source snapshot before eager root-histogram reuse; it must be
identified by source and binary hashes, not assigned either Git revision.
It also predates exact-byte checkpoint loading, identity-preserving artifact
export, and destination reservation for saving. Those changes do not execute in
the measured update calls, but setup differences cannot be isolated to histogram
reuse.
The combined candidate uses the production implementation committed in
`0f38d33f1b9c0409eff8307d167da819479741c5`; tests added after its source snapshot
mean its complete source tree must still be identified by the recorded hashes.

`analyze.py` reads the completed workstation campaign and emits compact JSON:
source-inventory digests and critical source hashes, actual native-binary hashes,
all fixture-file hashes, raw per-process cost statistics, output hashes,
prediction comparisons, the forward-gate comparison, and checkpoint continuation
results. It checks all 30 expected cost files and validates hashes before
discarding the large prediction arrays. The original arrays remain in the
workstation input directory recorded in the JSON.

```bash
python analyze.py /path/to/fpm-model-publication-v1 > measurements.json
```

The analysis performs no fitting or benchmarks. Repeating the cost probe requires
separate candidate Python environments and the recorded fixture bytes. Preparing
the fixture again also requires the recorded sklearn version; fresh training
must not be silently treated as the same input.
`protocol/` contains maintained copies of the execution scripts, with SPDX
headers, import cleanup, and lint/formatting changes. The JSON records execution
originals under `protocol_sha256` and these copies under
`maintained_protocol_sha256`; the hashes intentionally differ. Their executable
statements are compared after excluding import declarations. The controllers
retain the workstation's original absolute paths: adapt their environment paths
for another machine. The two probe scripts already accept a fixture path.

The separate five-round forward gate is a **one-case smoke check**, selected from
a 64-case matrix. Its cold metric is a steady-state unseen query, unlike the
fresh-process first-query metric above. It checks existing operator-level
prediction cost; it does not qualify the learned model for phase-only engine
replay. The checkpoint check saves standalone online state with three pending
labels, reloads it in the combined-capable binary as ordinary `online`, admits
four more labels, and flushes. Combined selector persistence remains unsupported.

## Historical accuracy comparison

The completed PR493 selector sweep provides controlled comparisons against
standalone hybrid calibration. These are historical exploratory results, not a
rerun of the current combined native binary. [accuracy-comparison.json](accuracy-comparison.json)
contains exact selected source records, three replica scores, source SHA256s,
all protocol-level metrics, settings, guards and extraction limits. The source
hash chain from findings to comparisons/ranking and from comparisons to the
sealed campaign/manifest was checked before extracting these values.

All values below are MAPE percentages; lower is better. `64/16` and `32/4` mean
calibration observation capacity / bucket count, corresponding to 4×4 and 2×2
grids. The 64/16 calibration uses PR493 defaults; 32/4 is the tuned calibration.
Both use factor bounds [0.5, 2] and readiness 5. Each combined column uses the
same calibration as the adjacent standalone column, with selector alpha 0.025,
warmup 10, relative margin 0.025 and persistence 1.

| Protocol | Hybrid 64/16 | Combined 64/16 | Hybrid 32/4 | Combined 32/4 |
| --- | ---: | ---: | ---: | ---: |
| B+C: mean of nine seed ratios | 3.3144% | 3.2002% | 3.3214% | 3.2422% |
| Paged C | 11.6698% | 7.2415% | 11.5264% | 7.3816% |
| Strict C | 8.2151% | 5.8821% | 8.2475% | 5.8835% |
| A+C: one supported capture | 5.5466% | 4.1561% | 5.6950% | 3.9033% |
| Cross-workload: six directions | 3.9665% | 3.2197% | 4.0170% | 3.3917% |

Both online experts use the explicit W64 research profile: structural fitting,
updates after each admitted label, history mass 0.1 with ramp 32, grouped history,
lazy derivatives with eager interval 32, split sampling 0.5 and rank tolerance
0.01, with all trees mutable. These are not whole-model API defaults. Each matched
pair starts from the same verified PR460-recipe seed (150 trees, at most 15 leaves,
learning rate 0.2, 17 `fpm5` features). The five-second admission schedule is
unchanged, and predictions precede feedback.

The 223 evaluation cells comprise 126 B+C cells (14 captures × nine seed ratios),
14 paged-C, 14 strict-C, one supported A+C capture, and 68 transfer cells across
six directions. B+C trains chronological 10–90% prefixes and scores the remaining
suffix. Paged C bootstraps each worker/store with 20 consecutive labels; strict C
uses only scheduled labels even during bootstrap. A+C starts from a separate
measured self-benchmark seed; only Kimi GB300 supports it. Transfers train on a
source workload family without target-family labels. Cold-start protocols have
different readiness masks; each comparison within a protocol uses identical
ready rows for all methods.

Per cell/query, MAPE is `100/n * sum(abs(prediction/observed - 1))`. Three native
process replicas are averaged at the score level before hierarchical aggregation;
predictions are not averaged. The Gym hierarchy equally weights captures, query
types, backends and datasets in that order. Transfers equally weight captures
within direction/backend/role, then those groups, then six directions. B+C gives
each of its nine ratios equal weight. There is no single pooled score across
these protocols, and repeated captures do not count as independent data.

Combined 32/4 is the sweep's declared minimax winner, candidate 783, and passes
all three replicas' campaign guards. Combined 64/16, candidate 5283, improves
hybrid calibration in all five displayed means but fails the A+C guard against
the original FIFO reference: 4.1561% exceeds 3.9953%. Mean gains are not uniform:
combined 32/4 regresses in 14 of 24 transfer direction/backend/role groups versus
PR493-default hybrid, and AgentX → LongBench worsens from 1.6822% to 1.7629%
against its own matched 32/4 hybrid.

All captures were already examined, and the sweep selected among 9,000 settings
on these same evaluation populations. The stable expert was replayed with pinned
PR493 code; online experts retain cached predictions from the earlier runtime.
The causal selector was scored over those trajectories. These results establish
neither current combined-native replay parity nor accuracy on untouched data.
The three process replicas measure implementation variation, not independent
dataset uncertainty. The compact receipt does not bundle raw captures or the
full research replay archive.
