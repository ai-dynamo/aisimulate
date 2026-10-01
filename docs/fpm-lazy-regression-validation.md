<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Configurable linear regression and lazy-update validation

All 44 complete-case Gym accuracy runs completed with zero prediction or tuning errors. The four approved complete cases contain 101,237 observations. Each run scores every eligible observation prequentially, with no fixed warmup skip. The scope and reproducibility records are described below.


## Scope and method

Measured on 2026-09-29 (local time), macOS 26.7, Apple Silicon. This is a
bounded implementation check on existing FPM Gym cases. It does not select a
universal optimum, validate Linux CPU costs, or include the private
AgentX/ShareGPT/LongBench sweep captures.

These measurements predate the safeguard that preserves the previous linear
serving model when an identifiable all-zero candidate is rejected. They have
not been rerun for that behavior change.

The dataset is the pinned cached `nvidia/aisimulate-fpm-dataset` revision
`5487a4599a7fbc012c07bcd3699754bdf4a8bef7`. Its catalog contains 20
configurations, including five without measurements. The unchanged
`scripts/fpm_accuracy/hf` loader and `score()` evaluator supply the membership,
rank grouping, worker separation, coverage, and predict-before-tune order.
Published records do not declare chronology: the loader preserves manifest
file order and source row order (`file_order_fallback`). Persistent model state
is preserved in that order; these are not claimed to be chronological deployment
traces. No artificial warmup exclusion or request-length reconstruction is used.

| Label | Registered configuration | Snapshot | Full observations |
|---|---|---|---:|
| Flash dep4 | DeepSeek-V4-Flash-0731 / B200 / vLLM 0.25.1 / dep4 | `aisim-commit-unknown-1cb1c4de` | 1,356 |
| Flash tep4 | DeepSeek-V4-Flash-0731 / B200 / vLLM 0.25.1 / tep4 | `aisim-commit-unknown-900fc3ca` | 1,368 |
| MiniMax tep4 | MiniMax-M2.7 / H200 / vLLM 0.25.1 / tep4 | `aisim-commit-unknown-4baad028` | 286 |
| Pro SGLang dep8 | DeepSeek-V4-Pro / B300 / SGLang `git-71de97b264b0-fpm-ed18d64951b9` / dep8 | `aisim-commit-unknown-bf0f277d` | 98,227 |

These cases contain prefill and decode observations with one, four, or eight
attention-DP ranks. They cover dedicated workers and an aggregated worker;
they do not provide locally mixed workload evidence. Request-list features
are covered by hand-derived core tests, not by these scalar-only Gym cases.

Configuration labels below use A=`attention`, M=`moe`. Defaults are the
existing nonnegative A/M fit, A/M 4×4 retention grid, and eager updates.
`n64`/`n512` set per-store capacity. `k256` means the periodic rebuild interval
is 256 insertions plus actual evictions; it is not a minimum gap between repairs.
All other intervals are disabled. `amn` selects signed A/M/n fitting and
A/M/n retention with `[4,4,2]` bins. `am_logn_n2` selects signed A/M/logN/n2
fitting and retains the A/M 4×4 grid. Every lazy variant uses a 5% relative
threshold, 0.1 ms absolute threshold, window 8, trigger 2, cooldown 4, and
startup 10. The A/M/logN/n2 eager/lazy pair also changes rebuilding, so its
cost difference cannot be attributed to lazy updates alone.

## Accuracy on complete Gym cases

Each cell is MAPE percent / unavailable observations. Available predictions form the MAPE denominator; every eligible observation contributes to the coverage denominator. These are implementation checks, not an optimized hyperparameter selection.

| Candidate configuration | Flash dep4 | Flash tep4 | MiniMax tep4 | Pro SGLang dep8 |
|---|---:|---:|---:|---:|
| default_n64_none | 27.6057 / 21 | 19.7815 / 20 | 18.7837 / 10 | 2.1678 / 15 |
| default_n64_k256 | 26.1426 / 21 | 20.0786 / 20 | 20.2948 / 10 | 2.1800 / 15 |
| default_n512_none | 24.4595 / 21 | 20.7757 / 20 | 25.7772 / 10 | 2.2880 / 15 |
| am_lazy_n64 | 26.8603 / 21 | 40.5203 / 20 | 19.9979 / 10 | 2.3673 / 15 |
| amn_eager_n64 | 25.9772 / 20 | 25.5624 / 20 | 24.5151 / 10 | 2.1835 / 10 |
| amn_lazy_n64 | 28.2900 / 20 | 20.4368 / 20 | 25.8697 / 10 | 2.3299 / 10 |
| am_logn_n2_eager_n512 | 18.3324 / 20 | 43.2416 / 20 | 17.9903 / 10 | 1.9568 / 10 |
| am_logn_n2_lazy_n512_k256 | 21.0835 / 20 | 55.4967 / 20 | 19.2066 / 10 | 2.1024 / 10 |

Case lengths, in table order: 1,356; 1,368; 286; 98,227 observations. The full inventory contains 2,155,267 eligible observations across 15 ready configurations; five current configurations have no measurements. Those extra cases were inventoried but are outside this bounded validation.

## Default controls and sampler variability

The preexisting HashMap tie-breaking in bounded retention is randomized across independent model instances. This affects which observations survive capacity eviction. Single-run default MAPEs therefore need not match across binaries. Seven independent runs per binary on each small case quantify that variation; all six baseline/candidate ranges overlap. This does not establish statistical equivalence. The N512 controls have identical counts and MAPE within 1e-12 percentage points on all three small cases before capacity eviction. Production sampler semantics were preserved.

| Case | Control | Baseline median [min,max] MAPE % | Candidate median [min,max] MAPE % |
|---|---|---:|---:|
| Flash dep4 | default_n64_none | 25.797 [22.708, 30.078] | 27.573 [24.616, 29.205] |
| Flash dep4 | default_n64_k256 | 26.984 [23.200, 31.725] | 29.492 [23.875, 32.513] |
| Flash tep4 | default_n64_none | 21.740 [19.447, 31.639] | 20.489 [18.871, 34.382] |
| Flash tep4 | default_n64_k256 | 21.189 [18.465, 26.189] | 31.666 [20.604, 33.972] |
| MiniMax tep4 | default_n64_none | 19.711 [19.469, 20.843] | 19.805 [18.796, 20.524] |
| MiniMax tep4 | default_n64_k256 | 20.426 [18.815, 20.676] | 20.166 [19.731, 20.674] |

## Native CPU cost

Values are microseconds per observation: median of seven whole-stream averages. Each pass starts from fresh models, then persists their state across all rows. The measured calls include canonical API feature extraction, retention, statistics, and fitting. Parsing, construction, worker index preparation, and final diagnostics are excluded. Matched controls alternate baseline/candidate order. Four timing streams use the first min(5,000, full case rows), totaling 8,010 observations per pass. No builds, tests, or other benchmarks ran concurrently.

### Flash dep4 (1,356 consecutive observations)

| Binary / configuration | Update µs/observation | Predict + update µs/observation |
|---|---:|---:|
| baseline / default_n64_none | 1.1492 | 1.1869 |
| candidate / default_n64_none | 1.2492 | 1.1335 |
| baseline / default_n64_k256 | 1.1500 | 1.1408 |
| candidate / default_n64_k256 | 1.1588 | 1.2056 |
| baseline / default_n512_none | 1.3208 | 1.2834 |
| candidate / default_n512_none | 1.3217 | 1.3466 |
| candidate / am_lazy_n64 | 0.5918 | 0.6012 |
| candidate / amn_eager_n64 | 2.9686 | 2.9449 |
| candidate / amn_lazy_n64 | 1.1440 | 1.1033 |
| candidate / am_logn_n2_eager_n512 | 1.1587 | 1.2204 |
| candidate / am_logn_n2_lazy_n512_k256 | 0.7880 | 0.8220 |

### Flash tep4 (1,368 consecutive observations)

| Binary / configuration | Update µs/observation | Predict + update µs/observation |
|---|---:|---:|
| baseline / default_n64_none | 1.1969 | 1.2251 |
| candidate / default_n64_none | 1.1561 | 1.1578 |
| baseline / default_n64_k256 | 1.1809 | 1.2222 |
| candidate / default_n64_k256 | 1.2217 | 1.2833 |
| baseline / default_n512_none | 1.4594 | 1.3469 |
| candidate / default_n512_none | 1.3744 | 1.3632 |
| candidate / am_lazy_n64 | 0.5811 | 0.5984 |
| candidate / amn_eager_n64 | 3.0016 | 3.0927 |
| candidate / amn_lazy_n64 | 1.0400 | 1.0964 |
| candidate / am_logn_n2_eager_n512 | 1.2305 | 1.2585 |
| candidate / am_logn_n2_lazy_n512_k256 | 0.7987 | 0.8766 |

### MiniMax tep4 (286 consecutive observations)

| Binary / configuration | Update µs/observation | Predict + update µs/observation |
|---|---:|---:|
| baseline / default_n64_none | 1.6603 | 1.5459 |
| candidate / default_n64_none | 1.6330 | 1.5587 |
| baseline / default_n64_k256 | 1.5679 | 1.5975 |
| candidate / default_n64_k256 | 1.4975 | 1.5533 |
| baseline / default_n512_none | 2.2158 | 2.4279 |
| candidate / default_n512_none | 2.3313 | 2.4009 |
| candidate / am_lazy_n64 | 0.9591 | 1.0054 |
| candidate / amn_eager_n64 | 5.2788 | 5.1737 |
| candidate / amn_lazy_n64 | 1.5364 | 1.5554 |
| candidate / am_logn_n2_eager_n512 | 2.1048 | 2.1645 |
| candidate / am_logn_n2_lazy_n512_k256 | 1.6536 | 1.8821 |

### Pro SGLang dep8 (5,000 consecutive observations)

| Binary / configuration | Update µs/observation | Predict + update µs/observation |
|---|---:|---:|
| baseline / default_n64_none | 1.7884 | 1.7650 |
| candidate / default_n64_none | 1.7898 | 1.8116 |
| baseline / default_n64_k256 | 1.7866 | 1.7851 |
| candidate / default_n64_k256 | 1.8520 | 1.8924 |
| baseline / default_n512_none | 5.7260 | 5.8794 |
| candidate / default_n512_none | 6.1227 | 6.1048 |
| candidate / am_lazy_n64 | 1.0833 | 1.0929 |
| candidate / amn_eager_n64 | 6.2255 | 6.2225 |
| candidate / amn_lazy_n64 | 1.2526 | 1.2707 |
| candidate / am_logn_n2_eager_n512 | 5.7138 | 5.8625 |
| candidate / am_logn_n2_lazy_n512_k256 | 5.4894 | 5.6267 |

All 616 timing trials completed. Repetitions and update/combined runs agree on final observable readiness and retained-count diagnostics. These diagnostics do not expose exact retained membership or coefficients. Raw timings, min/max variability, counts, binary hashes, and execution order are preserved in timing.jsonl, timing-summary.json, and timing-provenance.json.

## Interpretation and validation

Within the candidate implementation, enabling lazy updates with the same A/M
features and N64 grid reduced median update cost by 39.5–52.6% across these four
streams. With A/M/n fitting and its fixed 3D grid, the reduction was 61.5–79.9%.
Accuracy changed in both directions; for example, the Pro SGLang A/M MAPE
changed from 2.1678% to 2.3673%. Defaults remain eager. These configurations
are feature checks, not recommendations from a new sweep.

Matched default controls show small increases and decreases. Across the 24
case/configuration/operation comparisons, the median change was +1.02% and
the range was −5.83% to +8.70%; short streams and randomized retention contribute
to the variation. For N64 with rebuilding disabled, update-only changes ranged
from -3.4% to +8.7%. Pro SGLang N512 update medians were
5.726 vs 6.123 µs, with ranges [5.609, 5.852] and
[5.750, 6.173] µs. This remains an observed cost increase.

A diagnostic of the earlier candidate found zero full rebuilds in three
SGLang-prefix replays at either N64 or N512; an interval-1 positive control
confirmed the counters worked. Full rebuilds do not explain that earlier
slowdown. Specializing the default two-axis guard and the constraint mode
removed avoidable inner-loop checks; the final eager path also avoids
lazy-monitor prediction and duplicated finite-statistics scans. The residual
cost is not causally isolated by these measurements. No numerical repair or
coverage was disabled to improve timings.

Core checks passed: 141 FPM Rust tests, 14 external Rust API tests, and 623
Python/API/parity tests. Existing engine goldens were unchanged. Commands:

```bash
cargo test --locked --offline -p aisimulate-core --lib perfmodel::fpm
PYO3_PYTHON="$PWD/python/aisimulate/.venv/bin/python" CARGO_TARGET_DIR="$PWD/target" \
  cargo test --locked --offline --manifest-path crates/tests/public-api/Cargo.toml
python/aisimulate/.venv/bin/pytest -p no:timeout -c pytest.ini \
  python/aisimulate/tests/unit/sdk/test_fpm_spline.py \
  python/aisimulate/tests/unit/sdk/test_rust_engine_step.py \
  python/aisimulate/tests/cross_package/test_core_public_api.py \
  python/aisimulate/tests/cross_package/test_single_oracle_contract.py \
  tests/test_spline_estimator_config.py \
  crates/core/parity_tests/perfmodel/test_engine_step_parity.py \
  crates/core/parity_tests/perfmodel/test_compile_engine_parity.py
```

## Reproducibility

Baseline source: `ce98a7e2394ce7ff3acf4fbe6b4b3b0f99039dd4`. Candidate source is
the implementation in the commit containing this report. The measured core
source index SHA-256 is
`15a9daf5c97141abd38823194e78540360f86b2d18d4239a577b29d1dc7dd6ab`.
Both original native callers use release opt-level 3, codegen-units 1, thin LTO,
empty RUSTFLAGS, and registry dependencies checked against source lockfiles.
They call the public canonical API; no estimation math is duplicated.

| Artifact | SHA-256 |
|---|---|
| Baseline native timing caller | `2c05a288b041a5c75368598fa1c88340a48bd150f3100a186af12ab431e78a8f` |
| Candidate native timing caller | `a8d1f8c5554cedcdbd941d0f7bf5a2c142fc369ab067568d76d2d397c69b617b` |
| Shared native caller source | `d0884ce1c57549ac1e94fe46c8274f889e43acc32fab529b73eaf95f8f2e1749` |
| Baseline Python native module | `d6e35834e203f251b3f2e6aed7ad61d933b963fddd549af727dd0ad4bc6ccad1` |
| Candidate Python native module | `e059a60644e7d0a7617460bbf899e6b83fcc552628630079972bbe633f236241` |

The full local evidence bundle is retained in
`work/fpm_unified_regression_commit_20260930/`: `METHODS.md`, `cohort.json`,
`suite.json`, `validation.json`, build records, accuracy repetitions, original
validation drivers, and raw timing data. It is not part of the commit. The
native control has a separate local `work/fpm_commit_native_control/` bundle.
The committed tables contain the results needed to review this bounded check.
