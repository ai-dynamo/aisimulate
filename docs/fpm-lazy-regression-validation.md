<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Configurable linear regression: accuracy and CPU validation

These measurements compare the canonical default regression configuration with
signed lazy configurations on the same integrated source, `5a7e86dd`. This is a
configuration comparison, not a before/after executable comparison. It includes
the `VecDeque` retention queues, backend request-array compatibility, and
preservation of the previous serving snapshot after rejecting an identifiable
all-zero candidate.

## Configurations

The baseline is requested with `estimation_mode: fpm_regression`, fallback denied,
and an empty `estimator_config`. Rust resolves it to attention/MoE features,
nonnegative coefficients, a 4×4 attention/MoE retention grid, capacity 64 per store,
eager updates, minimum five observations, ridge 1e-9, and no scheduled rebuilds.
All comparisons below use the default rebuild interval (`null`); numerical
recovery and batch fallbacks remain enabled.

The FPM Gym candidate uses the full signed attention/MoE fit, a 4×1 retention
grid, capacity 64, relative tolerance 1%, absolute tolerance 0.1 ms, window 8,
trigger 2, cooldown 1, and startup 10. “Full signed” means the implementation's
full feature-set solve. The historical subset-enumeration diagnostic patches
are not part of this change or these measurements.

For AgentX, ShareGPT, and LongBench, the two lazy selections were chosen in the
earlier exploratory report. “Balanced” and “fast” describe that earlier
accuracy/CPU tradeoff; neither is an exhaustive optimum or a result on an
untouched holdout.

| Configuration | Role | Fit features | Retention grid / capacity | Relative / absolute tolerance | Window / trigger / cooldown |
|---|---|---|---|---|---|
| Balanced lazy | Prefill | attention, moe, nE, logF | 1×2 / 4096 | 5% / 0.1 ms | 8 / 2 / 4 |
| Balanced lazy | Decode | attention, moe, logN, n2 | 1×3 / 512 | 2.5% / 0 ms | 1 / 1 / 1 |
| Fast lazy | Prefill | attention, moe, nE, logF | 1×2 / 2048 | 10% / 0.1 ms | 8 / 2 / 4 |
| Fast lazy | Decode | attention, moe, logN, n2 | 1×3 / 512 | 2.5% / 0 ms | 8 / 2 / 4 |

All four capture configurations use signed coefficients, attention/MoE retention
axes, minimum five observations, ridge 1e-9, and startup 10. See the
[feature definitions and policy contract](core-api.md#linear-features-and-lazy-coefficient-updates).
Features, retention, slope constraints, and update policy vary together; the
CPU differences cannot be attributed to the lazy gate alone.

## Measurement method

Every native prediction precedes its corresponding update. Accuracy uses each
configuration's available predictions, accompanied by coverage and scores on
the intersection of available predictions. Missing predictions are never treated
as zero error. FPM Gym and captured workloads have different scoring protocols,
described with their tables, so their MAPEs should not be pooled.

The controlled comparisons use the same fixed bucket hasher in all configurations
to make retention tie-breaking reproducible. This is the only modification in
the controlled source copy. Separate runs of an unmodified production build
measure sensitivity to randomized hash-map ordering; production results need
not match a single controlled run.

CPU timings measure process user plus system time around native prediction and
update loops on an Apple M5 Pro macOS desktop. Parsing, model construction,
dispatch preparation, diagnostics, serialization, and destruction are outside
the clocks. Fresh models are used for each pass. Short inputs repeat fresh
passes to target at least 100 ms per timing block. Twelve matched rounds balance
configuration order; suite weights count original observations once, regardless
of repetitions. CPU changes are medians of within-round ratios and need not equal
ratios of the displayed medians. Timing campaigns run serially, after builds and tests complete.
Reported intervals describe variation across those rounds, not population
confidence intervals or Linux performance.

## FPM Gym

The pinned [FPM Gym dataset](https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset/tree/5487a4599a7fbc012c07bcd3699754bdf4a8bef7) contains 20 registered configurations; all 15 with measurements are included here: **2,155,267 observations**. Five catalog entries have no measurements. No measured case is removed. Inputs retain manifest-file and source-row order; the dataset does not declare deployment chronology. Dedicated workers and workload stores keep separate models.

Full-stream accuracy has no warmup exclusion. The reused final 30% of each worker/store (**646,607 observations**) is also scored after updating on its 70% prefix. Updating continues in the suffix; this is online evaluation, not a frozen fitted model. Macro MAPE gives the 15 cases equal weight. CPU uses complete case streams with original-row weighting.

### Aggregate results

MAPE uses equal-case means. The common-support columns score the exact two-way intersection of available predictions. Own coverage is each configuration's unfiltered availability.

| Configuration | Full own MAPE % | Full common MAPE % | Suffix own MAPE % | Suffix common MAPE % | Own full coverage % | Own suffix coverage % | CPU µs/observation median [IQR] |
|---|---:|---:|---:|---:|---:|---:|---:|
| current_defaults | 241.2654 | 241.2654 | 89.7721 | 89.7721 | 99.9056 | 99.9831 | 1.038822 [1.033350, 1.043013] |
| signed_lazy | 23.8874 | 23.5111 | 18.3061 | 18.2027 | 99.9865 | 100.0000 | 0.418777 [0.417996, 0.419755] |

The largest default full-sequence case error is `MiniMaxAI--MiniMax-M2.7/b200-sxm/vllm/0.25.1/tep4` at 3474.1972% MAPE; this case contributes 231.6131 percentage points to the equal-case default macro. The per-case table shows where the aggregate difference comes from.

Signed lazy paired CPU change: **-59.66%**,  IQR [-59.88%, -59.46%], observed 12-round range [-60.07%, -59.25%]. Paired speedup median: 2.479×.

### Per-case results

MAPE columns below use each configuration's own predictions; [the Gym result JSON](fpm-lazy-gym-results.json) also includes common support, coverage, and workload-specific results.

| Dataset configuration | Default full MAPE % | Signed full MAPE % | Default suffix MAPE % | Signed suffix MAPE % | Default CPU µs/obs | Signed CPU µs/obs | Paired CPU change % |
|---|---:|---:|---:|---:|---:|---:|---:|
| MiniMaxAI--MiniMax-M2.7/b200-sxm/vllm/0.25.1/tep4 | 3474.1972 | 251.5229 | 1179.0050 | 176.0061 | 0.912553 | 0.313917 | -66.10 |
| MiniMaxAI--MiniMax-M2.7/h200-sxm/vllm/0.25.1/pure-tp4 | 4.5333 | 2.2858 | 7.9223 | 3.1517 | 1.368732 | 0.782350 | -42.82 |
| MiniMaxAI--MiniMax-M2.7/h200-sxm/vllm/0.25.1/tep4 | 19.9016 | 14.6188 | 38.7928 | 21.6947 | 1.318796 | 0.726862 | -44.94 |
| deepseek-ai--DeepSeek-V4-Flash-0731/b200-sxm/vllm/0.25.1/dep4 | 26.9496 | 20.0727 | 19.5610 | 11.8448 | 0.971931 | 0.433395 | -55.20 |
| deepseek-ai--DeepSeek-V4-Flash-0731/b200-sxm/vllm/0.25.1/tep4 | 28.8899 | 17.7679 | 28.4043 | 12.6344 | 0.974858 | 0.437223 | -55.15 |
| deepseek-ai--DeepSeek-V4-Pro/b200-sxm/vllm/0.28.0/dep8 | 4.5963 | 4.2532 | 4.6008 | 4.3984 | 0.846337 | 0.275066 | -67.30 |
| deepseek-ai--DeepSeek-V4-Pro/b300-sxm/sglang/git-71de97b264b0-fpm-ed18d64951b9/dep8 | 2.1844 | 2.1575 | 2.2565 | 2.2163 | 1.005151 | 0.385488 | -61.75 |
| deepseek-ai--DeepSeek-V4-Pro/gb300/vllm/0.26.0/dep8 | 1.4508 | 0.6340 | 1.9755 | 0.5644 | 1.503218 | 0.905035 | -39.79 |
| deepseek-ai--DeepSeek-V4-Pro/gb300/vllm/0.26.0/tep8 | 4.3945 | 0.5296 | 6.8681 | 0.5328 | 0.908317 | 0.271644 | -70.10 |
| moonshotai--Kimi-K3/gb300/vllm/0.1.dev19262/tep8 | 3.6738 | 1.8308 | 2.5042 | 1.1180 | 0.945598 | 0.315878 | -66.46 |
| nvidia--GLM-5.2-NVFP4/b200-sxm/sglang/git-71de97b264b0-fpm-ed18d64951b9/pure-tp8 | 1.0421 | 0.8728 | 1.0790 | 0.7910 | 1.013841 | 0.348112 | -65.38 |
| nvidia--GLM-5.2-NVFP4/b200-sxm/vllm/0.25.1/dep8 | 13.9148 | 12.3711 | 19.5488 | 14.2522 | 1.316935 | 0.758803 | -42.43 |
| nvidia--GLM-5.2-NVFP4/b200-sxm/vllm/0.25.1/tep8 | 27.8864 | 25.9502 | 29.1391 | 22.5896 | 1.362145 | 0.796355 | -41.57 |
| nvidia--GLM-5.2-NVFP4/gb200/vllm/0.28.0/dep16 | 3.3975 | 2.0714 | 3.1491 | 1.7232 | 0.897427 | 0.291331 | -67.56 |
| nvidia--MiniMax-M3-NVFP4/b200-sxm/vllm/0.28.0/pure-tp4 | 1.9689 | 1.3726 | 1.7757 | 1.0731 | 0.879079 | 0.265536 | -69.79 |

### Production hashing crosscheck

Five fresh production-hasher runs per configuration are separate, unpaired accuracy reproducibility checks.

| Configuration | Full macro MAPE % median [min, max] | Suffix macro MAPE % median [min, max] |
|---|---:|---:|
| current_defaults | 239.7443[232.6318,241.4980] | 88.1559[82.5496,91.1879] |
| signed_lazy | 23.8594[23.5347,24.0514] | 18.2843[18.1449,18.5900] |


The MiniMax-M2.7 / B200 / TEP4 case dominates the aggregate error for both configurations. Its very large errors are retained in the tables. The suffix is a reused benchmark partition, and the aggregate improvement does not imply every case improves. Coverage also differs: default misses 2,034 full-stream rows and 109 suffix rows; signed lazy misses 290 full-stream rows and none in the suffix. The common-support columns remove that denominator mismatch.

## AgentX, ShareGPT, and LongBench

These are **separate captured workloads, not FPM Gym datasets**. Their source is [AISim E2E Gym MR 91](https://gitlab-master.nvidia.com/dl/ai-dynamo/aisim-e2e-gym/-/merge_requests/91): 34 captures, 17 per role, totaling **5,854,891 observations**, across SGLang and vLLM. Accuracy starts a fresh model for each original capture and excludes the first ten input rows regardless of prediction availability, while still updating on those rows.

CPU uses 12 timestamp-merged backend/workload/role streams. Thus the accuracy and CPU rows refer to the same workload families with explicitly different model histories. No cross-workload merged trace is used. Both the fast and balanced selections are retained below; the main comparison uses the prior balanced selection.

### Primary controlled comparison

Accuracy uses fresh original-capture models, predicts before tuning, and excludes the first ten input rows regardless of prediction availability. Values below use only rows where both default and balanced predict; the MAPE denominator is that common predicted-row count. The role macro gives captures equal weight within each backend/workload group, then gives the six groups equal weight. CPU uses separately replayed continuous timestamp-merged streams; it includes every input row and is weighted by original row counts.

| Role | Default MAPE % | Balanced MAPE % | Relative MAPE change | Default CPU µs/row | Balanced CPU µs/row | Paired CPU change | Paired speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| prefill | 22.7083 | 3.6898 | -83.75% | 0.761124 | 0.245886 | -67.70% | 3.10× |
| decode | 4.1372 | 2.3969 | -42.06% | 0.786096 | 0.430375 | -45.21% | 1.83× |

### Every workload and backend

These accuracy values pool scored rows from matching original captures; CPU values use the matching continuous runtime streams. Balanced SGLang LongBench prefill regresses slightly, from 1.4590% to 1.5083% MAPE; all groups are retained.

| Backend / workload / role | Default MAPE % | Balanced MAPE % | Default CPU µs/row | Balanced CPU µs/row | Paired CPU change % [min,max] |
|---|---:|---:|---:|---:|---:|
| sglang-1.6.0-dev.1 / agentx / prefill | 7.5194 | 6.8464 | 0.777910 | 0.283524 | -63.39 [-64.20,-62.83] |
| sglang-1.6.0-dev.1 / agentx / decode | 2.6568 | 1.9741 | 0.768107 | 0.464831 | -39.95 [-41.57,-37.86] |
| sglang-1.6.0-dev.1 / sharegpt / prefill | 2.9659 | 2.4088 | 0.746051 | 0.221234 | -70.27 [-72.23,-68.78] |
| sglang-1.6.0-dev.1 / sharegpt / decode | 2.9663 | 1.7642 | 0.764330 | 0.324170 | -57.86 [-60.24,-55.84] |
| sglang-1.6.0-dev.1 / longbench / prefill | 1.4590 | 1.5083 | 0.754871 | 0.200716 | -73.42 [-73.76,-72.72] |
| sglang-1.6.0-dev.1 / longbench / decode | 3.0493 | 2.0547 | 0.785109 | 0.362137 | -53.91 [-54.89,-53.48] |
| vllm-1.4.0 / agentx / prefill | 95.8504 | 2.9376 | 0.743760 | 0.239849 | -67.64 [-69.15,-58.03] |
| vllm-1.4.0 / agentx / decode | 8.7679 | 3.9848 | 0.781583 | 0.567552 | -27.36 [-28.08,-25.40] |
| vllm-1.4.0 / sharegpt / prefill | 31.6913 | 6.8034 | 0.749558 | 0.300013 | -59.98 [-60.98,-52.89] |
| vllm-1.4.0 / sharegpt / decode | 3.4440 | 2.0511 | 0.803782 | 0.334120 | -58.46 [-61.61,-57.64] |
| vllm-1.4.0 / longbench / prefill | 3.0387 | 1.9823 | 0.783565 | 0.231383 | -70.48 [-70.91,-69.86] |
| vllm-1.4.0 / longbench / decode | 3.1460 | 2.1110 | 0.799882 | 0.426648 | -46.42 [-48.07,-45.57] |

### Fast selection and production verification

Controlled accuracy below uses each model’s own available predictions. Actual production verification uses the same source without the diagnostic fixed hasher; its single fresh replay is separate because production retention is randomized. The [capture result JSON](fpm-lazy-capture-results.json) preserves matched-support results, resolved configurations, and source/input/timing fingerprints.

| Hasher / configuration / role | Own-support macro MAPE % | Scored predictions / eligible rows | Missing | Controlled CPU µs/row |
|---|---:|---:|---:|---:|
| controlled / default / prefill | 22.7083 | 226,409 / 226,787 | 378 | 0.761124 |
| controlled / default / decode | 4.1372 | 5,625,897 / 5,627,764 | 1,867 | 0.786096 |
| controlled / fast / prefill | 3.7750 | 226,787 / 226,787 | 0 | 0.199750 |
| controlled / fast / decode | 2.5127 | 5,627,764 / 5,627,764 | 0 | 0.301322 |
| controlled / balanced / prefill | 3.6925 | 226,787 / 226,787 | 0 | 0.245886 |
| controlled / balanced / decode | 2.3967 | 5,627,764 / 5,627,764 | 0 | 0.430375 |
| production / default / prefill | 22.5698 | 226,409 / 226,787 | 378 | not measured |
| production / default / decode | 4.1782 | 5,626,770 / 5,627,764 | 994 | not measured |
| production / fast / prefill | 3.7732 | 226,787 / 226,787 | 0 | not measured |
| production / fast / decode | 2.5111 | 5,627,764 / 5,627,764 | 0 | not measured |
| production / balanced / prefill | 3.6918 | 226,787 / 226,787 | 0 | not measured |
| production / balanced / decode | 2.3939 | 5,627,764 / 5,627,764 | 0 | not measured |


The capture production check is one fresh run per configuration, not a multi-seed variability interval. The historical report's signed baseline and archival rebuild policy are different from this canonical nonnegative baseline, so historical baseline values and subset-enumeration diagnostic results are not substituted into these tables.

## Validation and reproduction limits

Source anchor: `5a7e86ddd331c0aee25eccb57bb28a815089280c`. The final PR adds this
report and result summaries after that source commit. Release builds use the
repository dependency lock, optimization level 3, one codegen unit, and thin LTO.
The native Python module used for tests has SHA-256
`faa25c067cf9d563d1d6d6a1730ab2de299e4ec0d1caeb282084f6f04d81bd9b`.

Completed checks:

- `cargo test --locked --offline -p aisimulate-core perfmodel::fpm`: **172 passed**.
- `CARGO_TARGET_DIR=target cargo test --locked --offline --manifest-path crates/tests/public-api/Cargo.toml`: **14 passed**.
- `cargo fmt --all --check`, affected Python Ruff checks, generated CODEOWNERS,
  and `git diff --check`: passed.
- **614 Python tests passed, zero skipped**, using an import loader that explicitly
  selects the newly built release module before calling `pytest.main` with
  `-p no:timeout -c pytest.ini` and these targets:

  ```text
  crates/core/parity_tests/perfmodel/test_engine_step_parity.py
  crates/core/parity_tests/perfmodel/test_compile_engine_parity.py
  python/aisimulate/tests/unit/sdk/test_rust_engine_step.py
  python/aisimulate/tests/unit/sdk/test_fpm_spline.py
  python/aisimulate/tests/cross_package/test_core_public_api.py
  python/aisimulate/tests/cross_package/test_single_oracle_contract.py
  ```

No frozen golden values were changed. Source review against `main` found no
actionable defects after integration; a separate review checked the benchmark
scoring, clocks, weighting, coverage, and provenance assertions. Hosted CI and
CodeRabbit review are outside this local evidence.

The accompanying JSON summaries record configurations, source/input fingerprints,
per-case accuracy, coverage, and timing distributions. Raw private captures,
large prepared inputs, native binaries, and local orchestration scripts are not
shipped in the PR. Reproduction requires access to the pinned Gym revision and,
for the separate capture comparison, MR 91's data and the same input ordering.
The results qualify these measured configurations and workload histories;
they do not establish an optimum, unseen-data accuracy guarantee, or end-to-end
simulator speedup.
