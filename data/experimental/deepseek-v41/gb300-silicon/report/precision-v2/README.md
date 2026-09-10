# GB300 independent native forward comparison

The complete fixed holdout contains 38 configurations per Decoder profile. SILICON predicts all 38 with Decoder OFF and 28 with Decoder ON. The ten missing ON cases remain missing; HYBRID uses analytical fallback for those cases. These measurements do not establish HTTP E2E or whole-forward FPM accuracy.

## Observed versus predicted

Each observation is the median of ten independently invoked forwards, after taking the maximum of all four TP ranks for each invocation. One warmup per configuration is retained and excluded. Error is `(prediction / observation - 1) * 100`. Configurations have equal weight for signed bias, MAPE and APE percentiles. MAPE is `mean(abs(prediction / observation - 1)) * 100`; WAPE is `sum(abs(prediction - observation)) / sum(observation)`. The p90 column describes errors across configurations, not request tail latency.

| Profile | Mode | Predicted / measured | Mean signed error | Median APE | p90 APE | MAPE | WAPE |
|---|---|---:|---:|---:|---:|---:|---:|
| Decoder OFF | SOL | 38/38 | -96.19% | 94.99% | 99.65% | 96.19% | 96.04% |
| Decoder OFF | HYBRID | 38/38 | +0.57% | 7.09% | 13.17% | 7.31% | 7.09% |
| Decoder OFF | SILICON | 38/38 | +0.57% | 7.09% | 13.17% | 7.31% | 7.09% |
| Decoder ON | SOL | 38/38 | -96.26% | 95.04% | 99.66% | 96.26% | 96.13% |
| Decoder ON | HYBRID | 38/38 | -9.89% | 7.39% | 20.97% | 9.89% | 10.04% |
| Decoder ON | SILICON | 28/38 | -5.69% | 6.80% | 8.61% | 5.69% | 5.66% |

SILICON enforces measured V4.1 table coverage with shared-layer reuse disabled. Generic embedding, normalization, activation and memory operations still use the existing empirical models; a stage total is not entirely measured. SOL is the analytical lower-bound model and its large negative bias is reported explicitly. No correction factor was fitted to these holdouts.

![Prediction versus native forward](forward-comparison.png)

Horizontal whiskers show the minimum and maximum of the ten measured repetitions, not confidence intervals. The dashed line denotes exact agreement. Missing predictions have no point on the scatter and remain in the coverage denominator.

## Phase breakdown

| Profile | Mode | Phase | Coverage | Mean signed error | Median APE | MAPE | WAPE |
|---|---|---|---:|---:|---:|---:|---:|
| Decoder OFF | SOL | Prefill | 28/28 | -94.97% | 94.83% | 94.97% | 94.97% |
| Decoder OFF | SOL | Decode | 10/10 | -99.61% | 99.61% | 99.61% | 99.61% |
| Decoder OFF | HYBRID | Prefill | 28/28 | +5.34% | 5.57% | 5.35% | 5.40% |
| Decoder OFF | HYBRID | Decode | 10/10 | -12.78% | 12.95% | 12.78% | 12.77% |
| Decoder OFF | SILICON | Prefill | 28/28 | +5.34% | 5.57% | 5.35% | 5.40% |
| Decoder OFF | SILICON | Decode | 10/10 | -12.78% | 12.95% | 12.78% | 12.77% |
| Decoder ON | SOL | Prefill | 28/28 | -95.06% | 94.93% | 95.06% | 95.06% |
| Decoder ON | SOL | Decode | 10/10 | -99.62% | 99.62% | 99.62% | 99.62% |
| Decoder ON | HYBRID | Prefill | 28/28 | -10.89% | 8.18% | 10.89% | 10.94% |
| Decoder ON | HYBRID | Decode | 10/10 | -7.10% | 7.11% | 7.10% | 7.10% |
| Decoder ON | SILICON | Prefill | 18/28 | -4.91% | 3.82% | 4.91% | 4.98% |
| Decoder ON | SILICON | Decode | 10/10 | -7.10% | 7.11% | 7.10% | 7.10% |

Compare modes on the same supported subset as well: a smaller coverage set can otherwise conceal hard cases.

| Profile | Mode | Common strict subset | Mean signed error | Median APE | p90 APE | MAPE | WAPE |
|---|---|---:|---:|---:|---:|---:|---:|
| Decoder OFF | SOL | 38/38 | -96.19% | 94.99% | 99.65% | 96.19% | 96.04% |
| Decoder OFF | HYBRID | 38/38 | +0.57% | 7.09% | 13.17% | 7.31% | 7.09% |
| Decoder OFF | SILICON | 38/38 | +0.57% | 7.09% | 13.17% | 7.31% | 7.09% |
| Decoder ON | SOL | 28/38 | -96.82% | 95.76% | 99.66% | 96.82% | 96.67% |
| Decoder ON | HYBRID | 28/38 | -5.69% | 6.80% | 8.61% | 5.69% | 5.66% |
| Decoder ON | SILICON | 28/38 | -5.69% | 6.80% | 8.61% | 5.69% | 5.66% |

## Repeat stability

| Profile | Median within-point CV | Largest within-point CV | Case with largest CV | Repetitions (ms) |
|---|---:|---:|---|---|
| Decoder OFF | 0.40% | 3.57% | prefill-0005 | 198.277, 202.289, 202.790, 198.454, 210.331, 195.472, 193.314, 190.188, 189.286, 188.062 |
| Decoder ON | 0.41% | 7.86% | prefill-0004 | 193.943, 193.860, 194.363, 193.176, 193.669, 193.188, 192.527, 192.530, 192.920, 242.580 |

This is a separate precision attempt for all 38 holdouts in both profiles. The original three-repeat attempt is preserved [alongside this report](../README.md). The ON B2 / 96-new-token / zero-prefix observation changed from 379.867 ms to 187.947 ms (within-point CV 0.353%). Both slow original forwards remain in the original report. No sample was deleted, pooled across attempts, or used to refit the model. The largest new ON CV is 7.857% and is also retained. Stable repeats within this attempt do not establish stability across runtime lifecycles.

## Coverage and limitations

The frozen calibration contains 126 configurations per profile (100 prefill, 26 decode); the separate holdout contains 38 (28 prefill, 10 decode). Calibration has one warmup and three measured repetitions per configuration. This separate precision holdout has one warmup and ten measured repetitions. Per profile that is 378 calibration and 380 precision holdout measured invocations, each requiring four rank records. TP ranks and component keys are not independent workloads. Both final calibration/holdout attempts passed their raw evidence admission checks. See the [study evidence and attempt notes](../../study/README.md) for collection details.

Decoder OFF has 836 calibrated V4.1 physical module keys; Decoder ON has 830. Each has 80 GEMM/MoE/NCCL baseline keys. These calibration rows are never counted as independent accuracy results. The forward-only holdout ran without the component recorder and contributed no component rows to calibration.

The ON holdout exposes exact-prefix coverage gaps after the bounded layers shift a long extension to its last 128 tokens. The current lookup requires the shifted prefix bucket to exist. Missing cases are listed below, with their full native failure retained in the result JSON. Filling these from holdout module timings would contaminate the test; this report performs no such refill.

| Missing strict ON case | Batch | New tokens / request | Initial prefix |
|---|---:|---:|---:|
| prefill-0002 | 1 | 192 | 0 |
| prefill-0006 | 1 | 192 | 128 |
| prefill-0010 | 1 | 192 | 512 |
| prefill-0011 | 1 | 384 | 512 |
| prefill-0014 | 1 | 192 | 1536 |
| prefill-0015 | 1 | 384 | 1536 |
| prefill-0018 | 2 | 192 | 0 |
| prefill-0021 | 2 | 192 | 128 |
| prefill-0024 | 2 | 192 | 512 |
| prefill-0027 | 2 | 192 | 1536 |

The observation boundary is the native SGLang synchronized wall time around preparation, forward and sampling. It includes shared Engram hash/history work omitted from the module graph. Native mHC retains its statistics stream; component measurement serializes that stream. Baseline expert routing is seeded uniform synthetic routing, while held-out forward uses real text. These are explicit possible sources of error, not measured causal attributions. Profiles were collected in separate runs, so their timing ratio is not a controlled paired estimate of Decoder optimization speedup.

This is a fixed geometry grid with ten repetitions per point. It does not provide a production-workload confidence interval. Independent E2E trials will supply separate trial-level uncertainty estimates. Other corpora, larger batches, long contexts, candidate saturation, CUDA graphs, other parallel layouts, vision and DSpark remain outside this result.

## Reproduction and identity

Use the [pinned runtime, kernel and input evidence](../../study/README.md). The measurement uses four GB300 GPUs, pure TP4/EP1/DP1/PP1, SGLang `0.0.0.dev0`, ARM64 image `sha256:800cc9adea5be1e18f48185451220c4bc487c545b7095c720d2ccc9ba9bb3b5d`, checkpoint `fb2764a5cf321eaa5070ca8f9e892818f477c16d`, eager execution, DSpark off, Engram in HBM, and explicit NCCL with custom and FlashInfer AR fusion disabled. Resident unused vision weights are a runtime loading limitation; no vision work is executed. Do not reinterpret this as a graph-enabled or fused-AR result.

1. In the SILICON checkout, rebuild the native extension and run `verify_study.py` with `PYTHONPATH=python/aisimulate/src:python/aisimulate`.
2. Use the analysis utility from the sibling FPM PR at [`afcad7e2`](https://github.com/ai-dynamo/aisimulate/tree/afcad7e2/data/experimental/deepseek-v41/verification-plan). That utility calls the shared native forward API; it does not require FPM calibration to produce these op-level predictions. Keep the SILICON checkout as the working directory and Python source path.
3. For each profile/mode, run `compare_forward.py --observations data/experimental/deepseek-v41/gb300-silicon/study/<profile>/precision-v2/forward-results.json --heldout-plan <FPM-checkout>/data/experimental/deepseek-v41/verification-plan/heldout.json --prediction-config data/experimental/deepseek-v41/gb300-silicon/report/precision-v2/<profile>-<mode>-config.json --output <new-results-file.json>`. The utility refuses to overwrite results.
4. Run `render_report.py` to regenerate this document and PNG/PDF figures from the six stored comparison files. [SHA-256 receipt](artifact-hashes.json) pins the input/configuration/results and rendering source. Each result also pins the actually imported model, engine and native binary, independent of installed package metadata. Internal cluster logs and identifiers are kept separately.

## Separate attempt comparison

The model and calibration tables are unchanged. All 228 profile/mode predictions and availability decisions exactly match the original report; these are 76 unique profile/configuration pairs, not 228 independent measurements. Observations changed between separately launched native forward attempts. The following comparison keeps the original results visible rather than selecting the better error.

| Profile | Mode | Original 3-repeat median APE | Precision 10-repeat median APE | Original MAPE | Precision MAPE | Original WAPE | Precision WAPE |
|---|---|---:|---:|---:|---:|---:|---:|
| Decoder OFF | SOL | 95.32% | 94.99% | 96.42% | 96.19% | 96.26% | 96.04% |
| Decoder OFF | HYBRID | 4.13% | 7.09% | 6.88% | 7.31% | 6.38% | 7.09% |
| Decoder OFF | SILICON | 4.13% | 7.09% | 6.88% | 7.31% | 6.38% | 7.09% |
| Decoder ON | SOL | 95.30% | 95.04% | 96.32% | 96.26% | 96.22% | 96.13% |
| Decoder ON | HYBRID | 5.61% | 7.39% | 10.87% | 9.89% | 12.41% | 10.04% |
| Decoder ON | SILICON | 4.73% | 6.80% | 7.31% | 5.69% | 9.29% | 5.66% |

The prediction receipt pins the resolved checkpoint configuration and inferred precision, the selected system specification and complete table overlay inventory, the native extension, and all comparison/axis/statistics sources. The native extension was rebuilt from source 8a4caf9c; its SHA is 3e4de42a013ff5f37cdc7b5df7f7fdd70b8d70840f8c3c8cc2dcb968109451b4. Its predictions were checked against the original binary on all configurations.
