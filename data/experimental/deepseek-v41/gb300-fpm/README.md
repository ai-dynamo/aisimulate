# GB300 TP4: measured serving data and whole-forward FPM predictions

These packets compare independently collected SGLang DeviceTimer and HTTP
observations against whole-forward FPM predictions. Decoder replay OFF and ON
have separate calibration tables and runtime identities. Each error below is
**MAPE / WAPE (%)** over the same supported pairs.

| Decoder | Verification scope | Native forward | TTFT | Mean TPOT | Output throughput |
| --- | --- | ---: | ---: | ---: | ---: |
| OFF | Core | 1.2556 / 1.5594 | 13.1621 / 13.2815 | 2.6574 / 2.7261 | 3.1702 / 2.9208 |
| OFF | Field notes | 0.7513 / 0.9682 | 16.1361 / 16.0925 | 1.4787 / 1.5117 | 2.3120 / 2.3869 |
| OFF | Service records | 0.7200 / 0.9424 | 14.6925 / 14.4668 | 1.4870 / 1.5236 | 2.1926 / 2.2429 |
| ON | Core | 2.0535 / 2.3311 | 13.1382 / 13.2701 | 3.8090 / 3.8503 | 4.0948 / 3.6066 |
| ON | Field notes | 1.1918 / 1.5964 | 16.7871 / 16.9198 | 2.5895 / 2.6213 | 3.6209 / 3.3529 |
| ON | Service records | 1.0313 / 1.2563 | 16.5980 / 16.8026 | 2.1891 / 2.2047 | 3.4813 / 3.3344 |

[OFF: real and predicted means, all seven metrics, CSV and figures](off-v1/report/README.md).
[ON: real and predicted means, all seven metrics, CSV and figures](on-v1/report/README.md).
The full reports also include mean exact ITL, request completion latency and
last-token latency. Mean TPOT and mean ITL do not measure tail ITL.

## Coverage and qualification

| Decoder | Scope | Independent trials | HTTP cohorts supported / observed | Requests supported / observed | Native intervals supported / observed |
| --- | --- | ---: | ---: | ---: | ---: |
| OFF | Core | 40 | 440 / 600 | 600 / 880 | 14,220 / 15,756 |
| OFF | Field notes | 30 | 120 / 120 | 180 / 180 | 3,466 / 3,503 |
| OFF | Service records | 30 | 120 / 120 | 180 / 180 | 3,478 / 3,509 |
| ON | Core | 100 logical | 1,100 / 1,500 | 1,500 / 2,200 | 35,544 / 39,393 |
| ON | Field notes | 30 | 120 / 120 | 180 / 180 | 3,466 / 3,503 |
| ON | Service records | 30 | 120 / 120 | 180 / 180 | 3,468 / 3,504 |

Each profile planned 125 geometries, one complete warmup grid and ten fixed main
attempts per geometry. Each published table retains **124 points: 99 prefill and
25 decode**, selecting the first geometry-qualified planned attempt. The missing
B2 prefill point has 512 total query tokens and 3,072 total past-KV tokens; all ten
attempts split into separate native dispatches. No fastest-sample selection or
verification-based correction fit is used. The original attempts and missing
point remain recorded in each collection receipt.

The v1 calibration domain covers batches 1–2 and at most 512 total prefill tokens.
Batch-3 decode and larger prefill queries are unsupported, explaining the
coverage gaps above. Native telemetry and simulated HTTP schedules can request
different shapes; complete HTTP coverage does not imply complete native coverage.
Every unavailable prediction and its original reason remains in the compressed
comparison outputs. A separate extension is being prepared from source request
geometry, without selecting points by prediction error; it is not part of v1.

Both calibration tables pass 124 exact native consumer queries and reject the
opposite Decoder profile. Their self-query MAPE and WAPE are zero **only as
calibration integrity checks**; these values are excluded from every accuracy
table. Calibration and verification have disjoint run, observation, cohort and
plan identities. The real verification observations remain unchanged from the
separate SILICON study.

## How to interpret the errors

For supported positive observations `y` and predictions `p`, MAPE is
`100 * mean(abs(p - y) / y)` and WAPE is
`100 * sum(abs(p - y)) / sum(y)`. Missing predictions remain in coverage but do
not acquire an error value. The scopes and Decoder profiles are reported
separately; no pooled accuracy or confidence interval is claimed.
Op-level and FPM reports can have different supported subsets, so comparing their
aggregate errors alone does not establish a ranking between methods.

Native forward errors pair individual recorded DeviceTimer intervals, using the
native inclusive-to-past decode-axis conversion. HTTP errors pair client
observations with scheduler replay using the predicted FPM timings; HTTP timing
includes costs outside the DeviceTimer interval. Equal aggregate coordinates do
not imply identical request contents or schedules, and a point inside raw axis
bounds does not prove which interpolation branch ran.

The frozen `query_domain_coverage.region_metrics.weighting` string comes from a
generic older helper and mentions a median of rank maxima and logical
configurations. Here its numeric values use **individual recorded DeviceTimer
interval pairs**, with no additional median reduction. The primary metric rows
and this report state the applicable weighting explicitly; the original JSON
evidence is preserved.

Native intervals are correlated. Scenario bootstrap intervals resample whole
trials, conditional on the frozen calibration and observed lifecycle. Core ON
retains two lifecycles with 73 and 26 complete observed trials; conditional
scenario intervals require complete applicable prediction coverage, and missing
predictions suppress those intervals. The split boundary trial 73 is descriptive.
There is no pooled cross-lifecycle CI. The original ON precision target was not met. The supplemental precision
limitations also remain unchanged: field OFF and ON each miss one of twelve
targets, service OFF meets twelve, and service ON misses one. These error reports
do not replace the original precision assessment.

Output and prefix-cache disagreements remain included. Content controls do not
establish actual Engram addresses/cache hits or causal locality effects; clearing
KV does not flush Engram, HBM or L2. Neither these results nor the analytical
memory inventory validate runtime allocator usage.

## Runtime identity and reproduction

The study uses one GB300 node with four GPUs, pure TP4, text only, DSpark disabled,
SGLang `0.0.0.dev0`, and the source-qualified Dynamo DeviceTimer implementation.
The ARM image is pinned to
`sha256:800cc9adea5be1e18f48185451220c4bc487c545b7095c720d2ccc9ba9bb3b5d`.
The checkpoint revision is `fb2764a5cf321eaa5070ca8f9e892818f477c16d`.
Actual installed sources, precision flags and immutable controls are recorded in
each collection receipt. This is a composite image/source qualification; it is
not a claim that the entire installed runtime equals an upstream SGLang commit.

`fpm_fmha_dtype=fp8` selects the SDK table identity. It does not assert that all
physical V4.1 KV or attention storage is FP8. The actual MoE path is
`w4a8_mxfp4_mxfp8_trtllm`, with checkpoint-native mixed KV/index storage.

The OFF table was produced with predictor source `be0c44df`; the ON table with
`831836ff`. Their native binary is identical, and the intervening commit changes
only the report helper and its tests. Both independent analyses use `831836ff`;
the original observation analysis remains recorded separately as `be0c44df`.
The source-bound Parquet, metadata and hardware YAML are preserved byte for byte.

Use the profile's `prediction-config.json` from the AISimulate repository root.
Its `systems_path` points to that profile's repository-relative table directory.
To reproduce the figures and CSV from the frozen public comparison pairs, without
predictions or private cluster files:

```bash
python data/experimental/deepseek-v41/gb300-fpm/off-v1/render_report.py \
  --input-root data/experimental/deepseek-v41/gb300-fpm/off-v1 \
  --output /tmp/dsv41-gb300-fpm-off-reproduced
```

Use `on-v1` and a new output directory for ON. The renderer checks input hashes
and recomputes MAPE/WAPE before rendering. Collection receipts, point manifests,
comparison pairs, coverage, failure records, source hashes and allowed path
transformations are included in each packet. Internal scheduler and host details
remain in the separate private archives.

Detailed summaries are stored as `reports/*/summary.json.gz`. Their packaging
receipts bind the original JSON hashes, lossless gzip hashes and renderer update;
decompression recovers the exact original JSON bytes. The original export
derivation receipts and source inventories remain as history. Packaging does not
change measurements, predictions, coverage, CSV or figures.
