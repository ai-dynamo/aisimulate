# GB200 TP4 historical whole-forward FPM observations

**Admission status: historical, environment-specific observations; not admitted
for prediction or reusable serving calibration.** The 126-point table remains
here for forensic reproduction of the original experiment. Its presence in the
experimental `systems` layout and successful loading do not qualify it for a new
serving environment. Re-admission requires a paired, same-input/runtime study
that explains the discrepancy described below. No scaling factor, replacement
measurement or removal of unsuccessful observations has been applied.

All **126 original calibration points** completed: 100 prefill and 26 real
decode, with no skipped or missing points. The separate 38-point holdout and
ordinary HTTP/Dynamo verification have also completed. The
[2026-09-14 prediction refresh](../../prediction-refresh-20260914/fpm/README.md)
reports **4.8942% MAPE** on the 38 independent holdout geometries, but
**432.3291% native-forward MAPE** on 12,531 ordinary-serving intervals and
**838.9658% TTFT MAPE** on 450 HTTP cohorts. These results are bound to that
report's predictor revision; the small holdout error does not establish
transfer to ordinary serving.

The separate DL GB200 diagnostic likewise reproduced the faster serving
regime: predictions overestimate its native intervals by **431.4790% MAPE**
(30 points) and its uniquely matched ordinary intervals by **430.8405% MAPE**
(20 of 30 planned points). The historical calibration durations are about
5.1–5.6 times the new observed durations at those coordinates. Matching many
runtime settings did not reproduce the historical regime, and the original
prompt was not fixed for a paired comparison. The evidence does not isolate a
cluster fault or timing bug. See the refresh for unmatched observations,
warmup exclusions, correlated repetitions and source bindings.

The exact 126 rows load through both the native static engine and the
whole-forward timing interface without changing their measured values.
`consumer-verification.json` records the latter check and rejection of an
incompatible BF16 attention identity. Matching the input table is a consumer
integration check; it is not a zero-error holdout result.

## Runtime and timing boundary

The runtime uses one node with four GB200 GPUs, pure TP4/EP1/DP1/PP1,
`vllm 0.1.dev20904+g179dd0fa9`, native NCCL 2.30.7, text autoregressive
inference, Engram in HBM,
DSpark disabled and eager execution. Decoder replay is OFF. ON remains an
unverified runtime dependency and cannot consume these observations.

The base ARM64 image is pinned to
`sha256:d84a123255b822fc22508635218000187221794f59c0694c33b0650d1e377d58`.
The prepared runtime image has its own immutable hash in
[the collection receipt](collection-receipt.json). Dynamo Python and native
runtime packages are 1.4.2, with four unchanged instrumentation modules from
`54960177085413259859c88bd34ed0734d4c2ea9`. This is a composite runtime;
`dynamo_revision=null` deliberately avoids describing the full worker as that
instrumentation revision. The checkpoint/tokenizer revision is
`fb2764a5cf321eaa5070ca8f9e892818f477c16d`.

Each row contains **one actual native FPM timing sample**. vLLM FPM reports
the existing CPU schedule/output or adjacent-output interval; these values
are not interchangeable with SGLang GPU-event durations. Real token history
and initialized KV were validated for original artifact acceptance. Past KV excludes the
current decode query. The exact frozen input geometry is retained in
[point-manifest.json](point-manifest.json).

There are also 16 prefill and two decode native warmup records, all retained
in private source evidence and excluded from the 126 rows. The table's
`warmup_repeats=0` and `global_warmup_iterations=0` are the native Collector
policy fields, not a claim that runtime initialization executed no warmup.
No repeat variability or confidence interval can be inferred from one
measurement per calibration point.

## Coverage and identity

| Property | Original observed range or value |
|---|---|
| Homogeneous batch | 1–2 |
| Total new prefill tokens | At most 512 |
| Per-request prefix plus new tokens | At most 2048 |
| Per-request decode past KV | At most 2048; inclusive query length 2049 |
| Context capacity used by native collection | 2050 |
| Measurement profile | Full Decoder execution, eager, real text/KV |
| Main GEMM / expert precision | FP8 block / MXFP4 weights with MXFP8 activations |
| Attention / KV-cache identity | FP8 / FP8 |
| Table schema | 7, exact config/profile/residency/text identity |

The 127/128/129 boundaries, cached prefill and real decode are included.
The collection receipt binds each phase's native artifact, input corpus,
producer sources, execution identity and timeout policy. The original
plan/attempt-bound admission was independently repeated on the host and is
recorded in [admission-receipt.json](admission-receipt.json). That historical
receipt establishes artifact integrity and the original collection contract;
it does not override the current prediction-admission status above. Internal storage,
allocation details, launch logs and complete token-stream witnesses are
retained separately, bound by hashes.

Do not mix this table with a different backend/version, execution profile,
input modality, Engram placement or resolved precision. The explicit FP8
attention selection in `prediction-config.json` matches the native Collector
identity. Ordinary serving verification additionally requires a receipt from
the actually initialized worker; calibration acceptance does not establish
HTTP prefix reuse, allocator accuracy, Engram locality or serving parity.

## Historical reproduction

Run from the repository root, with this revision's built native extension:

```sh
PYTHONPATH=python/aisimulate/src:python/aisimulate \
python data/experimental/deepseek-v41/gb200-fpm/calibration-v1/verify_calibration.py \
  --output /path/to/new-consumer-verification.json
```

The verification refuses to overwrite its output. It checks all 126 exact
cells through the real native whole-forward API, preserves the table bytes,
and proves an incompatible attention identity cannot reuse the data.
`artifact-hashes.json` records the published files; the Parquet checksum also
matches its original producer sidecar.

FPM's canonical loader layout is
`systems/data/gb200/vllm/<actual-version>/fpm_forward_perf.parquet`.
The shared Python operation-data discovery currently emits a legacy-layout
warning for this FPM-only directory. The actual FPM loader requires this
layout and all 126 native queries pass; moving it to an operation-family
directory would violate that loader contract.

Completed verification retains independent measured forward, TTFT, mean
ITL/TPOT, response-completion latency and throughput. Calibration timings were
not reused as holdout observations. The [refresh report's numerical
export](../../prediction-refresh-20260914/fpm/results/summary.csv) includes
coverage and error metrics; these historical predictions are evidence of the
unresolved calibration-transfer problem, not permission to deploy the table.
