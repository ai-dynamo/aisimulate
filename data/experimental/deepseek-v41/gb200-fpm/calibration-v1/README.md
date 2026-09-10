# GB200 TP4 whole-forward FPM calibration

All **126 calibration points** completed: 100 prefill and 26 real decode,
with no skipped or missing points. These are calibration observations, not
independent accuracy results. The separately frozen 38-point holdout and
ordinary HTTP/Dynamo FPM serving verification are still pending.

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
and initialized KV are validated before admission. Past KV excludes the
current decode query. The exact frozen input geometry is retained in
[point-manifest.json](point-manifest.json).

There are also 16 prefill and two decode native warmup records, all retained
in private source evidence and excluded from the 126 rows. The table's
`warmup_repeats=0` and `global_warmup_iterations=0` are the native Collector
policy fields, not a claim that runtime initialization executed no warmup.
No repeat variability or confidence interval can be inferred from one
measurement per calibration point.

## Coverage and identity

| Property | Qualified range or value |
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
recorded in [admission-receipt.json](admission-receipt.json). Internal storage,
allocation details, launch logs and complete token-stream witnesses are
retained separately, bound by hashes.

Do not mix this table with a different backend/version, execution profile,
input modality, Engram placement or resolved precision. The explicit FP8
attention selection in `prediction-config.json` matches the native Collector
identity. Ordinary serving verification additionally requires a receipt from
the actually initialized worker; calibration acceptance does not establish
HTTP prefix reuse, allocator accuracy, Engram locality or serving parity.

## Reproduction

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

The remaining verification reports will compare independent measured and
predicted forward, TTFT, mean ITL/TPOT, response-completion latency and
throughput. Calibration timings will not be reused as those holdouts.
