# Conditional B300 GLM B52 warm-history preparation fit

The 44 measured B52 points support a narrow estimate of warm persisted-history preparation on the measured B300/SGLang/NVFP4 path. Both variants meet the 20% point-error gate after the first chronological point is accounted for separately. There is no independent 15% group validation and no full collection GPU-hour claim.

| Variant | Warm fit / held out | Intercept | Slope per history token/request | Maximum held-out error |
|---|---:|---:|---:|---:|
| indexer | 17 / 4 | 0.037286336 s | 0.000113351257 s/token | 9.541% |
| skip_indexer | 17 / 4 | 0.039497174 s | 0.000113414274 s/token | 8.451% |

`preparation_seconds = intercept + slope × past_history_tokens_per_request`.

The domain is real batch 52, 8 local query heads, FP8 KV, BF16 compute/projections, prefix 0 and history 4–720,896, on `nvidia/GLM-5.2-NVFP4`. The native graph has 56 slots with 52 real requests. Full and skipped indexers are fitted independently. Coefficients already incorporate fixed batch 52; multiplying by batch again would double count.

The input is job **1024142**, signed collector `c4b44d81f526b3c5299f7c15ccafe02d90b39229`. The JSON embeds all **44 exact sample IDs**, cases, observed phase values, source-result/log SHA256 values, deterministic fit/held-out IDs, coefficients and observed-versus-estimated errors. The frozen 46-record normalization snapshot is hash-identified; the two extra records are inclusive worker groups. Raw artifacts and original/published hashes remain in [the production recollection bundle](https://github.com/ai-dynamo/aisimulate/tree/bfdb5f727ad601c9dde43933f6e18f2d97aad1d4/benchmarks/evidence/collector/pr665-b300-dsa-boundary/recollection).

Exactly the first chronological point in each worker (index 0, history 1) is excluded before fitting as a separate first-use stratum. Its full observed preparation/execution/inclusive cost remains in the JSON. This rule does not discard a point based on its residual. It assumes first-use work will be counted separately; one worker per mode cannot independently validate its repeatability. Preassigned held-out indices **4, 9, 14, 19** remain unchanged, leaving **17 fitting and 4 held-out points per variant**.

Preparation is the logged inclusive point total minus `bench + prewarm`. Nested `bench_*` fields are not added. Group residual setup and forward execution are outside this fitted component. Phase logs round to 0.01 seconds, so the errors describe the logged observations rather than sub-rounding accuracy or confidence bounds.

For the optional history ≥8,192 subset, each variant has 8 fitting and 2 held-out points. Maximum preparation errors are 0.940% (indexer) / 0.280% (skip_indexer). This subset remains the same exact model and geometry.

Do not transfer these rates to a BF16 checkpoint, another batch/head geometry, another framework/system, or an unmeasured history range. Checkpoint initialization, cold compilation, graph behavior, execution, failures and actual memory admission require their own evidence. A whole-worker or fleet GPU-hour total remains unqualified.
