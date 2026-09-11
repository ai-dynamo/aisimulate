# GB300 TP4: real serving data and whole-forward FPM predictions

The current packets extend the original calibration with separately collected
source-defined geometries. Each Decoder profile has **142 / 144 planned points**:
the original 124 / 125 and a new 18 / 19. The six independent verification scopes
reuse their original observations; no verification timing or error is fitted.

Every error below is **MAPE / WAPE (%)**, calculated on the same supported pairs.

| Decoder | Scope | Native forward | TTFT | Mean TPOT / mean ITL | Output throughput |
| --- | --- | ---: | ---: | ---: | ---: |
| OFF | Core | 1.2409 / 1.5244 | 13.4865 / 13.6283 | 2.6206 / 2.6674 | 3.1243 / 2.8097 |
| OFF | Field notes | 0.7507 / 0.9648 | 16.1361 / 16.0925 | 1.4787 / 1.5117 | 2.3120 / 2.3869 |
| OFF | Service records | 0.7217 / 0.9423 | 14.6925 / 14.4668 | 1.4870 / 1.5236 | 2.1926 / 2.2429 |
| ON | Core | 2.0114 / 2.2897 | 12.9572 / 13.0736 | 3.7935 / 3.8406 | 4.1050 / 3.6237 |
| ON | Field notes | 1.1930 / 1.5929 | 16.7871 / 16.9198 | 2.5895 / 2.6213 | 3.6209 / 3.3529 |
| ON | Service records | 1.0321 / 1.2546 | 16.5980 / 16.8026 | 2.1891 / 2.2047 | 3.4813 / 3.3344 |

[OFF: real/predicted means, seven metrics, CSV and figures](off-union-v3-tracewait2/report/README.md).
[ON: real/predicted means, seven metrics, CSV and figures](on-union-v3-tracewait2/report/README.md).
The additional metrics are mean exact ITL, request completion latency and last-token
latency. Mean ITL is not tail ITL.

## Coverage and sampling

| Decoder | Scope | Trials | HTTP cohorts supported / observed | Native intervals supported / observed |
| --- | --- | ---: | ---: | ---: |
| OFF | Core | 40 | 600 / 600 | 15,756 / 15,756 |
| OFF | Field notes | 30 | 120 / 120 | 3,503 / 3,503 |
| OFF | Service records | 30 | 120 / 120 | 3,509 / 3,509 |
| ON | Core | 100 | 1,400 / 1,500 | 39,293 / 39,393 |
| ON | Field notes | 30 | 120 / 120 | 3,503 / 3,503 |
| ON | Service records | 30 | 120 / 120 | 3,504 / 3,504 |

The original warmup and ten attempts per each of 125 geometries remain intact.
Each extension separately completed ten control suites, a full 19-point warmup,
and ten fixed main attempts per point. Each contributes seven prefill and eleven
decode cells; the union contains 106 prefill and 36 decode cells per profile.
Selection is the first geometry-qualified planned attempt, without fastest-sample
selection or output-equality filtering. Control/warmup timings are excluded.

Two calibration geometries remain missing per profile. Both retain the original
B2/Q512/past-KV3072 point. The new OFF gap is B3/Q1281/past-KV0; the new ON gap is
B3/Q1278/past-KV0. All ten formal attempts of each missing point split across
native dispatches. Those outcomes do not establish a hardware capacity limit.
Missing-point counts and source bindings remain in the public receipts; complete
attempt evidence remains in the private archives.

OFF predictions now cover all observed HTTP cohorts and native intervals.
ON core still lacks predictions for 100 heterogeneous HTTP cohorts (300 requests)
and 100 native prefill intervals. Decoder replay requires the per-request new-token
and cached-prefix lengths; the current mean/aggregate input cannot represent these
heterogeneous batches. All 100 native failures are this representation guard,
not missing-table lookups. Their observed dispatches are B2/Q895 (22), B2/Q511 (37),
B2/Q1152 (30), and B3/Q1279 (11), all with zero prefix and nonzero length variance.
This is separate from the missing balanced B3/Q1278 calibration point; collecting
that point alone would not make these requests representable. The same HTTP
cohorts first hit an earlier calibration-coverage gate in v1, so their original
and current failure reasons remain separately preserved. Missing predictions
remain in the original coverage denominators. All field/service HTTP cohorts
and native intervals are supported.
Calibration-coordinate coverage, observed coverage and prediction coverage are
distinct. Balanced aggregate interpolation does not establish equivalence of
heterogeneous per-request shapes.

The original core ON precision target remains unmet. Its two physical lifecycles
retain 73 and 26 complete observed trials, with the split trial descriptive and
no pooled lifecycle confidence interval. Applicable scenario intervals remain
conditional on the frozen calibration and prediction coverage. Field OFF/ON and
service ON each retain one unmet precision target out of twelve; service OFF
meets twelve. These limitations and output/prefix disagreements are preserved.

## Interpretation and provenance

For positive observations y and predictions p, MAPE is
`100 * mean(abs(p - y) / y)` and WAPE is
`100 * sum(abs(p - y)) / sum(y)`. Missing predictions receive no fabricated error.
The scopes and Decoder profiles are separate statistical populations.
Native metrics pair individual recorded DeviceTimer intervals, which are
correlated; HTTP metrics pair client cohorts with scheduler replay. HTTP includes
service costs outside DeviceTimer. The retained older region-weighting label
does not imply a second median reduction. Op-level and FPM aggregates can use
different supported subsets and do not by themselves establish a method ranking.

Each union preserves both physical collection identities, every source row and
both immutable component tables. Calibration and independent verification runs,
observations, cohorts and plans are disjoint. Exact native consumer queries and
opposite-profile rejection are integrity checks; self-query MAPE/WAPE are never
included as accuracy. New collection/union analysis records predictor 40c9541b;
the historical v1 producer and observation-analysis identities stay distinct.

Runtime scope is one GB300 node/four GPUs, pure TP4/EP1, text, HBM Engram, eager
execution and DSpark OFF. The checkpoint revision is
`fb2764a5cf321eaa5070ca8f9e892818f477c16d`; ARM image digest is
`sha256:800cc9adea5be1e18f48185451220c4bc487c545b7095c720d2ccc9ba9bb3b5d`.
Actual installed SGLang/Dynamo/kernel sources, precision and execution arguments
are bound in component identities. The SDK FP8 selector does not imply all
physical V4.1 storage or arithmetic is FP8. No task-quality, causal Engram-locality
or runtime allocator-accuracy claim follows from these measurements.

The original [v1 report and coverage](README-v1.md), [OFF packet](off-v1/README.md)
and [ON packet](on-v1/README.md) are preserved. Their different prediction coverage
means a before/after aggregate-error change is not a controlled accuracy comparison.

## Reproduction

Use a profile prediction-config.json from the repository root; systems_path is
repository-relative. Public packets retain all numeric comparison data, coverage,
failures, source hashes and allowed portability transformations. Internal cluster
identifiers and logs remain private.

```bash
python data/experimental/deepseek-v41/gb300-fpm/off-union-v3-tracewait2/render_report.py \
  --input-root data/experimental/deepseek-v41/gb300-fpm/off-union-v3-tracewait2 \
  --output /tmp/dsv41-gb300-union-off-reproduced
```

Use on-union-v3-tracewait2 and a fresh output for ON. The standalone renderer
checks input hashes and recomputes MAPE/WAPE from frozen public pairs. Detailed
summaries use lossless gzip; decompression reproduces the exact original JSON.
Packaging preserves all calibration and comparison bytes, and its CSV, PNG, PDF
and report README are byte-identical to the uncompressed report render.
