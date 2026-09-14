# Strict SILICON prediction refresh — 2026-09-14

These are actual CPU predictions from merged predictor commit
`f21ed55168074f8d32153a04d3cdfc524484722b`, with a freshly built native extension
SHA-256 `23bb3240683c1e2a18ce188200267c1fcb26294752cca50c022eca24e421a6d0`.
The mode is **SILICON with `forward_model=op_level`**, fixed profile-specific
GB300 operation tables, TP4 and the checkpoint's original mixed precision.
There is no new GPU measurement, calibration change or correction fitting.

| GB300 comparison | Predicted / observed | MAPE | WAPE | Fully predicted cohorts |
|---|---:|---:|---:|---:|
| OFF 46-point holdout | 46 / 46 | 7.3679% | 7.2321% | — |
| ON 46-point holdout | 30 / 46 | 6.8538% | 6.8723% | — |
| OFF primary native intervals | 14,476 / 15,756 | 13.0147% | 13.5565% | 560 / 600 |
| ON primary native intervals | 35,797 / 39,393 | 4.8107% | 5.1083% | 1,104 / 1,500 |
| OFF field native intervals | 3,503 / 3,503 | 11.5855% | 11.9119% | 120 / 120 |
| ON field native intervals | 3,466 / 3,503 | 3.7355% | 4.1381% | 83 / 120 |
| OFF service native intervals | 3,509 / 3,509 | 11.4120% | 11.7321% | 120 / 120 |
| ON service native intervals | 3,468 / 3,504 | 3.2848% | 3.5117% | 84 / 120 |

MAPE is the equal-weight mean of `100 × abs(predicted_ms / observed_ms − 1)`
over supported rows. WAPE is `100 × sum(abs(predicted_ms − observed_ms)) /
sum(observed_ms)` over the same rows. Missing predictions remain in every
observed denominator and have no fabricated zero error. Holdout observations
are the frozen median of repeated rank-maximum measurements; serving rows are
individual, correlated native forward intervals. These are different units.

**The lower ON MAPE accompanies reduced coverage.** The current API rejects
aggregate multi-request prefill inputs because they lack each request's extend
length, including aggregates with zero variance. This preserves 16 holdout
misses and adds 296 primary, 37 field and 36 service interval misses relative to
the historical reports. It is not evidence of improved prediction accuracy.
The primary study retains 40 OFF / 100 ON trials per scenario; each field and
service scope retains 30. ON's two physical segments remain distinct via
`segment_ordinal`. This refresh supplies descriptive statistics, without new
confidence intervals or a new claim about the original precision target.

The holdout comparisons consume original qualified observation files. Native
serving comparisons re-query complete observed geometry and timing retained in
the immutable original reports. Their original raw lifecycle artifacts could
not be reopened in this session; this is **not a fresh raw-source admission**.
Historical qualification, precision and failure records remain unchanged.

For core HTTP replay, 560 OFF and 1,300 ON cold workloads were recovered with
**exact equality to each original SILICON ReplaySpec SHA-256**. Original public
plans provide token arrays. Retained request metadata provides submit offsets;
40 OFF rows borrow those same-observation offsets from the FPM report and still
match the original SILICON hash. No predicted duration becomes an input.
These exact workloads were actually replayed through the current native engine:

| Core HTTP metric | OFF MAPE / WAPE | ON MAPE / WAPE |
|---|---:|---:|
| TTFT | 25.5211% / 25.3890% | 37.2799% / 37.2435% |
| Average TPOT | 12.9186% / 12.9586% | 6.2212% / 6.2757% |
| Output tokens/s | 16.9935% / 15.8194% | 10.8505% / 9.8814% |

OFF supports **520 / 600** original cohorts: 560 exact inputs were attempted,
40 returned model errors, and 40 prefix-reuse inputs lack the seed-A submit
offset. ON supports **1,300 / 1,500** original cohorts: 100 prefix-reuse inputs
lack seed timing and 100 lack request metadata. All are retained. Field/service
HTTP remains **0 / 120** in each scope because original token-bearing plans are
unavailable; the historical observed metrics remain present. These are input
availability limits, not zero-error predictions or new runtime failures.
Current cache-reuse disagreement remains included in the metrics for 40 of the
520 supported OFF cohorts and 102 of the 1,300 supported ON cohorts.

Current SGLang replay emits its first output token at final prefill completion.
The historical extra first-output decode charge is not retained. Hence current
TTFT results use different replay behavior and a smaller supported subset than
the historical overall tables. Output tokens/s measures finite-cohort
throughput, not saturation or serving capacity. See
[e2e-summary.csv](e2e-summary.csv), [e2e-rows.json.gz](e2e-rows.json.gz) and
[e2e-source-bindings.json](e2e-source-bindings.json) for exact values and input
closure. Native-forward and HTTP errors are separate metrics.

For GB200, all 38 retained holdout geometries were actually queried in strict
op-level SILICON mode on the original experimental overlay. All 38 returned
`PerfDataNotAvailableError`: `OneCCL data not configured for this system
(no misc.oneccl_version in YAML)`. Therefore strict op-level SILICON MAPE is
**N/A (0 / 38 supported)** on that overlay. No different hardware/version table
or shared-layer fallback was substituted. The separate whole-forward FPM mode
also uses the database-mode name SILICON; it is a different prediction model.
The DL benchmark-versus-serving measured timing APE is likewise not model MAPE.

[summary.csv](summary.csv) and [summary.json](summary.json) contain exact values.
[rows.json.gz](rows.json.gz) preserves all 92 holdout rows, 69,168 native rows and
38 coverage probes, including errors. `scope`, `profile` and zero-based
`input_ordinal` identify each original ordered input; native rows additionally
retain segment/cohort/interval ordinals and observed geometry. Private run,
request, worker, node and dispatch identifiers are omitted.
[original-input-bindings.json](original-input-bindings.json) binds original
public reports, unchanged prediction configurations and complete table
inventories. [source-bindings.json](source-bindings.json) binds the actual fresh
private comparison reports and exporter. No old report is overwritten.

The current source passed 147 targeted Python model/database/Collector contract
tests, 35 Rust DSv41 tests and 40 existing report/statistics tests. Packaged legal
files and generated ownership files match their source definitions.

Actual prediction reproduction is available directly from this public bundle.
`predict_native_rows.py` queries the native predictor for every published
geometry, including all missing rows and all 38 GB200 probes, then compares the
new results with the recorded outputs. Recorded predictions are never model
inputs. It checks the exact operation-table inventories and permits report-only
descendants of the recorded predictor source. A native binary rebuilt on another
platform may have a different file hash; its actual identity is recorded.

The cold recovery and replay programs are portable adaptations of the actual
executed sources, with CLI paths replacing session-specific constants. The
input reconstruction, original-spec hash check and prediction calls are
unchanged; [prediction-runner-provenance.json](prediction-runner-provenance.json)
binds the executed source and validation. No captured cluster paths or IDs are
embedded in these programs. Recovered workload files contain request/token
inputs and should be written to a private output directory. Relocating their
source files changes the outer packet hash, while each original ReplaySpec hash
remains unchanged and is checked before replay.

Use the merged SILICON checkout for `W`, and an FPM checkout retaining the
original experimental reports/GB200 overlay for `F`. `O` must name a new private
output directory; these programs do not overwrite old output files:

```bash
W=/path/to/merged-silicon-checkout
F=/path/to/fpm-checkout
O=/path/to/new-private-reproduction
P="$W/data/experimental/deepseek-v41/prediction-refresh-20260914/silicon"
cd "$W"
CARGO_BUILD_JOBS=6 uv sync --project python/aisimulate --extra dev --reinstall-package aisimulate
export PYTHONPATH="$W/python/aisimulate/src:$W/python/aisimulate"
PY="$W/python/aisimulate/.venv/bin/python"
"$PY" "$P/predict_native_rows.py" --repo "$W" --fpm-repo "$F" --output "$O/native"
"$PY" "$P/extract_analysis_tools.py" --repo "$W" --output "$O/analysis-tools"
"$PY" "$P/recover_cold_workloads.py" --repo "$W" --fpm-repo "$F" --output "$O/cold-inputs"
"$PY" "$P/replay_cold_workloads.py" --repo "$W" --workloads "$O/cold-inputs" --analysis-tools "$O/analysis-tools" --output "$O/cold-replay"
```

The public native runner actually reproduced **69,298 / 69,298** statuses,
predictions and typed errors. The portable recovery reproduces all 1,860 exact
specs and preserves all 240 missing-input cohorts; the portable cold runner
reproduces the current HTTP results on those same inputs. These are CPU
prediction checks, with no GPU execution, new raw admission or fitted factors.

Separately, reproduce the public projection from the preserved fresh comparison
artifact root (the output directory must not already contain these filenames):

```bash
python export.py --input-root /path/to/fresh-comparison-artifacts --output /path/to/new-public-projection
python export_e2e.py --input-root /path/to/fresh-comparison-artifacts --output /path/to/new-public-projection
```

Actual predictions use `RustForwardPassPerfModel.from_native` from the pinned
current build. For native rows the supplied metrics are version 1 with the
exact retained `scheduled_requests`; no request-level values are inferred to
bypass a guard. Original comparison/statistics helpers are AISimulate sources
at `be0c44dfd1e319139d16659f2d96bf9a88337c53` under
`data/experimental/deepseek-v41/verification-plan`. For both 46-point holdouts,
the original `compare_forward.py` is invoked with the corresponding fixed
`gb300-silicon/report/sol-review-v2/forward/*-silicon-config.json`, original
`prefix-refinement-v1/{full,decoder_bounded}/heldout/forward-results.json` and
`prefix-refinement-v1/heldout-points.json`, while importing the current build.
