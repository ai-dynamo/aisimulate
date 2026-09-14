# FPM predictions after review fixes

Actual CPU predictions were rerun at `430ed5e79abaf667c026d980c18ec240261fc800`, after the shared SOL scoring/storage fixes and integration of main `be8f69749589aea5a376083436e054d75810d379`. The native extension was built at `f541c436067d5d185e9cb712d72b75e44ae524f9`; subsequent Collector-only changes were verified to leave all predictor/build inputs unchanged; its SHA-256 is `50d3010ee04daaa23158fe5ef873b363a61b096ec755b1bc46a7a1ba6a8117b0`. Observations, calibration files and model precision were preserved. There was no fitting, fallback predictor, new GPU measurement or replacement of historical reports.

**GB200 admission remains blocked:** the retained 126-point table is historical, environment-specific evidence and is not admitted for prediction or reusable serving calibration. See the [admission disclosure](../../gb200-fpm/calibration-v1/README.md). These replays document its failures without requalifying it.

The GB300 results are development validation: coverage gaps in the original verification informed the 18-point calibration extension per profile, after which the same verification population was reused. Calibration and verification timings are distinct, but this is not a blind final test. No current prediction is fitted to its observed target.

The refreshed FPM remains accurate on the GB300 native-forward observations and strongly overpredicts GB200 ordinary-serving latency. The separate DL diagnostic reproduces this discrepancy on both native benchmark and ordinary-serving paths.

| Scope | Native-forward MAPE | Native coverage | TTFT MAPE | TPOT MAPE | Throughput MAPE | HTTP coverage |
|---|---:|---:|---:|---:|---:|---:|
| GB300 Decoder OFF, core | 1.2412% | 15,756 / 15,756 | 40.4646% | 2.4796% | 7.1762% | 560 / 600 cold cohorts |
| GB300 Decoder ON, core | 2.0103% | 38,997 / 39,393 | 42.0729% | 3.7009% | 8.5979% | 1,300 / 1,500 cold cohorts |
| GB200 Decoder OFF, ordinary serving | 432.3291% | 12,531 / 12,531 | 838.9658% | 435.9172% | 82.2571% | 450 / 450 cohorts; 660 requests |

GB300 native and HTTP columns use different populations. The HTTP columns are a recovered cold subset, not the complete historical study. Forty OFF prefix-reuse cohorts and 200 ON cohorts lack complete original replay inputs. All remain in the denominator. The ON omissions comprise 100 prefix-seed cases and 100 cases without complete request metadata. No arrival time or seed timing was inferred from a predicted duration. Each recovered workload reproduces its original SILICON replay-specification hash before substituting the current FPM timing provider. Current SGLang replay emits the first output token at prefill completion; these fresh TTFT predictions are not the historical native binary's predictions. The GB200 vLLM aggregated replay semantics remain separately applied. Current GB300 cache disagreements remain included: 40 / 560 OFF and 102 / 1,300 ON predicted cohorts.

Additional native-forward results:

| Scope | MAPE | Coverage |
|---|---:|---:|
| GB300 OFF field / service | 0.7507% / 0.7217% | 3,503 / 3,503; 3,509 / 3,509 |
| GB300 ON field / service | 1.1918% / 1.0313% | 3,466 / 3,503; 3,468 / 3,504 |
| GB300 OFF / ON, 46-point forward validation corpus | 3.2423% / 7.4274% | 45 / 46; 29 / 46 |
| GB200 independent 38-point holdout | 4.8942% | 38 / 38 |
| GB200 DL native diagnostic | 431.4790% | 30 / 30 |
| GB200 DL ordinary diagnostic | 430.8405% | 20 / 30 |

The GB300 46-point corpus was originally an op-level holdout. One coordinate per profile exactly overlaps the unchanged FPM calibration: prefill B1, 64 new tokens, 256 past-KV tokens. This report therefore calls it forward validation, without claiming new physical-observation independence. Both profiles retain the B2 decode/past-KV3840 missing-domain result. ON additionally retains 16 current multi-request prefill guard failures. The six native-serving scopes retain all 69,168 original intervals; the current ON guard rejects 396 core, 37 field and 36 service intervals. These are unsupported results, not zero-error predictions.

The DL diagnostic uses five correlated repetitions at each of six existing calibration coordinates. It is not an independent geometry holdout, a formal serving study or a calibration update. Ten ordinary points lack a unique matching dispatch and remain unavailable; its 20-point MAPE and the native 30-point MAPE have different populations. Six prospectively designated warmup points and prefix setup are excluded from those MAPEs. Benchmark/serving observed-latency ratios are distinct from prediction MAPE. These results do not establish a cause for the historical performance difference.

GB300 field/service HTTP token plans could not be recovered locally. Each of their four 120-cohort populations is explicitly exported with unavailable fresh predictions and original observed metrics. Their old predictions are not reused. Original source-qualified raw GB300 trace files were unavailable; native refresh re-queries complete workload descriptors preserved in immutable published reports and does not perform new raw-data admission.

The GB200 ordinary study retains 30 full trials, all 450 primary cohorts, 660 requests and 12,531 native intervals. Its 13 cache-disagreeing cohorts remain in accuracy. Native intervals are correlated. HTTP includes costs outside the device timer. The CSVs retain all six HTTP metrics, including mean ITL; mean ITL does not describe tail ITL. MAPE weights supported intervals/cohorts/configurations equally within each stated population; WAPE weights their observed magnitudes. Original current-run whole-trial bootstrap results are retained in [gb200-statistics.json](results/gb200-statistics.json). No new pooled GB300 lifecycle confidence interval is claimed.

## Files and checks

[summary.csv](results/summary.csv) contains full-precision MAPE, WAPE and coverage. The native, HTTP, configuration and diagnostic CSVs preserve every row, including unsupported results; runtime identifiers are replaced by ordinal positions. [source-bindings.json](results/source-bindings.json) binds completed raw prediction outputs, original inputs, checkpoint identity and unchanged system tables. [artifact-hashes.json](results/artifact-hashes.json) verifies the numerical export. The original GB200 and GB300 [OFF](../../gb300-fpm/off-union-v3-tracewait2/README.md)/[ON](../../gb300-fpm/on-union-v3-tracewait2/README.md) table bytes remain unchanged. The [GB200 admission disclosure](../../gb200-fpm/calibration-v1/README.md) now reflects completed contrary serving evidence.

Run `python verify_public.py` from this directory to verify hashes and independently recompute every published MAPE, WAPE and denominator without a model or private files. The integrated source passed 501 Python SDK/Collector tests and 706 Rust perfmodel tests (one pre-existing local-data test ignored). Public API/wire checks and repository policy validation are recorded with the PR. All new Python sources pass project Ruff lint and formatting. The public export preserves the exact current numerical pairs; it does not rerun predictions.

## Reproduce actual predictions

Build the checked-out FPM source with `uv sync --project python/aisimulate --extra dev`. Run from its repository root. The original measured predictor revision is recorded above; a later commit containing only these reports has the same predictor source. Always record the actual checked-out revision and freshly built native hash. Example shell variables:

```sh
FPM_REPO="$PWD"
FPM_PY="$FPM_REPO/python/aisimulate/.venv/bin/python"
FPM_TOOLS="data/experimental/deepseek-v41/prediction-refresh-20260914/fpm"
FPM_REV="$(git rev-parse HEAD)"
FPM_NATIVE="$($FPM_PY -c 'import hashlib,pathlib,aisimulate._runtime as n; print(hashlib.sha256(pathlib.Path(n.__file__).read_bytes()).hexdigest())')"
```

Re-query all six GB300 native corpora from their unchanged published descriptors:

```sh
"$FPM_PY" -B "$FPM_TOOLS/refresh_reported_native.py" \
  --repo "$FPM_REPO" --output-dir /path/to/new-native-output \
  --expected-head "$FPM_REV" --expected-native-sha256 "$FPM_NATIVE"
```

For the 46-point corpus, use a checkout containing PR #160's original public observations. Preparation restores only three byte-pinned qualification modules from this repository at `5b6309570bb7e1ccf5fa461afa92392df11d9a50`; it retains their Apache-2.0 headers and writes adjacent provenance. It does not import an old SDK or native extension.

```sh
"$FPM_PY" -B "$FPM_TOOLS/prepare_forward46_inputs.py" \
  --repo "$FPM_REPO" --silicon-repo /path/to/silicon-checkout --output /path/to/new-forward-inputs
"$FPM_PY" -B "$FPM_TOOLS/refresh_forward46.py" \
  --repo "$FPM_REPO" --input-task /path/to/new-forward-inputs/off.json \
  --qualifiers /path/to/new-forward-inputs/qualifiers --profile off \
  --output /path/to/new-forward-off.json --expected-head "$FPM_REV" --expected-native-sha256 "$FPM_NATIVE"
```

Repeat with `on.json` and `--profile on`. The actual FPM selector is `fpm_fmha_dtype=fp8`; it does not override analytical activation precision. An explicit selector now emits a WARNING with the original model mode and matched cell IDs. Exact recorded-label matching rejects other table precisions; it does not independently establish the engine-resolved runtime precision. The shared SOL layout changes the roofline used for interpolation, so a few predictions change despite unchanged calibration timings.

For cold replay, PR #160's portable `prediction-refresh-20260914/silicon/recover_cold_workloads.py` accepts `--repo` for the SILICON checkout, `--fpm-repo` for this checkout and a new `--output` directory. It reconstructs the shared 560 OFF / 1,300 ON exact original specifications. Regenerated packet hashes depend on provenance paths, so pass the freshly recorded packet SHA explicitly:

```sh
"$FPM_PY" -B "$FPM_TOOLS/refresh_cold_replay.py" \
  --repo "$FPM_REPO" --packet /path/to/recovered/off.json.gz --profile off \
  --expected-packet-sha256 SHA256_OF_RECOVERED_PACKET \
  --output /path/to/new-cold-off.json.gz --expected-head "$FPM_REV" --expected-native-sha256 "$FPM_NATIVE"
```

The runner still rechecks every original replay-spec hash. Only the timing provider is replaced; request tokens, arrivals, scheduler and topology remain intact. Repeat for ON.

GB200 ordinary reruns use the existing `verification-plan/compare_trace.py` and `compare_e2e.py` with the source-bound original normalized `audit.json`, `plan.json`, `measurement.json`, plus `client-summary.json` and `scheduler-receipt.json` for HTTP. Both use `gb200-fpm/holdout-sol-review-v2/fpm-prediction-config.json`. The 38-point rerun uses `compare_fpm_holdout.py`, the decompressed unchanged `holdout-v1/holdout.json.gz`, its admission receipt and `export_holdout.py`, and the unchanged `calibration-v1` directory. Their existing `--help` exposes all input arguments; exact input/output hashes are in the source bindings. Those original private observations are required for a fresh raw-qualified rerun.

`refresh_diagnostic.py --diagnostic-root /path/to/closed-diagnostic-evidence` consumes the already qualified DL archive and frozen point index; supply the same `--repo`, `--expected-head`, `--expected-native-sha256`, and a new `--output`. It preserves all native/ordinary mismatches and does not repeat GPU work. Each prediction runner rejects an output that already exists. `export_refresh.py` separately projects completed results; `verify_public.py` checks the standalone public export.
