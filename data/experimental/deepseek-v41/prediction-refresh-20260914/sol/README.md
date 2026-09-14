# SOL predictions refreshed after the main merge

Actual CPU predictions use source/build commit `8dc02dca8d7f33cc3c0448474f417aeb99345828` and native SHA `c30d6f8864bb9f047c8fb283993e1d9a6924dfcf9ce1bb8b2fed5761302d1226`. The report parent `af5f6efefa63c358bde1e10e46f77c930975fbe0` changes routing only. No new GPU measurements, fitting, calibration replacement, or predictor corrections were performed. Historical predictions are not reused as current results.

`observations-and-predictions.csv.gz` retains 100,081 scalar observation rows, including failures, missing predictions, cache disagreements, and six diagnostic warmup points in both populations. `summary.csv` and `summary.json` contain overall and per-scenario/per-phase values. Missing errors are N/A, not zero. MAPE is the equal-row mean absolute percentage error; WAPE is summed absolute error divided by summed observation. Both use exactly the same supported subset. Different studies/populations are not pooled.

## Native forward latency

| Dataset | Population | Predicted / planned | MAPE % | WAPE % |
|---|---|---:|---:|---:|
| gb200-fresh-diagnostic | native-benchmark | 30/30 | 96.6351 | 96.5395 |
| gb200-fresh-diagnostic | ordinary-matched | 20/30 | 96.6880 | 96.6158 |
| gb200-ordinary-trace | native-forward | 12531/12531 | 99.2273 | 99.2011 |
| gb200-retained-holdout | native-forward | 38/38 | 99.1204 | 99.1023 |
| gb300-holdout-off | native-forward | 46/46 | 96.1513 | 96.0327 |
| gb300-holdout-on | native-forward | 30/46 | 96.8892 | 96.7890 |
| gb300-retained-native-core-off | native-forward | 15756/15756 | 99.3541 | 99.3110 |
| gb300-retained-native-core-on | native-forward | 38997/39393 | 99.4258 | 99.4004 |
| gb300-retained-native-field-off | native-forward | 3503/3503 | 99.3802 | 99.3417 |
| gb300-retained-native-field-on | native-forward | 3466/3503 | 99.4481 | 99.4260 |
| gb300-retained-native-service-off | native-forward | 3509/3509 | 99.3733 | 99.3328 |
| gb300-retained-native-service-on | native-forward | 3468/3504 | 99.4424 | 99.4188 |

GB300 ON core retains 100 previously unidentifiable heterogeneous source intervals and 296 current multiple-prefill API rejections. Field ON retains 37 and service ON 36 current rejections. The independent GB300 ON holdout retains 16 rejected batch-two prefill configurations. The current aggregate FPM input cannot identify individual extend tails, including equal-length tails; no guessed per-request input is injected to restore coverage.

GB300 report-retained native geometry is rederived and checked against the original axis/variance bridge. This is a prediction refresh over immutable qualified observations, not a repeat of raw lifecycle admission. GB200 ordinary serving passes the full retained audit, plan, measurement, HTTP, source, and scheduler comparison gates. Native intervals remain correlated within trials; no interval-level independence or fresh confidence interval is claimed.

## HTTP cohort metrics

| Dataset | Metric | Predicted / planned cohorts | MAPE % | WAPE % |
|---|---|---:|---:|---:|
| gb200-ordinary-e2e | average_tpot_ms | 450/450 | 99.5751 | 99.5751 |
| gb200-ordinary-e2e | output_tokens_per_second | 450/450 | 12470.8689 | 12148.9281 |
| gb200-ordinary-e2e | ttft_ms | 450/450 | 94.0700 | 95.3421 |
| gb300-retained-e2e-core-off | average_tpot_ms | 560/600 | 99.5716 | 99.5703 |
| gb300-retained-e2e-core-off | output_tokens_per_second | 560/600 | 14535.9547 | 13751.7927 |
| gb300-retained-e2e-core-off | ttft_ms | 560/600 | 96.6770 | 96.6504 |
| gb300-retained-e2e-core-on | average_tpot_ms | 1300/1500 | 99.5892 | 99.5880 |
| gb300-retained-e2e-core-on | output_tokens_per_second | 1300/1500 | 15224.5380 | 14628.0342 |
| gb300-retained-e2e-core-on | ttft_ms | 1300/1500 | 96.9048 | 96.8937 |

All six HTTP metrics, including request latency, last-token latency and exact mean ITL, are present in the CSV. GB300 OFF replays 560/600 cohorts; ON replays 1,300/1,500. Reconstructed request tokens and arrival offsets reproduce each original SILICON ReplaySpec SHA before replacing only the timing-provider configuration with SOL. No prior predicted duration enters the model. OFF has 40 unavailable prefix-seed cohorts; ON has 100 unavailable prefix-seed cohorts and 100 cohorts lacking original arrival metadata. All remain in coverage.

Predicted cache reuse disagrees with native evidence in 40/560 supported OFF cohorts and 102/1,300 supported ON cohorts. These results remain in the error metrics and have row-level cache flags. All four field/service HTTP scopes retain 120 observed cohorts each with current predictions unavailable because the original token-bearing plans and complete submission inputs could not be recovered. Their historical predicted timings are not used as inputs or reported as current.

TTFT is the current replay engine's first output token; for chunked SGLang prefill the first token follows final prefill completion. HTTP predictions retain observed arrivals and scheduler geometry, but do not represent network/runtime overhead or prove measured/predicted output equivalence. Reconstructed-input replay is not fresh raw admission.

## Fresh GB200 diagnostic boundary

The two diagnostic rows above compare SOL predictions separately against 30 measured native benchmark repetitions and 20 exact ordinary-serving matches from 30 planned points. Ten batch-two ordinary targets have no exact match and remain N/A. Six native warmup points and their ordinary counterparts are preserved but excluded from summary errors. Five measured repetitions share a native process per mode and one ordinary process; they are not independent process lifecycles. The retained native/ordinary output comparison has 5 differences among 40 measured output requests. These diagnostic prediction MAPEs are separate from the earlier native-versus-ordinary measured APE. They do not admit new calibration, a full HTTP study, GB300 behavior, or a causal cluster explanation.

## Reproduction and provenance

Build the predictor at the recorded source commit in a fresh environment. With that environment active, run:

```sh
python -B replay.py --dataset gb200-retained-holdout --output /absolute/new-output.json
python -B replay.py --dataset gb300-retained-e2e-core-off --output /absolute/new-http-output.json
```

Run from this directory, or pass its absolute script path. The output must not exist. The script verifies the published inputs/system overlays and the explicit native SHA; `--native-sha256` supports a separately verified fresh platform build, whose hash is recorded in the output. It never tunes the model or invokes GPU collection. The portable script covers native rows and recovered GB300 core HTTP workloads. Repeating the complete GB200 HTTP qualification requires the hash-bound private closed packet; this publication provides its current observed/predicted metrics and input hashes.

All copied/adapted material comes from this Apache-2.0 AISimulate repository. Qualification/HTTP metric definitions are from commit `df04d9edec7efc1a97dbf67782c026a94f4b0761`, `data/experimental/deepseek-v41/verification-plan/`. Existing GB300 observation reports and system overlays are from commit `5b6309570bb7e1ccf5fa461afa92392df11d9a50`; the GB200 overlay is from `df04d9edec7efc1a97dbf67782c026a94f4b0761`. `input-origins.json`, `input-pins.json`, `systems-pins.json`, and the recovered workload packet source hashes identify exact inputs. The copies of the system overlays are unchanged; only config paths are made portable. SOL does not select the included FPM calibration table. `predictor-identity.json` binds actual Python/native code; `artifact-hashes.json` binds this publication.
