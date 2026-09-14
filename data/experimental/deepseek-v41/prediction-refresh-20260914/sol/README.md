# SOL predictions refreshed after review fixes

Actual CPU predictions use source/build commit `dd43dd3b87f6ef19067b2e372a2458605dfc58b3` and native SHA `eaa67bc03d410356b767f9fd285452613ae3a4c0e9f83734c751590d667b686b`. The predictor includes full-context index scoring before candidate masking, explicit SGLang physical KV payload/traffic, and the fixed main snapshot `be8f69749589aea5a376083436e054d75810d379`. No new GPU measurements, fitting, or calibration replacement were performed. Historical predictions are not reused as current results.

`observations-and-predictions.csv.gz` retains 100,081 scalar observation rows, including failures, missing predictions, cache disagreements, and six diagnostic warmup points in both populations. `summary.csv` and `summary.json` contain overall and per-scenario/per-phase values. Missing errors are N/A, not zero. MAPE is the equal-row mean absolute percentage error; WAPE is summed absolute error divided by summed observation. Both use exactly the same supported subset. Different studies/populations are not pooled.

## Native forward latency

| Dataset | Population | Predicted / planned | MAPE % | WAPE % |
|---|---|---:|---:|---:|
| gb200-fresh-diagnostic | native-benchmark | 30/30 | 96.6351 | 96.5395 |
| gb200-fresh-diagnostic | ordinary-matched | 20/30 | 96.6880 | 96.6158 |
| gb200-ordinary-trace | native-forward | 12531/12531 | 99.2273 | 99.2011 |
| gb200-retained-holdout | native-forward | 38/38 | 99.1204 | 99.1023 |
| gb300-holdout-off | native-forward | 46/46 | 96.1291 | 96.0098 |
| gb300-holdout-on | native-forward | 30/46 | 96.8764 | 96.7760 |
| gb300-retained-native-core-off | native-forward | 15756/15756 | 99.3505 | 99.3067 |
| gb300-retained-native-core-on | native-forward | 38997/39393 | 99.4241 | 99.3985 |
| gb300-retained-native-field-off | native-forward | 3503/3503 | 99.3781 | 99.3393 |
| gb300-retained-native-field-on | native-forward | 3466/3503 | 99.4472 | 99.4250 |
| gb300-retained-native-service-off | native-forward | 3509/3509 | 99.3712 | 99.3304 |
| gb300-retained-native-service-on | native-forward | 3468/3504 | 99.4414 | 99.4177 |

GB300 ON core retains 100 previously unidentifiable heterogeneous source intervals and 296 current multiple-prefill API rejections. Field ON retains 37 and service ON 36 current rejections. The independent GB300 ON holdout retains 16 rejected batch-two prefill configurations. The current aggregate forward-metric API cannot identify individual extend tails, including equal-length tails; no guessed per-request input is injected to restore coverage.

GB300 report-retained native geometry is rederived and checked against the original axis/variance bridge. This is a prediction refresh over immutable qualified observations, not a repeat of raw lifecycle admission. GB200 ordinary serving passes the full retained audit, plan, measurement, HTTP, source, and scheduler comparison gates. Native intervals remain correlated within trials; no interval-level independence or fresh confidence interval is claimed.

## HTTP cohort metrics

| Dataset | Metric | Predicted / planned cohorts | MAPE % | WAPE % |
|---|---|---:|---:|---:|
| gb200-ordinary-e2e | average_tpot_ms | 450/450 | 99.5751 | 99.5751 |
| gb200-ordinary-e2e | output_tokens_per_second | 450/450 | 12470.8689 | 12148.9281 |
| gb200-ordinary-e2e | ttft_ms | 450/450 | 94.0700 | 95.3421 |
| gb300-retained-e2e-core-off | average_tpot_ms | 560/600 | 99.5707 | 99.5694 |
| gb300-retained-e2e-core-off | output_tokens_per_second | 560/600 | 14468.9420 | 13681.5075 |
| gb300-retained-e2e-core-off | ttft_ms | 560/600 | 96.6417 | 96.6140 |
| gb300-retained-e2e-core-on | average_tpot_ms | 1300/1500 | 99.5886 | 99.5874 |
| gb300-retained-e2e-core-on | output_tokens_per_second | 1300/1500 | 15178.2845 | 14580.2611 |
| gb300-retained-e2e-core-on | ttft_ms | 1300/1500 | 96.8854 | 96.8740 |

All six HTTP metrics, including request latency, last-token latency and exact mean ITL, are present in the CSV. GB300 OFF replays 560/600 cohorts; ON replays 1,300/1,500. Reconstructed request tokens and arrival offsets reproduce each original SILICON ReplaySpec SHA before replacing only the timing-provider configuration with SOL. No prior predicted duration enters the model. OFF has 40 unavailable prefix-seed cohorts; ON has 100 unavailable prefix-seed cohorts and 100 cohorts lacking original arrival metadata. All remain in coverage.

Predicted cache reuse disagrees with native evidence in 40/560 supported OFF cohorts and 102/1,300 supported ON cohorts. These results remain in the error metrics and have row-level cache flags. All four field/service HTTP scopes retain 120 observed cohorts each with current predictions unavailable because the original token-bearing plans and complete submission inputs could not be recovered. Their historical predicted timings are not used as inputs or reported as current.

TTFT is the current replay engine's first output token; for chunked SGLang prefill the first token follows final prefill completion. HTTP predictions retain observed arrivals and scheduler geometry, but do not represent network/runtime overhead or prove measured/predicted output equivalence. Reconstructed-input replay is not fresh raw admission.

## Fresh GB200 diagnostic boundary

The two diagnostic rows above compare SOL predictions separately against 30 measured native benchmark repetitions and 20 exact ordinary-serving matches from 30 planned points. Ten batch-two ordinary targets have no exact match and remain N/A. Six native warmup points and their ordinary counterparts are preserved but excluded from summary errors. Five measured repetitions share a native process per mode and one ordinary process; they are not independent process lifecycles. The retained native/ordinary output comparison has 5 differences among 40 measured output requests. These diagnostic prediction MAPEs are separate from the earlier native-versus-ordinary measured APE. They do not admit new calibration, a full HTTP study, GB300 behavior, or a causal cluster explanation.

## Reproduction and provenance

GB300 SGLang predictions now use the explicitly serialized 584-byte FlashMLA main/window payload and 68-byte low-ratio index. GB200 vLLM retains the labeled theoretical packed layout; its physical runtime KV storage is not qualified here. Payload capacity excludes page allocator overhead. These are analytical assumptions, not new calibration admissions.

Build the predictor at the recorded source commit in a fresh environment. With that environment active, run:

```sh
python -B replay.py --dataset gb200-retained-holdout --output /absolute/new-output.json
python -B replay.py --dataset gb300-retained-e2e-core-off --output /absolute/new-http-output.json
```

Run from this directory, or pass its absolute script path. The output must not exist. The script verifies the published inputs/system overlays and the explicit native SHA; `--native-sha256` supports a separately verified fresh platform build, whose hash is recorded in the output. It never tunes the model or invokes GPU collection. The portable script covers native rows and recovered GB300 core HTTP workloads. Repeating the complete GB200 HTTP qualification requires the hash-bound private closed packet; this publication provides its current observed/predicted metrics and input hashes.

All copied/adapted material comes from this Apache-2.0 AISimulate repository. Qualification/HTTP metric definitions are from commit `df04d9edec7efc1a97dbf67782c026a94f4b0761`, `data/experimental/deepseek-v41/verification-plan/`. Existing GB300 observation reports and system overlays are from commit `5b6309570bb7e1ccf5fa461afa92392df11d9a50`; the GB200 overlay is from `df04d9edec7efc1a97dbf67782c026a94f4b0761`. `input-origins.json`, `input-pins.json`, `systems-pins.json`, and the recovered workload packet source hashes identify exact inputs. The copies of the system overlays are unchanged; only config paths are made portable. SOL does not select the included FPM calibration table. `predictor-identity.json` binds actual Python/native code; `artifact-hashes.json` binds this publication.

Validation: 692 Rust performance-model tests (one ignored), 483 Python/model/parity tests, four Rust public-API tests and 11 portable-consumer tests passed. A second public-consumer execution reproduced all 92,564 supported native/recovered-HTTP scalar pairs exactly; the separate complete GB200 HTTP comparison supplies another 2,700 pairs. All 100,081 observation rows, geometry, missing statuses and cache flags remain unchanged. `validation.json` records the checks.
