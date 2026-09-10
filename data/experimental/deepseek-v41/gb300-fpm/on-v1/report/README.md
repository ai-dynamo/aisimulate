# GB300 TP4 independent whole-forward FPM verification

Decoder replay: **ON**. Core, field and service remain separate.

| Scope | Metric | Supported / observed | Real mean | Prediction mean | MAPE | WAPE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| core | ttft_ms | 1100/1500 | 367.3589 | 390.7521 | 13.1382 | 13.2701 |
| core | average_tpot_ms | 1100/1500 | 160.2423 | 154.1376 | 3.8090 | 3.8503 |
| core | exact_itl_ms | 1100/1500 | 160.2423 | 154.1376 | 3.8090 | 3.8503 |
| core | output_tokens_per_second | 1100/1500 | 7.9991 | 8.1379 | 4.0948 | 3.6066 |
| core | request_latency_ms | 1100/1500 | 4070.4987 | 3963.6888 | 3.7659 | 3.4679 |
| core | last_token_latency_ms | 1100/1500 | 4070.0274 | 3963.6888 | 3.7591 | 3.4638 |
| core | native_forward_interval_ms | 35544/39393 | 158.0861 | 154.7081 | 2.0535 | 2.3311 |
| field | ttft_ms | 120/120 | 372.2376 | 415.1192 | 16.7871 | 16.9198 |
| field | average_tpot_ms | 120/120 | 159.4551 | 155.3943 | 2.5895 | 2.6213 |
| field | exact_itl_ms | 120/120 | 159.4551 | 155.3943 | 2.5895 | 2.6213 |
| field | output_tokens_per_second | 120/120 | 8.9204 | 8.9163 | 3.6209 | 3.3529 |
| field | request_latency_ms | 120/120 | 4689.3444 | 4631.8558 | 3.1296 | 2.9778 |
| field | last_token_latency_ms | 120/120 | 4688.8154 | 4631.8558 | 3.1268 | 2.9762 |
| field | native_forward_interval_ms | 3466/3503 | 157.0250 | 155.3264 | 1.1918 | 1.5964 |
| service | ttft_ms | 120/120 | 371.3825 | 415.1200 | 16.5980 | 16.8026 |
| service | average_tpot_ms | 120/120 | 158.4320 | 155.3943 | 2.1891 | 2.2047 |
| service | exact_itl_ms | 120/120 | 158.4320 | 155.3943 | 2.1891 | 2.2047 |
| service | output_tokens_per_second | 120/120 | 8.9781 | 8.9163 | 3.4813 | 3.3344 |
| service | request_latency_ms | 120/120 | 4658.6337 | 4631.8566 | 2.8795 | 2.7001 |
| service | last_token_latency_ms | 120/120 | 4658.1053 | 4631.8566 | 2.8768 | 2.6986 |
| service | native_forward_interval_ms | 3468/3504 | 155.9715 | 155.3405 | 1.0313 | 1.2563 |

![MAPE and WAPE](comparison.png)

Both errors use identical supported pairs. Missing predictions remain in coverage and keep their original failure reasons.
Observed coverage, prediction coverage and measured calibration-coordinate coverage are separate in each source summary.
Native intervals are correlated. Original scenario CIs remain in the compressed comparison outputs; core ON stays segmented, with no pooled lifecycle CI.
Core scenario rows spanning lifecycle segments are descriptive error aggregates, without a pooled CI.
HTTP timing includes service costs outside the native DeviceTimer interval; mean TPOT is not tail ITL.
The calibration-only consumer self-query checks appear separately under ../calibration and are excluded from every accuracy table and plot.
