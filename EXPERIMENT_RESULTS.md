# Simulation regression experiment results

**All three restored regressions were detected. The unchanged-revision control passed.**

[GitHub Actions run](https://github.com/ai-dynamo/aisimulate/actions/runs/37498928548) — October 6, 2026. The workflow completed with failure because all three candidate benchmark steps reported performance regressions. All five release builds and all setup steps passed.

Baseline: `8a709362b5f0692d263eceab4cba0a13790fe138`. Workflow revision: `6f28febeff57951b6b39f07967af851cb8ec62d6`. See [EXPERIMENT.md](EXPERIMENT.md) for source revisions, reverse-patch adjustments, and local validation.

## Detection results

| Comparison | Result | Cases flagged | Invalid cases | Benchmark seconds | Job |
|---|---|---:|---:|---:|---|
| control | pass | 0/12 | 0 | 620.2 | [run](https://github.com/ai-dynamo/aisimulate/actions/runs/37498928548/job/112393672908) |
| without-295 | detected | 12/12 | 0 | 678.6 | [run](https://github.com/ai-dynamo/aisimulate/actions/runs/37498928548/job/112393672770) |
| without-321 | detected | 4/12 | 0 | 673.2 | [run](https://github.com/ai-dynamo/aisimulate/actions/runs/37498928548/job/112393672925) |
| without-386 | detected | 3/12 | 0 | 643.3 | [run](https://github.com/ai-dynamo/aisimulate/actions/runs/37498928548/job/112393672919) |

All 48 case comparisons completed five paired rounds: 240 pairs and 480 successful worker responses. The downloaded raw records were checked again with the unchanged comparator. Case definitions, source revisions, request/token completion, model/data identity, and paired build settings passed validation. There were no invalid or missing measurements.

Control median changes ranged from -0.9% to +1.9%. The regression rule remained greater than 10% and 100 ms in at least four of five paired rounds. No workloads, thresholds, or samples were changed, and no benchmark was rerun.

## What the suite caught

- **Without #295:** all 12 cases failed, each in 5/5 rounds. Median slowdowns ranged from +12.3% to +87.0%. Both AgentX cases also failed.
- **Without #321:** dense SGLang (+352.1%), P/D SGLang (+339.2%), cache-pressure SGLang (+131.4%), and long MoE decode (+32.0%) failed in 5/5 rounds.
- **Without #386:** dense vLLM (+11.0%) and P/D vLLM (+11.1%) failed in 5/5 rounds; dense TRT-LLM (+11.4%) failed in 4/5 rounds.

The synthetic cases were necessary for this result: neither AgentX case flagged the #321 or #386 variant. The #321 disaggregated SGLang AgentX median rose 6.3%, below the relative threshold.

## All case results

Changes below are ratios of head/base medians. The verdict uses individual paired rounds, not the ratio of medians. Benchmark seconds exclude builds and installation.

### control

Candidate: `8a709362b5f0692d263eceab4cba0a13790fe138`.

| Case | Classification | Base ms | Candidate ms | Median change | Rounds above both thresholds |
|---|---|---:|---:|---:|---:|
| dense-vllm | PASS | 2990.90 | 2992.89 | +0.1% | 0/5 |
| dense-sglang | PASS | 1692.09 | 1724.26 | +1.9% | 0/5 |
| dense-trtllm | PASS | 3429.18 | 3481.06 | +1.5% | 0/5 |
| moe-long-prefill | PASS | 3090.25 | 3101.52 | +0.4% | 0/5 |
| moe-long-decode | PASS | 3295.63 | 3290.03 | -0.2% | 0/5 |
| cache-pressure-vllm | PASS | 2459.55 | 2451.57 | -0.3% | 0/5 |
| cache-pressure-sglang | PASS | 1998.46 | 1985.80 | -0.6% | 0/5 |
| mla-multiworker-dp | PASS | 2973.49 | 2950.36 | -0.8% | 0/5 |
| pd-vllm | PASS | 2367.66 | 2368.96 | +0.1% | 0/5 |
| pd-sglang | PASS | 1321.99 | 1314.23 | -0.6% | 0/5 |
| agentx-vllm-aggregated | PASS | 2462.90 | 2467.70 | +0.2% | 0/5 |
| agentx-sglang-disaggregated | PASS | 2720.24 | 2695.82 | -0.9% | 0/5 |

### without-295

Candidate: `4fe1aff3e1003f4b8994f5b9721800f42c8cf1f2`.

| Case | Classification | Base ms | Candidate ms | Median change | Rounds above both thresholds |
|---|---|---:|---:|---:|---:|
| dense-vllm | PERFORMANCE_REGRESSION | 2989.74 | 3732.36 | +24.8% | 5/5 |
| dense-sglang | PERFORMANCE_REGRESSION | 1711.32 | 1967.93 | +15.0% | 5/5 |
| dense-trtllm | PERFORMANCE_REGRESSION | 3483.89 | 4225.61 | +21.3% | 5/5 |
| moe-long-prefill | PERFORMANCE_REGRESSION | 2997.10 | 4076.90 | +36.0% | 5/5 |
| moe-long-decode | PERFORMANCE_REGRESSION | 3307.87 | 4038.50 | +22.1% | 5/5 |
| cache-pressure-vllm | PERFORMANCE_REGRESSION | 2433.20 | 4549.84 | +87.0% | 5/5 |
| cache-pressure-sglang | PERFORMANCE_REGRESSION | 1953.73 | 3011.07 | +54.1% | 5/5 |
| mla-multiworker-dp | PERFORMANCE_REGRESSION | 2946.37 | 4167.50 | +41.4% | 5/5 |
| pd-vllm | PERFORMANCE_REGRESSION | 2319.99 | 3016.15 | +30.0% | 5/5 |
| pd-sglang | PERFORMANCE_REGRESSION | 1310.40 | 1471.10 | +12.3% | 5/5 |
| agentx-vllm-aggregated | PERFORMANCE_REGRESSION | 2449.83 | 4316.59 | +76.2% | 5/5 |
| agentx-sglang-disaggregated | PERFORMANCE_REGRESSION | 2703.90 | 4572.32 | +69.1% | 5/5 |

### without-321

Candidate: `71e54cb06b24e2e774c4583fe017cdb49298b834`.

| Case | Classification | Base ms | Candidate ms | Median change | Rounds above both thresholds |
|---|---|---:|---:|---:|---:|
| dense-vllm | PASS | 2931.64 | 3001.21 | +2.4% | 0/5 |
| dense-sglang | PERFORMANCE_REGRESSION | 1672.22 | 7559.91 | +352.1% | 5/5 |
| dense-trtllm | PASS | 3428.17 | 3438.18 | +0.3% | 0/5 |
| moe-long-prefill | PASS | 2956.84 | 2891.86 | -2.2% | 0/5 |
| moe-long-decode | PERFORMANCE_REGRESSION | 3263.55 | 4307.21 | +32.0% | 5/5 |
| cache-pressure-vllm | PASS | 2433.72 | 2421.58 | -0.5% | 0/5 |
| cache-pressure-sglang | PERFORMANCE_REGRESSION | 1898.32 | 4392.28 | +131.4% | 5/5 |
| mla-multiworker-dp | PASS | 2927.42 | 2911.05 | -0.6% | 0/5 |
| pd-vllm | PASS | 2297.13 | 2291.18 | -0.3% | 0/5 |
| pd-sglang | PERFORMANCE_REGRESSION | 1256.18 | 5517.02 | +339.2% | 5/5 |
| agentx-vllm-aggregated | PASS | 2453.48 | 2443.86 | -0.4% | 0/5 |
| agentx-sglang-disaggregated | PASS | 2658.55 | 2824.78 | +6.3% | 0/5 |

### without-386

Candidate: `00732f0f4bb5637f52d15c4f8fec612602db10e8`.

| Case | Classification | Base ms | Candidate ms | Median change | Rounds above both thresholds |
|---|---|---:|---:|---:|---:|
| dense-vllm | PERFORMANCE_REGRESSION | 2992.16 | 3321.58 | +11.0% | 5/5 |
| dense-sglang | PASS | 1733.70 | 1814.00 | +4.6% | 0/5 |
| dense-trtllm | PERFORMANCE_REGRESSION | 3518.59 | 3920.98 | +11.4% | 4/5 |
| moe-long-prefill | PASS | 3179.12 | 3237.38 | +1.8% | 0/5 |
| moe-long-decode | PASS | 3375.88 | 3679.45 | +9.0% | 1/5 |
| cache-pressure-vllm | PASS | 2450.80 | 2578.21 | +5.2% | 0/5 |
| cache-pressure-sglang | PASS | 2013.30 | 2042.96 | +1.5% | 0/5 |
| mla-multiworker-dp | PASS | 3040.46 | 3121.21 | +2.7% | 0/5 |
| pd-vllm | PERFORMANCE_REGRESSION | 2386.36 | 2652.30 | +11.1% | 5/5 |
| pd-sglang | PASS | 1363.70 | 1410.27 | +3.4% | 0/5 |
| agentx-vllm-aggregated | PASS | 2472.50 | 2461.51 | -0.4% | 0/5 |
| agentx-sglang-disaggregated | PASS | 2762.02 | 2762.69 | +0.0% | 0/5 |

## Artifacts and limits

Each result artifact contains `raw.json`, `comparison.json`, `summary.md`, worker logs, build/package provenance, the candidate patch, and the experiment source record. CI retains these artifacts for 30 days.

| Artifact | ID |
|---|---:|
| `simulation-regression-without-321` | 11430645305 |
| `simulation-regression-control` | 11430505364 |
| `simulation-regression-without-295` | 11430047003 |
| `simulation-regression-without-386` | 11429169846 |

This is one complete experiment on current main with each optimization separately reversed. It tests current CI coverage, not the original historical revision pairs. The gate checks complete work and comparable model settings; it is not a full output-equivalence test. Timing is the native replay `wall_time_ms`; reported percentages are not predicted GPU-serving slowdowns. CPU affinity follows normal CI and does not prove exclusive physical-host ownership.
