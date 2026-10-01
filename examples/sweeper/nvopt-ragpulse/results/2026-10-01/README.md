<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ComputeLab results — all three broad searches complete

All three native studies exhausted their **256-suggestion budgets**. The selected winners passed **1 / 3 / 3 independent full-day replays**. All raw results were checksum-verified before the owned containers, node directories and allocations were released. Report assembled at **2026-10-01T17:34:43.430737+00:00**.

[Search spaces and protocol](../../README.md) · [Exact configurations and results](snapshot.json) · [Replay timing CSV](attempt-timings.csv) · [Coverage references](coverage-references.md)

## Validated winners

| Scenario | Search reference | Fresh mean [min, max] tok/s/GPU | Mean total goodput tok/s | Mean SLA pass | Mean allocated GPUs | Fresh replays |
|---|---:|---:|---:|---:|---:|---:|
| Static | 34.3868 | 34.3868 [34.3868, 34.3868] | 1375.471 | 78.51% | 40.000 | 1 |
| + KV Router | 23.2101 | 23.2134 [23.1997, 23.2211] | 1671.362 | 89.38% | 72.000 | 3 |
| + Planner | 41.9101 | 41.7679 [41.6052, 41.8529] | 1049.319 | 58.32% | 25.123 | 3 |

The objective is compliant output tokens / 86,400 seconds / average GPUs, with **no minimum request SLA coverage constraint**. Planner has the highest observed efficiency but lower total goodput and coverage. These are independently optimized configurations, not a controlled causal ablation of Router or Planner, and not proven global optima.

A useful search reference is Router **candidate-000227**: 80 GPUs, 23.1996 output tokens/s/GPU and **99.7809% request SLA coverage**. Its observed search score is only 0.0453% below selected candidate-000228, which uses 72 GPUs with 89.3540% search coverage. Candidate-000227 was **not independently replayed**; no statistical equivalence or reliable ordering of this small difference is claimed. Formal selections remain unchanged. The [0/90/95/99% coverage table](coverage-references.md) is a read-only filter of original search observations, not another optimization or fresh validation.

## Configurations

All winners use DeepSeek-V3 on modeled H200 SXM, BF16 attention, FP8 KV cache, 64-token blocks, prefix caching, chunked prefill and zero worker startup delay. Exact fields and hashes are in [snapshot.json](snapshot.json).

| Setting | Static | + KV Router | + Planner |
|---|---|---|---|
| Backend | SGLang 0.5.14 | SGLang 0.5.14 | vLLM 0.24.0 |
| Serving mode | Disaggregated | Disaggregated | Disaggregated |
| Initial GPUs | 40 | 72 | 72 |
| Prefill | 3 × TP8 / attention-DP1 / MoE-TP8 EP1 | 4 × TP8 / attention-DP1 / MoE-TP8 EP1 | 5 × TP8 / attention-DP1 / MoE-TP8 EP1 |
| Decode | 2 × TP1 / attention-DP8 / MoE-TP8 EP1 | 5 × TP1 / attention-DP8 / MoE-TP1 EP8 | 4 × TP8 / attention-DP1 / MoE-TP8 EP1 |
| Prefill max tokens / sequences | 32768 / 256 | 32768 / 1 | 16384 / 2 |
| Decode max tokens / sequences | 16384 / 512 | 16384 / 1024 | 8192 / 1024 |
| Routing | Round-robin placement | KV; load model none, overlap 1.5, load scale 64, temperature 0 | KV; load model none, overlap 1, load scale 64, temperature 0 |
| Planner | Disabled | Disabled | Throughput + load scaling, 60 s / 5 s; constant_last predictor |

Planner's final load sensitivity is 95 with 12 observations; FPM sampling is 256 samples / bucket 16. Four-day predictor selection/bootstrap selected **constant_last** at 60-second cadence. Earlier intermediate Kalman selections are not the final winner. These results do not establish learned diurnal seasonality; earlier days did not populate the evaluation engine KV cache.

Static's fresh modeled summaries matched within the declared tolerance. Router's three fresh scores were 23.219283, 23.199737, 23.221081; Planner's were 41.845607, 41.605250, 41.852928. Native routing can remain stochastic at temperature 0 because ties are random. Differences were retained; configurations were not changed during validation and no best fresh repeat was selected.

## Search time and accounting

![Accepted incumbent and replay/optimizer time](search-comparison.png)

[Timing PDF](search-comparison.pdf). These curves show the replacement studies, including Planner preparation. [Selected-study replay durations](attempt-timings.csv) are retained with interrupted durations explicitly marked as bounds. The [all-replay timing ledger](all-replay-timings.csv) also includes the three archived failed studies and all seven agreed fresh validations: 816 scoped replay records in total. Recorded status and duration kind remain separate; an old raw running status with an exit bound denotes a censored attempt, not a live process.

| Scenario | Supervisor wall time | Instrumented optimizer calls | Complete replay p50 / p90 / max | Longest suggest |
|---|---:|---:|---:|---:|
| 1 | 4.0397 h | 2100.803 s (14.45%) | 188.27 / 490.05 / 2903.98 s | 400.36 s |
| 2 | 11.3755 h | 2105.293 s (5.14%) | 267.37 / 596.31 / 7040.32 s | 450.61 s |
| 3 | 5.7666 h | 3320.622 s (16.00%) | 276.22 / 801.17 / 5661.43 s | 846.43 s |

Planner preparation took **1,767.553 seconds** for 55 freshly recomputed native predictor tasks. Optimizer timing covers AIS sampler constructor/suggest/observe calls and native projection, not isolated Vizier GP CPU time. Parallel replay wall-seconds and overlapping callbacks do not form an additive partition of study wall time.

| Scenario | Actual full-cohort replays | Successful / failed cache feedback | Other actual outcomes | Suggestions |
|---|---:|---:|---|---:|
| Static | 162 | 84 / 0 | 5 virtual-cap incomplete, 3 native invariant errors, 2 timeouts | 256 |
| + KV Router | 200 | 13 / 12 | 26 virtual-cap incomplete, 5 timeouts | 256 |
| + Planner | 242 | 13 / 0 | 1 timeout | 256 |

Each study used 128 aggregated and 128 disaggregated suggestions, with exactly one terminal receipt for every issued branch/trial ID. No OOM or resource-limit termination occurred. Incomplete or timed-out replays never receive fabricated zero scores. Static's invariant errors remain separately recorded; their internal cause was not proven and the numerical runtime was not changed.

### Restart and validation cost

The first three studies exited when timeout cleanup could not stop the executor manager within the pinned nightly's two 2-second joins. A controlled 32-worker / 216.67-GiB fixture reproduced it; larger join budgets completed cleanup in about 4.69 seconds. Wrapper commit `38a89d02` adds a 60-second executor join grace while retaining ownership checks and escalation. All three replacement studies subsequently continued successfully through actual timeout/pool replacement.

Only `runtime/run_sweep.py` changed in copies of original Static/Router bundle `71fbad95` and Planner bundle `9ebfc7de`. Numerical engine, image, data, search domains, objective, seed, budgets and 7200-second native wave deadline stayed frozen. These were **new studies, not resume**; previous optimizer observations were not silently reused.

Failed-study wall times were Static **8,416.411 s**, Router **7,230.824 s**, Planner **14,413.437 s**. Fresh validation wall times were 234.840 s for Static; 441.856 / 366.765 / 367.815 s for Router; and 275.454 / 192.277 / 193.042 s for Planner. First repeats capture request/telemetry detail; these timings are not controlled speedup comparisons.

The overlapping execution window from first formal start **03:25:27 UTC** to last fresh completion **17:29:00 UTC** was **14.06 hours**. It includes failed studies and repair gaps, but excludes earlier image creation/initial staging and final archival/cleanup. Parallel study times must not be naively summed as campaign elapsed time.

## Full-day serving comparison

![Traffic, goodput, SLA coverage, cache reuse and GPU allocation](winner-day-comparison.png)

[Comparison PDF](winner-day-comparison.pdf). Curves use the first agreed detailed replay; the results table uses all agreed fresh repeats. All request/token counters and the 24-hour offered workload match across reports.

| First detailed replay | Initial / average / peak GPUs | Native GPU-hours | First-admission token-weighted cache reuse |
|---|---:|---:|---:|
| Static | 40 / 40 / 40 | 959.382 | 12.1279% |
| + KV Router | 72 / 72 / 72 | 1726.884 | 20.9708% |
| + Planner | 72 / 25.106 / 136 | 602.172 | 7.9918% |

GPU-hours match exact lifecycle integration of active, starting and draining workers. Cache ratios were independently reconstructed from admissions and matched to native summaries; decode/readmission counters were not added as extra prefill savings. These are observations of jointly optimized configurations, not isolated Router/scaling effects.

Detailed latency/serving plots: [Static](scenario-1-day.png), [Router](scenario-2-day.png), [Planner](scenario-3-day.png).
Hourly data: [Static CSV](scenario-1-hourly.csv), [Router CSV](scenario-2-hourly.csv), [Planner CSV](scenario-3-hourly.csv).

![Native Planner decisions](planner-decisions.png)

[Planner decisions PDF](planner-decisions.pdf). Requested replica targets are separate from allocated/ready workers. The first Planner replay falls to about 16 GPUs in the low-load period and reaches a 136-GPU peak; lower average allocation accompanies lower SLA coverage.

## Frozen runtime, inputs and evidence

Dynamo: `c7241c2f153efba10b57c38c2144b70d82194a4d`.
AISimulate: `0.13.0.dev202609300000000061`; release Rust bindings.
Image: `nvcr.io/nvidian/dynamo-dev/aisimulate@sha256:3d38aa7e325da409c3a7b1d33298fbce5843fbc1c2f93b0643a69d6e907be670`.
CPU JAX/JAXlib 0.4.38 with x64 enabled; Vizier 0.1.21.

Each study used 32 evaluation workers in a 40-CPU / 384-GiB container, with a 320-GiB tree-RSS guard, 96-GiB worker address-space limit and 16-GiB host-available floor. Peak study RSS was **242.708 / 236.561 / 248.507 GiB**. Shared CPU nodes do not imply dedicated cores; no physical GPUs were required.

Input provenance is in the [experiment contract](../../experiment-contract.yaml). Source: [RAGPulse revision 7da286be](https://github.com/flashserve/RAGPulse/tree/7da286becf0f049b2bcb1e5a11d9ba8eb638eff4). Traffic uses 512 disjoint copies with bounded session jitter, without time compression. Days 1–4 supply predictor history only; day 5 supplies 537,600 requests, 1,476,413,952 input tokens and 160,730,624 output tokens. SLAs are TTFT ≤ 1000 ms and request-mean ITL ≤ 50 ms.

Day 5 is also optimizer feedback: fresh replay checks same-day reproducibility, not a new held-out day or physical GPU accuracy. Zero startup delay is a modeling assumption. Prefix hashes derive from opaque component IDs, not original token blocks.

These reports and figures are original aggregate experiment measurements; source traces and per-request reports are not redistributed. Full receipts, raw reports, source hashes and independent audits remain in the local campaign archive. Cross-node SHA checks verified Static **1,587 files / 1,523,744,763 bytes**; Router **1,867 files / 1,534,674,981 bytes**; Planner **2,008 files / 1,635,879,947 bytes**. All owned containers/directories were removed and allocations released. Static/Router cancellation was intentional post-completion release; Planner allocations completed naturally.
