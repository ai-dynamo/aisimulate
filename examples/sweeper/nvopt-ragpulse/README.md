<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVOpt: three RAGPulse search-space designs

This is a **review-stage experiment design**, frozen against Dynamo
`c7241c2f153efba10b57c38c2144b70d82194a4d` and AISimulate nightly
`0.13.0.dev202609300000000061`. The local broad sweep was stopped by request;
these files do not resume it or launch another local sweep. The environment
image is separate from the experiment data and execution harness.

| Experiment | Configuration | Search scope |
|---|---|---|
| 1. Static engine | [01-static.yaml](01-static.yaml) | Aggregated/disaggregated engine mapping and scheduler; round-robin placement, no Planner |
| 2. Engine + KV Router | [02-kv-router.yaml](02-kv-router.yaml) | The entire experiment 1 space plus KV Router knobs |
| 3. Engine + KV Router + Planner | [03-kv-router-planner.yaml](03-kv-router-planner.yaml) | The entire experiment 2 space plus Planner knobs; four-day predictor history |

These are three independent broad Bayesian searches, **256 native suggestions
each**, seed `20260929`, eight concurrent evaluations, and a 7,200-second native
evaluation deadline. There is no four-hour whole-study deadline. Failures,
cache hits and projected duplicate configurations consume suggestions; 256
suggestions do not mean 256 unique successful replays or exhaustive coverage.
Native early projection-stall termination must be reported with the actual
budget consumed. Engine parameters remain searchable in experiments 2 and 3;
the winner of an earlier experiment is not fixed as their engine configuration.

## Shared engine space and measurement

- Model: `deepseek-ai/DeepSeek-V3`; hardware: `h200_sxm`.
- Backends: vLLM `0.24.0`, SGLang `0.5.14`; aggregated and disaggregated serving.
- Maximum initial deployment: 256 simulated GPUs. No artificial minimum GPU
  floor or preferred 100-GPU filter. The optimizer can choose fewer GPUs.
- Parallelism uses the pinned native `default` legal mapping menu, including
  independent prefill/decode attention DP. Observed menu DP sizes are 1, 8, 16;
  these are legal coupled TP/DP/MoE/replica mappings, not independent Cartesian
  TP/DP/EP dimensions. PP remains 1. Initial replica counts are optimized; in
  experiment 3 Planner subsequently changes replicas within the GPU budget,
  not the per-worker TP/DP mapping.
- Op-level timing with fallback denied; BF16 attention, FP8 KV; KV blocks of
  64 tokens, prefix caching and chunked prefill enabled, memory fraction 0.9.
- Disaggregated transfer: 35,136 bytes/token, 64 GB/s, `full_prompt` timing.
- Context length 8,192. Pinned Dynamo's SGLang parser does not accept the
  corresponding `max_model_len` argument: the runner must omit it as in the
  local baseline. The evaluation trace maximum ISL+OSL is 7,599, so that
  omission does not truncate evaluation requests.

| Scheduler knob | Aggregated | Prefill | Decode |
|---|---|---|---|
| Max batched tokens | 1024, 2048, 4096, 8192, 16384, 32768 | same | 2048, 4096, 8192, 16384 |
| Max sequences | 64, 128, 256, 512, 1024 | 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 | 64, 128, 256, 512, 1024 |

All experiments evaluate the same 512-copy, fifth-day trace: 537,600 requests,
1,476,413,952 input tokens and 160,730,624 output tokens. Arrivals remain
open-loop over 24 hours; replay may drain until virtual hour 25. Incomplete
request/token cohorts are infeasible, never fabricated zero-score trials.

The objective is **SLA-compliant output tokens / 86,400 / average GPUs**.
Use TTFT <= 1,000 ms and **per-request mean ITL <= 50 ms**, not the maximum
single-token gap. Average GPUs are `native_gpu_hours / native_duration_hours`;
for static deployments this is exactly the fixed GPU count. For Planner this
uses provisioned GPU time over the native replay, including drain. Normalize
the compliant-token rate to 86,400 seconds but preserve native duration and
GPU-hours. Report native GPU-hours as well; do not charge Planner for its
256-GPU ceiling or initial allocation as though they were used all day.
There is no minimum SLA-pass-fraction filter: always show coverage and total
goodput next to efficiency so a small but overloaded deployment is visible.

## Router: defaults plus three points per numeric knob

The default domains below come from the pinned Dynamo
[Router adapter](https://github.com/ai-dynamo/dynamo/blob/c7241c2f153efba10b57c38c2144b70d82194a4d/components/src/dynamo/router/simulation/config.py).
Experiments 2 and 3 fix `policy: kv_router`; experiment 1 supplies the
round-robin comparison. Keep both default load-model choices, `none` and `ais`.

| Knob | Native default search values | Added values |
|---|---|---|
| `overlap_score_credit` | 0, 0.5, 1 | **0.25, 0.75, 1.5** |
| `prefill_load_scale` | 0, 0.25, 0.5, 1, 2, 4, 8, 16, 32 | **0.125, 64, 128** |
| `temperature` | 0, 0.2, 0.5, 1 | **0.1, 0.75, 2** |

The numeric domains bracket the existing settings while retaining every
default value. This gives 1,008 nominal Router combinations per compatible
engine mapping (2 load models × 6 credits × 12 scales × 7 temperatures).
Admission-control thresholds are excluded: the pinned replay adapter explicitly
does not support them. They are not silently accepted no-op knobs.

## Planner: defaults plus three expanded groups

Experiment 3 fixes `policy: enabled`. It retains all six **enabled** native
scaling presets; the `disabled` preset belongs to experiment 2 and is excluded
so experiment 3 cannot silently turn Planner off. Source:
[Planner presets](https://github.com/ai-dynamo/dynamo/blob/c7241c2f153efba10b57c38c2144b70d82194a4d/components/src/dynamo/planner/simulation/presets.py).

| Group | Native default candidates retained | Added combinations |
|---|---|---|
| Scaling policy | `throughput_180_5`, `throughput_600_5`, `load_180_5`, `load_180_10`, `hybrid_180_5`, `hybrid_600_5` | **hybrid (60s, 5s), (300s, 10s), (900s, 10s)** |
| FPM sampling | `small` (32,4), `default` (64,16), `large` (128,16), `fine` (128,64) | **(256 samples,16 buckets), (256 samples,64 buckets)** |
| Load sensitivity | `aggressive` (70,3), `default` (80,5), `conservative` (90,8) | **(60,2), (85,10), (95,12)**: down-sensitivity %, minimum observations |

Custom entries are self-contained mappings accepted by the public schema,
not invented preset names or a Cartesian product of mutually incompatible
preset and independent fields. Bucket sizes remain perfect squares.
FPM sampling matters only for throughput-enabled policies, and load
sensitivity only for load-enabled policies; native materialization/cache
handles those conditional identities. Do not interpret their full Cartesian
product as an equal number of distinct replays.

Keep the predictor default menu unchanged: `constant_last`, `arima_raw`,
`arima_log1p`, Prophet windows 20/50 with raw/log1p, and Kalman default/reactive
with raw/log1p: **11 candidates**. Predictor selection is a separate native
pre-search per throughput interval (60, 180, 300, 600, 900 seconds), not another
11-fold Vizier dimension. Fit and select only on days 1–4, recording its
separate timing, losses and selected family. Native fallback/no-winner must
remain explicit in the report.

The runtime ceiling is **256 GPUs**, with at least one aggregated worker or
one worker in each prefill/decode role. Preserve the default
`max_throughput_scaling_replicas: 8`: this is a **per-observation change limit**,
not an eight-worker total cap. Initial replicas come from the engine search.
Worker startup delay remains the baseline default (zero); these experiments
do not establish accuracy for real model-loading/cold-start delays.

## Four-day history, fifth-day evaluation

The source is [RAGPulse revision 7da286be](https://github.com/flashserve/RAGPulse/tree/7da286becf0f049b2bcb1e5a11d9ba8eb638eff4).
Use source time `[0, 345600)` only as Planner history and `[345600, 432000)`
only for evaluation. Both use the same 512-copy transform and seed
`20260928`, disjoint hash namespaces per copy, and deterministic per-session
phase shifts within ±30 seconds tapered at day boundaries. Time is not
compressed. Session IDs stay out of replay rows to preserve open-loop arrivals.

RAGPulse supplies component identifiers, not native token-block hashes. The
existing conversion packs ordered opaque component slices into 64-token blocks
and chains each block to its preceding prefix. Unlocated residual tokens are
row-specific. Router reuse results are consequently **RAGPulse-derived
opaque-component workload predictions**, not measured original token reuse.
Keep this qualification in plots and final findings.

The history trace contains 2,024,448 scaled requests. Before remote execution,
materialize or supply it separately, verify its four-day provenance/counts,
and record its SHA256. A saved 180-second observation file is not valid for
the other cadence choices. Preserve idle/zero-request intervals.
Days 1–4 train/bootstrap only the load predictor: they do not warm engine KV
caches or count toward GPU-hours, goodput, or evaluation requests. Start all
three evaluation caches cold. Each Planner candidate receives a fresh
predictor instance bootstrapped with its matching-cadence historical data.

## Execution boundary and review checks

The YAML files are valid public recommendation **search-space declarations**;
they are not a complete portable replacement for the experiment runner.
**Do not launch them with plain `aisimulate recommend` and treat that as the
specified protocol.** [experiment-contract.yaml](experiment-contract.yaml)
records the additional execution requirements:

1. Preserve the local baseline's fixed-day scoring and precision/canonical
   lowering bridge. The pinned stock Dynamo factory does not advertise the
   two precision controls even though its lowering/runtime implements them.
2. During Planner `compile_recommendation`, replace only its
   `RecommendationAdapterContext.sweep.workload` trace with the four-day
   history. The native adapter otherwise selects its predictor from the
   evaluation workload, which would leak day 5 into predictor selection.
3. During Planner hook materialization, add
   `planner_config.load_predictor_warmup_trace` pointing to the history trace.
   The native runtime supports it; the public Planner recommendation schema
   currently does not expose it. Do not add an unrecognized YAML field or
   mistake predictor selection for per-candidate warmup.
4. Keep the simulated GPU ceiling independent of CPU-host memory limits;
   require full cohort completion; retain native rejection/cache/timeout
   receipts. The 7,200-second deadline applies to native worker waves and can
   replace the pool; it does not guarantee every candidate finishes.

These integration requirements are explicit before launch. This branch
designs the spaces and publishes the environment; it does not claim that the
three full campaigns or this execution bridge have been run or implemented.

Run the lightweight checker **inside the pinned environment**:

```bash
python examples/sweeper/nvopt-ragpulse/validate_design.py
```

It validates core/adapter schemas, default preservation, domain extensions,
cross-scenario nesting, and predictor cadence coverage. It never trains a
predictor, builds an estimator, or launches a simulation. Runtime smoke-test
evidence belongs to the separately published container manifest.

Remote resource starting point: one study per CPU pod, 8 requested / 16
limited CPU cores, 112 GiB requested / 128 GiB limited memory, eight evaluations,
96 GB execution budget, disk-backed temporary/output volumes, and **zero
physical GPUs**. Mount data read-only under `/data/ragpulse`. The source bundle
and result volumes stay separate from the environment image.

For each trial record the actual engine/Router/Planner configuration, native
acceptance, simulation wall time, optimizer timing, goodput/GPU and SLA
coverage. Plot cumulative best against elapsed wall time together with trial
durations. For final winners also plot hourly offered traffic, goodput,
latencies, allocated GPUs, cache reuse and Planner scale decisions; separately
replay the selected configuration before calling it validated.
