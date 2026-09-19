<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Learned forward-pass model from real-traffic FPM telemetry

The learned forward-pass model predicts one engine iteration's wall time from
the scheduled batch composition, using a tree ensemble trained offline on
`ForwardPassMetrics` (FPM) records that a real deployment emitted while
serving real traffic. It is the third `ForwardPassPerfModel` mode next to the
native AIC estimator and the online linear regression:

| Mode | Base estimate | Learns from | Needs |
| --- | --- | --- | --- |
| `from_native` | Compiled AIC engine | Online correction grid | Model supported by AIC, perf database |
| `from_regression` | Two-feature linear fit | Online, per workload store | Nothing (cold start) |
| `from_learned` | Offline tree ensemble | Offline FPM stream, plus the same online correction grid | An FPM stream of the deployment |

The learned mode does not use the anchor-library collector or a kvwarm
self-benchmark. Its only input is the telemetry the engine already produces.

## 1. Collect the FPM stream

Run the deployment you want to model with Dynamo's FPM trace enabled on every
worker:

```bash
export DYN_FPM_TRACE=1
export DYN_FPM_MODE=full          # every iteration, not sampled
export DYN_FPM_OUTPUT_PATH=/results/fpm   # one jsonl.gz per producer
```

Every scheduler iteration writes one record per attention-DP rank with the
scheduled counts (`num_*_requests`, `sum_prefill_tokens`,
`sum_prefill_kv_tokens`, `sum_decode_kv_tokens`, variances) and the observed
`wall_time` (schedule to update-from-output, in seconds). Drive the deployment
with the traffic you want the model to cover: production traffic, a trace
replay (for example AA-RWLT over Claude Code traces), or an agentic generator
(AgentX). Sweep the concurrency range you intend to simulate; a learned model
is only reliable inside the batch/KV space it has seen.

The trainer also accepts the event-plane sink format (one flat FPM JSON per
line, as written by an `FpmEventSubscriber` sink) and the Dynamo trace envelope
(`{"event": {"fpm": {...}}}`).

Keep the engine posture fixed while collecting: async scheduling, CUDA-graph
capture list, and max batched tokens all change `wall_time` and belong to the
model's identity.

## 2. Train

```bash
uv sync --project python/aisimulate --extra learned   # installs scikit-learn
python -m aiconfigurator_core.sdk.fpm_learned train \
  --fpm decode_worker/*.jsonl.gz \
  --worker-type decode \
  --join-ranks counter \
  --holdout-frac 0.2 \
  --out models/decode_learned.json
```

- `--worker-type` binds the artifact to `prefill`, `decode`, or `aggregated`.
  Aggregated engines get up to four stores (`pure_decode`,
  `contains_locally_mixed`, `cross_rank_aggregated`, `pure_prefill`), selected
  by the same rule the regression fallback uses.
- `--join-ranks counter` groups per-rank records into one iteration by
  `(worker_id, counter_id)` for attention-DP engines; the label is the maximum
  positive `wall_time` across ranks. Use `none` for single-rank engines.
- Features are named quantities computed identically in `fpm_learned.py` and
  `learned.rs` (135 names, three groups):
  - 21 aggregate features from the FPM v1 counts/sums/variances (preset `v1`);
  - the 18 per-request features of the SGLang simulator `MLTimePredictor`
    (`req_*`: sum/max/min extend and past, cross terms, attention FLOPs proxy,
    `is_decode`/`is_prefill`; preset `sglang18`);
  - 32 HiSim-style request slots sorted by past KV descending, each
    `(present, past, extend)` (preset `hisim`).
  The per-request groups need the producer to emit `extend_lengths` /
  `past_kv_lengths` in `scheduled_requests`. Both Dynamo backends have an
  additive patch for it: vLLM in `InstrumentedScheduler._extract_scheduled`,
  SGLang in `metrics_reporter._build_scheduled_request_metrics` (taken from
  the schedule-time `batch.extend_lens` / `batch.prefix_lens`, because the
  per-request attributes are already reset when the metrics are emitted). On
  aggregates-only streams they are NaN and the trees route them through
  `missing_left`.
  `--features` takes a preset name or a comma-separated list; the default is
  `sglang18` (the 18 per-request features). Aggregate-only streams must opt in
  with `--features v1`.
- The model is a scikit-learn `HistGradientBoostingRegressor` on
  `log(wall_ms)` per store, exported to plain JSON
  (`schema = aic_fpm_learned_forward_perf`, version 1). No pickle is involved
  and inference does not need scikit-learn.

The command prints the train-set fit and the holdout accuracy per store
(MAPE, median and p95 APE). `evaluate --model ... --fpm ...` scores an
artifact against any other FPM files; add `--rust` to score through the
compiled model and confirm the exported trees match.

## 3. Use it

```python
from aiconfigurator_core.sdk import RustForwardPassPerfModel

model = RustForwardPassPerfModel.from_learned("models/decode_learned.json")
ms = model.estimate_forward_pass_time_ms([fpm_rank0, fpm_rank1])  # one iteration
model.tune_with_fpms([[observed_rank0, observed_rank1]])           # optional online correction
```

`estimate_forward_pass_time_ms` returns the learned prediction multiplied by
the native-style online correction factor (median observed/predicted ratio in
the workload region, bounded by `min_faster_correction_factor` /
`max_slower_correction_factor`). Iterations whose workload kind has no trained
store return `None`; empty iterations return `0.0`. `diagnostics()["source"]`
reports `learned` or `learned_with_correction`.

Consumers that already hold a `RustForwardPassPerfModel` (the Dynamo planner
engine-query layer, the mocker's AIC callback) need no code change beyond
choosing the constructor.

## 4. Accuracy reference

Per raw iteration, APE = |predicted − observed| / observed, step-weighted MAPE
(median in parentheses). Real AgentX traffic (SemiAnalysis CC traces,
`inferencex-agentx-mvp` scenario) on Dynamo 1P1D, one 4×GB300 node per role,
TP4, vLLM runtime 1.4.0, `DYN_FPM_TRACE` full mode with per-request lists,
features = the 18 per-request features (`sglang18`), one HGB per engine.
Train and test are two independent boots with different seeds and different
concurrency tiers (2026-09-18, dlcluster).

| Deployment | Train tiers → test tiers | decode | prefill |
| --- | --- | --- | --- |
| Qwen3-32B-FP8, 262k YaRN ctx | c16/32/64 → c24/48/96 | 2.41% (0.97%) over 122k steps | 1.64% (0.55%) over 13k steps |
| DeepSeek-V4-Flash, 262k ctx | c16/32/64/128 → c24/48/96/64 | 3.14% (0.97%) over 457k steps | 4.46% (4.48%) over 32k steps |

Leave-one-tier-out inside a single run lands at 1–4.5% for decode and
1.2–3.7% for prefill on both deployments; the largest errors are the tiers
outside the trained concurrency range (extrapolation).

The DeepSeek-V4-Flash prefill number is a boot-to-boot offset, not scatter:
predicted/observed sits at 1.044 median with a 1.002–1.058 p10–p90 band, i.e.
the test boot's prefill engine ran ~4% faster than the training boot. The
online correction grid on top of the learned model (`tune_with_fpms`) is
designed to absorb exactly this kind of constant factor.

On aggregates-only features (`v1`) the same experiments give 3.04% / 2.04%
(Qwen) and comparable Flash numbers; the per-request features matter most
where batches are large and heterogeneous.

### DeepSeek-V4.1-Flash on the Dynamo SGLang runtime

Same traffic and topology, SGLang runtime `1.6.0-deepseek-v4.1-flash-dev.1`
(TP4/EP4, page size 256, mooncake disaggregation, 262k context),
2026-09-19, `sglang18` features, train c16/32/64/128 → test c24/48/96:

| Test tier | decode steps | decode MAPE (median) | p95 |
| --- | --- | --- | --- |
| c24 | 140k | 0.65% (0.47%) | 1.8% |
| c48 | 119k | 0.94% (0.70%) | 2.5% |
| c96 | 92k | 1.66% (1.16%) | 5.0% |
| all | 350k | 1.01% (0.66%) | 3.0% |

The SGLang decode engine is predicted at least as well as the vLLM ones. A
second pair whose training boot lost the c64/c128 tiers (prefill-engine OOM)
gives 0.67% / 1.72% on c24 / c48 but 13.7% on c96, three times the trained
maximum: cover the intended concurrency range when collecting, the model
does not extrapolate.

The SGLang prefill engine is a different story: 22% MAPE (11% median, p95
65%) on 5.4k steps, and the same 21% on a random split of a single run, so
it is not a train/test mismatch. The step time is bimodal for identical batch
shapes: grouping single-request steps by (extend, past) bucket, the
within-bucket p90/p10 spread averages 0.69× the median, with a ~200 ms floor
that sometimes doubles regardless of the extend length. Only the full 16k
chunks (the bulk of the prefill time) are tight (1–11% spread). Summed over a
tier the prediction is within 5% of the observed prefill time
(time-weighted MAE 21%). This is a property of the disaggregated SGLang
prefill loop's `wall_time`, not of the feature set: `v1`, `sglang18` and
`hisim` all land at 22%.

## Limitations

- Stock Dynamo FPM v1 carries aggregates only (counts, sums, variances). The
  per-request `extend_lengths` / `past_kv_lengths` lists are an additive
  extension of the scheduler's `_extract_scheduled`; without them only the
  `v1` preset carries signal.
- The model extrapolates poorly outside the collected batch/KV range. Check
  the holdout report and cover the intended operating range when collecting.
- One artifact is one deployment identity (model, engine version, parallelism,
  CUDA-graph list, scheduler settings). Record that identity in `metadata`
  and retrain when it changes.
