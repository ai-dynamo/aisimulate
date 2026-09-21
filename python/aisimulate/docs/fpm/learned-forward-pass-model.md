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

`wall_time` means different things per backend. The vLLM producer
(`InstrumentedScheduler`) measures each step from `schedule()` to
`update_from_output()` on the host clock, one value per step. The SGLang
producer reads a CUDA-event `DeviceTimer` around the forward, but it
accumulates every *finished* interval at emission time and drops the record
when nothing finished. With the overlap scheduler on, a step whose result
processing does long host work (the last chunk of a request in
disaggregated prefill hands the KV cache to the decode worker) lets the next
forward finish first, so that record carries two forwards and the next record
is lost (V4.1-Flash prefill: 18% of batches missing, a 2× bimodal step time,
22% MAPE). **Collect SGLang FPM with `--disable-overlap-schedule`**; the
non-overlap loop emits exactly one interval per batch and the measured GPU
time is the same quantity a production overlap deployment spends per forward.
The same requirement holds for the SGLang simulator's own collection hook.

### Per-request fields

Stock Dynamo FPM v1 (both backends) carries aggregates only. The per-request
presets (`sglang18`, `hisim`) need two additive, aligned lists in
`scheduled_requests`, one entry per scheduled request, prefill requests first:

```json
"extend_lengths":  [16384, 1, 1],      // tokens computed for the request in this step
"past_kv_lengths": [32768, 7100, 950]  // KV tokens already present before the step
```

Neither producer emits them natively yet. Two ways to add them without
touching the engine code bases:

**Runtime hooks (recommended, ships with this package).**
`aisimulate_core.fpm_hooks` patches the producers in memory when their
modules are imported inside the engine process, including SGLang's spawned
scheduler subprocesses:

```bash
# in the engine container (both backends), before launching the worker
export PYTHONPATH=$(python -c 'import aisimulate_core.fpm_hooks as h; print(h.hook_path())'):$PYTHONPATH
python -m dynamo.sglang ...        # or: python -m dynamo.vllm ...
# or, equivalently
python -m aisimulate_core.fpm_hooks dynamo.sglang -- ...
```

The hook is a no-op when a producer already carries the fields, and logs and
skips (aggregate-only FPM) when the engine internals it wraps are missing.

**Source patches (upstream proposals).** The same change as unified diffs
against ai-dynamo/dynamo and sgl-project/sglang, see
[`patches/`](patches/README.md). What they add (version stays 1, additive
fields):

- vLLM: `dynamo/common/forward_pass_metrics.py` (two optional list fields on
  `ScheduledRequestMetrics`) and `dynamo/vllm/instrumented_scheduler.py`
  (`_extract_scheduled` fills them from `num_computed_tokens` /
  `num_new_tokens`).
- SGLang: `sglang/srt/observability/forward_pass_metrics.py` (same two fields)
  and `metrics_reporter._build_scheduled_request_metrics` (values from the
  schedule-time `batch.extend_lens` / `batch.prefix_lens`; the per-request
  attributes are already reset when metrics are emitted). Mixed batches append
  the decode requests as `(1, seqlen)`.

Streams from unpatched producers train with `--features v1` only.

## 2. Train

```bash
uv sync --project python/aisimulate --extra learned   # installs scikit-learn
python -m aisimulate_core.sdk.fpm_learned train \
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
  The per-request groups need the `extend_lengths` / `past_kv_lengths`
  producer fields described above. Lists must be aligned and cover every
  scheduled request; the Rust validator rejects partial vectors, and an
  artifact that uses `req_*` / `slot*` features **refuses** (error, not a
  constant prediction) iterations that carry no consistent lists, because
  the trees never saw those features missing during training.
  `--features` takes a preset name or a comma-separated list; the default is
  `sglang18` (the 18 per-request features). Aggregate-only streams must opt in
  with `--features v1`.
- The model is a scikit-learn `HistGradientBoostingRegressor` on
  `log(wall_ms)` per store, exported to plain JSON
  (`schema = aic_fpm_learned_forward_perf`, version 1). No pickle is involved
  and inference does not need scikit-learn. Early stopping is off unless
  `--early-stopping` is passed (sklearn would otherwise enable it above 10k
  rows with a hidden 10 % validation split); `metadata.trees_fitted` records
  the trees per store.

The command prints the train-set fit and the holdout accuracy per store
(MAPE, median and p95 APE), scored through the compiled Rust model, which is
the only prediction path. `evaluate --model ... --fpm ...` scores an artifact
against any other FPM files.

## 3. Use it

```python
from aisimulate_core.sdk import RustForwardPassPerfModel

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
| Qwen3-32B-FP8, 262k YaRN ctx | c16/32/64 → c24/48/96 | 2.77% (1.02%) over 122k steps | 1.64% (0.55%) over 13k steps |
| DeepSeek-V4-Flash, 262k ctx | c16/32/64/128 → c24/48/96/64 | 3.29% (1.01%) over 457k steps | 4.43% (4.32%) over 32k steps |

All accuracy figures in this section are from the trainer defaults (400 trees,
learning rate 0.05, 31 leaves, early stopping off).

Leave-one-tier-out inside a single run lands at 1–4.5% for decode and
1.2–3.7% for prefill on both deployments; the largest errors are the tiers
outside the trained concurrency range (extrapolation).

The DeepSeek-V4-Flash prefill number is a boot-to-boot offset, not scatter:
predicted/observed sits at about 1.04 median with a narrow p10–p90 band, i.e.
the test boot's prefill engine ran ~4% faster than the training boot. The
online correction grid on top of the learned model (`tune_with_fpms`) is
designed to absorb exactly this kind of constant factor.

On aggregates-only features (`v1`) the same experiments land within about
half a percentage point of `sglang18` on both deployments; the per-request
features matter most where batches are large and heterogeneous.

### DeepSeek-V4.1-Flash on the Dynamo SGLang runtime

Same traffic and topology, SGLang runtime `1.6.0-deepseek-v4.1-flash-dev.1`
(TP4/EP4, page size 256, mooncake disaggregation, 262k context), collected
with `--disable-overlap-schedule` (see §1), `sglang18` features, train
c16/32/64/128 (seed 42) → test c24/48/96 (seed 7), 2026-09-20:

| Engine | Test tier | steps | MAPE (median) | p95 |
| --- | --- | --- | --- | --- |
| decode | c24 | 108k | 1.85% (1.41%) | 4.4% |
| decode | c48 | 89k | 1.99% (1.53%) | 5.0% |
| decode | all | 197k | 1.92% (1.47%) | 4.7% |
| prefill | c24 | 973 | 2.67% (1.41%) | 8.1% |
| prefill | c48 | 1,935 | 2.56% (1.33%) | 7.8% |
| prefill | c96 | 765 | 0.83% (0.58%) | 2.0% |
| prefill | all | 3,673 | 2.23% (1.00%) | 7.3% |

Leave-one-tier-out inside the training run: decode 1.5% / 1.5% / 4.9%
(c16 / c32 / c64), prefill 0.8–2.9%. `hisim` and `v1` land within 0.3 pp of
`sglang18` on both engines. The test boot lost its c96 decode tier and the
train boot its c128 prefill tier to a DeepGEMM prefill OOM at
`mem-fraction-static 0.8`; lower it for long-context V4.1 captures.

Why the overlap flag matters: an earlier pair collected with the overlap
scheduler on gave 22% prefill MAPE (11% median) with a 2× bimodal step time
for identical batch shapes and 18% of prefill batches missing from the FPM
stream, the accumulator effect described in §1. With the flag, drops are 0%
and identical-shape step times agree to 1%.

A model trained on overlap-off data still predicts overlap-on decode steps
(the GPU forward is the same quantity): scored against the overlap-on test
boot it gives 5.5% MAPE (5.6% median) with a p95 of 8.4%, i.e. a near
constant offset that the online correction grid absorbs. The overlap-on
prefill records themselves are unusable as truth because of the accumulator
merge, so no prefill cross-mode number is quoted.

### Workload coverage: AgentX and ShareGPT (chatbot) on DeepSeek-V4.1-Flash

Same V4.1-Flash SGLang deployment, overlap off, `sglang18` features. Two
AgentX boots (c16/32/64/128 seed 42, c24/48/96 seed 7) and two ShareGPT
chatbot boots (aiperf `--public-dataset sharegpt`, c32/64/128/256 seed 42,
c48/96/192 seed 7), 2026-09-20/21. The two workloads occupy different parts
of the input space: AgentX prefill is long-context with heavy prefix reuse
(extend up to ~16k tokens, past KV in the hundreds of thousands) and its
decode batches stay at or below 64; ShareGPT prefill is short (ISL median
~530, almost no prefix hits) and its decode batch equals the concurrency,
up to 256.

Pooled 60/40 split with no shared step: every (boot, tier) window is cut
into five consecutive time blocks, blocks 1/2/4 train and 3/5 test, so both
sides see every boot and tier. Step-weighted MAPE (median / p95):

| Training data | Test data | decode steps | decode | prefill steps | prefill |
| --- | --- | --- | --- | --- | --- |
| AgentX only | AgentX 40% | 200k | 1.70% (1.32% / 4.2%) | 3.5k | 2.52% (1.45% / 8.0%) |
| ShareGPT only | ShareGPT 40% | 97k | 1.85% (1.30% / 5.4%) | 7.2k | 2.58% (1.82% / 6.3%) |
| AgentX + ShareGPT | both, 40% | 297k | 1.69% (1.28% / 4.4%) | 10.7k | 2.45% (1.75% / 6.2%) |
| AgentX + ShareGPT, equal steps per workload | both, 40% | 197k | 1.71% (1.29% / 4.5%) | 7.1k | 2.54% (1.81% / 6.7%) |

Within the pooled model the AgentX tiers land at 1.5–2.0% decode /
0.7–2.8% prefill and the ShareGPT tiers at 1.2–2.1% decode / 1.7–3.4%
prefill, i.e. pooling costs neither workload anything against its own
single-workload model. Splitting by whole boot instead of by time block
(one boot per workload held out entirely) gives 1.46% / 2.52% for the
pooled model.

Training on one workload and testing on the other shows why both belong
in a release model:

| Train → test | decode | prefill |
| --- | --- | --- |
| AgentX → ShareGPT | 17.5% (c32 2.2%, c64 11%, c128 30%, c256 45%) | 4.7% |
| ShareGPT → AgentX | 4.0% | 26.1% (c16–c64 20–24%, c96/c128 47%) |

The decode error grows monotonically with batch sizes the AgentX model
never saw; the prefill error is the ShareGPT model extrapolating to prefix
lengths it never saw. Each direction is fine where the training data covers
the test inputs (AgentX → ShareGPT prefill 4.7%, ShareGPT → AgentX decode
4.0%). A release artifact should therefore be trained on the union of the
workloads it is expected to simulate, and the pooled numbers above are what
to expect from it.

## Limitations

- Learned mode is SDK-only in this release: `best_available(config)` and
  `EstimationMode` have no learned option yet, so the simulator's replay,
  sweeper and planner entry points still build native or regression models.
  Wiring a `learned_artifact` into `ForwardPassPerfModelConfig` is a
  follow-up.

- Stock Dynamo FPM v1 carries aggregates only (counts, sums, variances). The
  per-request `extend_lengths` / `past_kv_lengths` lists are an additive
  extension of the scheduler's `_extract_scheduled`; without them only the
  `v1` preset carries signal.
- The model extrapolates poorly outside the collected batch/KV range. Check
  the holdout report and cover the intended operating range when collecting.
- One artifact is one deployment identity (model, engine version, parallelism,
  CUDA-graph list, scheduler settings). Record that identity in `metadata`
  and retrain when it changes.
