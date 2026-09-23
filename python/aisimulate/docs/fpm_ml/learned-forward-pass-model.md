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

Design rationale, model details, evaluation method and the full result tables are in
[`design.md`](design.md).

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
`scheduled_requests`, one entry per scheduled request in any order (the
consumer sorts; only the alignment of the two lists matters):

```json
"extend_lengths":  [16384, 1, 1],      // tokens computed for the request in this step
"past_kv_lengths": [32768, 7100, 950]  // KV tokens already present before the step
```

Neither producer emits them natively yet. `aisimulate_core.fpm_hooks` adds
them at runtime, in memory, when the producer modules are imported inside the
engine process (including SGLang's spawned scheduler subprocesses); no engine
source is modified and any stock Dynamo image produces the fields:

```bash
# in the engine container (both backends), before launching the worker
export PYTHONPATH=$(python -c 'import aisimulate_core.fpm_hooks as h; print(h.hook_path())'):$PYTHONPATH
python -m dynamo.sglang ...        # or: python -m dynamo.vllm ...
# or, equivalently
python -m aisimulate_core.fpm_hooks dynamo.sglang -- ...
```

What the hooks add (FPM `version` stays 1, the two fields are additive and
ignored by aggregate-only consumers):

- vLLM path: `InstrumentedScheduler._extract_scheduled` fills the lists from the
  `SchedulerOutput` (`num_scheduled_tokens` → extend, `num_computed_tokens` → past).
- SGLang: `_build_scheduled_request_metrics` fills them from the schedule-time
  `batch.extend_lens` / `batch.prefix_lens` for prefill batches and from
  `batch.seq_lens_cpu` for decode batches (the per-request attributes are already
  reset when metrics are emitted).

The hook is a no-op when a producer already carries the fields, and logs and
skips (aggregate-only FPM) when the engine internals it wraps are missing.
Streams recorded without the hooks train with `--features v1` only.

Record the stream with Dynamo's own relay sink (`DYN_FPM_TRACE=1
DYN_FPM_OUTPUT_PATH=...`, files `fpm-relay*.jsonl.gz`), which writes the raw
payload. A consumer that decodes the msgpack with the *stock* typed
`ForwardPassMetrics` struct silently drops the two extra keys: a custom ZMQ
subscriber written that way produced aggregate-only files while the relay
files next to it carried the lists, and a model trained on those files landed
at 12–19% decode error instead of 2–3%. Check one record for `extend_lengths`
before training.

## 2. Train

Where each step runs: collection happens inside the engine container (the
Dynamo `sglang-runtime` / `vllm-runtime` image, nothing extra installed) and
only writes FPM `jsonl.gz` files. Training reads those files on any CPU
machine with the `aisimulate[learned]` extra installed (scikit-learn is used
only here; a login node or laptop is fine: 2.7M decode steps fit in ~25 s on
16 threads, loading the gzip stream takes longer than the fit). The output is
a plain JSON artifact that the simulator loads through the Rust model with no
Python ML dependency.

Training environment, pick one:

- **Plain Python on the host** (login node, workstation, laptop):
  `pip install "aisimulate[learned]"` or, from a checkout,
  `uv sync --project python/aisimulate --extra learned`. This is what the
  numbers in §4 were produced with (a venv on a cluster login node).
- **A container, when the site only allows containerised jobs** (Slurm with
  pyxis/enroot, Kubernetes): the Dynamo runtime images do **not** ship
  scikit-learn (only numpy/scipy), so either `pip install scikit-learn` inside
  the capture image, or use the NGC PyTorch image, which includes
  scikit-learn:

  ```bash
  srun -N1 --container-image=nvcr.io#nvidia/pytorch:25.08-py3 \
       --container-mounts=/path/to/traces:/traces,/path/to/aisimulate:/aisimulate \
       bash -c 'pip install -q /aisimulate/python/aisimulate[learned] && \
                python -m aisimulate_core.sdk.fpm_learned train --fpm /traces/decode/*.jsonl.gz \
                  --worker-type decode --join-ranks counter --out /traces/decode_learned.json'
  ```

  No GPU is used; request a CPU partition if the site has one.
- **Inside the capture container right after collection**: the same
  `pip install scikit-learn` + `train` command, so collection and training can
  be one Slurm job. The artifact is a small JSON file either way.

```bash
uv sync --project python/aisimulate --extra learned   # installs scikit-learn
# or: pip install "aisimulate[learned]"
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
    `(present, past, extend)` (preset `hisim`);
  - `core4`: the four of the 18 that an ablation found sufficient
    (`req_batch_size`, `req_sum_extend`, `req_sum_past`, `req_sum_attn_flops`).
    Same accuracy as `sglang18` when train and test share the workload mix
    (2.02 % vs 2.05 % decode, 1.97 % vs 1.98 % prefill on the pooled GB300
    runs of §4) and for decode across workloads, but prefill extrapolation
    degrades (LongBench → AgentX 6.5 → 15 %), so it is not the default. The
    full tables are in `design.md` §3.1.
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

The command prints the train-set fit and a random-holdout report per store
(MAPE, median and p95 APE), scored through the compiled Rust model, which is
the only prediction path. The random holdout is a smoke check only:
consecutive steps are correlated, so for a real accuracy number train on one
set of files and score another with `evaluate --model ... --fpm ...` (for
example a capture run with a different seed and different concurrency tiers,
as in §4).

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
Train and test are two independent capture runs (a capture run = one fresh Dynamo deployment on its own nodes, one aiperf seed, every concurrency tier once) with different seeds and different
concurrency tiers (2026-09-18, dlcluster).

| Deployment | Train tiers → test tiers | decode | prefill |
| --- | --- | --- | --- |
| Qwen3-32B-FP8, 262k YaRN ctx | c16/32/64 → c24/48/96 | 2.77% (1.02%) over 122k steps | 1.64% (0.55%) over 13k steps |
| DeepSeek-V4-Flash, 262k ctx | c16/32/64/128 → c24/48/96/64 | 3.29% (1.01%) over 457k steps | 4.43% (4.32%) over 32k steps |

All accuracy figures in this section use `HistGradientBoostingRegressor` with 400
trees, learning rate 0.05, 31 leaves, 5 samples per leaf and scikit-learn's default
`early_stopping='auto'` (prefill stores stop early, decode stores run all 400 trees).
The CLI defaults differ: `--max-iter 600` and early stopping off.

Leave-one-tier-out inside a single run lands at 1–4.5% for decode and
1.2–3.7% for prefill on both deployments; the largest errors are the tiers
outside the trained concurrency range (extrapolation).

The DeepSeek-V4-Flash prefill number is a run-to-run offset, not scatter:
predicted/observed sits at about 1.04 median with a narrow p10–p90 band, i.e.
the test run's prefill engine ran ~4% faster than the training run. The
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
`sglang18` on both engines. The test run lost its c96 decode tier and the
train run its c128 prefill tier to a DeepGEMM prefill OOM at
`mem-fraction-static 0.8`; lower it for long-context V4.1 captures.

Why the overlap flag matters: an earlier pair collected with the overlap
scheduler on gave 22% prefill MAPE (11% median) with a 2× bimodal step time
for identical batch shapes and 18% of prefill batches missing from the FPM
stream, the accumulator effect described in §1. With the flag, drops are 0%
and identical-shape step times agree to 1%.

A model trained on overlap-off data still predicts overlap-on decode steps
(the GPU forward is the same quantity): scored against the overlap-on test
run it gives 5.5% MAPE (5.6% median) with a p95 of 8.4%, i.e. a near
constant offset that the online correction grid absorbs. The overlap-on
prefill records themselves are unusable as truth because of the accumulator
merge, so no prefill cross-mode number is quoted.

### Workload coverage: AgentX, ShareGPT and LongBench on DeepSeek-V4.1-Flash

Three workloads on the same deployment (DeepSeek-V4.1-Flash, Dynamo SGLang runtime, one 4×GB300 node per role, TP4/EP4, `--disable-overlap-schedule`, per-request lists via `aisimulate_core.fpm_hooks`):

- **AgentX**: aiperf `inferencex-agentx-mvp`, SemiAnalysis CC traces (agent sessions, 262k context).
- **ShareGPT**: aiperf `--public-dataset sharegpt` (chatbot conversations).
- **LongBench**: single-turn long-document QA built from LongBench-v2 (503 real documents, 8k–2M words). Each prompt = a unique preamble line (so no cross-request prefix-cache hit is possible) + a document window of 8k–200k tokens on a fixed grid + the question and choices; `max_tokens` 1024; 4000 prompts in 8 files, one file per concurrency tier so no prompt is ever sent twice.

A *capture run* is one fresh deployment on its own nodes with one aiperf seed, every concurrency tier run once for 1200 s. Input distributions as the engine saw them, from the FPM per-request lists of one capture run each (p5 / p50 / p95 / max):

**Prefill**

| Quantity | AgentX | ShareGPT | LongBench |
| --- | --- | --- | --- |
| requests per step | 1 / 1 / 2 / 11 | 1 / 3 / 9 / 63 | 1 / 1 / 2 / 3 |
| new tokens per request this step (extend, 16k chunking) | 333 / 3,743 / 16,384 / 16,384 | 40 / 326 / 796 / 1,934 | 1,719 / 16,384 / 16,384 / 16,384 |
| KV already present per request (past) | 0 / 70,144 / 194,816 / 252,672 | 0 / 0 / 1,280 / 1,792 | 0 / 19,200 / 125,952 / 200,192 |
| request ISL | 6,656 / 82,949 / 213,369 / 254,199 | 40 / 531 / 1,642 / 2,134 | 1,536 / 14,336 / 128,151 / 200,624 |
| cross-request prefix-cache hit | 54% of requests; hit size p50 31k, p95 176k | ~none | none (by construction) |

**Decode**

| Quantity | AgentX | ShareGPT | LongBench |
| --- | --- | --- | --- |
| requests per step (batch) | 1 / 7 / 30 / 43 | 26 / 58 / 243 / 256 | 2 / 5 / 10 / 26 |
| context per request | 36,268 / 115,407 / 233,765 / 254,469 | 122 / 779 / 1,879 / 2,886 | 8,682 / 32,496 / 160,469 / 201,648 |
| KV tokens per step | 129k / 909k / 3.8M / 4.9M | 17k / 49k / 215k / 270k | 128k / 288k / 400k / 511k |
| output length | model-terminated (hundreds to thousands) | model-terminated (tens to hundreds) | capped at 1024 |

**Data volume**

| Workload | capture runs (seeds) | prefill steps | decode steps |
| --- | --- | --- | --- |
| AgentX | 4 (42 / 7 / 11 / 23) | 20.9k | 1,150k |
| ShareGPT | 4 (42 / 7 / 11 / 23) | 38.2k | 520k |
| LongBench | 2 (42 / 7) | 20.2k | 1,062k |

Accuracy, step-weighted MAPE per raw iteration, rows = training data, columns = test set. Same-workload cells (diagonal and the pooled row): the workload's runs are cut into five consecutive time blocks per run and tier, blocks 1/2/4 train, 3/5 test, no shared step, so the test set is 40% of that workload. Cross-workload cells: trained on all runs of the row workload, tested on all runs of the column workload.

**Decode**

| Training data \ test set | AgentX | ShareGPT | LongBench |
| --- | --- | --- | --- |
| AgentX only | 2.34% | 14.14% | 3.03% |
| ShareGPT only | 3.88% | 2.26% | 3.43% |
| LongBench only | 2.91% | 22.45% | 1.84% |
| all three pooled | 2.21% | 2.10% | 1.84% |

**Prefill**

| Training data \ test set | AgentX | ShareGPT | LongBench |
| --- | --- | --- | --- |
| AgentX only | 2.24% | 3.49% | 0.88% |
| ShareGPT only | 26.23% | 2.55% | 43.17% |
| LongBench only | 6.52% | 29.42% | 0.62% |
| all three pooled | 2.28% | 2.56% | 0.57% |

Each workload lacks a region another one has (ShareGPT never sees 16k chunks or large past; AgentX and LongBench never see decode batches above ~40; LongBench has no short requests and no prefix hits), so any single-workload model extrapolates badly on at least one other workload. The pooled model matches or beats every single-workload model on its own workload. None of the three covers large decode batches at long context; that region remains untested.

### The same workloads on the vLLM backend

DeepSeek-V4-Flash on `vllm-runtime:1.4.0` (V4.1 has no vLLM release yet), same GB300
nodes, two capture runs per workload, per-request lists from the `_dynamo_vllm` hook,
recorded with the Dynamo relay sink. Step-weighted MAPE, same layout as above:

Decode

| Training data \ test set | AgentX | ShareGPT | LongBench |
| --- | --- | --- | --- |
| AgentX only | 3.78% | 19.05% | 1.85% |
| ShareGPT only | 4.20% | 1.52% | 2.44% |
| LongBench only | 3.95% | 28.86% | 1.71% |
| all three pooled | 3.40% | 1.54% | 1.70% |

Prefill

| Training data \ test set | AgentX | ShareGPT | LongBench |
| --- | --- | --- | --- |
| AgentX only | 3.11% | 7.99% | 5.85% |
| ShareGPT only | 19.22% | 2.99% | 24.03% |
| LongBench only | 5.06% | 7.46% | 2.43% |
| all three pooled | 3.34% | 3.01% | 3.31% |

vLLM `wall_time` is host time between `schedule()` calls (includes scheduler
overhead; AgentX decode 3.8% vs 2.3% on SGLang), and its 4k prefill chunks make
each prefill step shorter (LongBench prefill 2.4% vs 0.6%). The pooled model
again matches every single-workload model on its own workload.

## 5. Training and inference time

Measured with this branch on an AI Hub login node (NVIDIA Grace, Arm Neoverse V2, 96
cores, 370 GB RAM, aarch64; training with `OMP_NUM_THREADS=16`, inference on one core),
DeepSeek-V4.1-Flash SGLang AgentX captures: the seed-42 run trains, the seed-7 run is
predicted. Inference does not use scikit-learn; it is the Rust tree walk in `learned.rs`,
called here once per step through the PyO3 binding, which is how the simulator uses it.

| Step | Data | Time |
| --- | --- | --- |
| load + featurize (gzip JSON lines) | 302,603 decode steps | 4.4 s |
| train decode model (scikit-learn HGB, 400 trees, early stopping off) | 302,603 steps | 54 s |
| train prefill model (same settings) | 5,245 steps | 20 s |
| predict decode, Rust, one call per step (through the Python wrapper) | 196,585 steps | 3.78 s (19 µs / step, mean batch 10, max 35) |
| predict prefill, Rust, one call per step (through the Python wrapper) | 3,803 steps | 0.08 s (22 µs / step, mean batch 1.2) |

The GBDT itself, measured as a Rust call on a prebuilt struct on one Grace core
(release build, `taskset` pinned):

| step | GBDT alone (Rust) | through the Python wrapper |
| --- | --- | --- |
| decode, batch 1 | 3.4 µs | 11 µs |
| decode, batch 16 | 4.2 µs | 15 µs |
| decode, batch 256 | 5.6 µs | 59 µs |
| prefill, 1 × 512 tokens, no prefix | 4.7 µs | 12 µs |
| prefill, 1 × 4096 tokens, no prefix | 6.3 µs | 14 µs |
| prefill, 1 × 4096 tokens, 64k prefix | 4.5 µs | 12 µs |
| prefill, 4 × 4096 tokens, 16k prefix each | 4.8 µs | 13 µs |

The tree walk is a few microseconds and the 18-feature build is one pass over the
per-request lists at about 8 ns per request, so prefill is flat and decode grows mildly.
The per-step numbers in the first table and the Python column here include the wrapper
serialising the FPM dict to JSON and parsing it in Rust on every call; the simulator calls
the Rust path directly. Training the pooled ten-run set (2.7M decode steps) took 25 s on
a dlcluster login node (AMD EPYC 7232P, 8 cores / 16 threads, x86_64); loading the
compressed stream (204 s) dominates there. Artifacts are 100–450 KB of JSON.

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
