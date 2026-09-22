# Learned forward-pass model: design

A gradient-boosted tree model that predicts the wall time of one engine forward pass
(one scheduler step of a prefill or decode worker) from the composition of the batch,
trained on `ForwardPassMetrics` (FPM) recorded from real traffic. It is a third variant
of `ForwardPassPerfModel` next to the native analytic model and the regression store,
and is exposed today through the SDK (`aisimulate_core.sdk.fpm_learned` for training,
`RustForwardPassPerfModel.from_learned` for inference).

Companion documents in this directory: [`learned-forward-pass-model.md`](learned-forward-pass-model.md)
(how to collect, train and use) and [`patches/`](patches/README.md) (proposed producer patches).
The existing FPM documentation under `../fpm/` is unchanged; everything about the learned model lives here.

## 1. Problem and design goals

The simulator needs, per scheduler step, the time the GPU forward will take for the
batch the scheduler just formed. The native model derives it from kernel-level
analytic formulas; the regression store interpolates a grid of measured operating
points. Both need per-model calibration effort and neither sees the batch at the
granularity the engine does.

Real deployments already emit exactly the needed observation: Dynamo's FPM stream
reports, for every step, the scheduled batch and the step's wall time. The learned
model turns that stream into a predictor with these goals:

1. **Per-request resolution.** The prediction must condition on each request's
   `(extend, past)` pair, not only on batch aggregates: two decode batches with the
   same total KV but different per-request context distributions do not take the
   same time, and a prefill chunk with 16k new tokens on 200k past is not a 16k chunk
   on 0 past.
2. **No engine source changes.** Stock Dynamo FPM v1 carries aggregates only. The
   per-request lists are added at runtime by import hooks that live in this repo
   (`aisimulate_core.fpm_hooks`), so any stock Dynamo image produces trainable data.
3. **Inference in Rust, microseconds per call.** The simulator calls the perf model
   once per simulated step; the model must be a pure function over a fixed feature
   vector, evaluated without Python.
4. **Refuse rather than degrade.** If an artifact needs per-request features and the
   input has none, the model errors with a pointer to the fix instead of returning a
   constant.
5. **Honest evaluation.** Accuracy is quoted per raw step on data the model never
   saw, with the split scheme stated, never on a random shuffle of correlated steps.

## 2. Data: what a training record is

One record = one FPM event from one engine worker:

| Field | Meaning |
| --- | --- |
| `worker_id`, `dp_rank`, `counter_id` | identity and step counter of the emitting worker |
| `wall_time` | seconds the step took (semantics below) |
| `scheduled_requests.num_prefill_requests`, `sum_prefill_tokens`, `sum_prefill_kv_tokens`, `var_prefill_length` | prefill aggregates |
| `scheduled_requests.num_decode_requests`, `sum_decode_kv_tokens`, `var_decode_kv_tokens` | decode aggregates |
| `scheduled_requests.extend_lengths[i]`, `past_kv_lengths[i]` | **per request**: new tokens this step, KV already present (added by the hooks) |

`wall_time` semantics differ by backend and matter for what is being learned:

- **vLLM** (`InstrumentedScheduler`): host clock between two `schedule()` calls, i.e.
  the engine step including scheduler overhead. Overlap is not an issue.
- **SGLang** (`SchedulerMetricsReporter` + `DeviceTimer`): CUDA events around the
  forward, accumulated and flushed when the batch result is processed. With the
  overlap scheduler on, the disaggregated prefill loop does KV hand-off work before
  the flush, so the next forward completes first and is merged into the current record
  (2× bimodal step times for identical batch shapes, 18% of records dropped, 22%
  prefill error). **All SGLang collection is done with `--disable-overlap-schedule`**,
  which makes each record exactly one forward. A model trained overlap-off still
  predicts overlap-on decode steps to within a near-constant 5.5% offset; overlap-on
  prefill records are not usable as truth.

A **capture run** is one fresh deployment on its own nodes, one aiperf seed, every
concurrency tier run once (20 min each). Runs are the unit of independence: steps
inside a run share machine state, warm-up and request sequence.

## 2a. Per-request fields and the runtime hooks that add them

### What stock FPM carries and what is missing

Stock Dynamo `ForwardPassMetrics` v1 describes a step by seven aggregates (counts,
token sums and variances for the prefill and decode sides). Two batches with identical
aggregates can differ a lot in cost: a decode batch of 8 requests all at 100k context
and one with seven at 10k plus one at 730k have the same `sum_decode_kv_tokens`; a
prefill step with one 16k chunk on 200k past and one with two 8k chunks on 0 past have
the same `sum_prefill_tokens`. The attention work Σ eᵢ·pᵢ, the extremes and the spread
are not recoverable from sums and variances. The learned model therefore needs, per
scheduled request, the pair

- `extend_lengths[i]`: tokens computed for request *i* in this step (chunk size for a
  prefill request, 1 for a plain decode request, `1 + k` with speculative drafts);
- `past_kv_lengths[i]`: KV tokens already present for request *i* before this step
  (prefix-cache hit plus earlier chunks for prefill; the context so far for decode).

Neither field exists in Dynamo main or SGLang main. Instead of patching either code
base, the two lists are added at runtime inside the engine process by
`aisimulate_core.fpm_hooks`. The producer patches under
`python/aisimulate/docs/fpm_ml/patches/` are the same change proposed upstream.

### How the hooks work

1. **Post-import patching.** `fpm_hooks.install()` registers a `sys.meta_path` finder
   for two module names, `sglang.srt.managers.scheduler_components.metrics_reporter`
   and `dynamo.vllm.instrumented_scheduler`. When the engine imports one of them the
   finder delegates to the real loader and runs the corresponding patch function right
   after `exec_module`. Modules that never get imported are never touched; nothing is
   imported eagerly.
2. **Extended `ScheduledRequestMetrics`.** FPM is a `msgspec` struct serialized to
   msgpack. The hook creates a frozen subclass of the producer's
   `ScheduledRequestMetrics` with two extra `list[int]` fields (`_struct.extend_struct`),
   keeps the class name and module, and copies the original instance into it
   (`_struct.with_pairs`). The msgpack payload keeps FPM `version` 1; aggregate-only
   consumers ignore the two extra keys, the Dynamo runtime decodes the event
   unchanged and the Rust trace sink writes them out. If the producer's struct already
   has both fields (a patched image), the hook is a no-op.
3. **SGLang** (`fpm_hooks/sglang.py`). The class that defines
   `_build_scheduled_request_metrics` is found by method name (it is
   `SchedulerMetricsReporter` in the V4.1 runtime, a mixin in other versions). Its
   method is wrapped: the wrapper calls the original, then derives one `(extend, past)`
   pair per request of the `ScheduleBatch` it received:
   - prefill (extend) batches: `batch.extend_lens[i]` and `batch.prefix_lens[i]`, the
     schedule-time values aligned with `batch.reqs` (the per-request attributes are
     reset after the step, so they cannot be used);
   - decode batches: `(1, batch.seq_lens_cpu[i])`, the same source the aggregate
     `sum_decode_kv_tokens` is computed from (`req.seqlen` is already incremented when
     the metrics are built and would be one token high).
4. **Dynamo vLLM** (`fpm_hooks/dynamo_vllm.py`). `InstrumentedScheduler._extract_scheduled`
   is wrapped; the pairs come from the `SchedulerOutput` the scheduler just produced:
   `num_scheduled_tokens[req_id]` is the extend, the request's `num_computed_tokens`
   the past (for newly scheduled requests from the request object, for cached
   requests from `scheduled_cached_reqs.num_computed_tokens`).
5. **Reaching every process.** Engines fork scheduler subprocesses that do not run the
   parent's Python code. `fpm_hooks/sitecustomize.py` calls `install()` at interpreter
   start, so putting the package directory on `PYTHONPATH`
   (`export PYTHONPATH=$(python -c 'import aisimulate_core.fpm_hooks as h; print(h.hook_path())'):$PYTHONPATH`)
   installs the hooks in the launcher and in every spawned interpreter. Alternatively
   `python -m aisimulate_core.fpm_hooks dynamo.sglang -- <args>` sets this up and
   execs the module.
6. **Failure mode.** The hooks depend on private engine internals (the two method
   names and the batch attributes). If a target does not look as expected the installer
   logs and skips, and the engine keeps emitting aggregate-only FPM; the learned model
   then refuses `sglang18` inference with a message pointing here rather than
   predicting from NaN features. During the V4.1 captures a class-name change (mixin →
   `SchedulerMetricsReporter`) was caught exactly this way and fixed by the
   method-name lookup.

### Validation of the recorded lists

On load, both lists must be absent or both of length
`num_prefill_requests + num_decode_requests` (a spec-decode extend of `1 + k` is
allowed). In the learned feature path, `sum(past)` must not exceed
`sum_prefill_kv + sum_decode_kv + sum_prefill_tokens + num_decode`; lists that do are
treated as absent for that step. On the V4.1 captures the per-request sums matched the
aggregates on 100% of decode steps and on all but a handful of prefill steps (page
padding makes `sum_prefill_tokens` slightly larger than Σ extend).

## 3. Features

The artifact is a fixed 135-slot feature ABI; a preset selects which slots a model
reads. Unused slots are NaN. Three presets exist:

| Preset | Slots | Source |
| --- | --- | --- |
| `v1` | 21 aggregates + their `log1p` and derived ratios | stock FPM v1, no lists needed |
| `sglang18` (default) | the 18 per-request features below | per-request lists |
| `hisim` | `req_batch_size` + 32 request slots × (present, past, extend), requests sorted by past descending | per-request lists |

The `sglang18` set is the feature definition of the SGLang simulator's
`MLTimePredictor`; the `hisim` slots follow HiSim's `_build_xgb_feature_maxbs_2`.
Both are feature definitions adapted from those projects (see
`THIRD_PARTY_NOTICES.md`), not code copies. Over the per-request pairs
`(e_i, p_i)` of a step:

| Feature | Definition |
| --- | --- |
| `req_batch_size` | number of requests |
| `req_sum_extend`, `req_max_extend`, `req_min_extend` | Σ e_i, max, min |
| `req_sum_past`, `req_max_past`, `req_min_past` | Σ p_i, max, min |
| `req_sum_extend_x_past` | Σ e_i·p_i |
| `req_sum_extend_squared` | Σ e_i² |
| `req_sum_past_squared` | Σ p_i² |
| `req_sum_attn_flops` | Σ e_i·(p_i + e_i/2), the attention work proxy |
| `req_sum_extend_x_max_past` | Σ e_i · max_j p_j |
| `req_log1p_sum_past`, `req_log1p_sum_attn_flops` | log1p of the two sums |
| `req_batch_size_x_sum_extend` | n · Σ e_i |
| `req_max_past_minus_min_past` | context spread inside the batch |
| `req_is_decode`, `req_is_prefill` | worker role flags |

Why these and not raw lists: trees need a fixed-width input; sums, extremes and the
attention proxy capture the two cost drivers (GEMM work ∝ Σ e_i, attention work
∝ Σ e_i·p_i) plus the batch shape (n, spread) that decides kernel selection and
padding. `v1` and `hisim` land within 0.3–0.6 pp of `sglang18` on every deployment
tested; `sglang18` is the default because it is the smallest set that carries the
per-request information.

## 4. Model

### 4.1 Estimator

One `HistGradientBoostingRegressor` (scikit-learn GBDT, histogram-binned, 255 bins)
per **store**, where a store is `(worker role, workload kind)`: `pure_prefill`,
`pure_decode`, and `mixed` when a worker schedules both in one step. Target is
`log(wall_ms)`; the prediction is `exp(raw)`. The log target makes the loss relative
(a 10% miss on a 4 ms decode step and on a 400 ms prefill step weigh the same) and
keeps predictions positive.

Hyperparameters used for every number in this document (`HistGradientBoostingRegressor`, `random_state=0`):

| | value |
| --- | --- |
| trees (`max_iter`) | 400 (CLI default is 600) |
| learning rate | 0.05 |
| max leaf nodes | 31 |
| min samples per leaf | 5 |
| L2 | 0 |
| early stopping | trainer CLI: **off**. The evaluation scripts behind §7 used sklearn's default `'auto'`, which above 10k rows holds out 10% of the *training* portion and stops when validation loss plateaus: decode fits ran all 400 trees, prefill fits stopped at ~116. The held-out 10% is taken from the training side only, so test sets are unaffected. |
| loss | squared error on log target |

`trees_fitted` is recorded in the artifact so the tree count is never assumed.

Why GBDT and not a random forest or a neural net: the mapping is a smooth-ish,
monotone-by-region function of a dozen numeric features with sharp regime changes
(kernel switches at batch-size and chunk-size thresholds). Boosted trees fit such
piecewise structure with a few hundred shallow trees, train in minutes on a million
rows on CPU, need no feature scaling, handle NaN natively, and evaluate as a
sum of leaf values, which is what makes the Rust inference trivial and fast.
Random forests need far more, deeper trees for the same accuracy; neural nets need
scaling, GPU-free training is slower and the artifact is not a portable JSON.

### 4.2 Artifact

A single JSON per worker role, schema version 1:

```
schema, schema_version, worker_type, target ("log_ms"), features (ordered names),
stores: {kind: {trees: [{children_left, children_right, feature, threshold, value, missing_left}], base_score}},
metadata: {trainer, model, hyperparameters, trees_fitted, train_rows, features_preset, deployment, ...}
```

`validate_artifact` checks the schema version is an `int` equal to 1, every tree is
well formed (array lengths match, leaves have no children, features are within the
ABI), and `missing_left` is optional per tree. Typical size 100–450 KB.

### 4.3 Inference (Rust)

`crates/core/src/perfmodel/fpm/learned.rs`:

1. `IterationFeatureVector::from_metrics(&[ForwardPassMetrics])` builds the 135-slot
   vector from all ranks of one step, walking the per-request lists once, and records
   whether request lists were present and consistent.
2. `predict_ms` selects the store by workload kind, walks each tree (NaN follows
   `missing_left`), sums leaf values onto `base_score`, and returns `exp`. A non-finite
   result is an error, not `None`.
3. `learned_base_ms` (in `model.rs`) refuses with `InvalidForwardPassMetrics` when the
   artifact uses any `req_*` / `slot*` feature and the step has no usable lists on an
   active rank; idle steps return `Some(0.0)`. Regression-store weights never affect the
   learned path (`classify_regression_workload` is weight-independent).

The online correction grid (`tune_with_fpms`) sits on top of the learned base exactly
as on top of the native model, so a constant boot-to-boot offset is absorbed at run
time.

## 5. Evaluation method

- **Unit:** one raw scheduler step. `APE = |predicted − observed| / observed`.
  Reported: step-weighted MAPE, median APE, p95 APE. No aggregation into tiers before
  scoring, so a model that is right on average but wrong per step is not rewarded.
- **Never a random split.** Consecutive steps of one run are strongly correlated
  (same requests, same KV state); a random shuffle puts near-duplicates of test steps
  into training and hides run-to-run variation, so it is not used anywhere.
- **Two split schemes are used:**
  1. *Independent runs*: train on one capture run, test on another with a different
     seed and different concurrency tiers. The strictest test; used for the
     per-deployment headline numbers.
  2. *60/40 time blocks over all runs*: every (run, tier) window is cut into five
     consecutive blocks; blocks 1, 2, 4 train, 3, 5 test. No shared step; both sides
     see every run and tier. Used when several runs of several workloads are pooled,
     so the comparison between single-workload and pooled models is on the same test
     set.
- **Cross-workload:** train on all runs of workload A, test on all runs of workload B.
  Measures extrapolation, i.e. what happens when a release model meets traffic it was
  not trained on.

## 6. The three workloads

All on one deployment: DeepSeek-V4.1-Flash, Dynamo SGLang runtime
`1.6.0-deepseek-v4.1-flash-dev.1`, 1P1D, one 4×GB300 node per role, TP4/EP4, page
size 256, 262k context, mooncake KV transfer, `--disable-overlap-schedule`,
`mem-fraction-static 0.8`, `max-prefill-tokens 16384`, hooks on. Load generator is
the SemiAnalysis aiperf fork at the InferenceX pin. GB300 only; no other GPU type is
mixed in.

| Workload | Source | Why it is in the set |
| --- | --- | --- |
| **AgentX** | aiperf `--scenario inferencex-agentx-mvp`, SemiAnalysis Claude-Code traces, 393 sessions, trajectory start 0.25–0.75 | agentic long-context traffic with heavy prefix reuse; the target workload |
| **ShareGPT** | aiperf `--public-dataset sharegpt` | chatbot traffic: short prompts, no prefix reuse, decode batch pinned at the concurrency |
| **LongBench** | LongBench-v2 (503 real documents, 8k–2M words). Each prompt = unique preamble line + a document window on a fixed 8k…200k-token grid + question and choices, `max_tokens` 1024; 4000 prompts in 8 files, one file per concurrency tier | long single-turn prefill with **no** cross-request prefix hit (the preamble makes every prompt's first token unique) and no prompt ever replayed |

Only workloads with real text tokens are used. Trace-replay datasets that carry
lengths and hash IDs only (Mooncake, Bailian, BurstGPT, the aiperf agentic-code
synthesizer) fill prompts with corpus slices; the MoE routing of such text is not the
routing of real requests, so they are excluded as truth.

Input distributions as the engine saw them, from the FPM per-request lists of one
capture run each (p5 / p50 / p95 / max):

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

| Workload | capture runs (seeds) | concurrency tiers | prefill steps | decode steps |
| --- | --- | --- | --- | --- |
| AgentX | 4 (42 / 7 / 11 / 23) | c16/32/64/128, c24/48/96 | 20.9k | 1,150k |
| ShareGPT | 4 (42 / 7 / 11 / 23) | c32/64/128/256, c48/96/192 | 38.2k | 520k |
| LongBench | 2 (42 / 7) | c16/32/64/128, c24/48/96 | 20.2k | 1,062k |

LongBench did not produce large decode batches even at concurrency 128: with
50k-token average prompts the prefill node is the bottleneck (three to four 16k
steps per request), so few requests are decoding at any moment. The region
"decode batch ≥ 64 at context ≥ 30k" is covered by none of the three workloads.

## 7. Results

### 7.1 Independent capture runs, one deployment each

Train on one run, test on another (different seed and tiers). Step-weighted MAPE
(median):

| Deployment | train → test tiers | decode | prefill |
| --- | --- | --- | --- |
| Qwen3-32B-FP8, vLLM 1.4.0, 262k YaRN | c16/32/64 → c24/48/96 | 2.41% (0.97%), 122k steps | 1.64% (0.55%), 13k steps |
| DeepSeek-V4-Flash, vLLM 1.4.0, 262k | c16/32/64/128 → c24/48/96/64 | 3.14% (0.97%), 457k steps | 4.46% (4.48%), 32k steps¹ |
| DeepSeek-V4.1-Flash, SGLang 1.6.0-dev, TP4/EP4, overlap off | c16/32/64/128 → c24/48/96 | 1.92% (1.47%), 197k steps | 2.23% (1.00%), 3.7k steps |

¹ a constant run-to-run offset (predicted/observed median 1.044, p10–p90
1.002–1.058), which the online correction grid absorbs.

### 7.2 Three workloads, V4.1-Flash SGLang GB300

Rows = training data, columns = test set. Same-workload cells (diagonal and the
pooled row): 60/40 time-block split of that workload's runs, test set = 40% of the
workload. Cross-workload cells: all runs of the row workload train, all runs of the
column workload test.

**Decode, step-weighted MAPE**

| Training data \ test set | AgentX | ShareGPT | LongBench |
| --- | --- | --- | --- |
| AgentX only | 2.34% | 14.14% | 3.03% |
| ShareGPT only | 3.88% | 2.26% | 3.43% |
| LongBench only | 2.91% | 22.45% | 1.84% |
| all three pooled | 2.21% | 2.10% | 1.84% |

**Prefill, step-weighted MAPE**

| Training data \ test set | AgentX | ShareGPT | LongBench |
| --- | --- | --- | --- |
| AgentX only | 2.24% | 3.49% | 0.88% |
| ShareGPT only | 26.23% | 2.55% | 43.17% |
| LongBench only | 6.52% | 29.42% | 0.62% |
| all three pooled | 2.28% | 2.56% | 0.57% |

Reading the off-diagonal cells:

- AgentX → ShareGPT decode 14%: error grows monotonically with concurrency (c32 2%,
  c64 5–7%, c128 24–26%, c256 41–43%); AgentX never has more than ~43 requests in a
  decode batch, so the model extrapolates.
- ShareGPT → AgentX prefill 26% and → LongBench prefill 43%: ShareGPT never has a 16k
  chunk or a past above 2k.
- LongBench → ShareGPT decode 22% (c32 2% … c256 50%) and prefill 29%: no short
  requests, no batch above 26.
- LongBench → AgentX prefill 6.5%: the full-chunk tiers are at 0.8%; the small-extend
  prefix-hit chunks that only AgentX has are at 7–8%.
- AgentX → LongBench 3.0% / 0.9%: LongBench's shapes lie inside AgentX's range, so no
  extrapolation is needed.

The pooled model matches or beats every single-workload model on that model's own
workload, so a release artifact should be trained on the union of the workloads it is
meant to simulate. Step balancing between workloads changes the pooled numbers by at
most 0.1 pp.

## 8. Speed

**Inference** (Rust tree walk through the PyO3 binding, one prediction = one
`ForwardPassMetrics` step, `sglang18` artifact with 400 trees):

| decode batch size | time per prediction |
| --- | --- |
| 8 | 12 µs |
| 64 | 23 µs |
| 256 | 58 µs |

The cost is dominated by building the feature vector from the per-request lists
(linear in batch size); the 400-tree walk itself is a few microseconds. A simulated
run of a million steps therefore spends well under a minute in the perf model.

**Training** (scikit-learn, CPU, 16 threads, log target, 400 trees, pooled 10 runs):

| store | rows | fit time |
| --- | --- | --- |
| decode | 2,732,395 | 25 s (400 trees) |
| prefill | 79,261 | 1 s (116 trees, early-stopped) |

Loading and featurizing the FPM stream (gzip JSON lines) dominates: 204 s for the 2.7M decode records against 25 s of fitting; both are one-off offline costs. scikit-learn's own batched `predict` runs at 2.5 µs per row on the same machine, the Rust single-step path above is what the simulator uses. Artifacts are 100–450 KB of JSON.

## 9. Limitations and follow-ups

- **Coverage.** No workload in the set has decode batches above ~43 at contexts above
  30k. A prefix-reuse long-context workload with many concurrent sessions (AgentX at
  higher concurrency with more prefill capacity, or a synthetic shape sweep, flagged as
  such) is needed before the model is trusted there.
- **Per-request fields upstream.** The lists come from runtime hooks; the producer
  patches under `docs/fpm_ml/patches/` are proposals, not merged in Dynamo or SGLang.
- **Simulator wiring.** `best_available(config)` and `EstimationMode` have no learned
  option yet; the model is reachable through the SDK only.
- **Speculative decoding.** Extends of `1 + k` are accepted, but no training data with
  MTP/EAGLE on exists yet.
- **GPU type.** All numbers are GB300. A model is per deployment (GPU, engine, model,
  parallelism); nothing here claims transfer across GPU types.
