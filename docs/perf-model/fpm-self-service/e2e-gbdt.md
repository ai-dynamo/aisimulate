<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# End-to-end offline profiling and GBDT

This chapter implements the end-to-end offline profiling route in
[FPM self-service onboarding](README.md#choose-a-profiling-route). Collect native
forward-pass measurements on your cluster, train a GBDT, and use it in AISimulate
Replay. You can then update the model with later measurements.

Start with **end-to-end offline profiling**: run a workload, collect its native
forward-pass measurements, and train a GBDT for frozen prediction. You can then
apply **online updates** using later measured feedback.
**Native FPM capture** is the collection mechanism used for both the initial
profiling run and later feedback; per-request data hooks are a future extension.

Follow [Collect](#1-collect) → [Check](#2-check-and-prepare-the-data) →
[Train](#3-train-and-qualify-accuracy) → [Replay](#4-use-the-model-in-replay) →
[Adapt](#5-optional-adapt-and-save).

## Before you start

You need a serving deployment with native FPM instrumentation, access to its
worker traces, and a client that can run your workload. Use your own cluster and
deployment tools. An existing completed capture can start at step 2 after reviewing its identity
and provenance through onboarding stages 1–3.

In an AISimulate source checkout, prepare Python 3.11–3.13 and the training extra:

```bash
uv sync --project python/aisimulate --extra learned
source python/aisimulate/.venv/bin/activate
export AIS_REPO="$PWD"
export PYTHONPATH="$AIS_REPO/python/aisimulate${PYTHONPATH:+:$PYTHONPATH}"
python -m aisimulate_core.sdk.fpm_learned train --help
aisimulate predict --help

export FPM_RUN="$PWD/fpm-run"
export FPM_ROLE=aggregated  # prefill or decode for P/D
mkdir "$FPM_RUN"
mkdir -p "$FPM_RUN"/{traces,receipts,selected,models}
```

Use a fresh run directory. Training runs on CPU; serving data collection needs
your GPUs. The online-seed exporter requires scikit-learn 1.9.1. Source-build
prerequisites are in the [installation guide](../../getting-started/installation.md#use-current-source).

Copy the [identity template](assets/identity.json) to `ROLE.identity.json` in the
run directory. Replace the model, packaged system, backend/version, role and
parallelism. Add the actual precision, KV block size and kernel settings from
the [configuration reference](../configuration.md#choosing-a-forward-pass-api).
The template's TP4/EP4 values are examples, not inferred deployment settings.
Review and save this identity, the runtime/model/tokenizer pins, launch settings
and expected worker IDs before capture. Keep unresolved resource facts explicit;
measured KV geometry and usable capacity are required before Replay. In an
agent-guided session, retain these decisions and artifacts in the
[onboarding checkpoint](implementation.md#checkpoint-and-resume-an-onboarding-session).

### What the model predicts

A forward pass is one engine iteration over a scheduled batch. Its duration
depends on the work in that batch: new prefill tokens, decode requests and the
KV context they attend to. One request usually participates in many passes.
The performance model predicts each pass; Replay supplies the scheduling,
queueing and cache behavior that turns those costs into request TTFT and TPOT.

End-to-end offline profiling supplies the initial training population. Online
updates use later measured feedback in the same telemetry format.
Keep the deployment identity fixed for each model instance; a change of
hardware, topology or backend needs its own model and qualification.

### How this fits onboarding

Follow the [six onboarding stages](README.md#workflow) to review the target and
choose this route. The detailed commands below implement its collection,
training and model operations:

| Onboarding stages | Work in this guide |
| --- | --- |
| 1–3. Inspect the target, choose the worker and limits, review collection inputs | Verify the deployment; record its model, runtime, topology and resource identity before capture. |
| 4. Plan collection | Choose the workload, capture settings, smoke checks and measurement boundaries. |
| 5. Collect and verify | Steps 1–2: collect native traces, inspect them and prepare training inputs. |
| 6. Validate and use | Steps 3–5: train, qualify accuracy, use Replay, and optionally apply online updates and save. |

The five steps below expand the corresponding onboarding stages; they are not
another onboarding procedure. Native JSONL is input to the standalone trainer;
`onboard finalize` expects self-benchmark timing tables and does not perform
this handoff. Keep measured resource capacity, prediction coverage and matched
serving accuracy as separate checks. The
[canonical agent procedure](implementation.md#onboard-with-an-agent) describes
both routes and their session checkpoint requirements.

## 1. Collect

### Enable native capture

Set these variables **on each serving worker**, using a persistent path visible
to that worker. `DYN_FPM_OUTPUT_PATH` is a filename prefix:

```bash
export DYN_FPM_TRACE=1
export DYN_FPM_MODE=full
export DYN_FPM_OUTPUT_PATH=/persistent/run/aggregated/fpm
export DYN_FPM_MAX_SEGMENTS=64  # size retention for your full run
```

| Backend | Aggregated | P/D disaggregated |
| --- | --- | --- |
| vLLM | Verify its native instrumented scheduler is active. Keep the role `aggregated` for prefill, decode and mixed batches. | Enable capture on both roles, with separate output prefixes and worker identities. Use real KV transfer. |
| SGLang | Verify forward-pass metrics are enabled and scheduled-prefill counters are correct. The qualified recipe uses serial scheduling with overlap disabled. | Apply those producer checks to both roles, with separate GPUs and reachable KV-transfer addresses. |

For SGLang, affected producers need the [native counter correction](examples.md#d1-pin-the-aggregated-sglang-runtime)
or an equivalent verified fix. The qualified capture configuration also disables
mixed chunks, speculation, attention DP and prefill CP. Changing these settings
needs new counter/timer checks.

Native labels are forward-pass durations in **seconds**. Keep them unchanged:
vLLM and SGLang have different timer and decode-context conventions. Keep all
five `scheduled_requests` counters and the complete raw envelope. Request
latency, TTFT and KV-transfer time are separate measurements.

The five counters describe the scheduled work on one attention-DP rank:

| Native field under `scheduled_requests` | Meaning |
| --- | --- |
| `num_prefill_requests` | Number of requests doing prefill work in this pass. |
| `sum_prefill_tokens` | Total new prefill tokens computed in this pass, including chunks. |
| `sum_prefill_kv_tokens` | Summed existing KV context for those prefill requests. |
| `num_decode_requests` | Number of requests doing decode work. |
| `sum_decode_kv_tokens` | Summed decode context, using the producer's convention. |

`--features fpm5` derives its feature vector from these counts and sums; the
name does not mean the fitted vector has exactly five entries. It requires no
per-request length lists or variance estimates. The trainer converts the
seconds-valued `wall_time` to a millisecond target internally. Keep the raw
label in seconds when supplying feedback through the API too.

### Smoke, then run your workload

Save a short request as `short.json`, replacing the model ID:

```json
{
  "model": "YOUR_SERVED_MODEL",
  "messages": [{"role": "user", "content": "List the integers from 1 to 8."}],
  "stream": false,
  "max_tokens": 32
}
```

Check your frontend from the client machine. These commands use an
OpenAI-compatible endpoint; apply your normal authentication/TLS settings:

```bash
export FPM_BASE_URL=http://your-serving-host:8000
curl --fail-with-body "$FPM_BASE_URL/v1/models"
curl --fail-with-body --max-time 120 \
  -H 'Content-Type: application/json' \
  --data-binary @short.json "$FPM_BASE_URL/v1/chat/completions"
```

**Proceed when** the model is registered, the response completes with consistent
token usage and multiple output tokens, and native busy rows appear for every
expected role. Check cold and repeated prefixes, chunked prefill if enabled,
and decode after idle. For P/D, confirm real transfer and both roles' records.
An HTTP success alone does not verify those properties.

Run a bounded pilot, then your own workload client. Adjust request rate or
concurrency if persistent queuing, retractions or stalls prevent stable
measurements. Record the change and retain genuine latency variation; collect
the batch sizes and context lengths you intend to model.

Preserve session structure, arrival timing and request termination semantics.
Record warmup, measurement and drain boundaries, request outcomes, and the
client command/exit status. Stop new sends, let outstanding work finish, verify
worker idle, then close or rotate capture files. Use the operator's procedure
for a shared deployment; stop only workers owned by your collection.

**Keep with the capture:** runtime/model/tokenizer identity, topology and serving
settings, expected worker/role IDs, workload configuration, phase boundaries,
completion counts, logs and file hashes. Identify smoke/pilot exclusions and any
other traffic sharing the workers.

**Completed SGLang example.** A native aggregated run used DeepSeek-V4-Flash-0731
on four B200 GPUs with TP4/EP4, FP8 KV and an 8192-token prefill chunk. One hour
at concurrency 15 completed 1,039 measured requests without errors or
cancellations. It captured pure-prefill and pure-decode observations, with no
mixed-row coverage. Use your own workload and load settings; the
[deployment example](examples.md#example-d-collect-native-fpm-from-an-aggregated-worker)
is optional background.

## 2. Check and prepare the data

Copy closed native files into `traces/ROLE/`. Validate them locally, replacing
the identity placeholders with your deployment's native source/worker values:

```bash
python3 "$AIS_REPO/scripts/fpm_accuracy/validate_native_capture.py" \
  --trace-dir "$FPM_RUN/traces/$FPM_ROLE" \
  --expect "$FPM_ROLE" NAMESPACE COMPONENT PRODUCER_ID WORKER_ID 0 \
  --report "$FPM_RUN/receipts/$FPM_ROLE.capture.json"
```

The last argument to `--expect` is the attention-DP rank. Repeat that option for
each expected stream. Validate P/D roles separately and keep worker restarts in
separate inputs. The helper reads full `dynamo.fpm.trace.v1` envelopes; it is a
source-checkout utility, not an installed AISimulate command.

**Proceed when** the report is `valid`, every expected worker has busy rows, and
your client completion checks passed. Inspect counts and warnings. Exit 1 means
invalid data; exit 2 requires review or corrected arguments. Readable files and
continuous counters do not prove lossless publication.

Choose the workload rows to retain. Save one reviewed inclusive line range per
file in `$FPM_RUN/$FPM_ROLE.selection.json`, replacing every value below with
actual evidence:

```json
[
  {
    "path": "traces/aggregated/fpm.YOUR_WORKER.000000.jsonl.gz",
    "sha256": "COPY_FROM_CAPTURE_REPORT",
    "first_line": 101,
    "last_line": 5000
  }
]
```

Exclude smoke/pilot rows and state whether warmup and drain are included. List
files in segment order. Then prepare an initial 50/50 split within each work kind:

```bash
python3 "$AIS_REPO/scripts/fpm_accuracy/prepare_native_training.py" \
  --run-dir "$FPM_RUN" --role "$FPM_ROLE"
```

This writes `selected/ROLE.train.jsonl`, `selected/ROLE.test.jsonl` and a selection
receipt. It preserves native rows and labels, checks source hashes and excludes
idle rows. Use separately collected training/test inputs when assessing another
run or workload. Keep decode-admission/mixed rows under their original role and
account for any trainer exclusions.

## 3. Train and qualify accuracy

Use the deployment identity reviewed before collection. Confirm it still
matches the selected native files and role; resolve mismatches before fitting.

Train one AISimulate-compatible GBDT per role:

```bash
python -m aisimulate_core.sdk.fpm_learned train \
  --fpm "$FPM_RUN/selected/$FPM_ROLE.train.jsonl" \
  --worker-type "$FPM_ROLE" --features fpm5 --join-ranks none \
  --identity "$FPM_RUN/$FPM_ROLE.identity.json" \
  --holdout-frac 0 --seed 42 \
  --out "$FPM_RUN/models/$FPM_ROLE.learned.json" \
  --online-checkpoint-dir "$FPM_RUN/models/$FPM_ROLE.online-seed"

python -m aisimulate_core.sdk.fpm_learned evaluate \
  --model "$FPM_RUN/models/$FPM_ROLE.learned.json" \
  --fpm "$FPM_RUN/selected/$FPM_ROLE.test.jsonl" --join-ranks none
```

Check each command's status and retain its output. The learned JSON is the
inference model; the fresh `online-seed` directory supports later adaptation.
Do not create that directory ahead of training. Keep early stopping disabled
for seed export. Repeat with separate identities/data for P/D.

### What training produces

The trainer fits separate forests for the workload classes supported by the
worker role. A *store* is the forest and fitting state for one such class;
having decode data does not create a mixed-batch store. `--join-ranks none`
treats each native row independently. It does not reconstruct synchronized
attention-DP iterations from unrelated rank counters.

For the default `log_ms` target, a forest predicts

$$
\begin{array}{rl}
g(x)&=b+\sum_{m=1}^{M}v_m(x),\\
\widehat y(x)&=\max(10^{-6},\exp(g(x))).
\end{array}
$$

Here $x$ is the derived feature vector, $b$ is the baseline, and each tree
contributes $v_m(x)$, including its fitted learning rate. The output
$\widehat y$ is milliseconds. Fitting squared error in log time emphasizes
multiplicative timing differences, but it does not directly minimize MAPE.

| Output | Use it for |
| --- | --- |
| `ROLE.learned.json` | Frozen prediction: feature ordering, target transform and fitted trees. |
| `ROLE.online-seed/` | Continuing training: the forest plus fitted bins, original observations and fitting state. |

The inference JSON cannot reconstruct the training checkpoint. Preserve both
when you want the option of online updates later. Evaluation reads separate
observations and scores predictions without updating either output.

**Proceed when** evaluation scores nonzero observations, required stores have
coverage, and error meets your chosen threshold. A successful exit alone is
insufficient. Report error, prediction coverage, train/test counts and excluded
rows together; missing stores need more representative data and another fit.

Use the [FPM Gym](https://ai-dynamo.org/aisimulate/fpm-accuracy/scenario-comparison.html)
as the accuracy qualification reference: select a GBDT trained through end-to-end
offline profiling and evaluated with online updates disabled. It evaluates
frozen predictions on held-out observations; MAPE covers successful predictions
only. Your own evaluation qualifies your own captured workload. Broader
generalization and Replay TTFT/TPOT accuracy need separate validation.

## 4. Use the model in Replay

Copy the [Replay template](assets/replay-aggregated.yaml) to `replay.yaml` and
replace every placeholder. It starts with aggregated vLLM, attention DP=1,
PP=1 and no context parallelism. Set the captured topology, precision and
scheduler limits, plus **measured** KV block geometry and usable capacity.
Point `traffic.source` at your request trace, not the FPM training data.

The worker's timing block selects the trained forest explicitly:

```yaml
timing:
  type: default
  estimation_mode: fpm_learned
  fallback_policy: deny
  estimator_config:
    fpm_learned:
      artifact_path: /absolute/path/fpm-run/models/aggregated.learned.json
      adaptation: frozen
    correction:
      enabled: false
```

```bash
aisimulate predict --config replay.yaml --output-dir replay-result
```

For P/D, use `engine.mode: disaggregated`, separate prefill/decode worker blocks
and role-matching artifacts. Set each role's resources and
[KV-transfer configuration](../../replay/engine/kv-transfer.md); omitting
transfer settings models zero transfer delay. Choose a supported
[request trace format](../../replay/workloads.md) that preserves your workload.

**Check coverage first.** Aggregated vLLM learned Replay needs prefill, decode and
mixed stores; dedicated roles need their corresponding stores. Attention DP>1
adds cross-rank requirements. This recipe does not qualify speculative decoding,
vision or bounded decoder replay; keep decode speedup at one.

**SGLang limitation:** native collection and training are supported, but the
Replay consumer's decode-context mapping and mixed scheduling need qualification
against SGLang. Successful SGLang capture alone does not establish Replay
compatibility or accuracy. Compare simulated metrics with a matched serving run
before making end-to-end accuracy claims.

### The prediction API behind Replay

`RustForwardPassPerfModel.best_available(config)` is the common constructor.
The configuration binds a model to its deployment identity and selects its
estimator. In this guide, explicit `fpm_learned` plus `fallback_policy: deny`
selects the supplied forest. The name `best_available` does not mean it measures
accuracy and chooses the best model for every query.

The core calls used by the examples are:

| Call | Purpose and contract |
| --- | --- |
| `iter_fpm_records(paths)` | Unwrap native JSONL records into FPM dictionaries. Parsing does not replace capture validation. |
| `best_available(config)` | Load and validate the selected source, identity and adaptation settings. |
| `estimate_forward_pass_time_ms(rank_records)` | Predict one iteration from its complete rank group. Returns milliseconds; an uncovered learned query can return `None`. |
| `tune_with_fpms(iterations)` | Admit measured iterations. Each item is a list of that iteration's rank records, including seconds-valued `wall_time`. |
| `readiness_report()` / `diagnostics()` | Inspect fitted stores, model state and resolved provenance. Readiness alone does not prove accuracy or complete workload coverage. |
| `flush_updates()` | Explicitly fit pending measured feedback, including a partial update batch. |
| `save_online_checkpoint(path)` | Save standalone online fitting state, including pending observations, without implicitly fitting them. |
| `export_learned_artifact()` | Return the current inference forest as a JSON-compatible dictionary. |

For attention DP=1, a query is `[row]` and one feedback iteration is `[[row]]`.
With attention DP>1, use one complete, synchronized list of rank records per
iteration; worker IDs and equal counter values do not establish synchronization.

## 5. Optional: adapt and save

Supply later measured observations from the same deployment as
`selected/ROLE.online.jsonl`. Validate/select them as before, keep them separate
from the frozen test set, and retain their order. Only role-compatible busy rows
belong in this feedback file.

Run the [online update example](assets/adapt_gbdt.py):

```bash
python "$AIS_REPO/docs/perf-model/fpm-self-service/assets/adapt_gbdt.py" \
  --run-dir "$FPM_RUN" --role "$FPM_ROLE" \
  --output-dir "$FPM_RUN/models/$FPM_ROLE.adapted"
```

The example loads the initial online seed through `best_available`, predicts
before admitting each label with `tune_with_fpms`, and saves an updated checkpoint
and inference forest. It also restores the checkpoint to check prediction
consistency. Use the exported forest with frozen adaptation in Replay.
The output directory contains `model.json`, `checkpoint/`, a restore
configuration and an adaptation receipt.

Saving retains pending fitting rows. Add `--flush-updates` only when you intend
to fit a partial update batch. Preserve the model trained offline for comparison.
For continued online operation, your application owns feedback delivery and the
last admitted position; coordinate that position with checkpoint saving.
Replay's simulated outputs are not new training evidence.

### Understand the update loop

The essential sequence inside the example is:

```python
# config identifies this deployment and its online-seed checkpoint.
online = RustForwardPassPerfModel.best_available(config)
for row in iter_fpm_records([feedback_path]):
    query = [{key: value for key, value in row.items() if key != "wall_time"}]
    predicted_ms = online.estimate_forward_pass_time_ms(query)
    # Score this prediction against 1000 * row["wall_time"] before updating.
    online.tune_with_fpms([[row]])  # one measured iteration, one DP rank
```

This excerpt shows ordering; the linked runnable example also checks role,
coverage, input integrity and persistence. For observation $t$, predict using
only the state available before its label is admitted:

$$
\begin{array}{rl}
\widehat y_t&=F_{t-1}(x_t),\\
y_t&=1000\,\mathrm{wall\_time}_t.
\end{array}
$$

`tune_with_fpms` then admits the label. Predictions change only after a
successful fitting update. `estimator_config.fpm_learned.update_every` counts
admitted observations, while
your application controls how often feedback is sampled. The standalone default
fits after each admission; larger batches can reduce update frequency.

Online fitting retains the seed history and a bounded recent window. It can
change leaf values and eligible splits, while preserving the seed's feature
bins and tree capacity. It does not add missing workload stores. This makes
representative initial coverage important even when you plan to adapt.

### Choose how online updates change the model

The example above uses the first option. The other two are alternatives when
you want to retain a stable forest or compare it with an adaptive one:

| Mode | Configuration under `estimator_config` | Effect of measured feedback |
| --- | --- | --- |
| Online GBDT | `fpm_learned.adaptation: online`, `correction.enabled: false` | Updates the existing trees; predicts from the latest fitted forest. |
| Hybrid calibration | `fpm_learned.adaptation: frozen`, `correction.enabled: true`, `correction.mode: hybrid` | Keeps the trees fixed and learns multiplicative timing factors. |
| Causal selector | `fpm_learned.adaptation: causal_selector`, correction enabled in `hybrid` mode | Maintains calibrated and adaptive experts; chooses using earlier errors. Requires a training checkpoint. |

Standalone online GBDT has **no automatic fallback to the initial forest**.
Its checkpoint/export example also does not persist the other two modes'
calibration or selector histories. Treat those modes as separate integrations;
do not reuse the standalone save loop as a complete state export for them.

### Calibration and fallback with sparse feedback

Let $F_0(x)$ be the initial frozen forest's prediction in milliseconds, and $z$
the workload coordinates used to group correction observations. For example, decode
correction groups by decode request count and summed decode KV context, rather
than the full GBDT feature vector. For a measured duration
$y_i$, hybrid calibration records a bounded ratio and applies a factor:

$$
\begin{array}{rl}
r_i&=\operatorname{clip}\left(\frac{y_i}{F_0(x_i)},0.5,2\right),\\
\widehat y_t&=c_{t-1}(z_t)F_0(x_t).
\end{array}
$$

The ratio always uses the **uncorrected** forest prediction, so repeated updates
do not compound earlier corrections. A 10 ms prediction and a 12 ms observation
provide a ratio of 1.2. This example does not imply that one observation makes
the correction history ready; the default requires five retained observations.

Hybrid calibration uses a fixed grid within a configured workload domain and a
dynamic grid outside it. The fallback depends on which grid applies:

| Query location | Factor used |
| --- | --- |
| Inside the fixed domain | Median ratio in its populated cell once the fixed history is ready; otherwise **1**. It does not fall through to the dynamic grid. |
| Outside the fixed domain, with a usable dynamic cell | Blend that cell's median with the median of recent feedback. |
| Outside the fixed domain, with no usable dynamic cell | Recent-feedback median, or **1** while that history is unready. |

For a usable dynamic cell with $n$ samples, local median $a$ and recent-global
median $g$, the blend is

$$
\begin{array}{rl}
w&=\frac{n}{n+20},\\
c&=\exp\left(w\log a+(1-w)\log g\right).
\end{array}
$$

Small local samples therefore rely mostly on recent feedback across workload
regions. An empty cell can still receive a global correction, while factor 1
preserves the base prediction when evidence is insufficient. These are
fallbacks for the **correction factor**; they cannot supply a missing forest.
The bounds also limit what calibration can repair: a 10 ms base prediction
cannot exceed 20 ms under the default factor limits.

### Choosing between calibrated and adaptive predictions

The causal selector starts with the calibrated expert. For each admitted
observation, both experts predict before either learns its label. It tracks
their fractional absolute errors with an exponentially weighted average:

$$
\begin{array}{rl}
L_t^{e}&=\left|\frac{\widehat y_t^{e}}{y_t}-1\right|,\\
E_t^{e}&=(1-\alpha)E_{t-1}^{e}+\alpha L_t^{e}.
\end{array}
$$

The first loss initializes the average. Error histories and warmup are separate
for each forest-store/query-kind pair; five decode comparisons do not warm up
prefill or mixed selection. Elapsed time alone does not change these averages.
After warmup, the challenger replaces
the active expert for subsequent predictions when

$$
E_t^{\mathrm{challenger}}<(1-\delta)E_t^{\mathrm{active}}.
$$

Defaults are $\alpha=0.4$, five paired observations for warmup, a relative margin
$\delta=0.05$, and one qualifying comparison to switch. For example, an active
error score of 0.10 requires a challenger score below 0.095. Both experts keep
receiving feedback, so selection can switch back. The margin reduces switching
on small differences; it is not a confidence bound or a guarantee of lower
error on the next pass.
These settings are `alpha`, `warmup`, `margin` and `persistence` under
`estimator_config.selector`.

### What if a workload has no fitted store?

For a model trained through end-to-end offline profiling, collect the missing
workload class and train a compatible seed. Without an offline seed, first collect
measured feedback and perform an application-managed initial fit.
`tune_with_fpms` does not turn an empty source into a forest.

**Self-benchmarking** measures controlled batch shapes directly.
That separate route can optionally use compatible profiling tables for
mixed-workload fallback, with optional online updates.
For one rank running both prefill and decode, its basic composition is

$$
I(x)=I_P(x)+\max\left(0,I_D(x)-I_{D,0}\right).
$$

Here $I_P$ and $I_D$ are prefill and decode interpolation estimates. The decode
baseline $I_{D,0}$ is the floor cost of the decode curve for that batch size;
subtracting it avoids charging that base cost twice. With multiple attention-DP
ranks, the iteration takes the maximum rank latency, not the sum of separate
prefill and decode ranks. Measured feedback can calibrate the resulting
estimate.

This route requires its own profiles and eligibility checks and is not part of
this guide's end-to-end offline profiling route. The top-level `fallback_policy`
governs estimator selection at construction and must remain `deny` for
`fpm_learned`. It does not disable
the explicitly configured correction or selector. Missing training coverage
remains a collection/training task.

## If a step fails

| Symptom | Next action |
| --- | --- |
| Requests succeed but no native rows appear | Inspect the active producer, worker output prefix, permissions and retention. |
| Queuing, retractions or stalled requests | Reduce offered load, check memory/transfer limits, then repeat in a fresh run directory. |
| Capture validation fails | Preserve the report; recover closed files or reconcile worker identities/restarts before selecting rows. |
| Missing stores or many excluded rows | Check role and work-kind coverage; collect the missing operating region and fit again. |
| Low FPM error but poor Replay results | Check workload semantics, native-feature mapping, scheduler settings, KV capacity and transfer behavior. |
