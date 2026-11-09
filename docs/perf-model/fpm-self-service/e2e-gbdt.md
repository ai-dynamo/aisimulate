<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# End-to-end offline profiling and GBDT

This chapter implements the end-to-end offline profiling route in
[FPM self-service onboarding](README.md#choose-a-profiling-route). Collect native
forward-pass metrics (FPM) on your cluster, train a gradient-boosted decision tree
(GBDT) model, and use it in AISim Replay. These metrics describe the
scheduled batch work and its measured duration.

Start with **end-to-end offline profiling**: run a workload, collect its native
forward-pass measurements, and train a GBDT for frozen prediction. You can then
apply **online updates** using later measured feedback.
**Native FPM capture** is the collection mechanism used for both the initial
profiling run and later **online observations**, which supply measured feedback
for online adaptation. This guide uses aggregate batch features, so per-request
length lists are not required. Per-request data hooks are a future extension.

For initialization from online observations alone, or self-benchmark seeds with
online feedback, follow
[Start and adapt a GBDT with online observations](online-gbdt.md).

Follow [Collect](#1-collect) → [Check](#2-check-and-prepare-the-data) →
[Train](#3-train-and-qualify-accuracy) → [Replay](#4-use-the-model-in-replay) →
[Adapt](#5-optional-adapt-and-save).

## Before you start

You need a serving deployment with native FPM instrumentation, access to its
worker traces, and a client that can run your workload. Use your own cluster and
deployment tools. An existing completed capture can start at step 2 after
reviewing its identity and provenance through onboarding stages 1–3.

In an AISim source checkout, prepare Python 3.11–3.13 and the training extra:

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

Copy the [identity template](../../../scripts/fpm_self_service/identity.json) to `ROLE.identity.json` in the
run directory. Replace the model, packaged system, backend/version, role and
parallelism. Add the actual precision, KV block size and kernel settings from
the [configuration reference](../configuration.md#choosing-a-forward-pass-api).
The template's TP4/EP4 values are examples, not inferred deployment settings.
Leave learned source paths out of this identity; training creates the source.

For Replay, omit both `moe_tp_size` and `moe_ep_size` from the training identity
when `moe_tp_size * moe_ep_size == 1`. The Replay YAML can still use
`parallelism.moe_tensor: 1` and `parallelism.moe_expert: 1`; its compiler represents
those unsharded dimensions as `None` in the canonical identity. Training with
explicit `1, 1` preserves a different identity and fails the exact identity check
when Replay loads the artifact. For a sharded topology, retain both fields,
including a dimension equal to 1 as in the template. Use the same reviewed
identity for training and subsequent SDK construction.

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
observations supply later measured feedback in the same telemetry format;
online updates use those observations to adapt the predictor.
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

For separate roles, the optional [P/D setup reference](native-pd-setup.md) gives
tested runtime pins and launch settings; use [P/D troubleshooting](native-pd-troubleshooting.md)
during the pilot and completion checks.

For SGLang, affected producers need the [native counter correction](examples.md#d1-pin-the-aggregated-sglang-runtime)
or an equivalent verified fix. The qualified capture configuration also disables
mixed chunks, speculation, attention DP and prefill CP. Changing these settings
needs new counter/timer checks. The
[capture reference](../fpm_ml/learned-forward-pass-model.md#1-collect-the-fpm-stream)
explains why overlap scheduling can merge or omit timing intervals; verify that
prefill counters describe the tokens scheduled in each pass, including partial
chunks.

Native labels are forward-pass durations in **seconds**. Keep them unchanged:
vLLM and SGLang have different timer and decode-context conventions. Keep all
five `scheduled_requests` counters and the complete raw envelope. Request
latency, TTFT and KV-transfer time are separate measurements.

The five counters describe the scheduled work on one attention-DP rank.
The symbols below refer to counts and sums, not mean request lengths:

| Symbol | Native field under `scheduled_requests` | Meaning |
| --- | --- | --- |
| $B_p$ | `num_prefill_requests` | Number of requests doing prefill work in this pass. |
| $B_d$ | `num_decode_requests` | Number of requests doing decode work. |
| $Q_p$ | `sum_prefill_tokens` | Total new prefill tokens computed in this pass, including chunks. |
| $K_p$ | `sum_prefill_kv_tokens` | Summed existing KV context for those prefill requests. |
| $K_d$ | `sum_decode_kv_tokens` | Summed decode context, using the producer's convention. |

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

**SGLang configuration example.** The tested
[aggregated recipe](examples.md#example-d-collect-native-fpm-from-an-aggregated-worker)
uses DeepSeek-V4-Flash-0731 on four B200 GPUs with TP4/EP4, FP8 KV and an
8192-token prefill chunk. It covers pure-prefill and pure-decode work, with no
mixed-row coverage. Use your own workload and load settings.

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
source-checkout utility, not an installed AISim command.

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
idle rows. Adjacent iterations can remain correlated across this split, so its
held-out error describes the captured population. Use independently collected
evaluation data when assessing another run or workload. Keep
decode-admission/mixed rows under their original role and account for any trainer
exclusions. Retain the original selection, capture report and both exports;
the optional online-selection step uses them to exclude offline observations.

The trainer's reader can skip malformed JSON and unrecognized records, and
training can exclude incompatible rows. Its iteration/store counts do not
account for every source record or establish capture completeness. Retain the
validation and selection reports, reconcile accepted/excluded counts, and check
trainer exclusions separately. See the
[training reference](../fpm_ml/learned-forward-pass-model.md#2-train) for supported
input and split options.

## 3. Train and qualify accuracy

Use the deployment identity reviewed before collection. Confirm it still
matches the selected native files and role; resolve mismatches before fitting.

Train one AISim-compatible GBDT per role. These CLI commands support
attention DP=1: `--join-ranks none` treats each record independently, while
`--join-ranks counter` refuses cross-rank joins. For higher attention DP, the
Python `train` and `evaluate` APIs accept complete iteration groups established
using a trustworthy shared iteration identity. Producing and validating those
synchronized groups is outside this walkthrough.

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
for seed export. `--holdout-frac 0` fits all accepted training observations;
the evaluation command reads only the separate held-out file. Repeat with
separate identities/data for P/D. Each `--fpm` option also accepts multiple
paths when the accepted population spans several files.

### What training produces

The trainer fits separate GBDT ensembles for the workload classes supported by
the worker role. A *store* is the ensemble and fitting state for one such class;
having decode data does not create a mixed-batch store. `--join-ranks none`
treats each native row independently. It does not reconstruct synchronized
attention-DP iterations from unrelated rank counters.

For the default `log_ms` target, an ensemble predicts

$$
\begin{array}{rl}
g(x)&=b+\sum_{m=1}^{M}v_m(x),\\
\widehat\tau(x)&=\max(10^{-6},\exp(g(x))).
\end{array}
$$

Here $x$ is the derived feature vector, $b$ is the baseline, $M$ is the number
of fitted trees, and $v_m(x)$ includes each tree's fitted learning rate. We write
$\widehat\tau$ for predicted duration and $y$ for measured duration, both in
milliseconds; native `wall_time` remains in seconds. Fitting squared error in
log time emphasizes multiplicative timing differences, but it does not directly
minimize MAPE.

| Output | Use it for |
| --- | --- |
| `ROLE.learned.json` | Frozen prediction: feature ordering, target transform and fitted trees. |
| `ROLE.online-seed/` | Continuing training: the ensemble plus fitted bins, original observations and fitting state. |

The inference JSON cannot reconstruct the training checkpoint. Preserve both
when you want the option of online updates later. Evaluation reads separate
observations and scores predictions without updating either output.

**Proceed when** evaluation scores nonzero observations, required stores have
coverage, and error meets your chosen threshold. A successful exit alone is
insufficient. Report error, prediction coverage, train/test counts and excluded
rows together; missing stores need more representative data and another fit.

Use the [FPM Gym](https://ai-dynamo.org/aisim/fpm-accuracy/scenario-comparison.html)
as the accuracy qualification reference: select a GBDT trained through end-to-end
offline profiling and evaluated with online updates disabled. It evaluates
frozen predictions on held-out observations; MAPE covers successful predictions
only. Your own evaluation qualifies your own captured workload. Broader
generalization and Replay TTFT/TPOT accuracy need separate validation.

## 4. Use the model in Replay

Copy the [Replay template](../../../scripts/fpm_self_service/replay-aggregated.yaml) to `replay.yaml` and
replace every placeholder. It starts with aggregated vLLM, attention DP=1,
PP=1 and no context parallelism. Set the captured topology, precision and
scheduler limits, plus **measured** KV block geometry and usable capacity.
Include additional precision and kernel controls from the reviewed identity
using the [engine configuration fields](../configuration.md#flat-engine-precision-and-execution-controls).
Point `traffic.source` at your request trace, not the FPM training data. Its hash
block size describes the trace's prefix hashes; the worker's KV block size
describes the serving cache. Set each from its own source. The
[KV-cache reference](../../replay/engine/kv-cache.md) explains the units for
`bytes_per_token` and capacity.

The worker's timing block selects the trained GBDT explicitly:

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
aisimulate predict --stack engine --config replay.yaml --output-dir replay-result
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
selects the supplied GBDT. The name `best_available` does not mean it measures
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
| `export_learned_artifact()` | Return the current inference ensemble as a JSON-compatible dictionary. |

For attention DP=1, a query is `[row]` and one feedback iteration is `[[row]]`.
With attention DP>1, use one complete, synchronized list of rank records per
iteration; worker IDs and equal counter values do not establish synchronization.

## 5. Optional: adapt and save

Collect later measured observations from the same deployment. Recheck the
model/runtime, topology and serving settings against `ROLE.identity.json`;
native worker IDs alone do not establish that identity. This example supports
one worker stream with attention DP=1 (rank 0), one worker lifetime per capture,
and separate files for each P/D role.

Copy the closed later capture into a new `traces/ROLE-online/` directory under
`$FPM_RUN`, keeping the original capture, selection, receipts and train/test
exports unchanged. Validate the later files with their actual worker identity:

```bash
python3 "$AIS_REPO/scripts/fpm_accuracy/validate_native_capture.py" \
  --trace-dir "$FPM_RUN/traces/${FPM_ROLE}-online" \
  --expect "$FPM_ROLE" NAMESPACE COMPONENT PRODUCER_ID WORKER_ID 0 \
  --report "$FPM_RUN/receipts/$FPM_ROLE.online.capture.json"
```

Require a `valid` report. Save `$FPM_RUN/$FPM_ROLE.online.selection.json` using
the [selection format above](#2-check-and-prepare-the-data): point each entry at
its later capture file, use its validated hash and reviewed line range, and list
segments in capture order. Exclude smoke/pilot rows and record warmup/drain
choices as before. Then prepare the feedback file:

```bash
python3 "$AIS_REPO/scripts/fpm_accuracy/prepare_native_training.py" \
  --run-dir "$FPM_RUN" --role "$FPM_ROLE" --mode online
```

This writes `selected/ROLE.online.jsonl` and
`receipts/ROLE.online.selection.json`. It preserves the native bytes and order
of eligible busy rows, excluding idle rows and work classes incompatible with
the selected role. The receipt records hashes, admitted counts and exclusions.
Require the completed receipt before adaptation.

Online selection verifies the original offline receipts and exports, rejects
reused offline source/export files, and requires feedback observation timestamps
to be strictly later than every offline train/test row. Within the feedback,
timestamps must not go backward and counters must increase. A continuing worker
stream must also advance beyond the original capture's last counter. A timestamp
tied with the offline boundary is rejected; choose a distinct later capture instead
of relabeling records. These checks establish selection boundaries, not native
timer alignment or deployment identity. Do not rename or concatenate the
offline train/test files into feedback.

Run the [online update example](../../../scripts/fpm_self_service/adapt_gbdt.py):

```bash
python "$AIS_REPO/scripts/fpm_self_service/adapt_gbdt.py" \
  --run-dir "$FPM_RUN" --role "$FPM_ROLE" \
  --output-dir "$FPM_RUN/models/$FPM_ROLE.adapted"
```

The example loads the initial online seed through `best_available`, predicts
before admitting each label with `tune_with_fpms`, and saves an updated checkpoint
and inference ensemble. It also restores the checkpoint to check prediction
consistency. Use the exported ensemble with frozen adaptation in Replay.
The output directory contains `model.json`, `checkpoint/`, a restore
configuration and an adaptation receipt. The public
[online fitting and persistence APIs](../fpm_ml/learned-forward-pass-model.md#online-tree-adaptation)
provide these operations; a restore check does not qualify prediction accuracy.

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
coverage, input integrity and persistence. Let $M_t$ be the scheduled workload
for pass $t$, $x_t=\phi(M_t)$ its feature vector, and $\theta_t$ the model state
before that pass's label is admitted. For workload class $s$, $F_{s,t}$ predicts
milliseconds using this state:

$$
\begin{array}{rl}
\widehat\tau_t&=F_{s,t}(x_t),\\
y_t&=1000\,\mathrm{wall\_time}_t.
\end{array}
$$

`tune_with_fpms` then admits the label. Predictions change only after a
successful fitting update. `estimator_config.fpm_learned.update_every` counts
admitted observations, while
your application controls how often feedback is sampled. The example admits
every supplied row. A five-second feedback interval requires the caller to
select at most one observation every five seconds after initialization;
`update_every` does not establish a wall-clock sampling interval. The standalone
default fits after each admission; larger batches can reduce update frequency.

Online fitting retains the seed history and a bounded recent window. It can
change leaf values and eligible splits, while preserving the seed's feature
bins and tree capacity. It does not add missing workload stores. This makes
representative initial coverage important even when you plan to adapt.

### Choose how online updates change the model

The example above uses the first option. The other two are alternatives when
you want to retain a fixed GBDT or compare its calibrated prediction with an
adaptive one:

| Mode | Configuration under `estimator_config` | Effect of measured feedback |
| --- | --- | --- |
| Online GBDT | `fpm_learned.adaptation: online`, `correction.enabled: false` | Updates the existing trees; predicts from the latest fitted ensemble. |
| Hybrid calibration | `fpm_learned.adaptation: frozen`, `correction.enabled: true`, `correction.mode: hybrid` | Keeps the trees fixed and learns multiplicative timing factors. |
| Causal Selector | `fpm_learned.adaptation: causal_selector`, correction enabled in `hybrid` mode | Maintains calibrated and adaptive experts; chooses using earlier errors. Requires a training checkpoint. |

Standalone online GBDT has **no automatic fallback to the initial GBDT**.
Its checkpoint/export example also does not persist the other two modes'
calibration or Selector histories. Treat those modes as separate integrations;
do not reuse the standalone save loop as a complete state export for them.

### Calibration and fallback with sparse feedback

Let $F_{0,s}(x)$ be the frozen GBDT prediction for workload class $s$.
Calibration of this fitted GBDT builds one vector per active rank: $(Q_p)$ for
pure prefill, $(B_d,K_d)$ for pure decode, and $(Q_p,K_d)$ for locally mixed
work. With multiple attention-DP ranks, it selects the vector with the highest
load score, currently the sum of its coordinates. These calibration coordinates
$z$ need not match the GBDT feature vector $x$.

Eligible mixed-interpolation fallback maintains its own calibration history.
Its key aggregates the complete attention-DP group:

$$
z^I=\left[\max_r A_r,\ \gamma\sum_r(Q_{p,r}+B_{d,r})\right].
$$

Here $A_r$ is rank $r$'s weighted attention term, $Q_{p,r}$ is its prefill-token
count, $B_{d,r}$ is its decode-request count, and $\gamma$ is the FFN token
weight. This fallback uses the group key rather than selecting one rank's
vector. See the [self-benchmark and online fallback walkthrough](online-gbdt.md#4-calibrate-fallback-and-fit-the-missing-mixed-forest)
for its source requirements and update workflow.

For calibration of the frozen GBDT, let $y_i>0$ be an earlier measurement and
$p_i=F_{0,s}(x_i)>0$ its uncorrected prediction. Hybrid calibration records a
bounded ratio. The factor $c_t$ used for pass $t$ depends only on previously
admitted feedback:

$$
\begin{array}{rl}
r_i&=\operatorname{clip}\left(\frac{y_i}{p_i},c_{\min},c_{\max}\right),\\
\widehat\tau_t&=c_t(z_t)F_{0,s}(x_t).
\end{array}
$$

The default limits are $c_{\min}=0.5$ and $c_{\max}=2$. The ratio always
uses the **uncorrected** GBDT prediction, so repeated updates
do not compound earlier corrections. A 10 ms prediction and a 12 ms observation
provide a ratio of 1.2. This example does not imply that one observation makes
the correction history ready; the default requires five retained observations.

Hybrid calibration combines a fixed-domain factor $c_t^{\mathrm{fix}}$ and an
observation-driven factor $c_t^{\mathrm{dyn}}$. Let $\mathcal R$ be the configured
fixed domain and $I_{\mathcal R}(z)$ its membership indicator, including endpoints:

$$
c_t(z)=I_{\mathcal R}(z)c_t^{\mathrm{fix}}(z)
      +[1-I_{\mathcal R}(z)]c_t^{\mathrm{dyn}}(z).
$$

The fixed component retains only in-domain observations; the dynamic component
receives observations from both inside and outside that domain. A query chooses
its factor as follows:

| Query location | Factor used |
| --- | --- |
| Inside the fixed domain | Median ratio in its populated cell once the fixed history is ready; otherwise **1**. It does not fall through to the dynamic grid. |
| Outside the fixed domain, with a usable dynamic cell | Blend that cell's median with the median of recent feedback. |
| Outside the fixed domain, with no usable dynamic cell | Recent-feedback median, or **1** while that history is unready. |

Once the dynamic primary history is ready, let $b$ identify a populated cell,
$n_{t,b}$ its sample count and $a_{t,b}$ its median ratio. Let $g_t$ be the
recent-feedback median, or 1 while that recent history is unready. The blend is

$$
\begin{array}{rl}
w_{t,b}&=\frac{n_{t,b}}{n_{t,b}+20},\\
c_{t,b}^{\mathrm{local}}&=\exp\left(w_{t,b}\log a_{t,b}+(1-w_{t,b})\log g_t\right).
\end{array}
$$

Small local samples therefore rely mostly on recent feedback across workload
regions. An empty cell can still receive a global correction, while factor 1
preserves the base prediction when evidence is insufficient. These are
fallbacks for the **correction factor**; they cannot supply a missing GBDT.
The bounds also limit what calibration can repair: a 10 ms base prediction
cannot exceed 20 ms under the default factor limits.

### Choosing between calibrated and adaptive predictions

The causal Selector starts with the calibrated expert. It keeps a comparison
history $k=(s,q)$ for each workload class $s$ and calibration kind $q$ (prefill,
decode or mixed). The complete rank group determines $s$; the rank selected for
calibration determines $q$. These histories are separate from calibration cells.

For each admitted observation, both experts predict before either uses its
label. Let $n$ count valid paired comparisons in history $k$, and let $e$
identify the calibrated or adaptive expert. Its fractional absolute error and
exponentially weighted mean are

$$
\begin{array}{rl}
L_{k,n}^{e}&=\left|\frac{\widehat\tau_{k,n}^{e}}{y_{k,n}}-1\right|,\\
E_{k,1}^{e}&=L_{k,1}^{e},\\
E_{k,n}^{e}&=(1-\alpha)E_{k,n-1}^{e}+\alpha L_{k,n}^{e}\quad(n>1).
\end{array}
$$

Five decode comparisons do not warm up prefill or mixed selection. Elapsed time
alone does not change these averages. The inactive expert $e'$ qualifies to
replace the active expert $e$ for subsequent predictions when

$$
n\ge N_0
\quad\text{and}\quad
E_{k,n}^{e'}<(1-\delta)E_{k,n}^{e}.
$$

The condition must hold for $P$ consecutive comparisons in that history. A failed
condition or a switch resets the streak; ties keep the active expert. Defaults
are $\alpha=0.4$, $N_0=5$, $\delta=0.05$ and $P=1$. They map to `alpha`,
`warmup`, `margin` and `persistence` under `estimator_config.selector`.
For example, an active error score of 0.10 requires a challenger score below
0.095. Both experts keep receiving feedback, so selection can switch back.
The margin reduces switching on small differences; it is not a confidence bound
or a guarantee of lower error on the next pass.

### What if a workload has no fitted store?

For a model trained through end-to-end offline profiling, collect the missing
workload class and train a compatible seed. Without an offline seed, first collect
measured feedback and perform an application-managed initial fit.
`tune_with_fpms` does not turn an empty source into a GBDT.

**Self-benchmarking** implements **engine-level offline profiling** with
controlled batch shapes. That separate route can optionally use compatible
profiling tables for mixed-workload fallback, with optional online observations.
For an ordinary autoregressive pass on one rank doing both prefill and decode,
let $T_p$ and $T_d$ be the corresponding interpolation estimates:

$$
\begin{array}{rl}
\widehat\tau_{\mathrm{mix}}
  &=T_p(B_p,Q_p,K_p)+\Delta T_d,\\
\Delta T_d&=\left[T_d(B_d,K_d)-T_0(B_d;K_d)\right]_+,
\qquad [z]_+=\max(z,0).
\end{array}
$$

For direct interpolation, the decode baseline $T_0(B_d;K_d)$ uses each selected
decode curve's timing at its smallest collected KV count, combined with the
same batch-axis interpolation weights as $T_d$. It can therefore depend on
$K_d$; it is neither an empty-pass measurement nor an extrapolation to zero KV.
Subtraction treats that baseline
cost as already represented in prefill and adds only the nonnegative increment.
This is an empirical composition assumption, not an exact separation of costs.

With multiple attention-DP ranks, the interpolation estimate $I_t$ is the maximum
rank estimate for pass $t$. Prefill and decode on separate ranks are not summed.
Online observations can calibrate $I_t$ using the ratio $y_i/I_i$; this fallback
has its own histories, separate from calibration of a fitted GBDT.

This route requires its own profiles and eligibility checks and is not part of
this guide's end-to-end offline profiling route. The top-level `fallback_policy`
governs estimator selection at construction and must remain `deny` for
`fpm_learned`. It does not disable
the explicitly configured correction or Selector. Missing training coverage
remains a collection/training task.

## If a step fails

| Symptom | Next action |
| --- | --- |
| Requests succeed but no native rows appear | Inspect the active producer, worker output prefix, permissions and retention. |
| Queuing, retractions or stalled requests | Reduce offered load, check memory/transfer limits, then repeat in a fresh run directory. |
| Capture validation fails | Preserve the report; recover closed files or reconcile worker identities/restarts before selecting rows. |
| Missing stores or many excluded rows | Check role and work-kind coverage; collect the missing operating region and fit again. |
| Low FPM error but poor Replay results | Check workload semantics, native-feature mapping, scheduler settings, KV capacity and transfer behavior. |
