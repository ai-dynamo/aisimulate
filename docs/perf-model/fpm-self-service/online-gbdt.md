<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Start and adapt a GBDT with online observations

A GBDT needs measured forward passes before it can predict their duration.
You can collect those measurements ahead of deployment or build the first model
from online observations. If you start from self-benchmark data, an interpolation
fallback can cover eligible mixed workloads while their GBDTs are being trained.
Later observations let the model compare the two predictors and choose between
them.

The three data sources describe how the measurements are collected:

| Data source | Collection method |
| --- | --- |
| **A: self-benchmark** | Prepare cache state and directly measure selected batch and token shapes. |
| **B: End-to-end offline profiling** | Run selected request traces through a real serving scheduler and record the resulting forward passes. The traces, offered load and cache state determine the batch shapes observed. |
| **C: online observations** | Record forward passes during deployment, then use selected observations to adapt the predictor to the traffic and execution conditions it encounters. |

B and C can use the same FPM telemetry format. Their distinction is the
collection workflow and the role of the measurements: B supplies an offline
training population; C supplies observations during operation. A+C and B+C
start from their respective offline populations and then learn from C.

For B and B+C, begin with the [End-to-end offline profiling guide](e2e-gbdt.md). Here, choose
[C-only initialization](#2-start-with-c-only) or
[self-benchmark initialization](#3-start-with-a-self-benchmark-seed), then follow
the feedback and persistence steps.

## Before you start

The examples use explicit model artifacts or training checkpoints. Mixed
fallback is configured through `temporary_mixed_composition`; its inputs and
controls are described in the [learned-model configuration reference](../fpm_ml/configuration.md).
Keep the original profiling data and model configuration available when you
save or restore an online model. The persistence section explains which state
each save operation retains.

Prepare the training environment from the [E2E guide](e2e-gbdt.md#before-you-start).
Initial fitting runs on CPU and checkpoint export requires scikit-learn 1.9.1.
Online prediction and supported updates run through the Rust model. Inspect your
installed trainer before preparing inputs:

```bash
python -m aisim_core.sdk.fpm_learned train --help
```

Keep one model instance per worker/deployment identity when workers need
independent adaptation. A telemetry `worker_id` does not create separate model
state inside one instance. The identity must match the model, hardware and
engine configuration that produced both the seed and subsequent feedback.
Use the E2E guide's [identity preparation](e2e-gbdt.md#3-train-and-qualify-accuracy)
and native capture checks; preserve measured resource capacity separately.

| Starting data | First available prediction | Application responsibility |
| --- | --- | --- |
| C only | After a successful initial fit, or from a separately qualified prior supplied by the application | Buffer valid online observations, fit the first checkpoint, construct the model, and preserve the feedback position. |
| A only | A-trained GBDT for covered classes; qualified interpolation for eligible missing mixed classes | Preserve original A and configure the prior explicitly. Keep adaptation frozen. |
| A+C | A-trained stores and the qualified mixed prior can predict while missing mixed forests are being trained | Buffer each missing mixed class, fit and append its checkpoint, and deliver subsequent feedback. |
| B / B+C | The accepted B-trained model, subject to its store coverage | Follow the E2E guide; the A-specific fallback does not add a prior to these scenarios. |

AISim predicts pass durations, updates the constructed model and applies
the configured mixed routing rule. Your application collects observations and
fits the initial model. It also decides which observations to admit, prevents
duplicates and handles recovery from a partially completed update.
`fallback_policy: deny` controls estimator selection at construction;
it does not disable explicitly configured mixed composition within learned mode.

## 1. Keep predictions ahead of feedback

Let $M_t$ contain the workload records from all participating ranks in iteration
$t$. The feature map gives $x_t=\phi(M_t)$, and $c_{\mathrm{exec}}$ identifies the
execution configuration. With predictor state $\theta_t$, the estimated pass
duration is

$$
\widehat\tau_t=f_{\theta_t}\bigl(\phi(M_t);c_{\mathrm{exec}}\bigr).
$$

The measured duration $y_t>0$ becomes available after execution. Score
$\widehat\tau_t$ against that label before using it to update $\theta_t$.
Here, *admitting* an observation means passing its validated workload records
and measured duration to the update API. The application chooses which
observations to admit and preserves their chronological order.

For example, an application can admit one observed pass every five seconds of
deployment time while continuing to predict every pass.
This sampling schedule is separate from `update_every`, which counts admitted
observations before fitting. An admitted observation can remain pending until
a later fit. The file-based examples below replay recorded online observations
to demonstrate the same prediction-before-update order.

Native `wall_time` labels are in **seconds**. Preserve them in the dictionaries
passed to the trainer and `tune_with_fpms`; the trainer converts the target to
milliseconds. The prediction API returns milliseconds. For a complete rank
group, the training target is the maximum rank wall time, multiplied by 1,000.
An aggregate self-benchmark label needs its documented carrier-rank convention,
described in step 3, rather than invented rank measurements.

```python
# measured_group is one iteration's list of complete rank records.
query = [{k: v for k, v in rank.items() if k != "wall_time"}
         for rank in measured_group]
prediction_ms = model.estimate_forward_pass_time_ms(query)
# Record prediction_ms, including None, before making the update.
model.tune_with_fpms([measured_group])
```

The outer list in `tune_with_fpms([measured_group])` contains iterations; the
inner list contains their rank records. Keep synchronized attention-DP ranks
together. Rank-local counters alone do not establish a common iteration.
Reject malformed or incompatible observations before admission, and retain the
reason. A supported query with no estimate remains a coverage miss; invalid
identity, corrupted sources and numerical errors need investigation.

### Track readiness per workload class

An aggregated model has up to four forest stores:

| Store | Scheduled work |
| --- | --- |
| `pure_prefill` | Prefill without decode. |
| `pure_decode` | Decode without prefill. |
| `contains_locally_mixed` | At least one rank runs both phases. |
| `cross_rank_aggregated` | Both phases occur in the group, on separate pure ranks. |

Count training observations independently for each worker and store. A ready
decode forest does not make a missing mixed forest ready. Also distinguish
three questions: does the forest exist, can the configured model answer this
query, and has comparison evidence selected that forest over the prior? The
answers can differ during startup.

## 2. Start with C only

With no seed and no qualified prior, return an unavailable prediction for
non-empty scheduled work while collecting the first training population.
Constructed AISim models return `0.0` ms for valid rank records with no
scheduled work. Before constructing a model, the application must handle idle
inputs itself if it accepts them. An empty list of rank records is invalid;
it does not describe an idle iteration. The example below accepts only busy
decode observations.

Continue collecting real serving measurements; simulation outputs are not
labels. Keep a buffer for each worker and workload class $s$. For the buffer
in this example, write

$$
\mathcal H_{s,t}=\{(i,x_i,y_i): i\leq t,\ \text{observation }i\text{ admitted to class }s\},\qquad
n_{s,t}=|\mathcal H_{s,t}|.
$$

The trainer's default minimum is twenty valid rows per store. Once the
buffer reaches that size, the application can attempt an initial fit. If fitting
succeeds after observation $t$, the new forest becomes available to subsequent
queries. Observation $t$ keeps its original unavailable result; fitting does
not retroactively create a prediction for it.

For the default `log_ms` target, the forest approximates log duration:

$$
g_s(x)=b_s+\sum_{m=1}^{M}v_{s,m}(x),\qquad
\widehat\tau_s(x)=\max(10^{-6},\exp g_s(x)).
$$

Here $b_s$ is the fitted intercept and each tree contributes $v_{s,m}(x)$,
including its learning rate. The forest output $g_s$ is in log milliseconds;
$\widehat\tau_s$ is in milliseconds. In this equation, $M$ counts trees, while
$M_t$ above denotes an iteration's workload records.

Twenty observations permit fitting; they do not establish accuracy over the
workload domain. The threshold applies to accepted training rows, so twenty raw
records with exclusions may still be insufficient. A rarely observed mixed
class can remain cold long after the pure classes are ready.

### Fit a first checkpoint

The following CPU example follows a **single decode worker with attention
DP=1**. Start with an already validated chronological file
`c.decode.jsonl`, containing only that worker's busy decode observations, and a
matching `decode.identity.json`. Each retained row is admitted in this example.
A sampled application must feed the buffer only the rows its admission policy
selected. Keep a fresh `c-first-seed` destination; the trainer creates it.

```python
import json
import math
from copy import deepcopy
from pathlib import Path

from aisim_core.sdk import RustForwardPassPerfModel, fpm_learned

identity = json.loads(Path("decode.identity.json").read_text())
if identity["worker_type"] != "decode" or identity["attention_dp"] != 1:
    raise ValueError("This example requires a decode worker with attention DP=1")

seed = Path("c-first-seed").resolve()
buffer = []
model = None
for row in fpm_learned.iter_fpm_records(["c.decode.jsonl"]):
    scheduled = row["scheduled_requests"]
    if (scheduled["sum_prefill_tokens"] != 0
            or scheduled["num_decode_requests"] <= 0
            or not math.isfinite(row["wall_time"]) or row["wall_time"] <= 0):
        raise ValueError("Validate busy decode observations before this example")
    group = [row]
    query = [{k: v for k, v in row.items() if k != "wall_time"}]
    prediction = None if model is None else model.estimate_forward_pass_time_ms(query)
    print(json.dumps({"prediction_ms": prediction, "measured_ms": 1000 * row["wall_time"]}))

    if model is None:
        buffer.append(group)
        if len(buffer) < 20:
            continue
        artifact = fpm_learned.train(
            buffer, "decode", min_store_rows=20, early_stopping=False,
            online_checkpoint_dir=seed, identity=identity,
        )
        if "pure_decode" not in artifact["stores"]:
            raise RuntimeError("Inspect accepted training rows before retrying")
        config = deepcopy(identity)
        config.update(estimation_mode="fpm_learned", fallback_policy="deny")
        config["estimator_config"] = {
            "fpm_learned": {"checkpoint_path": str(seed), "adaptation": "online"},
            "correction": {"enabled": False},
        }
        model = RustForwardPassPerfModel.best_available(config)
        buffer.clear()
    else:
        model.tune_with_fpms([group])
```

The first twenty labels are already training history in the checkpoint. The
example begins `tune_with_fpms` with later labels so those initial rows are
admitted once. It fails on an error and leaves the inputs for inspection; a live
application also needs a durable buffer and feedback cursor before training.
Use the same boundary when fitting asynchronously: record the last training
label and deliver later labels only after publishing the candidate.

For an aggregated C-only worker, collect separate buffers for the required
classes and train a checkpoint from their accepted union. Direct queries may
use the stores already fitted, but native Replay/Recommend require all stores
needed by the role. These APIs do not provide a general C-only store-append
operation. When another class first appears, the application must prepare and
qualify a replacement model; replaying retained observations may be necessary
to reproduce its intended history. The specialized append API in step 4 is
restricted to eligible A+C mixed stores.

**Proceed when** the initial checkpoint loads, its required stores exist, and
later queries return valid predictions. Report startup duration and coverage
before assessing accuracy on subsequent observations.

## 3. Start with a self-benchmark seed

Self-benchmarking often supplies pure-prefill and pure-decode data before it
supplies mixed data. Train the covered classes from A and preserve that original
population. For each mixed class, native construction verifies whether original
A contains enough accepted groups and a matching trained store. That decision
fixes fallback eligibility for the instance's lifetime. C can add an online
forest later without changing the meaning of original A.

### Prepare the A groups and timing prior

Retain the accepted self-benchmark table and metadata from
[self-service collection](README.md#workflow). Training needs complete FPM
iteration groups, while interpolation reads the measured timing table. These
are different representations of the same recorded experiment.

The training API below accepts FPM groups. Converting a self-benchmark table
requires an exporter whose grouping matches that table's schema. The Gym
`self_benchmark_iterations` adapter supports schema-6 `balanced_v1` artifacts in a validated measurement
case. It checks source hashes and row counts. It is a source-checkout helper,
not an installed onboarding API, and it rejects schema-7 inputs. Select one
worker's exported groups so repeated worker copies do not inflate training
counts. Other schemas need a separately verified conversion before this step.

Run this optional adapter from the repository root in the activated training
environment. It also needs the Gym dataset dependency, which is separate from
the learned-model extra:

```bash
uv pip install 'huggingface-hub==1.32.0'
```

The following exports a registered artifact. Replace the uppercase values with
the retained dataset's local root, immutable revision and exact manifest
identifiers. The case must contain the selected worker's observations. Check
that both imported modules are available before using the conversion example.

```python
import json
from pathlib import Path

from scripts.fpm_accuracy.hf import HfDataset
from scripts.fpm_accuracy.models.scenario_predictors import self_benchmark_iterations

dataset = HfDataset.from_local(DATASET_ROOT, revision=PINNED_DATASET_SHA)
case = dataset.measurement_case(
    CONFIGURATION_PATH, snapshot_id=SNAPSHOT_ID, fpm_artifact_ids=[ARTIFACT_ID],
)
source = next(a for a in case.fpm_artifacts if a.artifact_id == ARTIFACT_ID)
a_groups = [
    observation.tuning_payload()
    for observation in self_benchmark_iterations(case, source)
    if observation.ranks[0].worker_id == WORKER_ID
]
if not a_groups:
    raise ValueError("No accepted A groups for the selected worker")
metadata = json.loads(source.local_metadata_path.read_text())
provenance = {
    "source_sha256": source.sha256,
    "metadata_sha256": source.metadata_sha256,
    "artifact_id": source.artifact_id,
    "sources": [
        {"path": str(source.local_path.resolve()), "sha256": source.sha256},
        {"path": str(source.local_metadata_path.resolve()), "sha256": source.metadata_sha256},
    ],
    "grouping": {
        "coordinate_system": metadata["coordinate_system"],
        "partition_policy": "balanced_v1",
        "rank_workload_reconstruction": "replicated_table_coordinates",
        "target_scope": "recorded_aggregate_latency",
    },
}
for filename, value in (
    ("a.accepted-iterations.json", a_groups), ("a.provenance.json", provenance),
):
    with Path(filename).open("x") as out:
        json.dump(value, out, allow_nan=False)
```

For that schema-6 convention, each rank receives the table's recorded balanced
coordinates. Do not divide them by attention DP again. A recorded aggregate
latency is carried on rank zero, in seconds; other ranks use the established
zero sentinel. Those zeros do not claim measured zero-cost ranks. Preserve the
conversion convention and original table hashes with the output.

The example below expects `a.accepted-iterations.json` to contain a JSON array
of those reviewed groups, and `a.provenance.json` to identify the source files:

```json
{
  "source_sha256": "REPLACE_WITH_PARQUET_SHA256",
  "metadata_sha256": "REPLACE_WITH_METADATA_SHA256",
  "sources": [
    {"path": "/absolute/path/fpm.parquet", "sha256": "REPLACE_WITH_FILE_SHA256"},
    {"path": "/absolute/path/fpm.metadata.json", "sha256": "REPLACE_WITH_FILE_SHA256"}
  ],
  "grouping": {
    "partition_policy": "balanced_v1",
    "rank_workload_reconstruction": "replicated_table_coordinates",
    "target_scope": "recorded_aggregate_latency"
  }
}
```

Use these grouping values only for the schema-6 adapter described above.
The hashes identify the source files. To verify the conversion itself, compare
the exported counters and targets with the source before fitting.
Keep any A groups below a store's fitting threshold:
they still matter to eligibility and calibration geometry.

### Train and bind original A

Use a complete `aggregated.identity.json` for the measured deployment. Its
systems roots and any `fpm_profile` must also resolve the qualified interpolation
source. This example preserves its other estimator settings while adding
explicit learned controls.

```python
import json
from copy import deepcopy
from pathlib import Path

from aisim_core.sdk import RustForwardPassPerfModel, fpm_learned

identity = json.loads(Path("aggregated.identity.json").read_text())
if identity["worker_type"] != "aggregated":
    raise ValueError("Mixed composition requires an aggregated worker")
a_groups = json.loads(Path("a.accepted-iterations.json").read_text())
provenance = json.loads(Path("a.provenance.json").read_text())
a_seed = Path("a-seed").resolve()  # Must not exist yet.
artifact = fpm_learned.train(
    a_groups, "aggregated", min_store_rows=20, early_stopping=False,
    online_checkpoint_dir=a_seed, identity=identity,
)
original_a = fpm_learned.write_original_a_manifest(
    "original-a.json", a_groups, "aggregated", identity=identity,
    min_store_rows=20, seed_artifact_paths=[a_seed / "model.json"],
    provenance=provenance,
)

config = deepcopy(identity)
config.update(estimation_mode="fpm_learned", fallback_policy="deny")
estimator = config.setdefault("estimator_config", {})
estimator["fpm_learned"] = {"checkpoint_path": str(a_seed), "adaptation": "online"}
estimator["correction"] = {"enabled": False}
estimator["temporary_mixed_composition"] = {
    "scenario": "a_c",
    "mode": "compose_calibrated",
    "original_a": original_a,
    "quality_gate": {"alpha": 0.05, "min_comparisons": 5},
}
model = RustForwardPassPerfModel.best_available(config)
print(model.mixed_composition_diagnostics())
```

`write_original_a_manifest` writes a new immutable file and returns
`manifest_path` and `source_sha256`, ready for the configuration above. It binds
the original groups to the seed artifacts and their training checkpoint.
Here the returned `source_sha256` hashes the generated manifest; the digest
inside `provenance.source_sha256` hashes the original Parquet file.
Construction checks identity, ordered features and accepted per-store counts.
Create the manifest before mixing C into the population. Enough original A rows
without the corresponding fitted store is an input error, not permission to
hide the broken seed behind interpolation.

The prior uses the configured `fpm_interpolation` method. Registered analytical
models may use SOL; direct interpolation needs a compatible timing source and
measured support for the requested coordinates. A verified `fpm_profile` is the
guided onboarding route; registered tables and an explicit `fpm_parquet_path`
are other supported inputs. The model pins the resolved
sources and does not try a different interpolation method after a miss. Inspect
both mixed classes with representative queries before starting adaptation.

For **A-only**, use the same immutable A inputs with
`fpm_learned.adaptation: frozen`, `scenario: a`, and `mode: compose`; omit
`quality_gate` and submit no C updates. Eligible mixed queries then use raw
interpolation. Ordinary A-supported stores keep their trained forests.

**Proceed when** seed construction and source checks pass, fallback eligibility
matches the original data, and the prior answers the mixed queries you need.
Eligibility alone does not guarantee coverage of every batch/KV coordinate.

## 4. Calibrate fallback and fit the missing mixed forest

For an eligible class $s$, let $I_t$ be the raw interpolation estimate for the
whole iteration. To construct it, consider a rank running both prefill and
decode. Here, $B_p$ and $B_d$ count requests, $Q_p$ counts new
prefill tokens, and $K_p$ and $K_d$ sum past KV lengths. The rank's estimate is

$$
\widehat\tau_{\mathrm{mix}}=T_p(B_p,Q_p,K_p)+\Delta T_d,\qquad
\Delta T_d=\bigl[T_d(B_d,K_d)-T_0(B_d;K_d)\bigr]_+,
$$

where $[u]_+=\max(u,0)$. The prefill and decode estimates are $T_p$ and $T_d$;
$T_0$ is the decode baseline from the measured curves used by that query.
The composition treats this baseline as already represented by the prefill
estimate and adds the remaining nonnegative decode cost. For example, 2 ms
prefill, 3 ms decode and a 1 ms baseline give a 4 ms mixed estimate.

This is an empirical approximation. The baseline comes from the curves'
minimum measured KV coordinates, not an assumed empty pass. Native
interpolation applies the bracketing rules and combines rank estimates; pass
complete FPM records to the API rather than reconstructing that calculation
in the application. With attention DP=1, $I_t=\widehat\tau_{\mathrm{mix}}$.
With multiple ranks, use the whole iteration's $I_t$ for calibration so it has
the same scope as the measured duration $y_t$.

With A+C, online observations calibrate the interpolation estimate:

$$
r_i^I=\operatorname{clip}\left(\frac{y_i}{I_i},c_{\min},c_{\max}\right),\qquad
\widehat\tau^I_{s,t}=c^I_{s,t}(z_t)I_t.
$$

The factor $c^I_{s,t}$ uses only observations admitted before prediction $t$;
$z_t$ contains the attention and tokenwise-work scores used for calibration.
For each admitted observation $i$, calibration divides $y_i$ by **raw**
interpolation; dividing by the corrected prediction would compound previous
corrections. Before the relevant history is ready, the factor is one.

The default correction bounds are $[0.5,2]$. A four-by-four grid groups similar
workloads, and a correction history needs five observations before it is ready.
The fixed grid covers the domain established by the original self-benchmark
data. Queries outside that domain use the dynamic component, whose geometry
grows with online observations. Each mixed class keeps its own histories.
The [calibration reference](../fpm_ml/mixed-workload-fallback.md#independent-mixed-calibration)
describes these coordinates and retention rules.

While the prior predicts, collect a separate C buffer for each eligible missing
mixed class. Score the query first, call `tune_with_fpms([group])` once for an
admitted observation, and record successful admission before adding it to that
class's initial-fit population. Continue calibrating the prior while the new
forest is being prepared.

### Append a single trained mixed store

After at least twenty valid admitted observations of one class, train a fresh
single-class checkpoint. This continuation of step 3 reads reviewed, complete
groups from `c.admitted-iterations.json`, in chronological order. The file
contains only labels selected for admission; a live application also scores
queries between those admissions. Verify that the chosen class is absent and
eligible in the initial model's diagnostics before running this example.

```python
kind = "contains_locally_mixed"  # Or cross_rank_aggregated, in a separate fit.
mixed_seed = Path("c-first-locally-mixed").resolve()
mixed_groups = []
installed = False
for group in json.loads(Path("c.admitted-iterations.json").read_text()):
    query = [{k: v for k, v in rank.items() if k != "wall_time"} for rank in group]
    result = model.estimate_forward_pass_with_route(query)
    print(json.dumps(result))  # Persist this pre-update prediction and route.
    model.tune_with_fpms([group])
    if installed or fpm_learned.classify_workload(group, "aggregated") != kind:
        continue
    mixed_groups.append(group)
    accepted = fpm_learned.build_dataset(mixed_groups, "aggregated", artifact["features"])
    if kind not in accepted or len(accepted[kind][1]) < 20:
        continue
    mixed_artifact = fpm_learned.train(
        mixed_groups, "aggregated", features=artifact["features"],
        target=artifact["target"], min_store_rows=20, early_stopping=False,
        online_checkpoint_dir=mixed_seed, identity=identity,
    )
    if set(mixed_artifact["stores"]) != {kind}:
        raise RuntimeError("Expected exactly the missing mixed store; inspect input groups")
    model.append_mixed_learned_checkpoint(mixed_seed)
    installed = True
    mixed_groups.clear()
```

`append_mixed_learned_checkpoint` validates the checkpoint before mutation. It
accepts one absent, originally eligible mixed store for A+C online or selector
adaptation. It rejects a runtime checkpoint, duplicate store, mismatched
identity/features/target, or a class outside this policy. Existing forests and
their pending updates remain in place, along with calibration and comparison
histories. This avoids replacing the whole live A+C model.

The label that completes the initial-fit buffer was already predicted using the
prior. It can calibrate that prior and train the new forest, but it cannot
compare two pre-update experts: the new forest did not exist at prediction
time. Do not resubmit the initial training labels after append. Future labels
update the new forest and can supply valid paired comparisons.

**Proceed when** append succeeds and diagnostics show that the new online store
is ready. Selection may still favor interpolation, as the next step explains.

## 5. Select between fallback and the warmed-up GBDT

Choose a routing policy explicitly. The standalone quality gate in step 3
keeps interpolation until subsequent comparisons justify the GBDT. Removing
`quality_gate` instead prefers an ordinary GBDT prediction as soon as one is
available. Frozen hybrid calibration keeps its forests frozen and does not
train a mixed replacement from C.

### Continuous quality comparison

Once the new forest can predict, compare it with calibrated interpolation on
later observations. Both predictions must be made before either expert uses
the label. Keep a separate comparison history for each $k=(s,q)$: $s$ is the
workload class and $q$ is the critical rank's calibration kind (prefill, decode
or mixed). The kind follows the scheduled work, not the measured duration.

Superscript $I$ denotes interpolation and superscript $A$
denotes the adaptive GBDT. This expert label $A$ is separate from the data-source
label A for self-benchmark. For paired admitted observation $n$, compute each
expert's fractional absolute error:

$$
L^e_{k,n}=\left|\frac{\widehat\tau^e_{k,n}}{y_{k,n}}-1\right|,
\qquad e\in\{I,A\}.
$$

The index $n$ counts paired comparisons for $k$. The initial training rows and
queries without two valid predictions do not increase it. The first loss
initializes each expert's error average; later comparisons update it as

$$
E^e_{k,1}=L^e_{k,1},\qquad
E^e_{k,n}=(1-\alpha_{\mathrm{qual}})E^e_{k,n-1}
+\alpha_{\mathrm{qual}}L^e_{k,n},\quad n>1.
$$

After that comparison, the standalone gate prefers GBDT for later predictions
when

$$
n\geq N_{\mathrm{qual}}\quad\text{and}\quad E^A_{k,n}\leq E^I_{k,n}.
$$

The example uses the default $\alpha_{\mathrm{qual}}=0.05$ and
$N_{\mathrm{qual}}=5$, configured as `quality_gate.alpha` and
`quality_gate.min_comparisons`. Equality favors the adaptive GBDT. Both experts
continue receiving feedback, so a later error reversal can return predictions
to interpolation. Lower past error does not guarantee the better next
prediction. Each comparison history warms up independently; fitting a mixed
store does not supply five comparisons for every query kind in that store.

### A worked timeline

This illustrative sequence concerns one eligible mixed store and one query type.
Every row represents an admitted label; other unadmitted queries can occur
between them. Assume both experts answer all post-fit queries, and use the
five-comparison gate above.

| Admitted observation | Prediction made before this label | Action after observing the label | Choice for subsequent queries |
| --- | --- | --- | --- |
| 1-19 | Raw or calibrated interpolation | Calibrate the prior and grow the class buffer. | Interpolation; no online forest yet. |
| 20 | Calibrated interpolation | Admit once, fit the twenty-row checkpoint, then append it. | Interpolation; paired-comparison count is still zero. |
| 21-24 | Calibrated interpolation | Compare both pre-update experts, then update them. | Interpolation; only one through four comparisons exist. |
| 25 | Calibrated interpolation | The fifth pair gives, for illustration, $E^I_{k,5}=0.12$, $E^A_{k,5}=0.08$. | GBDT for the next covered query. |
| Later | GBDT | Continued paired feedback eventually gives $E^I_{k,n}=0.09$, $E^A_{k,n}=0.11$. | Return to interpolation for the next covered query. |

The values illustrate routing, not measured accuracy. With one admission every
five seconds, twenty admissions need about 100 seconds only if this class
appears at every tick. Sparse classes and missing paired predictions take
longer. Record the admission schedule, fitting threshold and comparison settings
with each evaluation so another run can reproduce its startup and switching
behavior.

### Causal selector and availability exceptions

`adaptation: causal_selector` maintains a calibrated expert $C$ and an adaptive
expert $A$. These superscripts name predictors, not data sources. For a class
with an offline GBDT, the two predictions are

$$
\widehat\tau^C_{k,t}=c_{k,t}(z_t)F_{0,s}(x_t),\qquad
\widehat\tau^A_{s,t}=F_{s,t}(x_t).
$$

$F_{0,s}$ is the fixed source GBDT and $F_{s,t}$ is its adaptive counterpart;
both return milliseconds. For an eligible mixed class without an A-trained
GBDT, calibrated interpolation supplies the stable prediction instead, and the
online observations train the adaptive forest.

The Selector uses the same causal loss calculation, with its own EWMA parameter
$\alpha$. Let $e$ be the active expert and $e'$ the alternative. It switches
only after at least $N_0$ paired comparisons and when

$$
n\geq N_0,\qquad E^{e'}_{k,n}<(1-\delta)E^e_{k,n}
$$

holds for $P$ consecutive comparisons. The relative margin $\delta$ prevents a
small error difference from triggering a switch; a tie keeps the active expert.
The reference settings are $\alpha=0.4$, $N_0=5$, $\delta=0.05$ and $P=1$.
They correspond to `alpha`, `warmup`, `margin` and `persistence` under
`estimator_config.selector`. The Selector rejects the standalone `quality_gate`
because it already has its own selection rule. See the
[selector reference](../fpm_ml/online-gbdt-and-causal-selection.md#hybrid-calibration-with-a-causal-selector)
for configuration details.

If interpolation cannot answer a supported query but the adaptive forest can,
native routing may use the forest before comparison warmup completes. Record this as
an availability decision. It creates no paired evidence and does not establish
that GBDT has passed the quality comparison. If neither expert answers, record
an unavailable query. Hard source, identity and prediction errors remain errors.

### Inspect the prediction and its route together

```python
result = model.estimate_forward_pass_with_route(query)
print(result["latency_ms"])
route = result["temporary_mixed_composition"]
if route is not None:
    print({key: route.get(key) for key in (
        "workload_kind", "fallback_eligible", "route", "online_ready",
        "paired_comparisons", "warmup_complete", "selected_expert",
        "preferred_expert", "coverage_escape",
    )})
# Save this result before admitting the measured group.
```

`route` identifies the actual interpolation or ordinary path. The preferred
expert and actual path can differ during an availability exception. Some fields
exist only for eligible mixed queries or enabled selection policies; a missing
field is not a zero counter. Expert availability can be null when that expert
was not queried. This combined API avoids predicting once and then querying
again solely to determine the route.

For general application-owned fallback outside this native A policy, use two
separately prepared compatible models and an explicit caller routing rule.
Fallback on a supported absence, not on every exception. Preserve each expert's
pre-update prediction if you compare their errors, and record which model
answered. That wrapper remains application logic; it is not automatically part
of Replay/Recommend.

## 6. Save, resume and hand the model to Replay

For **standalone online GBDT**, save to a fresh directory after recording the
last admitted input position. The causal-selector variant discussed above
rejects these single-forest export/checkpoint calls; see its persistence limit
in the table below.

```python
checkpoint = Path("online-checkpoint-0001").resolve()
model.save_online_checkpoint(checkpoint)
with Path("online-forest-0001.json").open("x") as out:
    json.dump(model.export_learned_artifact(), out, allow_nan=False)

resume_config = deepcopy(model.diagnostics()["provenance"]["config"])
learned = resume_config["estimator_config"]["fpm_learned"]
learned.pop("artifact_path", None)
learned.pop("source_sha256", None)
learned["checkpoint_path"] = str(checkpoint)
resumed = RustForwardPassPerfModel.best_available(resume_config)
before = model.estimate_forward_pass_time_ms(query)
after = resumed.estimate_forward_pass_time_ms(query)
if before != after:
    raise RuntimeError("Inspect checkpoint, identity and source dependencies")
```

This checkpoint route supports standalone online continuation, including its
mixed-composition state. Keep the original-A manifest and referenced seed/data
files available at the configured paths; the checkpoint is not a relocatable
bundle of every prior dependency. Preserve your application's buffers for
stores not yet fitted too. The model does not save an external stream cursor or
save it atomically with the checkpoint. Resume after the last admitted label,
and reconcile failures before replaying a partially accepted batch.

Saving retains pending online rows. `flush_updates()` fits pending rows now;
call it only when that is the intended publication boundary. A fit failure can
occur after a row was accepted, so a retry must follow the admission counters
rather than resubmit the label blindly. An exported forest contains the fitted
predictor, without pending rows or the complete calibration/selection policy.

| State to preserve | Save operation and limitation |
| --- | --- |
| Ordinary prepared model | `save_state_snapshot()` packages supported ordinary state; restore compatibility is tied to the writer's core version. |
| Standalone online model with mixed composition | `save_online_checkpoint()` plus the original sources/configuration and application state. Check restoration on representative queries. |
| Complete mixed policy as a portable snapshot | `save_state_snapshot()` rejects `temporary_mixed_composition`. |
| Frozen mixed calibration or full mixed causal selector | Partial mixed-state APIs do not supply complete continuation. Retain sufficient original inputs and causal history to reconstruct the policy, or keep the live instance; do not claim an exact restore from the forest alone. |

Native Replay/Recommend consumes a prepared model and does not collect or admit
C. The configured native mixed route can satisfy an eligible missing store's
readiness requirement, but an unsupported query still fails the replay. A
Python fallback wrapper cannot supply that native route. Where a consumer needs
a portable snapshot, prepare a sufficiently covered ordinary learned model or
use another qualified supported configuration; the mixed snapshot restriction
still applies.

Check the required stores: dedicated prefill/decode roles need their respective
pure stores; aggregated roles need both. Aggregated vLLM/TRT-LLM also needs
locally mixed support, and aggregated attention DP greater than one needs
cross-rank support. The supported SGLang scheduler does not require a locally
mixed store. Identity, resource capacity and consumer restrictions remain
separate checks. Use the [Replay walkthrough](e2e-gbdt.md#4-use-the-model-in-replay)
and [native consumer reference](../configuration.md#choosing-a-forward-pass-api).

Keep the last accepted model and its configuration until the new one passes
the intended query and serving checks. To roll back predictions, restore that
accepted version and record the version change. To resume its learning, also
restore its matching cursor and buffers; decide explicitly which later labels
to replay. Switching the forest file alone does not restore the application.

## 7. Verify coverage and accuracy

For $N$ eligible scored observations, let $\mathcal S$ contain the observations
with a successful prediction. Report

$$
\operatorname{coverage}=\frac{|\mathcal S|}{N},\qquad
\operatorname{MAPE}=\frac{100}{|\mathcal S|}
\sum_{t\in\mathcal S}\left|\frac{\widehat\tau_t}{y_t}-1\right|.
$$

MAPE is undefined when $\mathcal S$ is empty. C-only startup misses remain in
$N$. For a combined policy, report interpolation and GBDT query counts, plus
unavailable queries, per workload class. Define fallback share as successful
interpolation predictions divided by $|\mathcal S|$, and retain both counts.
The overall MAPE describes the combined policy; it is not a GBDT-only score.
When comparing methods with
different coverage, also compare a common successful-prediction population
and retain each method's full coverage result.

Keep the seed configuration and immutable data references with each result.
Record the admission schedule, actual initial-fit counts and the comparison
settings. Inspect switch points and availability exceptions as well as average
error. The [accuracy review guide](../../../benchmarks/evidence/fpm_gbdt_documentation_review_20261103/README.md)
explains how to compare these workflows and retain the evidence needed to
interpret their coverage and limits. Forward-pass accuracy, native replay completion and matched serving
TTFT/TPOT are separate results.

## If a step fails

| Symptom | Next action |
| --- | --- |
| C-only model never appears | Check admitted counts by worker/store, trainer exclusions and fitting errors. Raw trace length is not the accepted training count. |
| A seed lacks a pure forest | Collect or fit that class. The mixed policy cannot supply pure-prefill or pure-decode fallback. |
| Manifest or append rejects inputs | Check original source hashes, identity, ordered features, target and store membership. Use a new checkpoint destination for a corrected fit. |
| GBDT is ready but interpolation remains selected | Inspect paired-comparison count and error for this store/query key. Initial training labels do not count as comparisons. |
| Replay or restore fails | Check query coverage and native restrictions, then source paths and the persistence mode. Complete mixed portable snapshots are unsupported by the save operations described here. |
