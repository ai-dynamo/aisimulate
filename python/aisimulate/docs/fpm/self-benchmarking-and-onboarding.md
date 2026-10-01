<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Guide: self-benchmark and onboard a forward-pass performance model

This guide explains how to collect engine forward-pass measurements, turn them
into an FPM data profile, and use that profile in AISimulate. **Self-collection**
means running the engine's self-benchmark on your target configuration.
**Onboarding** means validating and loading its measured profile through the
canonical performance-model API, then using it for prediction.

The procedure has seven technical steps. For an agent-led session, follow the
[six onboarding stages](../../../../docs/fpm-self-service.md#onboard-with-an-agent),
which also cover input discovery, review, acceptance, and checkpointing. The
`aisimulate onboard ...` commands below implement that guided path. This guide
then follows the data from collection through consumption and validation.
The model, hardware, and engine settings are inputs to that procedure. [Kimi K3 TP8+DCP8](#example-a-onboard-the-collected-kimi-k3-tp8dcp8-profile)
and [MiniMax-M2.7 TP4](#example-b-collect-a-minimax-m27-tp4-profile) are worked
examples at the end; their values are not defaults for other deployments.

## When self-collection is useful

Use self-collection when you need iteration latency for a specific engine
configuration that the existing performance data does not represent well:

- A custom engine build, newer kernel, quantization mode, or CUDA Graph policy
  changes forward-pass execution time.
- Your GPU/topology or workload shapes lack measured coverage.
- You need the combined effect of operations, launch overhead, and communication
  within the measured engine iteration, rather than composing their costs from
  separate operation tables.
- You want to calibrate a supported configuration against the runtime you will
  deploy, while keeping the rest of AISimulate's serving simulation unchanged.

For example, collecting Kimi TP8+DCP8 measures that configuration's prefill and
decode costs, including DCP communication inside the timing boundary. A TP8
profile without DCP is a different measurement identity and cannot substitute
for those observations. If suitable measured data already exists, reuse it
starting at step 4 after checking its provenance, matching resource information,
and coverage. A measured timing table and an onboarding resource profile are
different artifacts; a timing table alone does not determine serving memory.

## What self-collection covers

FPM onboarding supplies **customized engine step times** for the measured
configuration and covered workload shapes. AISimulate Replay uses these timings
with its existing scheduling, routing, prefill–decode transfer, and KV-cache
models. Special or customized mechanisms that differ from supported Replay
behavior, such as layer-wise KV transfer or a custom scheduling/overlap policy,
require corresponding Replay-side changes; collecting new step times alone does
not implement those mechanisms.

The current collection workflow requires **PP=1**. Validate TTFT, ITL, and
end-to-end throughput separately against the target serving configuration;
matching engine step times alone does not establish their accuracy.

## Current support and architecture readiness

| Backend | Self-benchmark collection status in this workflow |
| --- | --- |
| vLLM | Available for supported configurations; validate the exact model, engine build, and benchmark path before a campaign |
| SGLang | Coming soon; not supported by this self-benchmark collection workflow today |
| TensorRT-LLM | Coming soon; not supported by this self-benchmark collection workflow today |

This table describes **self-benchmark collection**, not AISimulate's broader
operation-level modeling support. The dedicated Collector currently uses
`PP=CP=1`; its topology presets do not enumerate DCP. The Kimi DCP8 example
imports a profile produced by an adapted vLLM/Dynamo self-benchmark path.

**Self-benchmarking does not automatically support every model or architecture.**
An engine being able to serve a model does not prove the benchmark can seed its
state, schedule the requested work, and time it correctly. New architectures or
engine versions can require changes in:

- **Benchmark input and state preparation:** attention KV warmup, recurrent or
  convolution state, per-request state allocation, and distributed cache layout.
- **Scheduling and measurement hooks:** requested versus actually scheduled
  token counts, graph paths, completion checks, and per-rank timing aggregation.
- **AISimulate input and memory support:** architecture/configuration parsing,
  precision identity, and cache geometry/capacity used by the consumer. Direct
  FPM uses an explicit identity/resource profile without an op-level model class;
  unknown runtime layouts still need an audited observation adapter or supported
  resource declaration. A collection plan is not proof of consumer support.

Use a small smoke run to qualify those paths before collecting the full surface.
If it fails or silently schedules different work, fix and revalidate the
benchmark adapter before using the measurements. For a new architecture, start from a
[local model config](../../../../docs/fpm-self-service.md#start-from-a-local-model-config)
and investigate unresolved precision/cache facts before asking the user.
[How to add a new model](../add_a_new_model.md) describes the separate registered
analytical-model route when op-level or SOL estimation is wanted. Neither a
dedicated class nor per-operation silicon data is required for direct FPM.

## Workflow at a glance

| Step | Action | Required result |
| --- | --- | --- |
| [1](#1-check-support-and-prepare-the-environment) | Check backend/model support and prepare the environment | Compatible engine, benchmark, and AISimulate builds; known adaptation gaps |
| [2](#2-freeze-the-engine-configuration-and-measurement-plan) | `onboard init`, review/accept, then `onboard plan` | Reviewed role/configuration and a saved plan; runtime readiness still unchecked |
| [3](#3-smoke-test-and-collect) | `onboard collect-fpm`: preview, smoke/readiness, then execute | Measurements for each selected role/phase plus raw evidence and provenance |
| [4](#4-validate-and-install-the-fpm-profile) | Inspect the pair and run `onboard validate-collection` | Matching Parquet/metadata pair and separate validity, repeatability, and holdout results |
| [5](#5-construct-and-check-the-performance-model) | `onboard finalize` when memory is pending; check canonical queries | Reviewed resolved resource profile, matching estimator, and checked lookups |
| [6](#6-connect-the-model-to-replay) | `onboard validate-fpm` and ordinary `predict`/`recommend` for aggregated profiles | Reproducible replay and coverage reports, separate from accuracy |
| [7](#7-validate-coverage-and-accuracy) | Assess matched serving with `onboard validate-serving` | Coverage, forward errors, and serving accuracy reported separately |

With a validated profile from someone else, review steps 1–2 and start execution
at step 4. New GPU collection follows all seven steps. A failure in runtime or
simulator support is an adaptation task; additional sampling alone will not
resolve it.

<a id="1-check-the-aisimulate-commands"></a>

## 1. Check support and prepare the environment

Use Python 3.11–3.13 and a source revision containing the commands you need.
Keep your existing working checkout; switching branches can remove the guided
CLI. Follow the [installation guide](../../../../docs/installation.md#use-current-source)
when creating a new checkout. From its repository root:

```bash
uv sync --project python/aisimulate --extra dev
source python/aisimulate/.venv/bin/activate
export AIS_REPO="$PWD"
export PYTHONPATH="$AIS_REPO/python/aisimulate${PYTHONPATH:+:$PYTHONPATH}"
git rev-parse HEAD
python -c 'from importlib.metadata import version; print(version("aisimulate"))'
aisimulate predict --help
```

For the guided workflow and Example B, additionally verify:

```bash
aisimulate onboard --help
aisimulate onboard init --help
```

Source installation needs `uv`, Cargo/Rust, and a C/C++ compiler/linker. A missing
guided command is a build/version mismatch for that workflow. Existing-profile
consumers such as Example A can skip the `onboard` checks; they use the canonical
SDK and ordinary prediction in a compatible consumer build. Inspect each later
subcommand's help before using it. Keep the same activated Bash or Zsh session for the examples. No serving
GPU or model weights are needed to query existing timings or run simulation.
The Kimi example has [additional consumer-build prerequisites](#example-a-onboard-the-collected-kimi-k3-tp8dcp8-profile).

For **new GPU collection**, also check `python -m collector.fpm_forward --help`.
The host needs local model configuration metadata; the target runtime needs the
pinned checkpoint, allocated GPUs, and matching engine/benchmark code. Kubernetes
needs `kubectl`, an existing namespace, model-cache PVC and deployment permissions.
Slurm needs a caller-owned `sbatch`/`salloc` allocation, Pyxis/Enroot, shared storage
and matching CPU/GPU resources. Neither path needs a prestarted HTTP server:
AISimulate launches benchmark workers and Dynamo self-benchmark generates their
measurements. Record the immutable image/model revisions and actual full runtime
version, including custom-build suffixes. See
[executor prerequisites](../../../../docs/fpm-self-service.md#choose-the-collection-executor)
and [recovery and cleanup](#recovery-and-cleanup).

## 2. Freeze the engine configuration and measurement plan

Create one `onboard checkpoint` file outside fresh request/plan output roots,
then use `onboard init --model-config` (or an existing `--fpm-profile`) to prepare
the inputs. Resolve facts from the pinned checkpoint, hardware and runtime;
present supported precision options before requesting the user's choice. Review
the exact request, save edits and record acceptance as described in the
[checkpoint workflow](../../../../docs/fpm-self-service.md#preserve-partial-profile-review).
`onboard plan` saves the accepted request, collector command and simulation inputs;
it does not verify a target runtime. Keep a record of:

- **Identity:** model/config/checkpoint revision, GPU and interconnect topology,
  engine version and image/source revision, TP/PP/attention-DP/MoE TP/EP, and DCP
  when used. Record actual GPU count separately from logical parallel dimensions.
- **Execution policy:** weight and KV precision, attention/MoE backend settings,
  CUDA Graph capture policy, speculation, model context limit, token budget,
  maximum active requests, and prefix-cache policy.
- **Measurement contract:** which engine work is timed, units, rank aggregation,
  warmup and repeat policy, and the source/initialization of every cache or
  recurrent state consumed by a measurement.
- **Coverage:** prefill request/new-token/prefix coordinates and decode
  batch/context coordinates needed by the intended Replay workload.

Choose `--worker-type aggregated` for one serving worker with both collection
phases. For P/D disaggregation, onboard `prefill` and `decode` independently;
each may have its own topology, graph policy, scheduler bounds and memory budget.
Collection initializes state locally and does not need an actual P/D transfer.
Explicit aggregated probes receive the same intended serving configuration;
vLLM may naturally resolve different per-batch prefill/decode graph paths. See
[serving roles](../../../../docs/fpm-self-service.md#serving-roles-and-cuda-graph-settings).

AISimulate sets runtime limits and collection policies. Dynamo self-benchmark
uses the initialized engine, capture configuration and feasibility checks to
generate the exact grid. `--prefill-cudagraph-policy runtime` leaves graph
selection to the pinned runtime; it does not request a second trace-derived grid.
Validation trace lengths and SLAs do not determine collection coordinates.

The balanced profile uses iteration totals: prefill coordinates are
`(batch_size, total_prefill_tokens, total_kv_read_tokens)`; decode coordinates
are `(batch_size, total_kv_read_tokens)`. A rectangular min/max range does not
prove that all interior queries can be interpolated. Include graph boundaries
and the intended batch/context range in the plan.

For an existing profile, compare these requirements with its manifest and
resolved engine configuration. Keep unknown fields explicit rather than
inventing values or changing the recorded identity to obtain a match.

**Example — Kimi:** TP8+DCP8 uses eight GPUs, `max_num_seqs=32`, and a custom
vLLM build. Its data covers that measured engine configuration. It neither
provides a TP8-only profile nor qualifies PP>1 or arbitrary mixed/ragged shapes.
**Example — MiniMax:** [B1–B2](#b1-prepare-the-campaign-step-2) show configuration
and plan generation for a new four-H200 campaign.

**Result:** a frozen plan/configuration with known state-seeding and coverage
assumptions. Use a separate data identity or campaign when changing those conditions.

## 3. Smoke test and collect

Preview `onboard collect-fpm` without `--execute`, keeping the deployment
options identical to the intended launch. Inspect the generated read-only
collector plan too; the guided preview itself does not run that plan. Run
`onboard collect-fpm --check-readiness` against any existing native evidence.
Where evidence is missing, use `--execute --smoke` for every phase required by
each selected role; guided smoke checks all cells by default and publishes no
formal pair. Aggregated configurations require both prefill and decode. Check that the engine initializes, the requested work is actually
scheduled, timings are finite and use the intended boundary, and all required
ranks participate. Inspect whether attention KV and any recurrent state are
warmed, synthesized, or skipped. A successful launch or prefill alone does not
qualify decode.

If the architecture needs benchmark changes, implement them and repeat the
smoke before running the sweep. In particular, a token-based attention capacity
limit may not account for per-request linear/recurrent cache state. State
initialization can change performance; document and validate synthetic-state
measurements instead of treating them as equivalent to real request history.

After matching readiness passes, use `onboard collect-fpm --execute` to run
the full planned surface, omitting `--smoke` and `--limit`. Save raw FPMs, actual scheduled
shapes, effective engine configuration, versions/hashes, repeat counts, and
failure/skip reasons. Keep incomplete or alternate-state results distinguishable
from the primary profile. Validate that the intended phases and cells were
published; exit code 0 by itself is not evidence that old cells were replaced.

The [MiniMax collection example](#b3-run-smoke-then-formal-collection-step-3)
shows the guided commands and publication checks. Track independent
configurations separately; sharing GPUs does not make one configuration's
publication a prerequisite for another. Use
[recovery and cleanup](#recovery-and-cleanup) for retries and owned resources.

**Result:** measurements and provenance that satisfy the frozen plan, with
missing or unsupported regions reported explicitly.

<a id="5-inspect-the-published-pair"></a>

## 4. Validate and install the FPM profile

The runtime consumes a pair of files, not raw self-benchmark JSON:

```text
<systems-root>/
  <system>.yaml
  query_versions.yaml
  attention_lane_defaults.yaml
  data/<system>/<backend>/<version>/
    fpm_forward_perf.parquet
    fpm_forward_perf.metadata.json
```

Use the [whole-forward publication contract](../../collector/README.md#whole-forward-fpm-campaign)
for schema `aic_fpm_forward_perf`. New guided collection publishes version 7,
including its execution identity. The published Kimi example retains its
original version-6 pair and requires a consumer that supports that identity.
Do not rewrite a schema label to migrate data. Collector performs conversion
for its own campaigns. Importing raw output from another benchmark path
requires an explicit conversion preserving the recorded workload and identity.
The `systems_paths` configuration points to `<systems-root>`, not its `data/`
subdirectory. Whole-forward profiles use the layout above; operation tables may
have an additional family directory.

Check the file hash, schema, row count, phase coverage, identity fields, units,
and measurement policy. A recorded DCP value must be positive and divide TP;
missing DCP and explicit DCP1 remain different identities. For per-row repeated
measurements, the sidecar policy and each row's policy/repeat count must agree.
Keep the producer's aggregated latency; importing the pair does not train or
recompute it.

Run `onboard validate-collection` for native guided campaigns: point validity,
execution evidence, matched-context full-grid repetitions, and withheld-coordinate
interpolation are separate checks. Without `--execute`, the command launches no
GPU repetitions; it can assess compatible existing evidence, while missing
repetitions leave repeatability incomplete. Review the editable policy before
collecting repetitions; bounded subsets remain diagnostics. Imported historical
tables without the required native evidence stay unqualified rather than
acquiring a passing status from successful loading. See the
[collection-quality procedure](../../../../docs/fpm-self-service.md#validate-collection-and-serving-accuracy).

Keep the hardware/query-policy files and source manifest with the pair. Preserve
both files together when copying or renaming them to the canonical filenames.
A dataset may contain comparator profiles or nonuniform workloads: choose the
appropriate primary profile instead of merging every artifact into one table.

[Example A1](#a1-import-the-measured-pair-step-4) downloads our published DCP8
profile at a pinned HF revision and checks its hashes. [Example B4](#b4-inspect-the-published-pair-step-4)
checks a newly collected pair and prepares its quality assessment. Publishing a pair does not automatically bundle
it into an AISimulate release: use an external systems root or submit it through
the repository's data-publication process.

**Result:** a validated local profile and its provenance. No separate registry
service or regression-training command is needed for measured FPM interpolation.

<a id="6-load-the-new-data-and-query-one-forward-pass"></a>

## 5. Construct and check the performance model

For config-derived profiles with pending memory, first run `onboard finalize`
against the original collection, writing to a fresh resolved directory. This
verifies runtime observations and the formal pair; it does not infer a required
activation/non-KV bound from guesswork. Review and accept the resolved profile
separately. Timing publication or a quality pass does not itself resolve memory.
An unsupported runtime needs an audited
[observation adapter](../../../../docs/fpm-self-service.md#resolve-cache-geometry-with-a-runtime-probe);
a preexisting complete resource profile can be used without this finalization.
Historical version-6 tables cannot be finalized by the current native campaign
finalizer; retain their resource provenance and use the compatible consumer route.

Use `RustForwardPassPerfModel.best_available(ForwardPassPerfModelConfig(...))`.
Supply the exact model, system, backend/version, worker role, topology, precision,
and backend identity of the profile, together with its `systems_paths` root.
Use `estimation_mode="fpm_interpolation"` and `fallback_policy="deny"` to
select measured FPM data explicitly. The general defaults are `auto` + `deny`;
auto tries op-level, FPM interpolation, then regression at construction.
Guided config/profile onboarding instead supplies `fpm_profile` and explicitly
sets `estimator_config.fpm_interpolation.method="direct"`, avoiding analytical
model construction and SOL estimation. Keep this choice in generated configs.

Check `diagnostics()` for readiness, the selected estimator, resolved identity,
and systems root. Query a known measured prefill point and a known measured
decode point with `estimate_forward_pass_time_ms`. FPM inputs describe scheduled
iteration totals; they are not request TTFT/ITL. With correction disabled, verify
that measured-point queries reproduce their stored latency before testing
interpolation. Save the resolved provenance with your results.

The following additional options belong to the consumer build used by
[Example A](#example-a-onboard-the-collected-kimi-k3-tp8dcp8-profile); do not assume
a build with `onboard` already supports these independently added features:

- `dcp` records decode context parallelism within the TP group; it does not
  multiply physical GPU count. Current DCP timing uses measured vLLM FPM;
  SOL-dependent transfer paths are unsupported.
- `estimator_config.fpm_interpolation.text_only` permits language timing for
  a multimodal architecture while retaining encoder weights. It supplies no
  vision execution timing.
- `unrecorded_quant_modes` can select null FMHA/communication identity fields.
  It is an exact null match, not a wildcard; leave the corresponding explicit
  quantization overrides unset.

A missing cell or unsupported query is a coverage/support error. Do not change
its topology, precision, or version labels to make it load. The selected model
does not switch estimators on a query miss, and untrained regression cannot run
offline prediction.

[Example A2](#a2-query-the-canonical-model-step-5) checks both phases of the
collected Kimi profile. [Example B5](#b5-finalize-memory-and-check-queries-step-5) checks
the resolved profile through the same API for MiniMax. See the [Core API](../../../../docs/core-api.md#choosing-a-forward-pass-api)
for the complete contract.

**Result:** a ready estimator with the intended data identity and checked queries.

<a id="7-run-ais-predict-using-the-same-external-data"></a>

## 6. Connect the model to Replay

For an aggregated guided campaign, use the resolved directory's generated
`predict/pilot.yaml` and `recommend/pilot.yaml`; `onboard validate-fpm` generates
and runs a cold aggregated trace replay with direct-FPM coverage reporting.
It does not download traces or collect GPU data. An independent role exports
`worker.yaml`, not an executable full P/D prediction/recommendation; qualify the
counterpart independently and compose serving only where Replay supports it.
Current grouped-cache P/D handoff limits concern replay, not FPM collection.

Configure `aisimulate predict` with the same data root, estimator selection,
identity, and engine build assumptions. Then supply the serving behavior that
an FPM table does not define: worker layout, scheduler token/request limits,
context limit, block geometry and capacity, prefix-cache policy, traffic, and
any transfer/offload settings.

Relevant ordinary prediction YAML fields include the following. DCP and the
precision/backend override support require the consumer build in Example A;
they are not additional guided topology choices.

| Purpose | Configuration |
| --- | --- |
| Local profile and selection | `engine.systems_paths`, `engine.estimation_mode`, `engine.fallback_policy` |
| Estimator-specific controls | `engine.estimator_config`, or per-role `timing.estimator_config` |
| Recorded DCP | `engine.workers.<role>.parallelism.decode_context` |
| Precision and attention implementation | Per-role `timing.gemm_quant_mode`, `moe_quant_mode`, `fmha_quant_mode`, `kvcache_quant_mode`, `comm_quant_mode`, `attention_backend` |
| Scheduling and KV state | Per-role `scheduler` and `kv_cache` |

Start with a small in-domain workload and inspect completed requests, generated
tokens, estimator provenance, and GPU counts. Expand it only after validating
query coverage. New FPM measurements do not remove unsupported scheduler or
memory behavior. DCP currently requires explicit fixed KV capacity and explicit
bytes per token if using offload/P-D transfer. The generic cache does not model
KDA checkpoint/eviction behavior or native hybrid prefill chunk alignment.

For `decode_context`, use **`--stack engine`**. Dynamo Replay and Planner do not
yet support this field; their DCP integration requires a separate downstream update.

Prediction accepts the timing identity overrides above for regular language
workers with default timing. In the Example A consumer build, recommendation rejects those overrides,
and DCP is not a recommendation search dimension. [Example A3](#a3-run-and-check-replay-steps-67)
and [Example B6](#b6-run-replay-step-6) provide complete prediction inputs.

**Result:** a reproducible Replay configuration and a successful smoke run;
accuracy and unsupported regions are checked in step 7.

<a id="9-validate-accuracy-separately"></a>

## 7. Validate coverage and accuracy

Use `onboard validate-collection` for native collection quality and
`onboard validate-fpm` for aggregated trace coverage. Follow
[`onboard validate-serving`](../../../../docs/fpm-self-service.md#stage-6-matched-serving)
to prepare a frozen matched workload, retain serving measurements and assess it
in a separate output directory. The collection/report, replay/report and
serving/report remain separate; ordinary exploratory prediction is available
when memory is resolved even if accuracy is unqualified.

First verify exact measured-point queries, then test the interpolation and
Replay shapes required by your workload. Report unsupported queries as coverage
gaps. Successfully loading a table and replaying its calibration points does not
establish prediction accuracy; use independent measurements under matched conditions.

1. Freeze a validation workload with held-out shapes or traces. Record model and
   runtime versions, GPU/system, parallelism, quantization, CUDA Graph policy,
   context/KV limits, prefix policy, input/output lengths, and arrival or
   concurrency settings. Preserve the raw measurements.
2. For forward-level validation, compare predicted and measured latency for the
   same iteration coordinates, phase, rank aggregation, and timing boundary.
   Report prefill and decode separately. A collected-point lookup is a data-path
   check, not an independent accuracy test.
3. For request-level validation, compare AIS predictions against a real serving
   benchmark with the same traffic. Define TTFT, ITL/TPOT, throughput units, and
   aggregation identically. Include ragged batches, mixed/chunked execution,
   and higher concurrency when they are part of the intended deployment.
4. Report query coverage and failed/out-of-domain cases alongside errors. For
   positive measured latencies `m_i` and predictions `p_i`, per-point absolute
   percentage error is `100 * abs(p_i - m_i) / m_i`; MAPE averages those errors.
   WAPE is `100 * sum(abs(p_i - m_i)) / sum(m_i)` and weights larger latencies
   more heavily. Select acceptance thresholds before inspecting the results.

**Result:** a reproducible comparison with explicit conditions, coverage,
errors, and pass/fail criteria. FPM forward error alone does not establish
TTFT/TPOT or throughput accuracy. The
[E2E Accuracy Overview](https://ai-dynamo.org/aisimulate/e2e-accuracy/) reports existing
accuracy results; it is not a substitute for validating a new cell. See the
[snapshot and regeneration details](../../../../pages/e2e-accuracy/README.md) for how that evidence
is produced.

Retain known limitations with the profile so subsequent users can decide
whether it fits their workload.

<a id="use-an-existing-profile-kimi-k3-tp8dcp8"></a>

## Example A: onboard the collected Kimi K3 TP8+DCP8 profile

This example needs an AISimulate build containing the DCP profile-consumer
changes from [#284](https://github.com/ai-dynamo/aisimulate/pull/284)
(merged commit `982c18f4e8e33c124b5195945db269b83168ed3f`), including `dcp`,
`text_only`, `unrecorded_quant_modes`, and engine-stack `decode_context`. Verify
that build separately if your guided-onboarding checkout lacks these options.
`onboard init` and its collection presets do not enumerate DCP, so this published
profile uses the canonical SDK and ordinary prediction directly. Do not replace
DCP8 with TP8-only to make a guided request pass.

This worked example illustrates steps 4–7 using a profile already collected
through Dynamo self-benchmarking. Complete step 1 first and review the retained
collection configuration as described in step 2. It runs on CPU and downloads
the published
[Kimi K3 profile](https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset/tree/6fad3f9a0df5a24603108dcea0d201259254b904/data/moonshotai--Kimi-K3/gb300/vllm/0.29.0/tp8-dcp8),
checks one measured point from each phase, and completes a small Replay run.
It models **eight GB300 GPUs**: DCP8 reuses the TP8 group.

### A1. Import the measured pair (step 4)

Choose a new output directory. The public download uses Python's standard
library; no Hugging Face login or additional download package is required.

```bash
export FPM_RUN="$PWD/kimi-k3-fpm"
mkdir "$FPM_RUN"
python - <<'PY'
import hashlib
import json
import os
import shutil
from importlib.resources import files
from pathlib import Path
from urllib.request import urlopen

run = Path(os.environ["FPM_RUN"]).resolve()
revision = "6fad3f9a0df5a24603108dcea0d201259254b904"
base = f"https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset/resolve/{revision}/"
config_path = "data/moonshotai--Kimi-K3/gb300/vllm/0.29.0/tp8-dcp8"

def download(path):
    with urlopen(base + path, timeout=60) as response:
        return response.read()

manifest_bytes = download(f"{config_path}/manifest.json")
manifest = json.loads(manifest_bytes)
primary = next(item for item in manifest["fpm"] if item["role"] == "primary")
parquet = download(primary["path"])
sidecar = download(primary["metadata_path"])
metadata = json.loads(sidecar)
assert hashlib.sha256(parquet).hexdigest() == primary["sha256"] == metadata["parquet_sha256"]
assert metadata["schema_name"] == "aic_fpm_forward_perf" and metadata["schema_version"] == 6
assert primary["row_count"] == metadata["row_count"] == 669
assert manifest["dcp"] == metadata["configuration_selector"]["dcp"] == 8

systems = run / "systems"
target = systems / "data/gb300/vllm/0.29.0"
target.mkdir(parents=True)
(target / "fpm_forward_perf.parquet").write_bytes(parquet)
(target / "fpm_forward_perf.metadata.json").write_bytes(sidecar)
packaged = files("aisimulate_core") / "systems"
for name in ("gb300.yaml", "query_versions.yaml", "attention_lane_defaults.yaml"):
    shutil.copyfile(str(packaged / name), systems / name)
(run / "manifest.json").write_bytes(manifest_bytes)
(run / "engine-config.json").write_bytes(download(f"{config_path}/fpm/provenance/engine-config.json"))
(run / "dataset-revision.txt").write_text(revision + "\n")
print(f"Installed {primary['row_count']} primary rows under {systems}")
PY
```

**Expected:** 669 primary rows: 338 balanced prefill points and 331 decode
points. Only the filenames change to the loader's canonical names; the Parquet
and sidecar contents remain unchanged. Retain the manifest, dataset revision,
and engine configuration beside the imported data. The separate comparator
with synthetic attention KV and the nonuniform prefill records are not part of
this primary profile.

The decode measurements use warmed MLA KV and **random KDA state**. They are
calibration data, not held-out accuracy evidence. See the retained
`engine-config.json` for measurement conditions and known unknowns.

This profile uses a custom vLLM `0.29.0` build. Allow its literal version to be
queried outside the copied version-slot policy for this session:

```bash
export AIC_ALLOW_UNLISTED_VERSIONS=1
```

This permits version selection; model/topology/precision matching stays strict.
For another profile whose backend version is already queryable, omit the override.

### A2. Query the canonical model (step 5)

Run this in the same session. The script compares its predictions with the
stored measurements and records the model's resolved configuration.

```bash
python - <<'PY'
import json
import math
import os
from pathlib import Path

import pyarrow.parquet as pq
from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

run = Path(os.environ["FPM_RUN"]).resolve()
config = ForwardPassPerfModelConfig(
    model="moonshotai/Kimi-K3", system="gb300", backend="vllm",
    backend_version="0.29.0", worker_type="aggregated",
    tp=8, pp=1, attention_dp=1, moe_tp_size=8, moe_ep_size=1, dcp=8,
    gemm_quant_mode="bfloat16", moe_quant_mode="w4a16_mxfp4",
    kvcache_quant_mode="fp8", attention_backend="FLASHINFER_MLA",
    estimation_mode="fpm_interpolation", fallback_policy="deny",
    systems_paths=(str(run / "systems"),),
    estimator_config={
        "fpm_interpolation": {"text_only": True, "unrecorded_quant_modes": ["fmha", "comm"]},
        "correction": {"enabled": False},
    },
)
rows = pq.read_table(run / "systems/data/gb300/vllm/0.29.0/fpm_forward_perf.parquet").to_pylist()
assert len(rows) == 669
model = RustForwardPassPerfModel.best_available(config)
try:
    for phase, scheduled in (
        ("prefill", {"num_prefill_requests": 1, "sum_prefill_tokens": 128, "sum_prefill_kv_tokens": 0}),
        ("decode", {"num_decode_requests": 1, "sum_decode_kv_tokens": 128}),
    ):
        row = next(row for row in rows if row["workload_kind"] == phase and row["batch_size"] == 1
                   and row["total_prefill_tokens"] == (128 if phase == "prefill" else 0)
                   and row["total_kv_read_tokens"] == (0 if phase == "prefill" else 128))
        latency = model.estimate_forward_pass_time_ms({"version": 1, "scheduled_requests": scheduled})
        assert latency is not None and math.isclose(latency, row["latency_ms"], rel_tol=1e-9)
        print(json.dumps({"phase": phase, "predicted_ms": latency, "measured_ms": row["latency_ms"]}))
    diagnostics = model.diagnostics()
    assert diagnostics["readiness"] == "ready"
    (run / "estimator-provenance.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
finally:
    model.close()
PY
```

**Expected:** approximately 66.1715 ms for the selected prefill point and
10.5622 ms for the selected decode point. These are forward-pass latencies;
request TTFT and end-to-end latency also depend on scheduling and traffic.

`text_only` enables the language profile for Kimi's multimodal architecture
while retaining resident encoder weights. `unrecorded_quant_modes` matches
only null FMHA/communication identities in this dataset. Leave the corresponding
top-level quant fields unset. Missing DCP, DCP1, and DCP8 are distinct identities;
an explicit DCP8 request never substitutes another configuration.

### A3. Run and check Replay (steps 6–7)

Create the complete prediction YAML. The unquoted heredoc below expands
`FPM_RUN` to the absolute systems path; YAML itself does not expand shell variables.

```bash
cat > "$FPM_RUN/predict.yaml" <<YAML
engine:
  mode: aggregated
  model: moonshotai/Kimi-K3
  hardware: gb300
  backend: vllm
  backend_version: "0.29.0"
  context_length: 1048576
  systems_paths: ["$FPM_RUN/systems"]
  estimation_mode: fpm_interpolation
  fallback_policy: deny
  estimator_config:
    fpm_interpolation:
      text_only: true
      unrecorded_quant_modes: [fmha, comm]
    correction: {enabled: false}
  workers:
    aggregated:
      parallelism:
        tensor: 8
        pipeline: 1
        attention_data: 1
        moe_tensor: 8
        moe_expert: 1
        decode_context: 8
      scheduler:
        max_batched_tokens: 8192
        max_sequences: 32
      kv_cache:
        block_size: 12288
        prefix_caching: false
        capacity: {type: fixed, blocks: 2175}
      timing:
        gemm_quant_mode: bfloat16
        moe_quant_mode: w4a16_mxfp4
        kvcache_quant_mode: fp8
        attention_backend: FLASHINFER_MLA
traffic:
  source: {type: synthetic, input_tokens: 128, output_tokens: 2}
  load: {type: concurrency, concurrency: 1}
  stop: {requests: 2}
YAML

aisimulate predict --stack engine -c "$FPM_RUN/predict.yaml" \
  --output-dir "$FPM_RUN/prediction-fpm" --capture-per-request
```

**Expected:** `prediction.json` reports 2 completed requests, 4 output tokens,
and 8 GPUs; `requests.jsonl` contains the two request records. Use a fresh
output directory for another run.

The source engine reported 2,176 physical pool blocks, including one reserved
null block. This example supplies 2,175 usable blocks and the **logical DCP
block size 12,288**, derived from physical block size 1,536 × DCP8. The requested
engine block size of 64 is not this logical block size. The recorded engine's
`max_num_seqs` is 32. Keep these values tied to this profile, not to Kimi models
in general.

This example checks timing consumption. DCP Replay requires explicit capacity;
if enabling host offload or P/D transfer, also provide explicit bytes per token.
The generic Replay cache does not model KDA checkpoint/eviction behavior or
native hybrid prefill chunk alignment. Prefix caching is disabled here to keep
this smoke run independent of those behaviors. Text profiles do not supply
vision execution timing. Prediction accepts the precision/backend overrides
above; recommendation currently rejects those overrides, and DCP is not a
recommendation search dimension.

Only measured points and supported interpolation are available for DCP;
SOL-dependent transfer paths fail explicitly. Arbitrary cached-prefix/chunk
shapes may be unsupported even inside the axis minima/maxima. Expand traffic
only after checking query coverage and perform an
[independent accuracy check](#7-validate-coverage-and-accuracy) before using the
results for deployment decisions.

These checks verify that the profile loads and the example workload runs. Use
step 7 to validate broader workloads. Example B shows a new collection campaign;
use a separate directory if running both examples.

## Example B: collect a MiniMax-M2.7 TP4 profile

This example uses the guided config/profile route for an **aggregated** worker
on four H200 GPUs. It collects whole-forward timings without requiring a new
op-level class. Its topology and workload values illustrate one campaign, not
MiniMax defaults. Complete step 1 and the pinned-runtime investigation first;
replace every placeholder with inspected deployment facts. This example does
not collect DCP or compose independent P/D workers.

<a id="2-choose-the-examples-output-paths"></a>

### B1. Prepare the campaign (step 2)

Keep the local `config.json` and any quantization sidecar from the immutable
checkpoint. Derive supported precision/cache geometry and prepare a reviewed
`resource-overrides.yaml` only for facts the config cannot establish; its
[fields and source requirements](../../../../docs/fpm-self-service.md#start-from-a-local-model-config)
are documented separately. Do not infer FMHA or KV precision solely from the
weight format, copy a historical BF16 cell's identity onto a native FP8 runtime,
or invent activation/non-KV bounds. Memory may remain pending until initialization.
For unsupported runtime geometry, first use the
[probe/import workflow](../../../../docs/fpm-self-service.md#resolve-cache-geometry-with-a-runtime-probe)
with an audited adapter for the exact framework build; keep the resulting
reviewed request instead of regenerating it from unobserved config estimates.

Use a fresh absolute directory, on shared storage for a multi-node Slurm campaign:

```bash
export FPM_RUN=/absolute/new/path/m27-h200-tp4
export FPM_MODEL_CONFIG=/absolute/path/to/pinned-checkpoint/config.json
export FPM_MODEL_REVISION=REPLACE_WITH_IMMUTABLE_CHECKPOINT_REVISION
export FPM_VLLM_VERSION=REPLACE_WITH_FULL_PINNED_VLLM_VERSION
export FPM_RESOURCE_OVERRIDES=/absolute/path/to/reviewed-resource-overrides.yaml
mkdir -p "$(dirname "$FPM_RUN")"
mkdir "$FPM_RUN"
aisimulate onboard checkpoint --file "$FPM_RUN/onboarding-checkpoint.json"

init_args=(
  --model MiniMaxAI/MiniMax-M2.7
  --model-config "$FPM_MODEL_CONFIG"
  --model-revision "$FPM_MODEL_REVISION"
  --framework-version "$FPM_VLLM_VERSION"
  --resource-overrides "$FPM_RESOURCE_OVERRIDES"
  --gpu h200_sxm --interconnect nvswitch
  --worker-type aggregated
  --tensor-parallel 4 --attention-data-parallel 1
  --moe-tensor-parallel 4 --moe-expert-parallel 1
  --context-length 8192 --max-num-tokens 8192 --max-batch-size 256
  --gpu-memory-utilization 0.90 --prefill-cudagraph-policy runtime
  --input-tokens 1024 --output-tokens 32 --concurrency 4 --request-count 8
)
aisimulate onboard init "${init_args[@]}" --output "$FPM_RUN/draft-request.yaml"
```

The last four flags describe an optional synthetic validation workload. They do
not define the collection grid. The context, scheduler, memory fraction and graph
policy are collection inputs and must reflect the intended serving configuration.
Keep the actual checkpoint mounted in the runtime consistent with its declared pin.
A config hash alone does not prove a checkpoint revision.

The command saves a draft without GPU work. If inputs remain unresolved, inspect
the diagnostics, investigate the missing facts and regenerate the draft with
corrected overrides. Save the complete draft in the session checkpoint:

```bash
python - <<'PY'
import json
import os
from pathlib import Path
import yaml

run = Path(os.environ["FPM_RUN"])
request = yaml.safe_load((run / "draft-request.yaml").read_text())
update = {"configurations": {"tp4": {"draft_request": request}}}
(run / "draft-update.json").write_text(json.dumps(update, indent=2) + "\n")
PY
FPM_CHECKPOINT_REVISION=$(aisimulate onboard checkpoint \
  --file "$FPM_RUN/onboarding-checkpoint.json" | \
  python -c 'import json, sys; print(json.load(sys.stdin)["state"]["revision"])')
aisimulate onboard checkpoint --file "$FPM_RUN/onboarding-checkpoint.json" \
  --expect-revision "$FPM_CHECKPOINT_REVISION" --update "$FPM_RUN/draft-update.json"
```

Present the exact profile, sources, precision, topology and collection settings
for review. Save edits and show the revised draft; only after explicit acceptance
of those values, record it and regenerate the same request at a fresh final path:

```bash
FPM_CHECKPOINT_REVISION=$(aisimulate onboard checkpoint \
  --file "$FPM_RUN/onboarding-checkpoint.json" | \
  python -c 'import json, sys; print(json.load(sys.stdin)["state"]["revision"])')
aisimulate onboard checkpoint --file "$FPM_RUN/onboarding-checkpoint.json" \
  --expect-revision "$FPM_CHECKPOINT_REVISION" --accept-profile tp4
aisimulate onboard init "${init_args[@]}" --output "$FPM_RUN/request.yaml"
cmp "$FPM_RUN/draft-request.yaml" "$FPM_RUN/request.yaml"
```

Keep `init_args` and input files synchronized with reviewed edits; a differing
request needs review again. Register the final request and later reports using
[checkpoint artifact references](../../../../docs/fpm-self-service.md#record-artifacts-and-resume).
Record findings, commands/exit statuses and blockers throughout, not only at the
end. The checkpoint command does not capture other commands automatically.

<a id="3-freeze-and-inspect-the-plan"></a>

### B2. Freeze and inspect the plan (step 2)

Generate the plan from the accepted request and preview the intended deployment:

```bash
aisimulate onboard plan \
  --config "$FPM_RUN/request.yaml" --output-dir "$FPM_RUN/collection"

deployment_args=(
  --executor kubernetes
  --namespace REPLACE_WITH_EXISTING_NAMESPACE
  --image REPLACE_WITH_IMMUTABLE_COLLECTION_IMAGE
  --model-cache REPLACE_WITH_MODEL_PVC:/models:REPLACE_WITH_RELATIVE_CHECKPOINT_DIRECTORY
  --dynamo-version REPLACE_WITH_MATCHING_DYNAMO_RELEASE
  --transport nvlink
)
collect_args=(
  --config "$FPM_RUN/collection/request.yaml"
  --output-dir "$FPM_RUN/collection"
  "${deployment_args[@]}"
)
aisimulate onboard collect-fpm "${collect_args[@]}"
```

Set the final `--model-cache` component to the pinned checkpoint directory
relative to the PVC root; use `.` when its weights and config are at that root.
Keep the canonical Hugging Face model ID and declared revision in the request
and profile; this path locates the same checkpoint on the PVC.

For Slurm, replace the deployment array with the actual `--executor slurm`,
`--image`, repeated `--container-mount`, `--cpus-per-task`, `--cpu-bind`, template
version and transport options. Use a caller-owned allocation; Kubernetes-only
namespace/PVC options do not apply. Inspect the
[executor requirements](../../../../docs/fpm-self-service.md#choose-the-collection-executor)
before launch. Both executors are supported by the same guided commands.

**Expected:** `collection/` contains `request.yaml`, `fpm-model-profile.json`,
`support-plan.json`, `commands.json`, local `systems/`, and generated
`predict/pilot.yaml` and `recommend/pilot.yaml`. Planning copies the hardware and
query-policy files; do not create or relabel timing data. The preview prints the
collector invocation without running it or checking target readiness.

Inspect the generated command and the read-only collector plan in
`commands.json` (`fpm_plan_local`). When executing that lower-level plan command,
apply the same deployment options in its collector spelling as shown by the
guided preview; it must retain `--plan-only`. Preserve all generated model,
profile, runtime and output arguments. The collector plan is authoritative for
admitted cells: for this one-topology aggregated request, expect prefill and
decode cells, or investigate omissions. The exact grid and point count remain
runtime-determined. AISimulate bounds context and scheduled new tokens
separately; Dynamo generates feasible points from those limits and the initialized
engine's capture configuration. Runtime graph policy does not force a prefill
capture limit of 2,048 or assume prefill and decode execute the same graph.

#### Include multiple parallel configurations

Use `onboard init --model-config ... --parallel-configs FILE --output-dir ROOT`
for separately reviewed TP/DEP/TEP requests. Do not combine `--parallel-configs`
with the single-topology flags above. For this MoE model, a list can select:

```yaml
- worker_type: aggregated
  tensor_parallel: 4
  attention_data_parallel: 1
  moe_tensor_parallel: 4
  moe_expert_parallel: 1
- worker_type: aggregated
  tensor_parallel: 1
  attention_data_parallel: 4
  moe_tensor_parallel: 1
  moe_expert_parallel: 4
```

These are TP4 and DEP4, each requiring four GPUs per worker. Review each tuple's
own resource profile and collection limits; do not reuse TP rank-local bounds
for DEP. Read the emitted `onboarding.json` and plan each request separately.
Available total GPU allocation is a later execution/deployment choice, not a
sum of selected worker widths. Track each campaign independently and sequence
shared-resource reuse without coupling unrelated publication success. See
[multiple configurations](../../../../docs/fpm-self-service.md#onboard-multiple-parallel-configurations).

<a id="4-run-smoke-then-formal-collection"></a>

### B3. Run smoke, then formal collection (step 3)

First inspect any existing native evidence without launching workers:

```bash
aisimulate onboard collect-fpm "${collect_args[@]}" --check-readiness
```

A fresh campaign returns incomplete/nonzero because it has no measurements.
When GPU execution is authorized and the environment is prepared, collect the
bounded smoke and inspect readiness:

```bash
aisimulate onboard collect-fpm "${collect_args[@]}" --execute --smoke
aisimulate onboard collect-fpm "${collect_args[@]}" --check-readiness
```

Guided smoke covers all cells by default: both phases for this aggregated
request. `--limit` can omit a required phase and leave readiness incomplete.
Smoke saves raw diagnostics and `fpm-readiness.json`, but no formal pair. Check
the effective precision, graph settings, runtime pin, KV/state initialization,
actual scheduled shapes and per-rank evidence. In Slurm also inspect the observed
CPU pool and worker/scheduler affinity. A completed memory probe or successful
prefill alone does not qualify decode timing. Preserve failed evidence and fix
deterministic capability gaps before retrying unchanged work.

Only once matching readiness is established, run formal collection:

```bash
aisimulate onboard collect-fpm "${collect_args[@]}" --execute
```

Omit both `--smoke` and `--limit`. Formal and smoke checkpoints/artifacts are
separate. Inspect `fpm-checkpoint/fpm_forward.json`: publication must pass with
no missing cells and the expected published count. A successful process exit
alone does not prove it replaced an existing cell: check
`skipped_first_publisher_wins` too. A fresh two-cell campaign should publish both
cells with no skips. Keep worker completion, publication, memory and quality
statuses separate in the session checkpoint.

The plan's main paths are:

```text
$FPM_RUN/
  onboarding-checkpoint.json
  request.yaml
  collection/
    request.yaml
    support-plan.json
    commands.json
    fpm-model-profile.json
    fpm-readiness.json
    fpm-checkpoint/fpm_forward.json
    fpm-artifacts/<plan-prefix>/...
    systems/data/h200_sxm/vllm/<actual-version>/
      fpm_forward_perf.parquet
      fpm_forward_perf.metadata.json
```

Retain the frozen plan, manifests/scripts, raw measurements, runtime observations,
logs and pair. Global warm-up, per-point timing and KV/state preparation serve
different purposes. Keep KV warm-up enabled for both phases and inspect native
eligibility/regimes; a zero-KV prefill point needs no cached prefix. Formal
`fake_fallback` rows remain ineligible for direct FPM. Do not infer repeatability
from adjacent decode steps or consolidated provenance duplicates. The quality
stage below checks independent measurements under matched execution conditions.

### B4. Inspect the published pair (step 4)

Verify the actual container's full vLLM version matches `FPM_VLLM_VERSION` before
using that directory; the Generator's `--dynamo-version` is not the runtime
version. Keep the Parquet and metadata together:

```bash
export FPM_PARQUET="$FPM_RUN/collection/systems/data/h200_sxm/vllm/$FPM_VLLM_VERSION/fpm_forward_perf.parquet"
python - "$FPM_PARQUET" > "$FPM_RUN/pair-inspection.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
import pyarrow.parquet as pq

parquet = Path(sys.argv[1])
metadata = json.loads(parquet.with_suffix(".metadata.json").read_text())
with parquet.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
assert metadata["schema_name"] == "aic_fpm_forward_perf"
assert metadata["schema_version"] == 7
assert metadata["parquet_sha256"] == digest
table = pq.read_table(parquet)
assert table.num_rows == metadata["row_count"] and table.num_rows > 0
identity = [
    "model_path", "system", "backend", "backend_version", "workload_kind",
    "tp", "pp", "dp", "moe_tp", "moe_ep", "cp", "gemm_quant_mode",
    "moe_quant_mode", "fmha_quant_mode", "comm_quant_mode", "kv_cache_dtype",
    "moe_backend", "attention_backend", "enable_wideep", "enable_eplb",
    "model_config_sha256", "execution_profile", "engram_residency", "input_modality",
]
frame = table.to_pandas()
print(json.dumps({
    "hash_match": True,
    "row_count": table.num_rows,
    "phase_cells": frame[identity].drop_duplicates().to_dict(orient="records"),
    "kv_seed_regimes": frame["kv_seed_regime"].value_counts().to_dict(),
}, indent=2))
PY
```

Expect matching identities, positive rows and a verified hash. This structural
check does not replace the native loader or quality gates. Historical version-6
reuse in Example A follows its own schema check, not this new-publication check.
`systems_paths` points to `collection/systems`, while the collector database root
is `collection/systems/data`; there is no extra model-family directory.

Review and save the editable
[validation policy](../../../../docs/fpm-self-service.md#validate-collection-and-serving-accuracy)
at `$FPM_RUN/validation-policy.json`, then prepare point-validity, execution and
holdout checks without GPU work:

```bash
aisimulate onboard validate-collection \
  --config "$FPM_RUN/collection/request.yaml" --output-dir "$FPM_RUN/collection" \
  --validation-output-dir "$FPM_RUN/quality" --policy "$FPM_RUN/validation-policy.json"
```

The initial report normally remains incomplete until matched full-grid repeats
exist. Numerical failures return nonzero even during preparation. When repetition
execution is authorized, use the frozen assessment and source deployment:

```bash
aisimulate onboard validate-collection \
  --config "$FPM_RUN/collection/request.yaml" --output-dir "$FPM_RUN/collection" \
  --validation-output-dir "$FPM_RUN/quality" --resume --execute
```

Preserve valid slow samples, individual attempts and the independent-launch
median. Bounded subsets have different sweep history and cannot qualify the
full-grid source. Do not relax CV/error thresholds or replace a source merely
because a newer attempt is faster. Missing producer evidence keeps comparability
unestablished; stable numbers alone do not prove comparable execution.

### B5. Finalize memory and check queries (step 5)

When the resource profile is pending, finalize from verified native observations:

```bash
aisimulate onboard finalize \
  --config "$FPM_RUN/collection/request.yaml" --output-dir "$FPM_RUN/collection" \
  --collection-report "$FPM_RUN/quality/collection-validation.json" \
  --resolved-output-dir "$FPM_RUN/resolved"
```

The report preserves its passed/failed/incomplete quality independently from
memory readiness. Missing or incompatible runtime memory cannot be replaced by
a passing timing table. Review and accept the resolved profile as a new checkpoint
draft, retaining the original collection references. Use that fresh directory
for simulation only after review. An already complete reviewed profile can use
its existing plan directory instead; do not run finalization just to copy data.
Set `FPM_SIM` to the applicable complete, accepted plan:

```bash
export FPM_SIM="$FPM_RUN/resolved"
```

Use the resolved identity/resource profile for a CPU query of one measured point
per phase. This selects direct interpolation without an analytical class:

```bash
python - <<'PY'
import json
import math
import os
from pathlib import Path
import pyarrow.parquet as pq
from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

root = Path(os.environ["FPM_SIM"])
profile = json.loads((root / "fpm-model-profile.json").read_text())
deployment, = profile["deployments"]
config = ForwardPassPerfModelConfig(
    model=profile["model"], system=deployment["system"], backend=deployment["backend"],
    backend_version=deployment["backend_version"], worker_type="aggregated",
    tp=deployment["tp"], pp=deployment["pp"], attention_dp=deployment["dp"],
    moe_tp_size=deployment["moe_tp"], moe_ep_size=deployment["moe_ep"],
    fpm_profile=profile, systems_paths=(str(root / "systems"),),
    estimation_mode="fpm_interpolation", fallback_policy="deny",
    estimator_config={"fpm_interpolation": {"method": "direct"}, "correction": {"enabled": False}},
)
pair = root / "systems/data" / deployment["system"] / deployment["backend"] / deployment["backend_version"] / "fpm_forward_perf.parquet"
rows = pq.read_table(pair).to_pylist()
model = RustForwardPassPerfModel.best_available(config)
try:
    for phase in ("prefill", "decode"):
        row = next(row for row in rows if row["workload_kind"] == phase and row["kv_seed_regime"] != "fake_fallback")
        scheduled = (
            {"num_prefill_requests": row["batch_size"], "sum_prefill_tokens": row["total_prefill_tokens"],
             "sum_prefill_kv_tokens": row["total_kv_read_tokens"]}
            if phase == "prefill" else
            {"num_decode_requests": row["batch_size"], "sum_decode_kv_tokens": row["total_kv_read_tokens"]}
        )
        latency = model.estimate_forward_pass_time_ms({"version": 1, "scheduled_requests": scheduled})
        assert latency is not None and math.isclose(latency, row["latency_ms"], rel_tol=1e-9)
        print(json.dumps({"phase": phase, "predicted_ms": latency, "measured_ms": row["latency_ms"]}))
    (root / "estimator-provenance.json").write_text(json.dumps(model.diagnostics(), indent=2) + "\n")
finally:
    model.close()
PY
```

If a custom runtime version is outside the copied query-version policy, use
`AIC_ALLOW_UNLISTED_VERSIONS=1` for the query and later simulation only after
confirming the actual identity. This permits version selection, not mismatched
precision/topology or unsupported queries. A measured-point match validates the
lookup path; it is not held-out accuracy evidence.

### B6. Run Replay (step 6)

For a first trace check, select one complete local AgentX play and preserve its
request dependencies. Run the existing cold aggregated replay with coverage:

```bash
aisimulate onboard validate-fpm \
  --config "$FPM_SIM/request.yaml" --output-dir "$FPM_SIM" \
  --trace /absolute/path/to/selected-play.jsonl \
  --validation-output-dir "$FPM_RUN/replay-validation"
```

The result records completed replay separately from exact/interpolated/unsupported
queries. A missing query stops execution; the partial report does not audit the
remaining trace. The supplied file is replayed completely. See
[AgentX scope and coverage results](../../../../docs/fpm-self-service.md#validate-fpm-query-coverage-with-agentx-replay).
This does not qualify unsupported grouped-cache P/D handoff or full multimodal
execution. Independent prefill/decode plans must be composed separately where
Replay supports their memory/transfer semantics.

The generated synthetic examples preserve the same accepted profile, data root,
worker width and strict direct-FPM selection:

```bash
aisimulate predict --config "$FPM_SIM/predict/pilot.yaml" \
  --output-dir "$FPM_RUN/prediction-fpm" --capture-per-request
aisimulate recommend --config "$FPM_SIM/recommend/pilot.yaml" \
  --output-dir "$FPM_RUN/recommendation-fpm"
```

Use fresh output paths and check exit status, completed requests/tokens and
estimator provenance. For this example the prediction asks for eight completions
with 32 output tokens each; recommendation evaluates one exact worker preset,
not every possible topology. Actual deployment replicas/GPU budgets belong in
ordinary prediction/recommendation configs. Neither successful synthetic output
nor covered trace replay establishes accuracy: finish the matched serving
validation in step 7 and retain the collection, memory, coverage and serving
results separately in the checkpoint.

<a id="8-recovery-and-cleanup"></a>

## Recovery and cleanup

Resume the agent session with `aisimulate onboard resume --checkpoint PATH`
and inspect integrity issues before acting on saved progress. This checkpoint
restores inputs/conversation state; it does not automatically restart collection.
For Example B, retain the original request, plan and `deployment_args`:

```bash
aisimulate onboard collect-fpm "${collect_args[@]}" --resume --execute
```

Use the same executor, image, mounts, transport and CPU policy. For a failed cell
that needs another GPU attempt, preserve the old raw/log evidence first; the
collector removes those directories when it re-executes the cell. The guided CLI
does not expose `--resume-retry-failed`. Follow the emitted lower-level
`python -m collector.fpm_forward` command with its **complete frozen arguments**,
adding `--resume --resume-retry-failed`; then return through guided
`collect-fpm --resume --execute` for runtime compatibility checks. Do not rerun
GPUs for a post-processing failure when intact artifacts can be recovered.

A completed validated publication is terminal on compatible collector resume.
New native campaigns use schema-11 plans and schema-7 pairs. Historical plans and
schema-6 pairs have separate compatibility limits; see
[finalization and migration limits](../../../../docs/fpm-self-service.md#finalize-runtime-memory).
Changing the model, image, sampling/CPU policy or source can change the frozen
plan. Use fresh request/plan, checkpoint, artifact and database roots for a
changed campaign or independent A/B recollection. Preserve the prior outputs.
Schedule independent configurations independently; use success dependencies only
for actual prerequisites such as validation after its own publication. See
[campaign orchestration](../../../../docs/fpm-self-service.md#orchestrate-independent-collection-campaigns).

The collector normally cleans up each cell's workload in its finalization path;
cleanup failure is an error. After an abnormal interruption, preserve available
logs and inspect the manifests recorded under this campaign's cell artifacts.
For Kubernetes, run the following only for each owned manifest that still has
resources. Slurm cleanup concerns the collector's named job steps, not the
caller's allocation; preserve its shared-storage results.

```bash
kubectl delete -f /absolute/path/to/this/cell/k8s_deploy.yaml \
  --ignore-not-found --cascade=foreground --wait=true --timeout=180s
kubectl get -f /absolute/path/to/this/cell/k8s_deploy.yaml
```

**Expected:** the resources are absent. Also check child Pods using the actual
labels/owner references in that manifest.
If deletion times out, cleanup is not complete; inspect and remove the remaining
owned resources before ending the campaign.
Do not delete unrelated workloads or the namespace. Default `/results` storage
is Pod-local `emptyDir`, so deleting Pods before salvaging results loses those
files.

## FPM data and prediction troubleshooting

| Symptom | Check and next action |
| --- | --- |
| Setup or plan cannot resolve model facts | Use `onboard init --model-config` and reviewed overrides for unresolved supported fields; preserve the checkpoint identity and pin |
| Smoke succeeds but no Parquet appears | Expected for smoke; inspect both phase results before formal collection |
| Partial publication or skipped existing cells | Inspect `missing_cells` and `skipped_first_publisher_wins`; use a new database root for independent recollection |
| Pair hash/schema validation fails | Preserve both files, recover a matching published pair, and do not edit the sidecar to conceal a mismatch |
| FPM lookup fails | Check root layout, runtime version, exact identity and query coordinates; direct FPM does not use SOL to fill missing coverage |
| SDK query works but replay fails | Inspect missing coordinates, resolved memory and supported Replay behavior; one lookup does not establish trace coverage |
| Quality or memory remains incomplete | Preserve the reports and resolve the missing native/repeat/runtime evidence; publication alone is not qualification |
| Prediction output directory is not empty | Select a new directory to retain the previous report |
