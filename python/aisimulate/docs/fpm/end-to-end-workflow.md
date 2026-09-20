<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Collect FPM data and use it in AISimulate

This guide follows one offline whole-forward FPM campaign from planning to
request-level prediction. It uses MiniMax-M2.7 on four H200 GPUs with pure tensor
parallelism, then consumes the newly collected data from an external directory.
It assumes an existing, working GPU collection environment. Fill in the
image, model access, GPU and Kubernetes inputs from that environment before
executing GPU steps. An already-running vLLM server or HTTP endpoint is not
required; the collector deploys and launches its benchmark workers.

For a new model, start with the [self-service config/profile workflow](../../../../docs/fpm-self-service.md#onboard-with-an-agent)
for direct FPM interpolation without an op-level model class, and record the
[model metadata and intended execution route](model-integration.md). This
campaign's prediction examples use a registered analytical class; the
model-integration guide describes the optional registered-model/SOL procedure
for that route.

```text
Freeze plan -> GPU smoke -> GPU collection -> validate and publish pair
                                                |
                                                v
Independent measurements <- compare <- AIS predict <- SDK FPM query
```

The collector renders the worker deployment and launches Dynamo/vLLM. The engine
initializes the model, cache and graph configuration, then Dynamo self-benchmark
generates and times the admitted measurement points. The collector
publishes `fpm_forward_perf.parquet` together with
`fpm_forward_perf.metadata.json`. The Rust performance model loads this pair,
selects a matching cell, and uses exact lookup, interpolation, and supported SOL
transfer. There is no separate `train` or `fit` command in this workflow.
[Online telemetry regression](aic-fpm-regression-design.md) is a separate model.

## Scope and expected outcomes

| Stage | Runs on | Expected result |
| --- | --- | --- |
| Plan | Machine running Collector | Frozen model, topology, sampling policy, and phase cells; no Kubernetes workload |
| Smoke | GPU cluster | Minimal prefill and decode execution passes; no formal database publication |
| Collect and publish | Machine running Collector + GPU cluster | All planned cells pass; a matching Parquet/metadata pair is published; owned resources are removed |
| SDK query | Machine running AISimulate | The new pair loads and returns a finite, positive forward latency in milliseconds |
| AIS predict | Machine running AISimulate | `prediction.json` and, when requested, `requests.jsonl` for the configured traffic |
| Accuracy validation | Machine running AISimulate + independent GPU run | Matched prediction/measurement errors and coverage, assessed against thresholds chosen for the target use case |

The collector currently supports vLLM with `PP=CP=1`. This example uses
`tp=4, pp=1, dp=1, moe_tp=4, moe_ep=1, cp=1`, default checkpoint quantization,
and one worker replica. Other supported topologies need their own collected
cells. The examples specify expected artifact structure and success conditions;
they do not promise a fixed latency, row count, collection duration, or accuracy.

## 1. Check the AISimulate commands

Activate an existing environment containing the current AISimulate package,
then check the commands from the repository root. If AISimulate is not installed,
follow the [developer setup](../../../../DEVELOPMENT.md) first. Installing the
development extras is not a required step for each collection campaign.

```bash
export AIS_REPO="$PWD"
aisimulate predict --help
cd "$AIS_REPO/python/aisimulate"
python3 -m collector.fpm_forward --help
```

Use the same activated Bash or zsh session for the remaining commands. The
machine running Collector needs model configuration metadata and
`kubectl` access to the target namespace. It does not need a local GPU.

For this example, use four H200 GPUs on one node. Obtain the namespace,
checkpoint PVC/path, collection image reference, and Generator target release
from the existing collection configuration. Record the AIS commit, image
digest, model revision, GPU type, and resolved configuration with the results.

Before launching, review [recovery and cleanup](#8-recovery-and-cleanup), which
uses the campaign's generated manifests. Preserve logs before manual
teardown and verify that no resources from this campaign remain.

## 2. Choose the example's output paths

This guide uses an external directory to keep the example's results separate
from bundled databases and previous campaigns. The directory name and layout
are choices for this walkthrough, not mandatory FPM setup steps. The later
commands use these paths consistently; replace the absolute path below with a
new persistent directory.

```bash
export FPM_RUN=/absolute/new/path/m27-h200-tp4
mkdir -p "$(dirname "$FPM_RUN")"
mkdir "$FPM_RUN"
mkdir -p "$FPM_RUN/systems/data"
cp "$AIS_REPO/python/aisimulate/src/aisimulate_core/systems/h200_sxm.yaml" \
  "$FPM_RUN/systems/"
cp "$AIS_REPO/python/aisimulate/src/aisimulate_core/systems/query_versions.yaml" \
  "$FPM_RUN/systems/"
```

`h200_sxm.yaml` provides hardware specifications for the later SDK/prediction
steps when they load this external systems root. `query_versions.yaml` preserves
the version-slot policy chosen for this example. Neither file is measured FPM
data, and copying them does not collect data or train a model. The collector's
`--fpm-database-root` override is optional; the later examples select it
explicitly so they publish and consume data from the same directory.

Save the following as `$FPM_RUN/collector-k8s.yaml`, replacing every
`REPLACE_...` value with the corresponding existing collection setting:

```yaml
K8sConfig:
  k8s_namespace: REPLACE_WITH_EXISTING_NAMESPACE
  k8s_image: REPLACE_WITH_EXISTING_COLLECTION_IMAGE_DIGEST
  k8s_pvc_name: REPLACE_WITH_MODEL_PVC
  k8s_pvc_mount_path: /models
  k8s_model_path_in_pvc: REPLACE_WITH_RELATIVE_CHECKPOINT_DIRECTORY
  k8s_hf_home: /tmp/fpm-hf-cache
```

The dedicated collector accepts deployment-only `K8sConfig` fields. In
particular, do not copy `Workers.agg.extra_cli_args` or `ServiceConfig` from a
standalone Generator example into this file. The accepted fields are defined in
[the collector input resolver](../../collector/fpm_forward/entry.py).

## 3. Freeze and inspect the plan

Use the Generator target release from the existing collection configuration
and define the campaign arguments once:

```bash
export AIS_DYNAMO_RELEASE=REPLACE_WITH_MATCHING_DYNAMO_RELEASE
cd "$AIS_REPO/python/aisimulate"
fpm_args=(
  --backend vllm
  --model-path MiniMaxAI/MiniMax-M2.7
  --gpu h200_sxm
  --fpm-max-gpus 4
  --fpm-gpu-counts 4
  --fpm-parallel-presets pure_tp
  --fpm-kv-cache-dtypes auto
  --fpm-max-prefill-isl 8192
  --fpm-prefill-cudagraph-policy runtime
  --fpm-gpu-memory-utilization 0.90
  --dynamo-version "$AIS_DYNAMO_RELEASE"
  --generator-config "$FPM_RUN/collector-k8s.yaml"
  --checkpoint-dir "$FPM_RUN/checkpoints"
  --fpm-artifact-root "$FPM_RUN/artifacts"
  --fpm-database-root "$FPM_RUN/systems/data"
)
python3 -m collector.fpm_forward "${fpm_args[@]}" \
  --plan-only > "$FPM_RUN/plan.json"
```

If the checkpoint exists only inside the target Pod and its metadata cannot be
resolved on the machine running Collector, add `--fpm-model-config /absolute/config.json`
to `fpm_args`. Keep the canonical model ID in `--model-path`.

**Expected:** exit code 0 and JSON describing the frozen plan, without creating
Kubernetes workloads. Inspect the model/config hash, system, topology,
quantization, phase cells, sampling policy, and admission decisions before
continuing. For this single-topology, single-KV-setting example, expect one
prefill cell and one decode cell if both are admitted. The plan is authoritative;
do not proceed with an empty plan or unexplained omissions.
`counts.cells` should be 2 for that case, while `counts.points` is
`"runtime-determined"`: runtime initialization determines the actual point set.
With runtime capture policy, `point_generation.prefill_sampling` records
`cudagraph_policy: runtime`; capture sizes/counts and capture-dependent new-token
fields are null before engine initialization. The collector still supplies
runtime limits and the graph-independent KV-read sample cap. Inspect actual
graph mode and capture sizes in initialization logs and raw runtime artifacts.

The sampling settings have distinct meanings:

- `--fpm-max-prefill-isl 8192` bounds the scheduled prefill **new-token** axis.
  It does not set runtime context length.
- `--fpm-prefill-cudagraph-policy runtime` leaves prefill compilation and
  capture-dependent new-token sampling to the initialized runtime. It emits no
  prefill compilation or new-token sample-cap override. To select a reviewed
  extension instead, use `--fpm-prefill-cudagraph-policy explicit
  --fpm-max-prefill-cudagraph-size 2048`. Runtime policy and a numeric capture
  size conflict. Standalone callers omitting the policy retain the legacy
  explicit 2,048-token default. Align actual captures with the serving target.
- `--fpm-gpu-memory-utilization 0.90` declares a starting fraction of total GPU
  memory; supported values are finite and in `(0, 1]`. It is not proof of fit.
  Omitting the flag preserves the existing runtime/default behavior.
- `--fpm-max-model-len`, `--fpm-max-num-batched-tokens` and `--fpm-max-num-seqs`
  select runtime context and per-rank scheduler bounds. Omitted values use a
  supplied FPM profile's bounds; without a profile, context retains vLLM's
  `-1` auto-fit behavior. Record the effective values with the results.
- The default is five global warm-up iterations and one measurement per point.
  Published `latency_ms` uses the maximum rank wall time, converted to ms;
  it is not a mean or median over repeated measurements.

The complete option list is available from `python3 -m collector.fpm_forward
--help` and [its argument definitions](../../collector/fpm_forward/config.py).

### Include multiple GPU counts and parallel configurations

Both GPU counts and parallel presets accept multiple values. To expand this
MiniMax campaign, replace these three entries in `fpm_args` before generating
the plan:

```text
--fpm-max-gpus 8
--fpm-gpu-counts 2 4 8
--fpm-parallel-presets pure_tp tep dep
```

For each selected GPU count `N`, the MoE presets request these configurations
(`PP=CP=1`):

| Preset | Attention TP | Attention DP | MoE TP | MoE EP |
| --- | --- | --- | --- | --- |
| `pure_tp` | N | 1 | N | 1 |
| `tep` | N | 1 | 1 | N |
| `dep` | 1 | N | 1 | N |

The planner combines the selected counts and presets, deduplicates identical
topologies, and applies model/backend and memory admission. It then creates
prefill and decode cells for admitted topology, KV dtype, and backend-policy
combinations. Read `plan.json` for the actual configurations and cell count;
do not assume every requested combination is admitted. Dense models use the
`tp` preset; MoE `pure_tp` requires support in the model's capability profile.
See [topology enumeration](../../collector/fpm_forward/topology.py).

`--fpm-max-gpus 8` is the maximum GPU count for one cell. The runner executes
cells sequentially; multiple Pods belonging to one cell execute together.
Selecting multiple counts/presets does not launch all configurations
simultaneously.

Regenerate and inspect `plan.json` after changing these options. If the original
campaign has already run, first prepare a fresh campaign directory as in step 2
and redefine `fpm_args` with the new paths. The later TP4 SDK/predict examples
still select only their matching cell from the larger database; they do not
automatically compare all collected configurations.

## 4. Run smoke, then formal collection

First read the number of cells from the inspected plan and execute the minimal
profile for every cell. This works for both the two-cell TP4 example and an
expanded campaign:

```bash
FPM_CELL_COUNT=$(python3 -c \
  'import json, sys; n = len(json.load(open(sys.argv[1]))["cells"]); assert n > 0; print(n)' \
  "$FPM_RUN/plan.json")
python3 -m collector.fpm_forward "${fpm_args[@]}" --smoke --limit "$FPM_CELL_COUNT"
```

**Expected:** exit code 0, all selected smoke cells pass, raw runtime evidence is saved,
and no formal Parquet pair is published. `--smoke` without `--limit` runs only
the first cell. `--limit 2` is sufficient for the original two-cell example,
but in an expanded plan the first two cells can both be prefill: the planner
lists all prefill cells before decode cells. Use the full plan count to check
both phases of every admitted configuration.
In `checkpoints/fpm_forward_smoke.json`, check `smoke.status == "passed"`,
`smoke.cell_count == FPM_CELL_COUNT`, and `smoke.formal_database_written == false`.

Inspect initialized precision, graph dispatch/capture sizes, cache allocation
and padding against the intended serving configuration. The CLI does not infer
these effective values from a pasted launch command. Smoke's small sample caps
do not establish the formal point grid or exact count.

Keep benchmark seeding separate from replay prefix reuse. The collector may
keep a prefix cache to prepare timing points outside the measured forward pass;
real KV warm-up depends on the selected strategy and runtime support. Inspect
raw provenance and skipped/fallback evidence. A cold replay cache policy does
not require disabling benchmark seeding, and an environment variable alone
does not prove that a patched seeding implementation ran.

Only after the smoke command succeeds and its evidence is checked, run:

```bash
python3 -m collector.fpm_forward "${fpm_args[@]}"
```

Formal collection must omit both `--smoke` and `--limit`. Smoke and formal
checkpoints/artifacts are isolated automatically, so the roots above can be
shared; smoke samples are not reused as formal measurements.

**Expected:** all frozen cells pass and the formal pair is published. Check the
formal checkpoint's database status, that `missing_cells` is empty, and that
`published_cells` covers the planned cells. In this fresh database,
`skipped_first_publisher_wins` should be empty. Exit code 0 alone is insufficient
to prove a new attempt replaced existing data: publication preserves a cell
already owned by another attempt. `--fpm-publish-partial` is an explicit opt-in
for incomplete campaigns and still returns a nonzero result.

For this fresh, two-cell campaign, the formal checkpoint should contain:

```json
{
  "database": {
    "status": "passed",
    "published_cells": 2,
    "plan_cells": 2,
    "missing_cells": [],
    "skipped_first_publisher_wins": []
  }
}
```

This is an excerpt of the expected fields, not a complete checkpoint file.
For an expanded campaign, use its actual plan count in place of 2; a complete
fresh publication has `published_cells == plan_cells == FPM_CELL_COUNT`.

The resulting directory has this structure; `<plan-prefix>` is the first
16 characters of the plan hash:

```text
$FPM_RUN/
  plan.json
  checkpoints/
    fpm_forward_smoke.json
    fpm_forward.json
  artifacts/<plan-prefix>/
    collection-plan.json
    run-manifest.json
    cells/<cell-id>/...
    smoke/...
  systems/
    h200_sxm.yaml
    query_versions.yaml
    data/h200_sxm/vllm/<actual-vllm-version>/
      fpm_forward_perf.parquet
      fpm_forward_perf.metadata.json
```

Retain generated manifests/scripts, raw benchmark files, resolved configs,
provenance, logs, checkpoints, and the published pair. Inspect
`kv_seed_regime`; do not assume every row contains real-KV evidence. See the
[collector contract](../../collector/README.md#whole-forward-fpm-campaign) for
accepted regimes and publication behavior.

## 5. Inspect the published pair

The version directory is the actual Pod's `importlib.metadata.version("vllm")`,
also recorded in the published rows/metadata. It is not the AIS version or the
`--dynamo-version` used to select templates. Read it from the artifacts before
setting this variable:

```bash
export FPM_VLLM_VERSION=REPLACE_WITH_ACTUAL_COLLECTED_VLLM_VERSION
export FPM_PARQUET="$FPM_RUN/systems/data/h200_sxm/vllm/$FPM_VLLM_VERSION/fpm_forward_perf.parquet"
python3 - "$FPM_PARQUET" > "$FPM_RUN/pair-inspection.json" <<'PY'
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
assert metadata["schema_version"] == 6
assert metadata["parquet_sha256"] == digest
table = pq.read_table(parquet)
assert table.num_rows == metadata["row_count"] and table.num_rows > 0
identity = [
    "model_path", "system", "backend", "backend_version", "workload_kind",
    "tp", "pp", "dp", "moe_tp", "moe_ep", "cp", "gemm_quant_mode",
    "moe_quant_mode", "fmha_quant_mode", "comm_quant_mode", "kv_cache_dtype",
    "moe_backend", "attention_backend", "enable_wideep", "enable_eplb",
]
frame = table.to_pandas()
report = {
    "hash_match": True,
    "row_count": table.num_rows,
    "phase_cells": frame[identity].drop_duplicates().to_dict(orient="records"),
    "kv_seed_regimes": frame["kv_seed_regime"].value_counts().to_dict(),
}
print(json.dumps(report, indent=2))
PY
```

**Expected:** exit code 0, `hash_match: true`, a positive row count, and phase
identities matching the campaign. This structural check does not replace the
native loader's validation or prove interpolation coverage. Inspect the
`batch_size`, `total_prefill_tokens`, and `total_kv_read_tokens` coordinates
before selecting queries; axis minima/maxima alone do not prove that all
interior points can be answered.

Keep these paths distinct:

| Setting | Value in this guide |
| --- | --- |
| Collector `--fpm-database-root` | `$FPM_RUN/systems/data` |
| SDK `systems_path` / Python systems root | `$FPM_RUN/systems` |
| Pair directory | `$FPM_RUN/systems/data/h200_sxm/vllm/$FPM_VLLM_VERSION` |

Whole-forward FPM has no extra model-family directory between the system and
backend. Always transfer the metadata sidecar together with the Parquet file.

## 6. Load the new data and query one forward pass

This query explicitly selects the external root. The example asks for a
four-request prefill with 1,024 new tokens per request and no cached prefix;
confirm this coordinate is covered by the new collection.

```bash
AIC_ALLOW_UNLISTED_VERSIONS=1 \
python3 - "$FPM_RUN/systems" "$FPM_VLLM_VERSION" <<'PY'
import json
import math
import sys
from pathlib import Path

from aisimulate_core.sdk import EngineHandle

engine = EngineHandle.compile(
    "MiniMaxAI/MiniMax-M2.7", "h200_sxm", "vllm",
    backend_version=sys.argv[2],
    systems_path=str(Path(sys.argv[1]).resolve(strict=True)),
    forward_model="fpm",
    tp_size=4, pp_size=1, attention_dp_size=1,
    moe_tp_size=4, moe_ep_size=1,
)
latency_ms = engine.predict_prefill_latency(bs=4, isl=1024, prefix=0)
assert math.isfinite(latency_ms) and latency_ms > 0
print(json.dumps({"prefill_ms": latency_ms}))
PY
```

**Expected:** exit code 0 and a finite, positive `prefill_ms`. No serving GPU is
used for this query. A successful lookup confirms this query can consume the
pair, not that all replay steps will be in-domain or that the prediction is
accurate.

The model, hardware, backend version, parallel shape, quantization, and backend
knobs must match the collected identity. This example uses checkpoint defaults.
For explicitly collected nondefault precision, use the SDK's
`gemm_quant_mode`, `moe_quant_mode`, `fmha_quant_mode`, `kvcache_quant_mode`, and
`comm_quant_mode` arguments with the corresponding enum names as strings. See
[the compile API](../../src/aisimulate_core/sdk/engine.py).

This guide copies the version policy file and uses literal backend versions.
If the collected version is already queryable, omit
`AIC_ALLOW_UNLISTED_VERSIONS=1`. The override is a transitional measure, not a
support guarantee. Do not rename version directories or edit identities to
bypass a mismatch. External trees without `query_versions.yaml` use a different
policy that disables the version-slot gate; omitting the file is not needed for
this workflow.

## 7. Run AIS predict using the same external data

Save this as `$FPM_RUN/predict.yaml`. Replace the backend version with the
collected version. Confirm block size, context length, scheduler limits, and
prefix-cache policy against the resolved runtime configuration and the target
workload. These settings are explicit simulation inputs, not reconstructed
automatically from the FPM table.

```yaml
engine:
  mode: aggregated
  model: MiniMaxAI/MiniMax-M2.7
  hardware: h200_sxm
  backend: vllm
  backend_version: "REPLACE_WITH_ACTUAL_COLLECTED_VLLM_VERSION"
  context_length: 8192
  workers:
    aggregated:
      parallelism:
        replicas: 1
        tensor: 4
        pipeline: 1
        attention_data: 1
        moe_tensor: 4
        moe_expert: 1
      scheduler:
        max_batched_tokens: 8192
        max_sequences: 256
      kv_cache:
        block_size: 64
        prefix_caching: false
        capacity: {type: default}
      timing: {type: default, forward_model: fpm}
traffic:
  source: {type: synthetic, input_tokens: 1024, output_tokens: 32}
  load: {type: concurrency, concurrency: 4}
  stop: {requests: 8}
```

The unified CLI currently has no `--systems-paths` option. The following
inline Python example sets existing Python and Rust root-selection APIs to the
same directory, then invokes `predict` in that process. It needs no additional
helper file and does not add a new CLI flag:

```bash
AIC_ALLOW_UNLISTED_VERSIONS=1 \
python3 - "$FPM_RUN/systems" "$FPM_RUN/predict.yaml" \
  "$FPM_RUN/prediction-fpm" <<'PY'
import os
import runpy
import sys
from pathlib import Path

root = str(Path(sys.argv[1]).resolve(strict=True))
config = str(Path(sys.argv[2]).resolve(strict=True))
output = str(Path(sys.argv[3]).resolve())
os.environ["AICONFIGURATOR_SYSTEMS_PATH"] = root

from aisimulate_core.sdk.perf_database import set_systems_paths

set_systems_paths([root])
print(f"FPM systems_root={root}", file=sys.stderr)
sys.argv = [
    "aisimulate", "predict", "--stack", "engine",
    "--config", config, "--output-dir", output, "--capture-per-request",
]
runpy.run_module("aisimulate", run_name="__main__")
PY
```

**Expected:** exit code 0 and a new `prediction-fpm/` directory containing:

- `prediction.json`: the durable prediction report; check the completion count
  against the eight requested completions and inspect the reported metrics.
- `requests.jsonl`: per-request records enabled by `--capture-per-request`.

Keep the input YAML, printed external root, package/source revision, and pair
hash with the report so its data source can be reproduced. Use a new output
directory for another prediction; the CLI rejects nonempty output directories
unless overwrite is explicitly requested.

Every simulated step needs a matching identity and supported coordinates. A
missing cell fails rather than silently switching to `op_level`. With default
capacity estimation, FPM also caps KV capacity at the collected decode-KV
ceiling. Inspect that cap when explaining concurrency and throughput.

This example covers `predict --stack engine` in one process. It does not cover
propagating Python root overrides to multiprocessing `recommend` workers. The
unified YAML also has no explicit per-operation quantization overrides; use the
SDK or compatibility CLI for those cells. See the
[migration limitations](../../../../docs/cli/migrate-from-aiconfigurator.md).

## 8. Recovery and cleanup

Use the same frozen arguments and roots to resume:

```bash
python3 -m collector.fpm_forward "${fpm_args[@]}" --resume
```

Before retrying failed cells, copy their existing raw artifacts and logs to a
separate archive. The runner removes old raw/log directories for cells it
re-executes. After preserving that evidence:

```bash
python3 -m collector.fpm_forward "${fpm_args[@]}" --resume --resume-retry-failed
```

A completed, validated schema-v6 publication is terminal on resume even if raw
artifacts were pruned. Changing the model, image, sampling policy, or source can
change the frozen plan. Use fresh campaign, checkpoint, artifact, and database
roots for independent recollection or A/B comparisons. Do not run concurrent
campaigns with colliding workload names.

The collector normally cleans up each cell's workload in its finalization path;
cleanup failure is an error. After an abnormal interruption, preserve available
logs and inspect the manifests recorded under this campaign's cell artifacts.
For each owned manifest that still has resources, run:

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

## 9. Validate accuracy separately

Passing the preceding stages establishes that the collection and prediction
path works. Accuracy requires independent measurements under matched conditions.

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

**Expected:** a reproducible comparison with explicit conditions, coverage,
errors, and pass/fail criteria. FPM forward error alone does not establish
TTFT/TPOT or throughput accuracy. The
[E2E Accuracy Overview](https://ai-dynamo.org/aisimulate/e2e-accuracy/) reports existing
accuracy results; it is not a substitute for validating a new cell. See the
[snapshot and regeneration details](../../../../pages/e2e-accuracy/README.md) for how that evidence
is produced.

## FPM data and prediction troubleshooting

| Symptom | Check and next action |
| --- | --- |
| Plan fails to resolve a model | Supply a real local config with `--fpm-model-config`; preserve the canonical model identity |
| Smoke succeeds but no Parquet appears | Expected for smoke; inspect both phase results before formal collection |
| Partial publication or skipped existing cells | Inspect `missing_cells` and `skipped_first_publisher_wins`; use a new database root for independent recollection |
| Pair hash/schema validation fails | Preserve both files, recover a matching published pair, and do not edit the sidecar to conceal a mismatch |
| FPM lookup fails | Check root layout, actual backend version, full identity, and supported coordinates; some queries also require supported SOL operations |
| SDK query works but replay fails | Inspect the later simulated shapes and decode-KV ceiling; one successful prefill query does not establish replay coverage |
| Prediction output directory is not empty | Select a new directory to retain the previous report |
