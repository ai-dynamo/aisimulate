<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Collect FPM data and use it in AISimulate

Use measured forward-pass data to construct a performance model and predict
request-level serving behavior. Choose the path that matches your starting point:

| Starting point | Follow |
| --- | --- |
| Published self-benchmark data; no collection environment needed | [Check the package](#1-check-the-aisimulate-commands), then [Kimi K3 TP8+DCP8 quickstart](#use-an-existing-profile-kimi-k3-tp8dcp8) |
| Your own schema-v6 Parquet/metadata pair | [Inspect the pair](#5-inspect-the-published-pair), [construct the model](#6-load-the-new-data-and-query-one-forward-pass), then [predict](#7-run-ais-predict-using-the-same-external-data); substitute your recorded identity |
| New measurements to collect | [Prepare the environment](#1-check-the-aisimulate-commands), then follow steps 2–9 for MiniMax-M2.7 on four H200 GPUs |

```text
Existing measured pair ------------------------+
                                               |
GPU plan -> smoke -> self-benchmark -> publish -+-> SDK query -> AIS predict
                                                                |
                                                                v
                                                 Independent accuracy check
```

The data contract is `fpm_forward_perf.parquet` plus its matching
`fpm_forward_perf.metadata.json`. The Rust performance model selects a matching
cell and uses measured lookup/interpolation and, where supported, SOL transfer.
An existing model architecture needs no new model class or parallelization
mapping to consume another measured configuration. Place the pair in a systems
root and pass its exact identity to
`RustForwardPassPerfModel.best_available(ForwardPassPerfModelConfig(...))`.
There is no separate FPM registration service or `train` command.

Raw benchmark JSON is not a loadable FPM pair. Convert it to the
[publication schema](../../collector/README.md#whole-forward-fpm-campaign)
with its measurement provenance before step 5. Collector performs this
conversion for the campaigns it runs in steps 2–4.

A new architecture still requires a model definition; see
[How to add a new model](../add_a_new_model.md).
[Online telemetry regression](aic-fpm-regression-design.md), which learns from
`tune_with_fpms`, is a separate workflow.

## Scope and expected outcomes

| Stage | Runs on | Expected result |
| --- | --- | --- |
| Plan | Machine running Collector | Frozen model, topology, sampling policy, and phase cells; no Kubernetes workload |
| Smoke | GPU cluster | Minimal prefill and decode execution passes; no formal database publication |
| Collect and publish | Machine running Collector + GPU cluster | All planned cells pass; a matching Parquet/metadata pair is published; owned resources are removed |
| SDK query | Machine running AISimulate | The new pair loads and returns a finite, positive forward latency in milliseconds |
| AIS predict | Machine running AISimulate | `prediction.json` and, when requested, `requests.jsonl` for the configured traffic |
| Accuracy validation | Machine running AISimulate + independent GPU run | Matched prediction/measurement errors and coverage, assessed against thresholds chosen for the target use case |

The collection walkthrough uses vLLM with
`tp=4, pp=1, dp=1, moe_tp=4, moe_ep=1, cp=1`, checkpoint-default quantization,
and one worker replica. The dedicated Collector currently supports `PP=CP=1`;
its topology presets do not collect DCP configurations. The DCP quickstart
consumes an existing Dynamo self-benchmark profile.

## 1. Check the AISimulate commands

Use Python 3.11–3.13 and an AISimulate build containing
[PR #284](https://github.com/ai-dynamo/aisimulate/pull/284). A package version
alone does not establish that an unreleased feature is included. If you already
have that build, activate its environment and continue with the checks below.
For a source installation while the PR is open:

```bash
git clone https://github.com/ai-dynamo/aisimulate.git
cd aisimulate
git fetch origin pull/284/head
git switch --detach FETCH_HEAD
uv sync --project python/aisimulate --extra dev
source python/aisimulate/.venv/bin/activate
```

Source installation requires `uv`, Cargo/Rust, and a C/C++ compiler/linker.
See the [installation guide](../../../../docs/installation.md#use-current-source)
for platform requirements. No serving GPU or model weights are needed to query
an existing profile or run the engine-stack simulation.

From the repository root, record the source and package in the activated session:

```bash
export AIS_REPO="$PWD"
git rev-parse HEAD
python -c 'from importlib.metadata import version; print(version("aisimulate"))'
python -c 'from aisimulate_core.sdk import ForwardPassPerfModelConfig; assert "dcp" in ForwardPassPerfModelConfig.__dataclass_fields__'
aisimulate predict --help
```

Use the same activated Bash or Zsh session for the remaining commands.
For existing data, continue with the quickstart below or step 5.

For **new GPU collection**, additionally check:

```bash
cd "$AIS_REPO/python/aisimulate"
python -m collector.fpm_forward --help
```

The collection host needs model configuration metadata and `kubectl` access.
The MiniMax example needs four H200 GPUs on one node and an existing namespace,
checkpoint PVC/path, collection image, and matching Generator target release.
Record the image digest, model revision, GPU type, and resolved configuration.
Before launching, read [recovery and cleanup](#8-recovery-and-cleanup).

## Use an existing profile: Kimi K3 TP8+DCP8

This CPU-only quickstart downloads the published
[Kimi K3 profile](https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset/tree/6fad3f9a0df5a24603108dcea0d201259254b904/data/moonshotai--Kimi-K3/gb300/vllm/0.29.0/tp8-dcp8),
checks one measured point from each phase, and completes a small Replay run.
It models **eight GB300 GPUs**: DCP8 reuses the TP8 group.

### Download and install the measured pair

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
packaged = files("aiconfigurator_core") / "systems"
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

### Query prefill and decode through the canonical API

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

### Run a small Kimi Replay

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

This quickstart checks timing consumption. DCP Replay requires explicit capacity;
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
[independent accuracy check](#9-validate-accuracy-separately) before using the
results for deployment decisions.

The existing-data quickstart is complete. Continue below only to collect a new
MiniMax profile; use a separate campaign directory.

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
cp "$AIS_REPO/python/aisimulate/src/aiconfigurator_core/systems/h200_sxm.yaml" \
  "$FPM_RUN/systems/"
cp "$AIS_REPO/python/aisimulate/src/aiconfigurator_core/systems/query_versions.yaml" \
  "$FPM_RUN/systems/"
cp "$AIS_REPO/python/aisimulate/src/aiconfigurator_core/systems/attention_lane_defaults.yaml" \
  "$FPM_RUN/systems/"
```

`h200_sxm.yaml` provides hardware specifications for the later SDK/prediction
steps when they load this external systems root. `query_versions.yaml` preserves
the version-slot policy, and `attention_lane_defaults.yaml` preserves backend
lane defaults. These files describe hardware and query policy; they contain no
measured FPM rows. The collector's
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
  --fpm-max-prefill-cudagraph-size 2048
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

The sampling settings have distinct meanings:

- `--fpm-max-prefill-isl 8192` bounds the scheduled prefill **new-token** axis.
  It does not set runtime context length.
- `--fpm-max-prefill-cudagraph-size 2048` is the prefill capture-size policy.
  Align it with the deployment being modeled before freezing a formal campaign.
- The collector currently sets runtime `max_model_len=-1` for vLLM auto-fit and
  has no CLI override for that setting. Record the resolved value with the
  collection results.
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
    attention_lane_defaults.yaml
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
identity += ["dcp"] if "dcp" in table.column_names else []
frame = table.to_pandas()
report = {
    "hash_match": True,
    "row_count": table.num_rows,
    "phase_cells": frame[identity].drop_duplicates().to_dict(orient="records"),
    "kv_seed_regimes": frame["kv_seed_regime"].value_counts().to_dict() if "kv_seed_regime" in frame else {},
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
| Canonical SDK/Replay `systems_paths` entry | `$FPM_RUN/systems` |
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

from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

config = ForwardPassPerfModelConfig(
    model="MiniMaxAI/MiniMax-M2.7", system="h200_sxm", backend="vllm",
    worker_type="aggregated", backend_version=sys.argv[2],
    systems_paths=(str(Path(sys.argv[1]).resolve(strict=True)),),
    estimation_mode="fpm_interpolation", fallback_policy="deny",
    tp=4, pp=1, attention_dp=1, moe_tp_size=4, moe_ep_size=1,
    estimator_config={"correction": {"enabled": False}},
)
model = RustForwardPassPerfModel.best_available(config)
try:
    latency_ms = model.estimate_forward_pass_time_ms({
        "version": 1,
        "scheduled_requests": {
            "num_prefill_requests": 4,
            "sum_prefill_tokens": 4096,
            "sum_prefill_kv_tokens": 0,
        },
    })
    assert latency_ms is not None and math.isfinite(latency_ms) and latency_ms > 0
    print(json.dumps({"prefill_ms": latency_ms, "provenance": model.diagnostics()["provenance"]}))
finally:
    model.close()
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
[the canonical API](../../../../docs/core-api.md#choosing-a-forward-pass-api).

This guide copies the version policy file and uses literal backend versions.
If the collected version is already queryable, omit
`AIC_ALLOW_UNLISTED_VERSIONS=1`. The override is a transitional measure, not a
support guarantee. Do not rename version directories or edit identities to
bypass a mismatch. External trees without `query_versions.yaml` use a different
policy that disables the version-slot gate; omitting the file is not needed for
this workflow.

## 7. Run AIS predict using the same external data

Save this as `$FPM_RUN/predict.yaml`. Replace the backend version with the
collected version and `systems_paths` with the absolute `$FPM_RUN/systems`
directory. Confirm block size, context length, scheduler limits, and
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
  systems_paths: ["REPLACE_WITH_ABSOLUTE_SYSTEMS_DIRECTORY"]
  estimation_mode: fpm_interpolation
  fallback_policy: deny
  estimator_config:
    correction: {enabled: false}
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
      timing: {type: default}
traffic:
  source: {type: synthetic, input_tokens: 1024, output_tokens: 32}
  load: {type: concurrency, concurrency: 4}
  stop: {requests: 8}
```

`engine.systems_paths` selects the data root for this prediction. No global
Python root overrides are needed. Run the CLI directly:

```bash
AIC_ALLOW_UNLISTED_VERSIONS=1 \
aisimulate predict --stack engine --config "$FPM_RUN/predict.yaml" \
  --output-dir "$FPM_RUN/prediction-fpm" --capture-per-request
```

**Expected:** exit code 0 and a new `prediction-fpm/` directory containing:

- `prediction.json`: the durable prediction report; check the completion count
  against the eight requested completions and inspect the reported metrics.
- `requests.jsonl`: per-request records enabled by `--capture-per-request`.

Keep the input YAML, resolved estimator provenance, package/source revision,
and pair hash with the report so its data source can be reproduced. Use a new output
directory for another prediction; the CLI rejects nonempty output directories
unless overwrite is explicitly requested.

Every simulated step needs a matching identity and supported coordinates. A
missing cell fails rather than silently switching to `op_level`. With default
capacity estimation, FPM also caps KV capacity at the collected decode-KV
ceiling. Inspect that cap when explaining concurrency and throughput.

New configurations use `estimation_mode` and `fallback_policy`. Without
explicit selection, the defaults are `auto` + `deny`: construction tries
op-level, measured FPM interpolation, then regression. For a measured-profile
workflow, explicitly select `fpm_interpolation` + `deny` as above. An untrained
regression cannot run offline prediction, and a selected estimator never switches
models on a query miss.

For nondefault precision, put `gemm_quant_mode`, `moe_quant_mode`,
`fmha_quant_mode`, `kvcache_quant_mode`, `comm_quant_mode`, and
`attention_backend` under `engine.workers.<role>.timing`, using the exact
recorded identity. These overrides require default timing and regular language
workers. Recommendation rejects them until its feasibility preflight supports
the same identity. See the [Core API](../../../../docs/core-api.md) for estimator
controls and the [DCP quickstart](#use-an-existing-profile-kimi-k3-tp8dcp8) for
a complete configuration with text-only and unrecorded-precision settings.

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
