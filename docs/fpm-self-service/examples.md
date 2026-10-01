<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FPM self-service examples

These examples apply the [self-service workflow](README.md#workflow). Read its
[support boundaries](README.md#support) and use the
[implementation reference](implementation.md) for CLI options, profile review,
validation and recovery. Keep the same activated Bash or Zsh session throughout
each example; replace deployment placeholders with inspected runtime facts.

| Example | Starting point | Execution |
| --- | --- | --- |
| [A: Kimi K3 TP8+DCP8](#example-a-onboard-the-collected-kimi-k3-tp8dcp8-profile) | An existing version-6 timing pair with retained resource provenance | CPU import, SDK query and Replay; no GPU collection |
| [B: MiniMax-M2.7 TP4](#example-b-collect-a-minimax-m27-tp4-profile) | A pinned local model config and a prepared target runtime | Four-H200 aggregated collection, quality checks, memory finalization and Replay |

The examples illustrate different supported routes. Their model, topology,
runtime and memory values are not defaults for another deployment. Use separate
output directories and preserve each campaign's evidence.

## Example A: onboard the collected Kimi K3 TP8+DCP8 profile

This example needs an AISimulate build containing the DCP profile-consumer
changes from [#284](https://github.com/ai-dynamo/aisimulate/pull/284)
(merged commit `982c18f4e8e33c124b5195945db269b83168ed3f`), including `dcp`,
`text_only`, `unrecorded_quant_modes`, and engine-stack `decode_context`. Verify
that build separately if your guided-onboarding checkout lacks these options.
`onboard init` and its collection presets do not enumerate DCP, so this published
profile uses the canonical SDK and ordinary prediction directly. Do not replace
DCP8 with TP8-only to make a guided request pass.

This example imports a profile already collected through Dynamo self-benchmarking.
First [prepare the environment](implementation.md#prepare-the-environment) and
review the retained collection identity, resource evidence and coverage against
your intended deployment. It runs on CPU and downloads
the published
[Kimi K3 profile](https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset/tree/6fad3f9a0df5a24603108dcea0d201259254b904/data/moonshotai--Kimi-K3/gb300/vllm/0.29.0/tp8-dcp8),
checks one measured point from each phase, and completes a small Replay run.
It models **eight GB300 GPUs**: DCP8 reuses the TP8 group.

### A1. Import the measured pair

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

### A2. Query the canonical model

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

### A3. Run and check Replay

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
[independent accuracy check](implementation.md#interpret-coverage-and-accuracy) before using the
results for deployment decisions.

These checks verify that the profile loads and the example workload runs. Use
[matched serving validation](implementation.md#stage-6-matched-serving) to validate broader workloads. Example B shows a new collection campaign;
use a separate directory if running both examples.

## Example B: collect a MiniMax-M2.7 TP4 profile

This example uses the guided config/profile route for an **aggregated** worker
on four H200 GPUs. It collects whole-forward timings without requiring a new
op-level class. Its topology and workload values illustrate one campaign, not
MiniMax defaults. Complete [environment setup](implementation.md#prepare-the-environment) and the
[pinned-runtime investigation](implementation.md#investigate-runtime-constraints-and-precision-options) first;
replace every placeholder with inspected deployment facts. This example does
not collect DCP or compose independent P/D workers.

### B1. Prepare the campaign

Keep the local `config.json` and any quantization sidecar from the immutable
checkpoint. Derive supported precision/cache geometry and prepare a reviewed
`resource-overrides.yaml` only for facts the config cannot establish; its
[fields and source requirements](implementation.md#start-from-a-local-model-config)
are documented separately. Do not infer FMHA or KV precision solely from the
weight format, copy a historical BF16 cell's identity onto a native FP8 runtime,
or invent activation/non-KV bounds. Memory may remain pending until initialization.
For unsupported runtime geometry, first use the
[probe/import workflow](implementation.md#resolve-cache-geometry-with-a-runtime-probe)
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
[checkpoint artifact references](implementation.md#record-artifacts-and-resume).
Record findings, commands/exit statuses and blockers throughout, not only at the
end. The checkpoint command does not capture other commands automatically.

### B2. Freeze and inspect the plan

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
[executor requirements](implementation.md#choose-the-collection-executor)
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
[multiple configurations](implementation.md#onboard-multiple-parallel-configurations).

### B3. Run smoke, then formal collection

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

### B4. Inspect the published pair

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
[validation policy](implementation.md#validate-collection-and-serving-accuracy)
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

### B5. Finalize memory and check queries

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

### B6. Run Replay

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
[AgentX scope and coverage results](implementation.md#validate-fpm-query-coverage-with-agentx-replay).
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
[validation procedure](implementation.md#stage-6-matched-serving) and retain the collection, memory, coverage and serving
results separately in the checkpoint.

### Resume the MiniMax campaign

Retain the original request, plan and `deployment_args`. In the same shell that
defined `collect_args`, resume with:

```bash
aisimulate onboard collect-fpm "${collect_args[@]}" --resume --execute
```

Resume may launch missing or unfinished cells. For failed cells, changed inputs,
artifact recovery or cleanup after interruption, follow
[recovery and cleanup](implementation.md#recovery-and-cleanup) and
[troubleshooting](implementation.md#fpm-data-and-prediction-troubleshooting).
