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
| [C: Native P/D capture](#example-c-collect-native-fpm-with-separate-prefill-and-decode-workers) | Your instrumented vLLM or SGLang deployment and workload | Capture and check each role independently, with real KV transfer |
| [D: Native aggregated capture](#example-d-collect-native-fpm-from-an-aggregated-worker) | Your aggregated deployment or completed native files | Shared capture requirements and a pinned SGLang worker recipe |

For end-to-end offline profiling with your own workload, use the
[E2E-to-GBDT walkthrough](e2e-gbdt.md). Examples C and D provide native capture
settings for that route; Examples A and B use self-benchmark timing tables.

The examples illustrate different supported routes. Their model, topology,
runtime and memory values are not defaults for another deployment. Use separate
output directories and preserve each campaign's evidence.

## Example A: onboard the collected Kimi K3 TP8+DCP8 profile

This example needs an AISim build containing the DCP profile-consumer
changes from [#284](https://github.com/ai-dynamo/aisim/pull/284)
(merged commit `982c18f4e8e33c124b5195945db269b83168ed3f`), including `dcp`,
`text_only`, `unrecorded_quant_modes`, and engine-stack `decode_context`. Verify
that build separately if your guided-onboarding checkout lacks these options.
This is a supported measured-FPM and engine Replay path. The current
[`onboard init` and resource-profile schema](implementation.md#parallelism-support-and-guided-cli-coverage)
do not yet expose that DCP identity, so this published profile uses the canonical
SDK and ordinary prediction directly. Do not replace DCP8 with TP8-only to make
a guided request pass.

This example imports a profile already collected through Dynamo self-benchmarking.
First [prepare the environment](implementation.md#prepare-the-environment) and
review the retained collection identity, resource evidence and coverage against
your intended deployment. It runs on CPU and downloads
the published
[Kimi K3 profile](https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset/tree/dbfcdd9eb62ea733efbc3f0cbd78277f2960061a/data/moonshotai--Kimi-K3/gb300/vllm/0.29.0/tp8-dcp8),
checks one measured point from each phase, and completes a small Replay run.
It models **eight GB300 GPUs**: DCP8 reuses the TP8 group.

### A1. Import the measured pair

Choose a new output directory and an AISim build with the
[`kimi_k3_fpm` pin manifest and materializer](README.md#reuse-a-published-profile).
The materializer verifies the pinned files and reuses a verified cache on later
calls. This historical profile is marked `development`: the explicit
`allow_unqualified=True` below permits this reproduction without granting serving
admission. No Hugging Face login or additional download package is required for
the public files.

```bash
export FPM_RUN="$PWD/kimi-k3-fpm"
mkdir "$FPM_RUN"
python - <<'PY'
import json
import os
import shutil
from pathlib import Path

from aisim_core.sdk.fpm_dataset import list_fpm_profiles, materialize_fpm_profile

run = Path(os.environ["FPM_RUN"]).resolve()
key = "gb300-vllm-mxfp4-tp8-dcp8-primary-v0.29.0"
pin = next(item for item in list_fpm_profiles("kimi_k3_fpm") if item["profile"] == key)
systems = materialize_fpm_profile(pin["manifest"], key, allow_unqualified=True)
(run / "systems").symlink_to(systems, target_is_directory=True)

metadata = json.loads((systems / "data/gb300/vllm/0.29.0/fpm_forward_perf.metadata.json").read_text())
manifest = json.loads((systems / "provenance/manifest.json").read_text())
assert metadata["schema_name"] == "aic_fpm_forward_perf" and metadata["schema_version"] == 6
assert metadata["row_count"] == 669
assert manifest["dcp"] == metadata["configuration_selector"]["dcp"] == 8

shutil.copyfile(pin["manifest"], run / "pin-manifest.json")
for name in ("manifest.json", "engine-config.json"):
    shutil.copyfile(systems / "provenance" / name, run / name)
(run / "dataset-revision.txt").write_text(pin["revision"] + "\n")
print(f"Installed {metadata['row_count']} primary rows under {run / 'systems'}")
PY
```

**Expected:** 669 primary rows: 338 balanced prefill points and 331 decode
points. Only the filenames change to the loader's canonical names; the Parquet
and sidecar contents remain unchanged. `$FPM_RUN/systems` links to the immutable
materialized root; keep that cache directory available for subsequent commands.
Retain `pin-manifest.json`, the published `manifest.json`, dataset revision,
and engine configuration beside the imported data. The separate comparator
with synthetic attention KV and the nonuniform prefill records are not part of
this primary profile.

The publication lacks a collection-time system YAML. This development pin
uses `packaged_system: gb300` to copy the installed canonical `gb300.yaml` into
the root. Its actual hash participates in cache identity and verification.
The manifest marks it as `packaged_reference_not_collection`: this current
hardware reference does not establish the collected runtime's system or
communication-library configuration. If the installed hardware YAML changes,
materialize a new root before using it offline.

The decode measurements use warmed MLA KV and **random KDA state**. They are
calibration data, not held-out accuracy evidence. See the retained
`engine-config.json` for measurement conditions and known unknowns.

This profile uses a custom vLLM `0.29.0` build. The materialized root has no
release-slot policy, so use that exact `backend_version` below. No version-policy
override is needed; model, topology and precision matching remain strict.

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
from aisim_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

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

aisim predict --stack engine -c "$FPM_RUN/predict.yaml" \
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

For a new MiniMax-M2.7 campaign, keep the selected KV precision in the request
and profile. FP8 weights do not require FP8 KV: an explicit BF16 KV selection
plans BF16 attention and cache identity. Runtime initialization and readiness
checks must verify the actual precision before collection. Use a fresh output
directory when changing precision; do not relabel existing measurements.

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
export FPM_VLLM_VERSION=my-vllm-patch-3
export FPM_RESOURCE_OVERRIDES=/absolute/path/to/reviewed-resource-overrides.yaml
mkdir -p "$(dirname "$FPM_RUN")"
mkdir "$FPM_RUN"
aisim onboard checkpoint --file "$FPM_RUN/onboarding-checkpoint.json"

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
aisim onboard init "${init_args[@]}" --output "$FPM_RUN/draft-request.yaml"
```

`FPM_VLLM_VERSION` is the backend identity label for this campaign. Replace the
illustrative custom label with your chosen version or build label; it need not
equal the package version inside the image. Preserve the actual runtime build
and image evidence separately. To autodetect the identity instead, omit
`--framework-version` and use the detected label in later dataset paths. Both
routes [observe the actual image version](implementation.md#create-the-request)
before the first worker launch and continue in a `runtime-resolved/` child plan.
This example keeps an explicit label so the dataset label stays fixed.

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
FPM_CHECKPOINT_REVISION=$(aisim onboard checkpoint \
  --file "$FPM_RUN/onboarding-checkpoint.json" | \
  python -c 'import json, sys; print(json.load(sys.stdin)["state"]["revision"])')
aisim onboard checkpoint --file "$FPM_RUN/onboarding-checkpoint.json" \
  --expect-revision "$FPM_CHECKPOINT_REVISION" --update "$FPM_RUN/draft-update.json"
```

Present the exact profile, sources, precision, topology and collection settings
for review. Save edits and show the revised draft; only after explicit acceptance
of those values, record it and regenerate the same request at a fresh final path:

```bash
FPM_CHECKPOINT_REVISION=$(aisim onboard checkpoint \
  --file "$FPM_RUN/onboarding-checkpoint.json" | \
  python -c 'import json, sys; print(json.load(sys.stdin)["state"]["revision"])')
aisim onboard checkpoint --file "$FPM_RUN/onboarding-checkpoint.json" \
  --expect-revision "$FPM_CHECKPOINT_REVISION" --accept-profile tp4
aisim onboard init "${init_args[@]}" --output "$FPM_RUN/request.yaml"
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
export FPM_COLLECTION="$FPM_RUN/collection"
aisim onboard plan \
  --config "$FPM_RUN/request.yaml" --output-dir "$FPM_COLLECTION"

deployment_args=(
  --executor kubernetes
  --namespace REPLACE_WITH_EXISTING_NAMESPACE
  --image REPLACE_WITH_IMMUTABLE_COLLECTION_IMAGE
  --model-cache REPLACE_WITH_MODEL_PVC:/models:REPLACE_WITH_RELATIVE_CHECKPOINT_DIRECTORY
  --transport nvlink
)
collect_args=(
  --config "$FPM_COLLECTION/request.yaml"
  --output-dir "$FPM_COLLECTION"
  "${deployment_args[@]}"
)
aisim onboard collect-fpm "${collect_args[@]}"
```

If useful, add `--dynamo-version YOUR_DYNAMO_BUILD_LABEL` to record collection
provenance. It does not select a template or validate the backend/runtime.

Set the final `--model-cache` component to the pinned checkpoint directory
relative to the PVC root; use `.` when its weights and config are at that root.
Keep the canonical Hugging Face model ID and declared revision in the request
and profile; this path locates the same checkpoint on the PVC.

For Slurm, replace the deployment array with the actual `--executor slurm`,
`--image`, repeated `--container-mount`, `--cpus-per-task`, `--cpu-bind`, optional
Dynamo provenance and transport options. Use a caller-owned allocation; Kubernetes-only
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
runtime-determined. AISim bounds context and scheduled new tokens
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
aisim onboard collect-fpm "${collect_args[@]}" --check-readiness
```

A fresh campaign returns incomplete/nonzero because it has no measurements.
When GPU execution is authorized and the environment is prepared, collect the
bounded smoke and inspect readiness:

```bash
aisim onboard collect-fpm "${collect_args[@]}" --execute --smoke

# Use the resolved paths printed by the first execution.
export FPM_COLLECTION="$FPM_RUN/collection/runtime-resolved"
collect_args=(
  --config "$FPM_COLLECTION/request.yaml"
  --output-dir "$FPM_COLLECTION"
  "${deployment_args[@]}"
)
aisim onboard collect-fpm "${collect_args[@]}" --check-readiness
```

The first execution probes the actual image's package metadata without GPUs,
then writes the resolved child plan before running smoke. The selected
`FPM_VLLM_VERSION` label is preserved; the child request records the observed
runtime separately. Keep the original request, parent plan and version-binding
record. Later commands below use `FPM_COLLECTION` for the resolved child.

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
aisim onboard collect-fpm "${collect_args[@]}" --execute
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
    runtime-version.json
    runtime-resolved/
      request.yaml
      support-plan.json
      commands.json
      fpm-model-profile.json
      fpm-readiness.json
      fpm-checkpoint/fpm_forward.json
      fpm-artifacts/<plan-prefix>/...
      systems/data/h200_sxm/vllm/<backend-version-label>/
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

Verify the published `backend_version` matches `FPM_VLLM_VERSION`, the selected
identity label. Review the actual container's vLLM version in the retained
collector provenance separately; it may differ from a custom label. The optional
`--dynamo-version` is also provenance, not backend identity. Keep the Parquet and
metadata together:

```bash
export FPM_PARQUET="$FPM_COLLECTION/systems/data/h200_sxm/vllm/$FPM_VLLM_VERSION/fpm_forward_perf.parquet"
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
`systems_paths` points to `$FPM_COLLECTION/systems`, while the collector database
root is `$FPM_COLLECTION/systems/data`; there is no extra model-family directory.

Review and save the editable
[validation policy](implementation.md#validate-collection-and-serving-accuracy)
at `$FPM_RUN/validation-policy.json`, then prepare point-validity, execution and
holdout checks without GPU work:

```bash
aisim onboard validate-collection \
  --config "$FPM_COLLECTION/request.yaml" --output-dir "$FPM_COLLECTION" \
  --validation-output-dir "$FPM_RUN/quality" --policy "$FPM_RUN/validation-policy.json"
```

The initial report normally remains incomplete until matched full-grid repeats
exist. Numerical failures return nonzero even during preparation. When repetition
execution is authorized, use the frozen assessment and source deployment:

```bash
aisim onboard validate-collection \
  --config "$FPM_COLLECTION/request.yaml" --output-dir "$FPM_COLLECTION" \
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
aisim onboard finalize \
  --config "$FPM_COLLECTION/request.yaml" --output-dir "$FPM_COLLECTION" \
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
from aisim_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

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

If a custom backend identity label is outside the copied query-version policy, use
`AIC_ALLOW_UNLISTED_VERSIONS=1` for the query and later simulation only after
confirming the published identity. This permits version selection, not mismatched
precision/topology or unsupported queries. A measured-point match validates the
lookup path; it is not held-out accuracy evidence.

### B6. Run Replay

For a first trace check, select one complete local AgentX play and preserve its
request dependencies. Run the existing cold aggregated replay with coverage:

```bash
aisim onboard validate-fpm \
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
aisim predict --config "$FPM_SIM/predict/pilot.yaml" \
  --output-dir "$FPM_RUN/prediction-fpm" --capture-per-request
aisim recommend --config "$FPM_SIM/recommend/pilot.yaml" \
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
aisim onboard collect-fpm "${collect_args[@]}" --resume --execute
```

Resume may launch missing or unfinished cells. For failed cells, changed inputs,
artifact recovery or cleanup after interruption, follow
[recovery and cleanup](implementation.md#recovery-and-cleanup) and
[troubleshooting](implementation.md#fpm-data-and-prediction-troubleshooting).

## Example C: collect native FPM with separate prefill and decode workers

Use your own cluster, deployment tools and workload client. Start from a working
vLLM or SGLang deployment with **real KV transfer**, then enable native capture
on every relevant serving worker. A frontend or HTTP trace alone does not
contain the engine measurements needed for training.

For deployment details, use the optional [P/D setup reference](native-pd-setup.md)
and [troubleshooting and completion checks](native-pd-troubleshooting.md). Retain
the [completion record](native-pd-troubleshooting.md#keep-a-reviewable-completion-record)
for your configuration, request outcomes and exports.

Set these values in each worker's environment, substituting its role and a
persistent path:

```bash
export DYN_FPM_TRACE=1 DYN_FPM_MODE=full
export DYN_FPM_OUTPUT_PATH=/persistent/run/prefill/fpm  # decode/fpm on decode
export DYN_FPM_MAX_SEGMENTS=64
```

`DYN_FPM_OUTPUT_PATH` is a filename prefix. Size retention for the whole run and
preserve every segment before the cap can prune it. Keep native worker/source
identities, attention-DP ranks, seconds-valued `wall_time`, and all five
`scheduled_requests` counters. Do not rewrite decode-admission or mixed rows
into a different serving role.

| Backend | Producer and deployment checks |
| --- | --- |
| vLLM | Verify its native instrumented scheduler is active on both roles. Record the actual connector and transfer settings. Decode admission can carry prefill work; account for trainer exclusions. |
| SGLang | Enable forward-pass metrics and verify scheduled-prefill accounting. The pinned producer below needs its native counter correction. Use serial scheduling for this qualified timing path; give the roles separate GPUs and reachable KV-transfer endpoints. |

Before collecting your workload, complete a short request with multiple output
tokens. Require native busy rows from both roles and evidence of completed KV
transfer. Then check repeated prefixes, chunked prefill when enabled, and decode
after idle. Registration or successful HTTP completion alone does not establish
correct counters or transfer. Record the engine's timer and decode-context
conventions; they differ between backends.

Use a bounded pilot to find a stable load, preserving genuine latency variation.
Record request rate/concurrency changes and smoke, warmup, measurement and drain
boundaries. At completion, stop new sends, drain outstanding work, observe idle,
and close or rotate capture files using the deployment owner's procedure.

Continue with [capture validation and selection](e2e-gbdt.md#2-check-and-prepare-the-data),
then train separate `prefill` and `decode` models using `--features fpm5
--join-ranks none`. Capture success does not establish prediction accuracy,
Replay support or resource capacity; retain those checks separately.

## Example D: collect native FPM from an aggregated worker

An aggregated worker handles prefill and decode in one serving configuration.
Use the same [capture and smoke checks](#example-c-collect-native-fpm-with-separate-prefill-and-decode-workers)
with role `aggregated` and a persistent `aggregated/fpm` output prefix. Both
vLLM and SGLang require verified native instrumentation. Keep pure-prefill,
pure-decode and mixed observations under the original aggregated identity.

Run your own workload and retain its session semantics, arrival schedule,
termination behavior and completion evidence. Inspect coverage by work kind and
feature range; many decode rows do not compensate for missing prefill or mixed
work. The [standalone walkthrough](e2e-gbdt.md) supplies the validation, selection,
GBDT training and Replay steps. It accepts completed captures without cluster
access and does not depend on a particular launcher or registry.

The SGLang recipe below was used for a completed aggregated AgentX collection.
It supplies a concrete native worker configuration; use your own workload client
and deployment tools, applying the [acceptance checks below](#d2-smoke-collect-and-hand-off).
Another model, workload or cluster needs its own producer, resource and timing
checks. Historical hook-based timings must retain their original identity and
cannot be relabeled as native capture.

### D1. Pin the aggregated SGLang runtime

| Input | Recipe identity or setting |
| --- | --- |
| Model and tokenizer | `deepseek-ai/DeepSeek-V4-Flash-0731`, revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062` |
| Runtime image | `nvcr.io/nvidia/ai-dynamo/sglang-runtime@sha256:b9ddf09444b4c454605b68ab2dbd7e99b6bdbe5fa7a459935edaa78c61592549` |
| SGLang / Dynamo source | `e087e662ba1ac4ef7747537e2a9141085efd4561` / `7931147d403c554575438c2451da4248e53e10b8` |
| Worker | Aggregated, four NVLink-connected B200 GPUs, TP4/EP4/DP1/PP1 |
| Context / KV | 262144 tokens, DSV4 attention, page size 256, `fp8_e4m3`, prefix caching enabled |
| Memory / prefill | `mem_fraction_static=0.9`, chunk size 8192, `max_prefill_tokens=8192` |
| Scheduling | Overlap, mixed chunks, speculation, attention DP and prefill CP disabled |
| Capture | Native five scheduled counters and CUDA duration in seconds; `fpm5`, no rank join |

Acquire the image and model snapshot with your deployment's tools. Verify the
weights, tokenizer and official encoder, and use fresh writable caches. This
pinned producer requires the attributed
[native counter correction](../../../scripts/fpm_self_service/native_patches/README.md).
From the repository root, materialize it on CPU:

```bash
FPM_NATIVE_PATCH=/absolute/owned/path/native-sglang-producer
python3 -B scripts/fpm_self_service/native_patches/materialize_sglang_patch.py \
  --output "$FPM_NATIVE_PATCH"
```

The corrected `metrics_reporter.py` must have SHA256
`80ca324b60beb470b4ea6057fde7ee9553dd32e851e9741831459614e40bee63`.
Mount it read-only at
`/sgl-workspace/sglang/python/sglang/srt/managers/scheduler_components/metrics_reporter.py`
in the disposable runtime. Verify the original source hash before replacement,
and the actual imported module's path/hash afterward. Retain the patch, license
and materialization receipt. The correction uses scheduled batch lengths;
it preserves the native timer and decode counters and does not install a data
hook. A different runtime needs a verified compatible producer, not a blind
application of this patch.

Start etcd, NATS and a Dynamo frontend using your process supervisor and a unique
namespace. The following command starts only the worker; replace paths, GPU UUIDs
and endpoints with your deployment's values:

```bash
FPM_MODEL=/absolute/runtime/path/to/verified/model
FPM_RUN=/absolute/runtime/path/to/fresh/run
export CUDA_VISIBLE_DEVICES="your-four-allocated-GPU-UUIDs"
export DYN_NAMESPACE=your-unique-collection-namespace
export ETCD_ENDPOINTS=http://127.0.0.1:24000
export NATS_SERVER=nats://127.0.0.1:24002
export DYN_SYSTEM_PORT=24010 DYN_FORWARDPASS_METRIC_PORT=24020
export DYN_FPM_TRACE=1 DYN_FPM_MODE=full
export DYN_FPM_OUTPUT_PATH="$FPM_RUN/traces/aggregated/fpm"
export DYN_FPM_JSONL_GZ_ROLL_BYTES=134217728 DYN_FPM_MAX_SEGMENTS=64
unset DYN_FPM_BENCHMARK_OUTPUT_PATH DYN_FPM_SAMPLE_INTERVAL_MS DYN_FPM_WORKER_ID
mkdir -p "$FPM_RUN/traces/aggregated"
python3 -m dynamo.sglang \
  --namespace "$DYN_NAMESPACE" --discovery-backend etcd \
  --request-plane tcp --event-plane zmq \
  --model-path "$FPM_MODEL" --served-model-name deepseek-ai/DeepSeek-V4-Flash-0731 \
  --tp-size 4 --ep-size 4 --dp-size 1 \
  --page-size 256 --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.9 \
  --context-length 262144 --max-running-requests 256 \
  --chunked-prefill-size 8192 --max-prefill-tokens 8192 \
  --host 127.0.0.1 --port 24030 --nccl-port 24050 --watchdog-timeout 3600 \
  --disable-overlap-schedule --enable-forward-pass-metrics --enable-metrics \
  --trust-remote-code \
  --engine-route get_server_info --engine-route get_model_info \
  --engine-route flush_cache:tm
```

Disable FPM capture on the frontend and reserve these ports within your
deployment. Check the resolved configuration for every disabled mode in the
table; omitted flags alone are insufficient. In this runtime,
`max_prefill_tokens=8192` is not a universal hard batch-token bound. TP4 emits one
native stream from attention TP rank 0 on the final PP stage.

### D2. Smoke, collect and hand off

Before measured traffic, check short, chunked and cached-prefix requests against
actual server token usage and native rows. Drain, reset the prefix cache and
observe idle before the measurement window. Serial scheduling is part of this
timer qualification: the reporter accumulates completed CUDA intervals without
a batch key. SGLang decode KV includes the current token. Preserve those
conventions when training and querying.

Use your own workload client and its normal authentication/TLS settings.
Preserve warmup, measurement and drain boundaries, successful/error/cancelled
request counts, client exit status and native worker identity. If using an
agentic workload, preserve session trees and recorded conversation behavior;
a static prompt list describes a different workload.

After draining, observe idle, gracefully stop workers owned by the collection,
allow publisher cleanup, and require closed gzip files before hashing. Keep
smoke and idle rows in the raw archive and exclude them explicitly when
selecting training observations. Validate under role `aggregated` and train
with `--worker-type aggregated --features fpm5 --join-ranks none`.

This recipe disables mixed scheduling. A Replay configuration that needs mixed
predictions requires separate native mixed-work measurements and a fitted mixed
store. Successful collection does not establish publisher losslessness, model
accuracy or SGLang Replay compatibility; follow the walkthrough's
[consumer checks](e2e-gbdt.md#4-use-the-model-in-replay).
