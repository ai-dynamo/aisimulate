<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Self-service model/GPU support MVP

`aisimulate support` turns one unsupported model/GPU cell into a reproducible, bounded FPM
campaign and a fail-closed validation result. A passing workflow produces ordinary
`aisimulate recommend` configurations that select `forward_model: fpm` and point at the formal
FPM database collected for that exact cell.

The MVP onboards arbitrary model/GPU *combinations* for packaged AIC GPU system specifications.
It does not add a previously unknown GPU architecture. The initial execution boundary is vLLM,
aggregated serving, and no MTP.

## Contract

One support request freezes:

- model, tokenizer, and chat-template revisions;
- framework and framework version;
- GPU type, count, node topology, and interconnect;
- one fixed 8K-input/1K-output workload and one SHA-256-pinned AgentX trace;
- SLOs, AISimulate revision, and validation policy.

The `mvp-v1` search profile generates a deterministic dense- or MoE-aware topology list and hard
caps it at 16 candidates. It never expands an unrestricted Cartesian search.

## Local planning

Planning does not require a GPU:

```bash
aisimulate support init \
  --model Qwen/Qwen3-32B \
  --model-revision <exact-model-revision> \
  --model-kind dense \
  --framework vllm \
  --framework-version 0.10.1 \
  --gpu h200_sxm \
  --gpu-count 8 \
  --interconnect nvswitch \
  --agentx-trace /traces/agentx.jsonl \
  --agentx-digest <sha256> \
  --ttft-ms 2000 \
  --tpot-ms 30 \
  --recommendation-uplift-min 1.5 \
  --aisimulate-revision <exact-commit> \
  --output support-request.yaml

aisimulate support check --config support-request.yaml
aisimulate support plan \
  --config support-request.yaml \
  --output-dir aisimulate-support
```

`check` exits with status 1 when no published exact cell exists. `plan` then writes:

```text
aisimulate-support/
  request.yaml
  support-plan.json
  commands.json
  evidence.yaml
  recommend/
    fixed-8k-1k.yaml
    agentx.yaml
  systems/
    <gpu>.yaml
    data/
  fpm-artifacts/
  fpm-checkpoint/
```

The recommendation files already select the bounded candidates, the FPM forward model, and the
local `systems/` overlay. The FPM collector publishes its sealed Parquet and metadata pair into
that same overlay.

## GPU execution with Brev

FPM collection and matched E2E measurements require the requested GPUs. Brev is the default
execution provider. The workflow only generates commands and defaults to reusing an existing
instance; it does not create, stop, or delete instances.

Inspect `commands.json`, replace `<BREV_INSTANCE>` with a running instance, and run the plan and
campaign from a checkout or installed wheel at the request's exact AISimulate revision:

```bash
brev ls
brev exec <BREV_INSTANCE> \
  "aisimulate support collect-fpm --config support-request.yaml --output-dir aisimulate-support"
brev exec <BREV_INSTANCE> \
  "aisimulate support collect-fpm --config support-request.yaml --output-dir aisimulate-support --execute --resume"
```

Use `brev copy` to move the request and resulting support directory between the workstation and
the instance when they do not share a filesystem. Smoke or limited campaigns are diagnostic only;
they cannot satisfy the formal publication gate.

## Validation and readiness

Populate `evidence.yaml` from independent runs, then validate:

```bash
aisimulate support validate \
  --config aisimulate-support/request.yaml \
  --systems-root aisimulate-support/systems \
  --evidence aisimulate-support/evidence.yaml \
  --output-dir aisimulate-support/validation \
  --format json
```

Support passes only when all gates pass:

- the evidence identity and GPU count match the exact support cell;
- the formal schema-v6 FPM Parquet/metadata pair exists, is digest-sealed, contains the exact
  model/backend/system identity, and has the declared rows and columns;
- baseline plus top-1/top-2/top-3 have matched TTFT, TPOT, and throughput measurements for both
  workloads (eight E2E configuration/workload pairs per metric);
- held-out prefill and decode FPM MAPE and every E2E metric MAPE are at most 20% by default;
- the SLO-compliant top-1 measured throughput clears the product-signed uplift threshold for both
  workloads.

Missing formal data or an unsigned value threshold blocks readiness. Identity errors, incomplete
matched evidence, accuracy misses, or value misses fail it. After a pass, run the generated
`recommend/*.yaml` files with the normal `aisimulate recommend` command; no separate simulation
CLI is introduced.

When ordinary `aisimulate predict` or `aisimulate recommend` detects a model, GPU, backend, data,
or feasibility support gap, its error points back to `aisimulate support init --help`.
