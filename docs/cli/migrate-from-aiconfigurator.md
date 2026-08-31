# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---
title: Migrate from AIConfigurator
subtitle: Decide when and how to replace each AIConfigurator CLI workflow
---

> [!WARNING]
> **Experimental.** The AISimulate recommendation schema and search behavior may change without a
> standard deprecation period.

The `aisimulate` wheel installs both the unified `aisimulate` CLI and the compatibility
`aiconfigurator` CLI. Start new offline serving-prediction and configuration-search workflows with
`aisimulate predict` and `aisimulate recommend`. Keep using `aiconfigurator` where this guide says
there is no direct replacement.

This is not an executable rename. The two CLIs do not share flags or input files, and there is no
automatic AIC-to-AISimulate configuration converter. Migrate the workflow's intent and validate the
result on the target hardware.

## AIC workflow support

| AIC workflow | What AIC provides | Unified CLI status | What to do |
|---|---|---|---|
| `aiconfigurator cli estimate` | One FPM point for an explicit batch, parallel configuration, and estimation mode | **Supported for deployment-level prediction, not behaviorally equivalent** | Use `aisimulate predict` when the goal is to predict one concrete serving deployment. Keep AIC for exact batch-level FPM, static, AFD, detail, per-op, or power semantics. |
| `aiconfigurator cli default` | Capacity-oriented aggregated/disaggregated search and selection | **Partially supported** | Use `aisimulate recommend` after choosing explicit traffic, topology domains, GPU bounds, and an objective. Keep AIC when its capacity-sweep and ranking semantics are required. |
| `aiconfigurator cli recommend` | Minimum-GPU procurement sizing for a load target and SLA | **Partially supported** | Use `aisimulate recommend` for explicit traffic/SLA/GPU-bounded search. AISimulate does not yet promise identical minimum-GPU or ranking results. |
| `aiconfigurator cli exp` | AIC experiment YAML, including heterogeneous experiments | **Manual migration only** | Use `predict` for each concrete deployment or `recommend` for a search domain. AISimulate does not consume AIC experiment YAML directly. |
| `aiconfigurator cli generate` | Deployment artifacts for Dynamo, llm-d, or FPM targets | **Not supported** | Continue using `aiconfigurator cli generate`. The unified CLI emits prediction and recommendation artifacts, not deployment manifests. |
| `aiconfigurator cli support` | AIC command-level aggregated/disaggregated coverage | **Not supported as an `aisimulate` command** | Continue using `aiconfigurator cli support` or the published AIC support matrix. Do not substitute FPE estimator coverage for CLI coverage. |

## Common input mapping

| AIConfigurator CLI | AISimulate configuration | Migration boundary |
|---|---|---|
| `--model-path` / `--model` | `engine.model` | Same model identifier |
| `--system` | `engine.hardware` for a concrete prediction; `optimization.hardware` when recommendation uses `engine.hardware: auto` | One concrete hardware type per recommendation |
| `--backend` | `engine.backend` | One value or an explicit recommendation domain |
| `--backend-version` | `engine.backend_version` | Optional concrete version; not a search domain |
| `--total-gpus` on AIC `default` | `optimization.constraints.max_candidate_gpus` | Recommendation budget only; a prediction derives GPU use from concrete worker parallelism and replicas |
| `--isl` | `traffic.source.input_tokens` | Synthetic request traffic |
| `--osl` | `traffic.source.output_tokens` | Synthetic request traffic |
| `--target-request-rate` | `traffic.load.requests_per_second` | Select `constant_rate` or `poisson` explicitly |
| `--target-concurrency` | `traffic.load.concurrency` | Select `concurrency` explicitly |
| `--ttft` | `evaluation.sla.ttft_ms` | Add `optimization.strict_sla: true` when aggregate-mean rejection is required |
| `--tpot` | `evaluation.sla.itl_ms` | Request goodput uses ITL; strict mode compares aggregate mean TPOT |
| `--request-latency` | `evaluation.sla.e2e_ms` | Mutually exclusive with `ttft_ms` and `itl_ms` |
| `--strict-sla` | `optimization.strict_sla: true` | Recommendation-only aggregate-mean filter |
| `--prefix` | No general direct mapping | Session shared-prefix controls have different semantics |
| `--database-mode`, `--forward-model`, estimate detail/power flags | No unified CLI mapping | Continue using AIC when these controls are required |
| `--save-dir`, `--deployment-target` | No unified CLI mapping | Continue using `aiconfigurator cli generate` |

The examples below use the built-in engine runner. Use `--stack dynamo` only when `ai-dynamo` is
installed and the workflow needs its runner or Router/Planner adapters. The stack selection is a CLI
option; it is not written into the YAML.

## Migrate one concrete deployment

An AIC single-point estimate and an AISimulate serving prediction answer different questions. For
example, this AIC command estimates one FPM point at an explicit batch size:

```bash
aiconfigurator cli estimate \
  --model-path meta-llama/Meta-Llama-3.1-8B \
  --system h200_sxm \
  --backend vllm \
  --estimate-mode agg \
  --batch-size 64 \
  --tp-size 2 \
  --isl 1024 \
  --osl 128
```

If the actual goal is to predict a concrete serving deployment under traffic, save the following as
`prediction.yaml`:

```yaml
traffic:
  source:
    type: synthetic
    input_tokens: 1024
    output_tokens: 128
  load:
    type: concurrency
    concurrency: 64
  stop:
    requests: 100

engine:
  mode: aggregated
  model: meta-llama/Meta-Llama-3.1-8B
  hardware: h200_sxm
  backend: vllm
  workers:
    aggregated:
      parallelism:
        replicas: 1
        tensor: 2
        pipeline: 1
        attention_data: 1
        moe_tensor: 1
        moe_expert: 1
```

Run the engine-only prediction:

```bash
aisimulate predict \
  --stack engine \
  --config prediction.yaml \
  --output-dir ./aisimulate-prediction
```

This predicts deployment-level serving behavior. It does not reproduce AIC's batch-level estimate,
per-op detail, power report, or static/AFD estimation modes.

## Migrate a configuration search

An AIC sizing command may look like this:

```bash
aiconfigurator cli recommend \
  --model-path meta-llama/Meta-Llama-3.1-8B \
  --system h200_sxm \
  --backend vllm \
  --target-request-rate 4 \
  --isl 1024 \
  --osl 128 \
  --ttft 800 \
  --tpot 30
```

To search explicit AISimulate candidates, choose a GPU budget and save this as
`recommendation.yaml`:

```yaml
traffic:
  source:
    type: synthetic
    input_tokens: 1024
    output_tokens: 128
  load:
    type: constant_rate
    requests_per_second: 4
  stop:
    requests_per_load_unit: 10

engine:
  mode: aggregated
  model: meta-llama/Meta-Llama-3.1-8B
  hardware: auto
  backend: vllm
  workers:
    aggregated:
      parallelism: {preset: default}

evaluation:
  sla:
    ttft_ms: 800
    itl_ms: 30

optimization:
  target: goodput_per_gpu
  hardware: h200_sxm
  strict_sla: true
  constraints:
    max_candidate_gpus: 8
```

Run the search:

```bash
aisimulate recommend \
  --stack engine \
  --config recommendation.yaml \
  --output-dir ./aisimulate-recommendation
```

Every YAML under `aisimulate-recommendation/recommendations/` is concrete and can be passed to
`aisimulate predict`. The traffic model, objective, candidate space, and ranking differ from AIC, so
compare the resulting candidates rather than expecting identical ordering or minimum-GPU results.

## SLA translation details

Without `optimization.strict_sla`, a configured SLA classifies individual requests for goodput; a
slow request contributes no tokens to goodput but does not reject the whole candidate. With
`strict_sla: true`, AISimulate also compares every configured bound with aggregate mean metrics and
removes a violation before scalar ranking or Pareto analysis. Missing or non-finite aggregate metrics
and zero qualifying latency samples reject the candidate.

For a fixed-output-length synthetic workload, `--request-latency` maps numerically to
`evaluation.sla.e2e_ms`. The aggregate relationship is:

```text
mean_e2e_latency_ms = mean_ttft_ms + mean_tpot_ms * (output_tokens - 1)
```

`e2e_ms` participates in request-level goodput, and `strict_sla: true` additionally filters aggregate
mean E2E latency. AISimulate does not expose a separate aggregate-only E2E constraint that is excluded
from request-level goodput.

## Workflows that must remain on AIC

Continue using the compatibility command for:

- deployment manifest generation and `--deployment-target` outputs;
- exact AIC support-matrix checks;
- AIC experiment YAML that has not been manually translated;
- exact single-point FPM, static, AFD, per-op, detail, or power estimation;
- AIC-specific database modes, forward-model selection, and expert estimator flags.

The compatibility CLI is planned for deprecation, but its removal date is a separate release decision.
Do not infer removal from the availability of `predict` and `recommend`.

## Related documentation

- [AISimulate CLI schema and output contract](design.md)
- [AIConfigurator CLI and Python API](../../python/aisimulate/README.md)
- [AISimulate repository history](../repository-history.md)
- [AIC synchronization process](../aic-sync.md)
