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

## Repository and release transition

The standalone [AIConfigurator repository](https://github.com/ai-dynamo/aiconfigurator) will publish
its final 0.12.0 `aiconfigurator` and `aiconfigurator-core` artifacts and then be archived. AISimulate
is the canonical home for ongoing development, releases, issues, and pull requests; open all new
issues and pull requests in the [AISimulate repository](https://github.com/ai-dynamo/aisimulate).

AISimulate 0.12.0 keeps the `aiconfigurator` compatibility command for workflows that do not yet have
a unified CLI replacement. The command is targeted for removal in AISimulate 0.13.0, after every
remaining AIC workflow has a verified replacement in the `aisimulate` CLI.

## AIC workflow support

| AIC workflow | What AIC provides | Unified CLI status | What to do |
|---|---|---|---|
| `aiconfigurator cli estimate` | One FPM point for an explicit batch, parallel configuration, and estimation mode | **Supported for deployment-level prediction, not behaviorally equivalent** | Use `aisimulate predict` when the goal is to predict one concrete serving deployment, including analytical AFD. Keep AIC for exact batch-level FPM, multimodal/EPD outside the [analytical EPD scope](../sweeper/epd.md), static, detail, per-op, or power semantics. |
| `aiconfigurator cli default` | Capacity-oriented aggregated/disaggregated search and selection | **Partially supported** | Use `aisimulate recommend` after choosing explicit traffic, topology domains, GPU bounds, and an objective. Keep AIC when its capacity-sweep and ranking semantics are required. |
| `aiconfigurator cli recommend` | Minimum-GPU procurement sizing for a load target and SLA | **Partially supported** | Use `aisimulate recommend` to search explicit candidates under a chosen traffic shape, SLA, GPU budget, and objective. The unified CLI does not reproduce AIC's minimum-GPU sizing from either a target request rate or target concurrency; keep using the compatibility CLI when that sizing result is required. |
| `aiconfigurator cli exp` | AIC experiment YAML, including heterogeneous experiments | **Manual migration only** | AISimulate does not consume AIC experiment YAML directly. Use `predict` for each concrete homogeneous deployment, `recommend` for a unified-CLI search domain, or the Sweeper SDK example below for heterogeneous P/D hardware. |
| `aiconfigurator cli generate` | Deployment artifacts for Dynamo, llm-d, or FPM targets | **Not supported** | Continue using `aiconfigurator cli generate`. The unified CLI emits prediction and recommendation artifacts, not deployment manifests. |
| `aiconfigurator cli support` | AIC command-level aggregated/disaggregated coverage | **Not supported as an `aisimulate` command** | Continue using `aiconfigurator cli support` or the published AIC support matrix. Do not substitute FPE estimator coverage for CLI coverage. |

## Known gaps in the unified path

The following limits apply to `aisimulate predict`, `aisimulate recommend`, and the new
`aisimulate.sweeper` API. The `aisimulate` distribution also contains the compatibility AIC
implementation and lower-level estimator, collector, and result-schema primitives. Their presence
does not make a capability available through the unified path.

The [analytical EPD integration](../sweeper/epd.md) supports bounded fixed-image
E+agg/E+P+D prediction and search through both the unified CLI and Sweeper SDK.
This does not imply event-level encoder simulation or deployment generation.

| Capability | Current unified status | Migration action |
|---|---|---|
| Multimodal image inputs and EPD | **Analytical fixed-image support.** Unified `predict` and `recommend` accept `traffic.source.images` and `engine.workers.encoder` for E+agg/E+P+D, using fixed synthetic concurrency. Saved recommendation YAML preserves the encoder and can be reloaded by `predict`. No image traces, per-request EPD metrics, event-level encoder queueing, or deployment generation. | Use the [CLI examples and semantics](../sweeper/epd.md#unified-cli); retain AIC workflows when their additional semantics are needed. |
| Attention/FFN disaggregation (AFD) | **Supported analytically for fixed-length synthetic traffic.** `engine.mode: afd` lowers concrete A/F topologies for `predict` and finite, memory-qualified topology domains for `recommend`; single-phase AFD can be paired with a regular P/D companion. | Use the AFD YAML below for analytical prediction or recommendation. Keep AIC for exact batch-level estimates and native deployment generation; AISimulate does not claim physical AFD serving execution. |
| Power and energy analysis | **Not AIC-equivalent.** Sweeper results can preserve optional runner-supplied power or energy metadata, but the unified engine path does not currently provide AIC's predicted `power_w`, coverage gate, or `--detail energy` report. | Continue using AIC estimate/reporting, and confirm that the selected model/system data has sufficient energy coverage. |
| Static, per-operation, and source breakdowns | **Not supported.** Unified prediction simulates serving traffic; it does not expose AIC's `static`, `static_ctx`, or `static_gen` single-pass modes or `--detail` memory/time/source reports. | Continue using `aiconfigurator cli estimate`. |
| Estimator and performance-data selection | **Not exposed by the unified CLI.** `engine.backend_version` is available, but database mode, forward model, transfer policy, custom system roots, and estimator tuning remain outside the public YAML. | Continue using AIC when those controls are required. |
| Explicit quantization overrides | **Not exposed by the unified CLI.** There is no direct mapping for AIC's GEMM, KV-cache, FMHA, MoE, or communication quantization flags. | Let the unified engine resolve model/runtime defaults, or stay on AIC when an explicit estimator override is required. |
| Heterogeneous P/D hardware or backends | **Hardware is supported through Sweeper YAML/SDK only.** Ordinary `disagg` search accepts independent prefill and decode hardware SKUs. Both roles still share one model, backend, and backend version. The unified `predict` and `recommend` schemas still have one hardware field. | Translate AIC role system names to `search_space.prefill_hardware_sku` and `search_space.decode_hardware_sku` as shown below. Keep heterogeneous backends or versions, unified-CLI use, deployment generation, and unsupported adapter paths on AIC. |

## Common input mapping

| AIConfigurator CLI | AISimulate configuration | Migration boundary |
|---|---|---|
| `--model-path` / `--model` | `engine.model` | Same model identifier |
| `--system` | `engine.hardware` for a concrete prediction; `optimization.hardware` when recommendation uses `engine.hardware: auto` | One concrete hardware type per recommendation |
| `prefill_system_name` / prefill `--system` | `search_space.prefill_hardware_sku` in Sweeper YAML | Optional override; inherits `search_space.hardware_sku` when omitted |
| `decode_system_name` / `--decode-system` | `search_space.decode_hardware_sku` in Sweeper YAML | Optional override; inherits `search_space.hardware_sku` when omitted |
| `--backend` | `engine.backend` | One value or an explicit recommendation domain |
| `--backend-version` | `engine.backend_version` | Optional concrete version; not a search domain |
| `--total-gpus` on AIC `default` | `optimization.constraints.max_candidate_gpus` | Recommendation budget only; a prediction derives GPU use from concrete worker parallelism and replicas |
| `--isl` | `traffic.source.input_tokens` | Synthetic request traffic |
| `--osl` | `traffic.source.output_tokens` | Synthetic request traffic |
| `--target-request-rate N` | `traffic.load.type: constant_rate` or `poisson`, plus `traffic.load.requests_per_second: N` | Traffic-shape mapping only: `N` is offered request rate, not required fleet capacity or a minimum-GPU sizing target |
| `--target-concurrency N` | `traffic.load.type: concurrency`, plus `traffic.load.concurrency: N` | Traffic-shape mapping only: `N` is the closed-loop in-flight-request cap, not a minimum-GPU sizing target |
| `--ttft` | `evaluation.sla.ttft_ms` | Add `optimization.strict_sla: true` when aggregate-mean rejection is required |
| `--tpot` | `evaluation.sla.itl_ms` | Request goodput uses ITL; strict mode compares aggregate mean TPOT |
| `--request-latency` | `evaluation.sla.e2e_ms` | Mutually exclusive with `ttft_ms` and `itl_ms` |
| `--strict-sla` | `optimization.strict_sla: true` | Recommendation-only aggregate-mean filter |
| `--prefix` | No general direct mapping | Session shared-prefix controls have different semantics |
| `--forward-model` | `engine.workers.<role>.timing.forward_model` | `op_level` (default) or `fpm`; per worker role; `default` timing only |
| `--database-mode`, `--transfer-policy`, `--systems-paths` | No unified CLI mapping | Continue using AIC when performance-data source selection is required |
| GEMM, KV-cache, FMHA, MoE, and communication quantization flags | No unified CLI mapping | Continue using AIC when explicit estimator quantization overrides are required |
| `--detail` and per-op memory/time/energy/source reports | No unified CLI mapping | Continue using AIC for single-point diagnostic breakdowns |
| Predicted `power_w` and power coverage | No AIC-equivalent unified output | Optional runner metadata is not a replacement for AIC power analysis |
| `--save-dir`, `--deployment-target` | No direct flag mapping on the new `aisimulate` CLI | `aisimulate recommend --output-dir` writes recommendation results and prediction-ready configs, not deployment manifests. For programmatic generation, select a candidate and use `from_sweeper_candidate(...)` with `generate_from_request(...)`; continue using `aiconfigurator cli generate` for a direct CLI workflow. |

The examples below use the built-in engine runner. Use `--stack dynamo` only when `ai-dynamo` is
installed and the workflow needs its runner or Router/Planner adapters. The stack selection is a CLI
option; it is not written into the YAML.

## Migrate heterogeneous P/D hardware with Sweeper

AIC disaggregated experiment YAML uses `prefill_system_name` and `decode_system_name`. Translate
those values to the two optional role overrides under `search_space`. `hardware_sku` remains required
and is the fallback for either omitted role.

For H200 prefill and GB200 decode, save this as `heterogeneous-pd-sweep.yaml`:

```yaml
search_space:
  model_name: Qwen/Qwen3-VL-30B-A3B-Instruct-FP8
  hardware_sku: h200_sxm
  prefill_hardware_sku: h200_sxm
  decode_hardware_sku: gb200
  backend: [vllm]
  deployment_mode: [disagg]
  context_length: 4096
  gpu_budget: 8

workload:
  isl: 1024
  osl: 128
  request_rate: 4
  num_request_ratio: 10

goal:
  target: throughput

sweep:
  max_rounds: 1
  candidates_per_round: 4
  parallel_evals: 1
```

Run the configuration through the Sweeper SDK with the replay runtime used by your application. The
built-in engine runner is:

```python
from aisimulate.runner import EngineReplayRunnerFactory
from aisimulate.sweeper import SmartSearchConfig, Sweeper

config = SmartSearchConfig.from_yaml("heterogeneous-pd-sweep.yaml")
result = Sweeper(runner_factory=EngineReplayRunnerFactory()).run(config)
print(result.to_json())
```

In the example, the explicit prefill override equals the fallback and may be omitted. If only
`decode_hardware_sku: gb200` is present, prefill inherits `hardware_sku: h200_sxm`; the inverse
applies when only the prefill override is set. Each role is independently parallel-enumerated and
KV-qualified against its effective hardware, and the pair must fit the shared `gpu_budget`.

The two roles must use the same model, backend, and backend version. When `backend_version` is
omitted, both hardware SKUs must resolve to the same latest version; otherwise set one explicit
version supported by both. This path does not add heterogeneous hardware to `aisimulate predict` or
`aisimulate recommend`. It also does not support heterogeneous deployment-manifest generation. With
the Dynamo stack, do not use Router `prefill_load_model.type: aic` until that provider consumes
`prefill_hardware_sku`; its current compatibility path still reads the shared fallback SKU.

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

This text-only example predicts deployment-level serving behavior. For bounded analytical EPD
prediction, use the [EPD CLI examples](../sweeper/epd.md#unified-cli). These paths do not reproduce AIC's
batch-level estimate, per-op detail, power report, or static estimation modes. AFD has a separate
analytical deployment path described below; it is not an exact replacement for AIC's single-point
AFD estimator output.

## Preserve request-rate traffic during a configuration search

An AIC minimum-GPU sizing command may look like this:

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

To preserve its constant 4 requests-per-second traffic shape while searching explicit AISimulate
candidates, choose a GPU budget and save this as `recommendation.yaml`:

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
`aisimulate predict`. Here, `requests_per_second: 4` is offered open-loop traffic, and
`max_candidate_gpus: 8` is only a search bound. Neither field asks AISimulate to find the minimum GPU
count that can serve 4 requests per second. The traffic model, objective, candidate space, and ranking
differ from AIC, so compare the resulting candidates rather than expecting identical ordering or
minimum-GPU results.

## Preserve concurrency traffic during a configuration search

Legacy `--target-concurrency 32` also has a traffic-shape mapping, but not a minimum-GPU sizing
equivalent. To search with 32 requests kept in flight, replace the `traffic` section in the preceding
example with:

```yaml
traffic:
  source:
    type: synthetic
    input_tokens: 1024
    output_tokens: 128
  load:
    type: concurrency
    concurrency: 32
  stop:
    requests: 320
```

This is closed-loop traffic: AISimulate keeps up to 32 requests in flight and starts another as one
finishes. The completion rate is an outcome, not a configured capacity requirement. A slow one-GPU
candidate can still maintain 32 in-flight requests while completing them slowly, so
`traffic.load.concurrency` must not be interpreted as the number of concurrent users that a selected
minimum-GPU fleet can serve.

At fixed concurrency, `strict_sla` filters aggregate mean latency violations, while
`goodput_per_gpu` favors throughput efficiency. The selected configuration may use more GPUs than
the smallest SLA-compliant configuration.

## Keep minimum-GPU sizing on the compatibility CLI

If the required result is AIC's estimate of the minimum GPUs or replicas needed for
`--target-request-rate` or `--target-concurrency` under an SLA, continue using
`aiconfigurator cli recommend`. The corresponding AISimulate YAML fields preserve the workload shape
only; they do not preserve that capacity-sizing behavior.

## SLA translation details

For text-only replay, without `optimization.strict_sla`, a configured SLA classifies individual requests for goodput; a
slow request contributes no tokens to goodput but does not reject the whole candidate. With
`strict_sla: true`, AISimulate also compares every configured bound with aggregate mean metrics and
removes a violation before scalar ranking or Pareto analysis. Missing or non-finite aggregate metrics
and zero qualifying latency samples reject the candidate.

For a fixed-output-length synthetic workload, `--request-latency` maps numerically to
`evaluation.sla.e2e_ms`. The aggregate relationship is:

```text
mean_e2e_latency_ms = mean_ttft_ms + mean_tpot_ms * (output_tokens - 1)
```

For text-only replay, `e2e_ms` participates in request-level goodput, and `strict_sla: true` additionally filters aggregate
mean E2E latency. This path does not expose a separate aggregate-only E2E constraint that is excluded
from request-level goodput. [Analytical EPD](../sweeper/epd.md) uses aggregate-mean SLA bounds only and does not report per-request goodput.

## AFD translation

Attention-FFN Disaggregation (AFD) is implemented as layered contracts. The
[AFD topology contract](../sweeper/afd-topology.md) owns complete A/F parallel enumeration,
validation, and A/F GPU accounting; later layers own search composition, measurements, staged
execution, analytical replay, and the public CLI. The generic Sweeper accepts internal `afd` and
`afd+pd` branches, and the public CLI lowers `engine.mode: afd` into those typed branches.

Legacy AFD command:

```bash
aiconfigurator cli default \
  --model-path Qwen/Qwen3-32B \
  --system h200_sxm \
  --backend trtllm \
  --serving-mode afd \
  --total-gpus 32 \
  --isl 1024 \
  --osl 128 \
  --ttft 800 \
  --tpot 30 \
  --strict-sla
```

The AISimulate recommendation contract is:

<!-- afd-migration-contract-start -->
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
  mode: afd
  model: Qwen/Qwen3-32B
  hardware: h200_sxm
  backend: trtllm
  afd:
    phase: decode
    combined_with_pd: true
    a_batch_size: 128

evaluation:
  sla:
    ttft_ms: 800
    itl_ms: 30

optimization:
  target: throughput_per_gpu
  strict_sla: true
  constraints:
    max_candidate_gpus: 32
```
<!-- afd-migration-contract-end -->

This preserves the legacy command's default of decode-side AFD combined with a static prefill
companion. Internally this maps to the Sweeper's `afd+pd` branch. The GPU constraint covers the A
pool, F pool, and companion together; the Sweeper AFD contract accounts for each contribution
explicitly. Run the analytical search with:

```bash
aisimulate recommend --config recommendation.yaml
```

Each selected recommendation contains a concrete `engine.afd` topology and, for `afd+pd`, its
regular companion worker. That generated YAML can be passed directly to `aisimulate predict`.
The prediction output includes `afd-replay-spec.json`, which freezes the exact topology,
measurements, workload, and companion contract, plus `afd-qualification.json`, which validates A/F
pool routing and GPU accounting. Both artifacts explicitly mark native deployment unsupported.
The commands use AISimulate's analytical AFD foreground engine; they do not imply native
request-level AFD serving or Kubernetes/shell deployment generation. AFD currently requires
fixed-length synthetic request traffic and an absolute load; trace, session, random-length, and
`kv_capacity_fraction` traffic fail validation.

## Workflows that must remain on AIC

Continue using the compatibility command for:

- minimum-GPU procurement sizing from `--target-request-rate` or `--target-concurrency`;
- deployment manifest generation and `--deployment-target` outputs;
- exact AIC support-matrix checks;
- AIC experiment YAML that has not been manually translated;
- multimodal image-input and EPD workloads outside the documented [fixed synthetic-image analytical EPD scope](../sweeper/epd.md);
- exact single-point FPM, static, AFD, per-op, detail, or power estimation;
- heterogeneous P/D backends or versions, and heterogeneous hardware outside the documented
  Sweeper YAML/SDK boundary;
- AIC-specific database modes and expert estimator flags.

The compatibility command remains available in AISimulate 0.12.0 and is targeted for removal in
AISimulate 0.13.0. Removal is gated on verified unified CLI replacements for every remaining workflow
above; continue using the compatibility command until the applicable replacement is documented.

## Related documentation

- [AISimulate CLI schema and output contract](design.md)
- [AIConfigurator CLI and Python API](../../python/aisimulate/README.md)
- [AISimulate repository history](../repository-history.md)
- [AIC synchronization process](../aic-sync.md)
