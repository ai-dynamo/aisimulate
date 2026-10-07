<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Public configuration

The public CLI consumes strict, typed YAML. `predict` accepts concrete values;
`recommend` adds supported search domains and an objective. This document is
not the [`SmartSearchConfig` SDK schema](../sweeper/sdk.md).

<a id="configuration-model"></a>

## Configuration Model

Jump to [traffic](../replay/workloads.md), [engine](../perf-model/configuration.md),
[search domains](../sweeper/search-space.md#recommendation-domains), [presets](../sweeper/search-space.md#presets-and-default-ranges),
[optimization goals](../sweeper/optimization-goals.md#optimization-goal), [search controls](../sweeper/search-space.md#optimizer-controls), or [outputs](cli.md#outputs).

These fragments show the available top-level sections; they are not complete runnable inputs.
Use the [quickstart](../getting-started/quickstart.md) for complete configurations.

Both commands use one strict YAML model:

```yaml
traffic: {}
engine: {}
router: {}
planner: {}
evaluation: {}
execution: {}
```

`recommend` extends that model with:

```yaml
optimization: {}
optimizer: {}
```

The command determines the document type. There is no top-level `kind` or stack field.

| Section | `predict` | `recommend` | Purpose |
|---|---|---|---|
| `traffic` | Optional | Optional | Request source, load shape, and stopping condition. Uses the default synthetic request traffic when omitted. |
| `engine` | Required | Required | Model, hardware, backend, topology, and worker roles. |
| `router` | Optional adapter | Optional adapter | Dynamo routing policy; round robin when omitted. Requires the integration when configured. |
| `planner` | Optional adapter | Optional adapter | Dynamo runtime scaling; disabled when omitted. Requires the integration when configured. |
| `evaluation` | Optional | Optional | Service-level objective (SLA) thresholds used for reporting and goals. |
| `execution` | Optional | Optional | Host RAM/CPU budgets and supervisor deadlines; automatic defaults apply when omitted. See [local resources](local-resources.md). |
| `optimization` | Rejected | Required | Recommendation objective and candidate GPU constraints. |
| `optimizer` | Rejected | Optional | Public search controls. |

Unknown fields are rejected everywhere. Every semantic configuration knob is an explicit, typed YAML
field. The selected stack, backend, policy, timing model, or capacity model determines which
conditional fields are legal; there is no generic configuration passthrough mapping.

## Defaults and overrides

Concrete prediction values use each component's defaults when omitted.
Recommendation fields may use a concrete value, `choices`, `range`, or a
component preset only where the field supports that form. Search domains are
never accepted in a prediction. Presets expand to complete atomic mappings;
see [presets and search domains](../sweeper/search-space.md).

`--set PATH=YAML_VALUE` applies after YAML loading, in left-to-right order.
The final value wins, then full validation runs. Replace a complete sequence
rather than addressing its indices. When changing a tagged object such as
`traffic.load.type`, replace the whole mapping if the old fields no longer
apply. See [override semantics](cli.md#override-semantics).

## Component reference

| Fields | Canonical guide |
|---|---|
| `traffic.source`, `traffic.load`, `traffic.stop` | [Replay workloads](../replay/workloads.md) |
| `engine.model`, hardware/backend identity, estimator policy and worker controls | [Performance-model configuration](../perf-model/configuration.md) |
| Worker roles, parallel execution and transfer | [Replay topology](../replay/topology-and-scheduling.md) |
| Prefix caching, host and storage offload | [Replay cache](../replay/cache.md) |
| `router`, `planner` | [Dynamo integration](../replay/dynamo.md) |
| `execution.resources` | [Local execution resources](local-resources.md) |
| `optimization` | [Optimization goals](../sweeper/optimization-goals.md) |
| `optimizer`, domains and presets | [Search space](../sweeper/search-space.md) |
| Selected `--output NAME` section | [Output adapter ABI](../adapters/output-abi.md) |

Component tables use **Default** for omitted concrete prediction values and
**Default Range** for independent recommendation dimensions. `x` marks a
non-sweepable field; `-` marks a singleton default domain. A **Preset** column
identifies the smallest object that owns the complete atomic mapping. `auto`
in the parallelism preset asks Sweeper to generate feasible shapes; it is not
a free-form value accepted by every field.

<a id="evaluation"></a>

## Evaluation

```yaml
evaluation:
  sla:
    ttft_ms: 500
    itl_ms: 50
```

| Knob | Default | Default Range | Preset | Rules |
|---|---:|---|---|---|
| `evaluation.sla.ttft_ms` | `null` | `x` | `-` | Positive and independently optional; an unset field is unbounded. |
| `evaluation.sla.itl_ms` | `null` | `x` | `-` | Positive and independently optional; an unset field is unbounded. |
| `evaluation.sla.e2e_ms` | `null` | `x` | `-` | Positive; mutually exclusive with TTFT plus ITL. |

`goodput` and `goodput_per_gpu` optimization require at least one SLA bound. Planner throughput
scaling specifically uses the `ttft_ms` plus `itl_ms` form when the recommendation target is
SLA-based.

## Optimization fields

These fields belong to the public `recommend` schema. For scoring, goodput,
strict SLA, Pareto and minimum-GPU semantics, use [optimization goals](../sweeper/optimization-goals.md).
The `optimization` section is required, even when its target uses a default.

| Knob | Default | Default Range | Preset | Rules |
|---|---:|---|---|---|
| `optimization.target` | `throughput` | `x` | `-` | Maximize `throughput`, `throughput_per_gpu`, `throughput_per_user`, `goodput`, or `goodput_per_gpu`; minimize `ttft`, `e2e_latency`, or `min_gpus`; or compute `pareto`. |
| `optimization.hardware` | `null` | `x` | `-` | One nonempty hardware identifier; required for `engine.hardware: auto`. |
| `optimization.strict_sla` | `false` | `x` | `-` | When true, reject candidates whose aggregate mean metrics exceed any configured SLA bound before ranking or Pareto analysis. |
| `optimization.constraints.min_candidate_gpus` | `null` | `x` | `-` | Positive when set and no greater than the maximum. |
| `optimization.constraints.max_candidate_gpus` | `32` | `x` | `-` | Positive. |
| `optimization.constraints.min_goodput_rps` | `null` | `x` | `-` | Positive finite SLA-compliant requests/s floor for `min_gpus` only; required with request-rate traffic and cannot exceed the offered rate. Optional with fixed concurrency. |

`optimization.hardware` is one fallback SKU, never a list or inventory.
Concrete P/D role SKUs can override it. `optimizer` controls execution of the
search and is described in [search space](../sweeper/search-space.md#optimizer-controls).
