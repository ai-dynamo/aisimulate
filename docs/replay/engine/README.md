<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Replay engine

`engine` describes the deployment that replays the [workload](../workloads.md):
the model, hardware and backend, the worker pools and their parallelism,
scheduler limits, KV cache, P/D transfer and speculative decoding. Every
`predict` and `recommend` configuration has one `engine` block next to
`traffic`.

```yaml
engine:
  mode: aggregated            # deployment shape
  model: Qwen/Qwen3-32B-FP8
  hardware: h200_sxm
  backend: vllm
  workers:                    # one entry per worker role
    aggregated:
      parallelism: {}         # replicas, TP, PP, attention DP, MoE TP/EP, CP
      scheduler: {}           # batch limits and backend cadence
      kv_cache: {}            # G1 capacity, prefix caching, G2/G3 offload
      timing: {}              # forward-pass timing provider
  # kv_transfer: {...}        # P/D only
  # speculation: {...}        # optional ngram speculative decoding
```

The smallest valid block sets `model`, `hardware` and an empty worker role:

```yaml
engine:
  model: Qwen/Qwen3-32B-FP8
  hardware: h200_sxm
  workers:
    aggregated: {}
```

With every other field omitted, it is equivalent to:

```yaml
engine:
  mode: aggregated
  model: Qwen/Qwen3-32B-FP8
  hardware: h200_sxm
  backend: vllm
  context_length: max
  workers:
    aggregated:
      parallelism: {replicas: 1, tensor: 1, pipeline: 1, attention_data: 1, moe_tensor: 1, moe_expert: 1}
      scheduler: {max_batched_tokens: 8192, max_sequences: 256}
      kv_cache:
        block_size: 64
        prefix_caching: true
        bytes_per_token: auto
        capacity: {type: default, memory_fraction: 0.9}
      timing: {type: default}
      startup_seconds: 0
```

In `recommend`, omitted `mode`, `backend` and parallelism fields become search
domains instead of the single values above; see
[Sweeper search space](../../sweeper/search-space.md).

## Read by block

| Block | Page |
| --- | --- |
| `engine.mode`, `model`, `hardware`, `backend`, `backend_version`, `context_length`, `workers` roles | [This page](#deployment-fields) |
| `engine.workers.<role>.{hardware, startup_seconds, parallelism, scheduler}`, `engine.enable_chunked_prefill` | [Workers](workers.md) |
| `engine.workers.<role>.kv_cache` | [KV cache](kv-cache.md) |
| `engine.kv_transfer` | [P/D KV transfer](kv-transfer.md) |
| `engine.speculation`, `engine.nextn`, `engine.nextn_accepted` | [Speculative decoding](speculation.md) |
| `engine.afd`, `engine.workers.encoder` | [Analytical AFD and EPD](analytical.md) |
| `engine.workers.<role>.timing`, `estimation_mode`, `database_mode`, quantization and kernel selectors | [Performance-model configuration](../../perf-model/configuration.md) |

Field tables on these pages use the following columns:

- **Default** is the value used when the field is omitted in `predict`.
- **Recommend** is `fixed` when `recommend` accepts only a concrete value,
  `predict only` when `recommend` rejects the field, or the default search
  domain when `recommend` searches it.

<a id="deployment-fields"></a>

## Deployment fields

| Knob | Default | Recommend | Rules |
|---|---|---|---|
| `engine.mode` | `aggregated` | `{choices: [aggregated, disaggregated]}` | `aggregated`, `disaggregated`, or `afd`. `afd` must be explicit and is described in [Analytical AFD and EPD](analytical.md). |
| `engine.model` | Required | fixed | Hugging Face ID or local model directory. |
| `engine.hardware` | Required | `auto` allowed | System identifier, such as `h200_sxm`. `auto` is recommendation-only and resolves from `optimization.hardware`. |
| `engine.backend` | `vllm` | `{choices: [vllm, sglang]}` | `vllm`, `sglang`, or `trtllm`. Selects both the scheduler semantics and the performance data. |
| `engine.backend_version` | Latest data for the backend | fixed | Selects performance data only. It does not change scheduler behavior. Must be a queryable version for the hardware and backend: `current`, `previous` when configured, `next` when newer data is available, or a version one of those available aliases resolves to. See [backend versions](#backend-versions). |
| `engine.context_length` | `max` | fixed | Positive prompt-plus-output token limit, or `max`. With a number, prompts at or above the limit are rejected and generation stops at the limit. `max` applies the model config's maximum on vLLM; SGLang and TensorRT-LLM then run without a limit. |
| `engine.workers` | Required | Per mode | Role mappings; see [Worker roles](#worker-roles). A role may be `{}` to use all defaults. |

## Backend versions

The accepted versions are the populated slots defined by
[`query_versions.yaml`](../../../python/aisimulate/src/aisimulate_core/systems/query_versions.yaml)
and the derived `next` slot when newer data is available. Versions outside these
slots, including some
support-matrix `PASS` rows, are rejected with an error listing the accepted
versions. Set `engine.backend_version: current` or use a version in that list.
See [the support-matrix version boundary](../../perf-model/support-matrix.md#versions-the-cli-accepts).

<a id="worker-roles"></a>

## Worker roles

`engine.workers` holds one mapping per worker role. The roles that must be
present follow `engine.mode`:

| `mode` | Required roles | Rejected roles |
| --- | --- | --- |
| `aggregated` | `aggregated` | `prefill`, `decode`, and `engine.kv_transfer` |
| `disaggregated` | `prefill` and `decode` | `aggregated` |
| `afd` | None, or one P/D companion | See [AFD](analytical.md#afd) |

Each role has the same sub-blocks (`parallelism`, `scheduler`, `kv_cache`,
`timing`, plus `hardware` and `startup_seconds`), configured independently. In
`recommend`, a mode domain may declare all three roles; each candidate keeps
only the roles its mode uses. An optional `encoder` role adds an analytical
EPD encoder pool; see [EPD](analytical.md#epd).

All roles share `model`, `backend` and `backend_version`;
per-role overrides of these fields are rejected. Only disaggregated `prefill`
and `decode` workers may override `hardware`.
Workers may override the shared `context_length`; see
[context limits](workers.md#context-limits) for prediction and recommendation rules.

## A complete P/D example

```yaml
engine:
  mode: disaggregated
  model: Qwen/Qwen3-32B-FP8
  hardware: h200_sxm
  backend: sglang
  workers:
    prefill:
      parallelism: {replicas: 2, tensor: 2}
    decode:
      parallelism: {replicas: 1, tensor: 4}
      scheduler: {max_sequences: 512}
  kv_transfer:
    bandwidth_gb_per_second: 50
```

This deployment uses `2 × 2 + 1 × 4 = 8` GPUs. Run it with:

```bash
aisimulate predict --stack engine --config prediction.yaml --output-dir ./pd-prediction
```

## Where engine options are validated

Replay validates the `engine` block before running. Unsupported combinations,
such as a backend-specific field on another backend or G3 offload in P/D,
fail with an error naming the field. They are never silently ignored. For which
features combine with which backends and topologies, see
[feature support](../features.md).
