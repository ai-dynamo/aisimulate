# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---
title: Migrate from AIConfigurator
subtitle: Translate AIConfigurator CLI commands into AISimulate CLI commands
---

> [!WARNING]
> **Experimental.** The AISimulate recommendation schema and search behavior may change without a
> standard deprecation period.

This guide compares the compatibility `aiconfigurator` CLI with the public AISimulate CLI introduced
by the unified command surface. A legacy command becomes an AISimulate recommendation YAML plus one
`aisimulate recommend --config ...` command.

## Common input mapping

| AIConfigurator CLI | AISimulate recommendation | Notes |
|---|---|---|
| `--model-path` | `engine.model` | Same model identifier |
| `--system` | `engine.hardware` | Same hardware identifier |
| `--backend` | `engine.backend` | One backend or an explicit recommendation domain |
| `--backend-version` | `engine.backend_version` | Pin the selected backend's forward-pass performance-data version |
| `--total-gpus` | `optimization.constraints.max_candidate_gpus` | Maximum GPUs per candidate |
| `--isl` | `traffic.source.input_tokens` | Synthetic input length |
| `--osl` | `traffic.source.output_tokens` | Synthetic output length |
| `--ttft` | `evaluation.sla.ttft_ms` | Time-to-first-token bound in milliseconds |
| `--tpot` | `evaluation.sla.itl_ms` | Per-request goodput uses average ITL; strict mode compares aggregate mean TPOT |
| `--request-latency` | `evaluation.sla.e2e_ms` plus `optimization.strict_sla: true` | Fixed-output synthetic migration; applies per-request E2E and filters aggregate mean E2E |
| `--strict-sla` | `optimization.strict_sla: true` | Reject before scalar ranking or Pareto dominance |

AISimulate uses only `engine.backend_version` for this identity. The current performance database
is keyed by backend and version, so there is no separate `performance_data_version` field. When
`engine.backend_version` is omitted, the Sweeper resolves the latest available version once before
the search begins and records that concrete version on every candidate. Pin `engine.backend` to one
concrete backend when setting `engine.backend_version`.

## Illustrative strict SLA translation

The examples below demonstrate how the strict-SLA fields map, but they are not behaviorally
equivalent workloads. The legacy `cli default` command capacity-sweeps aggregated and
disaggregated configurations under the eight-GPU budget without a fixed offered load. The
AISimulate example makes the additional choices of aggregated mode and a constant 4 RPS so its
traffic and replay behavior are explicit.

Legacy command:

```bash
aiconfigurator cli default \
  --model-path meta-llama/Meta-Llama-3.1-8B \
  --system gb200 \
  --backend trtllm \
  --total-gpus 8 \
  --isl 1024 \
  --osl 128 \
  --ttft 800 \
  --tpot 30 \
  --strict-sla
```

Save this AISimulate configuration as `recommendation.yaml`:

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
  hardware: gb200
  backend: trtllm
  workers:
    aggregated: {}

evaluation:
  sla:
    ttft_ms: 800
    itl_ms: 30

optimization:
  target: throughput
  strict_sla: true
  constraints:
    max_candidate_gpus: 8
```

Run the illustrative AISimulate command:

```bash
aisimulate recommend --config recommendation.yaml
```

Without `optimization.strict_sla`, the configured thresholds classify individual requests for
goodput; a slow request contributes no tokens to goodput, but does not reject the whole candidate.
With `strict_sla: true`, AISimulate additionally compares every configured bound with the
candidate's aggregate mean metrics and removes a violation before ranking or Pareto analysis.
Comparisons are inclusive, and missing/non-finite aggregate metrics or zero qualifying latency
samples reject the candidate.

Request-level goodput and strict filtering both accept `ttft_ms` or `itl_ms` independently; an unset
field is unbounded. `strict_sla` changes only whether the configured bounds additionally reject a
candidate based on its aggregate means.

## Request-latency translation

For a fixed-output-length synthetic workload, translate AIConfigurator's `--request-latency` to
AISimulate's existing end-to-end SLA and enable strict candidate filtering.

Legacy command:

```bash
aiconfigurator cli default \
  --model-path meta-llama/Meta-Llama-3.1-8B \
  --system gb200 \
  --backend trtllm \
  --total-gpus 8 \
  --isl 1024 \
  --osl 128 \
  --request-latency 12000
```

Save this AISimulate configuration as `recommendation.yaml`:

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
  hardware: gb200
  backend: trtllm
  workers:
    aggregated: {}

evaluation:
  sla:
    e2e_ms: 12000

optimization:
  target: throughput
  strict_sla: true
  constraints:
    max_candidate_gpus: 8
```

Run the AISimulate command:

```bash
aisimulate recommend --config recommendation.yaml
```

With a fixed output length, the aggregate values satisfy:

```text
mean_e2e_latency_ms = mean_ttft_ms + mean_tpot_ms * (output_tokens - 1)
```

The existing `e2e_ms` field therefore expresses the same numerical bound without introducing a
second request-latency alias. It is a per-request E2E bound when computing goodput, and
`strict_sla: true` additionally rejects candidates whose aggregate mean E2E latency exceeds the
bound. `e2e_ms` is mutually exclusive with `ttft_ms` and `itl_ms`.

This translation is not an aggregate-only constraint. AISimulate does not currently expose a
public mapping for a bound that filters aggregate mean E2E latency but must not participate in
request-level goodput.

## Migration limits

- `--stack engine` is the default. Use `--stack dynamo` only when the independently installed Dynamo
  package provides the required runner and configuration adapters.
- Keep using the compatibility CLI for legacy flags that are not listed in this guide until their
  AISimulate mapping is implemented and qualified.
