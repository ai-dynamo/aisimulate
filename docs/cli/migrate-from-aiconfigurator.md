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
| `--total-gpus` | `optimization.constraints.max_candidate_gpus` | Maximum GPUs per candidate |
| `--isl` | `traffic.source.input_tokens` | Synthetic input length |
| `--osl` | `traffic.source.output_tokens` | Synthetic output length |
| `--ttft` | `evaluation.sla.ttft_ms` | Time-to-first-token bound in milliseconds |
| `--tpot` | `evaluation.sla.itl_ms` | Per-request goodput uses average ITL; strict mode compares aggregate mean TPOT |
| `--strict-sla` | `optimization.strict_sla: true` | Reject before scalar ranking or Pareto dominance |

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

Strict filtering can use `ttft_ms` or `itl_ms` independently. A `goodput` or `goodput_per_gpu`
optimization still requires either both fields or `e2e_ms`, because those targets also need a
complete per-request SLA.

## Migration limits

- `--stack engine` is the default. Use `--stack dynamo` only when the independently installed Dynamo
  package provides the required runner and configuration adapters.
- Keep using the compatibility CLI for legacy flags that are not listed in this guide until their
  AISimulate mapping is implemented and qualified.
