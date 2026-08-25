# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---
title: Migrate from AIConfigurator
subtitle: Translate legacy Sweeper commands into AISimulate configuration
---

> [!WARNING]
> **Experimental.** The standalone Sweeper configuration and execution APIs may change without a
> standard deprecation period.

This guide translates legacy `aiconfigurator cli default` inputs into the standalone AISimulate
Sweeper configuration. This is currently a command-to-workflow migration, not a direct
command-to-command replacement: `python -m aisimulate.sweeper` validates configuration, while an
actual sweep requires a `RunnerFactory` supplied through Python or an adapter-owned CLI.

## Common input mapping

| AIConfigurator | AISimulate Sweeper | Notes |
|---|---|---|
| `--model-path` | `search_space.model_name` | Same model identifier |
| `--system` | `search_space.hardware_sku` | AIConfigurator system identifier |
| `--backend` | `search_space.backend` | AISimulate accepts a list |
| `--total-gpus` | `search_space.gpu_budget` | Maximum GPUs per candidate |
| `--isl` | `workload.isl` | Synthetic input length |
| `--osl` | `workload.osl` | Synthetic output length |
| `--ttft` | `goal.sla.ttft_ms` | Time-to-first-token bound in milliseconds |
| `--tpot` | `goal.sla.itl_ms` | Per-request goodput uses average ITL; strict mode compares aggregate mean TPOT |
| `--strict-sla` | `goal.strict_sla: true` | Reject before scalar ranking or Pareto dominance |

## Strict SLA filtering

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

AISimulate configuration:

```yaml
search_space:
  model_name: meta-llama/Meta-Llama-3.1-8B
  hardware_sku: gb200
  backend: [trtllm]
  deployment_mode: [agg]
  gpu_budget: 8

workload:
  isl: 1024
  osl: 128
  request_rate: 4
  num_request_ratio: 10

goal:
  target: throughput
  strict_sla: true
  sla:
    ttft_ms: 800
    itl_ms: 30

sweep:
  max_rounds: 10
  candidates_per_round: 8
  parallel_evals: 4
```

Load and execute the configuration with an explicit replay runtime:

```python
from aisimulate.sweeper import SmartSearchConfig, Sweeper

config = SmartSearchConfig.from_yaml("sweep.yaml")
candidates = Sweeper(runner_factory=my_runner_factory).run(config)
```

Without `strict_sla`, the configured thresholds classify individual requests for goodput; a slow
request contributes no tokens to goodput, but does not reject the whole candidate. With
`strict_sla: true`, AISimulate additionally compares every configured bound with the candidate's
aggregate mean metrics and removes a violation before ranking or Pareto analysis. Comparisons are
inclusive, and missing or non-finite aggregate metrics reject the candidate.

## Migration limits

- The standalone module CLI validates YAML but does not select a replay implementation.
- Adapter-owned search fields and execution commands depend on the installed integration.
- Keep using the compatibility CLI for legacy flags that are not listed in this guide until their
  AISimulate mapping is implemented and qualified.
