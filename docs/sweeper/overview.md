---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Sweeper
subtitle: Experimental backend-neutral configuration search
---

> [!WARNING]
> **Experimental.** Sweeper is intended for evaluation and feedback, not production capacity
> planning. Its API, configuration schema, search behavior, and output may change without a
> standard deprecation period.

Sweeper searches deployment configurations with a black-box optimizer. It turns every suggestion
into a versioned `ReplaySpec`, sends that specification to an injected `RunnerFactory`, and returns a
schema-versioned `SweepResult` with a complete candidate ledger and ranked or Pareto views.

The `aisimulate` package owns only backend-neutral simulation behavior. Optional feature packages
can register a `SweepConfigProvider` that contributes search dimensions and materializes its part
of a replay. Sweeper imports a provider only when its adapter name appears in the configuration.

## Start Here

- [Quickstart](quickstart.md) runs a small backend-neutral sweep.
- [Tutorial](tutorial.md) explains a complete sweep configuration.
- [Architecture](architecture.md) shows the provider, replay, and worker boundaries.
- [Configuration](configuration.md) describes core and adapter-owned search spaces.
- [Traffic](traffic.md) defines trace, request-rate, concurrency, and KV-load workloads.
- [Optimization Goals](optimization-goals.md) defines scalar and Pareto objectives.
- [AFD Topology Contract](afd-topology.md) defines Attention-FFN parallel shapes, validation, and
  complete topology enumeration.
- [Results](results.md) describes `ReplaySpec`, the `SweepResult` envelope, and candidate records.
- [Migrate from AIConfigurator](../cli/migrate-from-aiconfigurator.md) maps legacy Sweeper inputs to
  the standalone configuration and execution workflow.
- [Sweep Configuration Providers](sweep-config-provider.md) documents the extension ABI.
- [Dynamo Integration](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/developer-guide/knowledge-base/modular-components/ai-simulate-experimental/sweeper-experimental/dynamo-integration.md)
  composes Dynamo's optional Planner, Router, and replay adapters with the standalone Sweeper core.

## Python Entry Point

`Sweeper` is the only public execution interface. Supply a replay runtime explicitly:

```python
from aisimulate.sweeper import SmartSearchConfig, Sweeper

config = SmartSearchConfig.from_yaml("sweep.yaml")
result = Sweeper(runner_factory=my_runner_factory).run(config)
```

The public `aisimulate recommend --config ...` command validates the unified schema and selects a
runner through `--stack`. The `Sweeper` Python API remains available for callers that inject a
`RunnerFactory` directly.

## Compatibility

- A provider is imported only when its adapter name appears under `adapters`.
- The runner advertises supported `ReplaySpec` versions, backend/topology pairs, and runtime hooks
  before a study starts.
- Every `Sweeper.run` call owns fresh optimizer studies, result caches, runners, and worker pools.
- KVBM search fields are rejected. The AI Simulate engine and replay path do not support those
  fields and provide no adapter migration for the old host or disk offload settings.
