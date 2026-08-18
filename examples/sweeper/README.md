<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sweeper Examples

> [!WARNING]
> **Experimental.** Sweeper is intended for evaluation and feedback, not production capacity
> planning. Its Python API, configuration schema, search behavior, and output may change without a
> standard deprecation period. Sweeper does not guarantee SLA compliance, prediction accuracy, or
> globally optimal configurations.

This directory is the canonical home for backend-neutral and Dynamo-integrated Sweeper examples.

## Run the Backend-Neutral Example

The backend-neutral example injects a deterministic replay runner and does not import Dynamo or
another application framework:

```bash
python examples/sweeper/run_sweep.py \
  --config examples/sweeper/sweep.yaml
```

## Run the Dynamo Integration Example

To use Dynamo Replay or the `dynamo.planner` and `dynamo.router` providers, follow Dynamo's
[Sweeper integration guide](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/developer-guide/knowledge-base/modular-components/ai-simulate-experimental/sweeper-experimental/dynamo-integration.md).
That guide owns the Dynamo build, installation, Planner, Router, and replay-composition steps.

With compatible AI Simulate and Dynamo packages installed, run the general search example with the
explicit Dynamo runner:

```bash
python examples/sweeper/tools/run_sweep.py \
  --config examples/sweeper/configs/smart_sweep.yaml
```

To use another replay implementation, pass its `RunnerFactory` to the `Sweeper` constructor.

The GLM-5-FP8 Pareto-front configuration captures a disaggregated search over `kv_load_ratio`.
Update `workload.trace_path` before running a trace-backed configuration.

## Generate a Synthetic Trace

Generate a Mooncake-format trace whose request rate follows a sine wave:

```bash
python examples/sweeper/tools/gen_sine_trace.py \
  --out /tmp/sweeper-sine-trace.jsonl
```

Compare Planner load predictors on that trace:

```bash
python examples/sweeper/tools/run_load_predictor_sweep.py \
  --trace /tmp/sweeper-sine-trace.jsonl \
  --policies throughput_180_5 throughput_600_5
```

## Documentation

Read the canonical [Sweeper documentation](../../docs/sweeper/overview.md)
for the search flow, workload schema, optimization goals, and search-space reference. Dynamo-specific
composition is documented in the [Dynamo integration
guide](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/developer-guide/knowledge-base/modular-components/ai-simulate-experimental/sweeper-experimental/dynamo-integration.md).
