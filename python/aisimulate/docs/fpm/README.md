<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Forward-pass models

Start with the [end-to-end FPM workflow](end-to-end-workflow.md) to collect
whole-forward measurements, publish a performance-data pair, load it through
the SDK, and run an AISimulate prediction. The guide includes prerequisites,
commands, expected artifacts, acceptance checks, and recovery steps.

There are two distinct workflows:

| Workflow | Input | Consumer |
| --- | --- | --- |
| Offline whole-forward FPM | Collector-produced `fpm_forward_perf.parquet` and its metadata sidecar | `forward_model="fpm"`: lookup, interpolation, and supported SOL transfer |
| Online regression | Observed per-iteration, per-rank telemetry | A role-bound model updated with `tune_with_fpms` |
| Learned forward-pass model | The same FPM telemetry recorded from a real deployment, trained offline | `RustForwardPassPerfModel.from_learned`: tree-ensemble artifact, see [learned-forward-pass-model.md](learned-forward-pass-model.md) |

Offline FPM does not require an additional regression-training step. Predicting
request-level TTFT, ITL/TPOT, and throughput also requires scheduler and traffic
simulation; a forward-pass latency alone is not an end-to-end serving metric.

## References

- [Collector usage and publication contract](../../collector/README.md#whole-forward-fpm-campaign)
- [Generator FPM target and runtime responsibilities](../generator_overview.md#fpm-v1-target)
- [AISimulate CLI configuration](../../../../docs/cli/user-guide.md)
- [Core API](../../../../docs/core-api.md)
- [Offline FPM modeling plan](aic-fpm-modeling-plan.md): design background;
  some milestones and implementation descriptions are historical.
- [Rust port design, August 2026](aic-fpm-rust-port-design-20260802.md): historical
  migration design, not the current API reference.
- [Online regression design](aic-fpm-regression-design.md): a separate telemetry
  workflow, not a prerequisite for consuming an offline FPM table.
