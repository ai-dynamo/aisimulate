<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Forward-pass models

Start with the [end-to-end FPM workflow](end-to-end-workflow.md) to import
published self-benchmark data or collect new whole-forward measurements, load
the performance-data pair through the canonical SDK, and run an AISimulate
prediction. The [Kimi K3 TP8+DCP8 quickstart](end-to-end-workflow.md#use-an-existing-profile-kimi-k3-tp8dcp8)
uses existing data and runs entirely on CPU. The guide also includes collection
prerequisites, expected artifacts, acceptance checks, and recovery steps.

There are two distinct workflows:

| Workflow | Input | Consumer |
| --- | --- | --- |
| Offline whole-forward FPM | Validated `fpm_forward_perf.parquet` and its metadata sidecar | `best_available` with `estimation_mode="fpm_interpolation"`: lookup, interpolation, and supported SOL transfer |
| Online regression | Observed per-iteration, per-rank telemetry | A role-bound model updated with `tune_with_fpms` |

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
