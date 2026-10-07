<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Forward-pass models

Start with the [FPM self-service overview](../../../../docs/fpm-self-service/README.md)
for motivation, support and the onboarding workflow. All self-service guidance
lives under `docs/fpm-self-service/`:

- [Implementation and CLI reference](../../../../docs/fpm-self-service/implementation.md):
  commands, artifacts, validation, checkpointing and recovery.
- [Worked examples](../../../../docs/fpm-self-service/examples.md): import a Kimi K3
  TP8+DCP8 profile or collect a MiniMax-M2.7 TP4 profile.

This directory contains developer integration guidance, FPM design background
and model-specific profiles.

There are two distinct workflows:

| Workflow | Input | Consumer |
| --- | --- | --- |
| Offline whole-forward FPM | Validated `fpm_forward_perf.parquet` and its metadata sidecar | `best_available` with `estimation_mode="fpm_interpolation"`: lookup, interpolation, and supported SOL transfer |
| Online regression | Observed per-iteration, per-rank telemetry | A role-bound model updated with `tune_with_fpms` |

Offline FPM does not require an additional regression-training step. Predicting
request-level TTFT, ITL/TPOT, and throughput also requires scheduler and traffic
simulation; a forward-pass latency alone is not an end-to-end serving metric.
The published parquet and same-stem metadata sidecar may live outside the
AISimulate checkout; pass its path as `estimator_config.fpm_interpolation.fpm_parquet_path` in the canonical model configuration, or use the legacy `timing.fpm_parquet_path` YAML field with `timing.forward_model: fpm`.

## References

- [Add a model architecture for SOL-assisted FPM](sol-model-integration.md):
  developer instructions for analytical model descriptions and SOL-path tests;
  not required for direct-FPM self-service.
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
