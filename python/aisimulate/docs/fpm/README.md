<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Forward-pass models

Start with the [self-benchmark and FPM onboarding guide](self-benchmarking-and-onboarding.md).
Its seven steps cover support checks, measurement planning, collection, profile
validation, canonical model construction, Replay integration, and accuracy
validation. It explains which engine-iteration differences self-collection can
capture and which behaviors, including PP, need separate simulator support.

Self-benchmark collection currently supports vLLM configurations that pass
model/runtime validation; new architectures can require benchmark adaptation.
SGLang and TensorRT-LLM support is coming soon. The guide includes the collected
[Kimi K3 TP8+DCP8 profile](self-benchmarking-and-onboarding.md#example-a-onboard-the-collected-kimi-k3-tp8dcp8-profile)
and a new MiniMax collection campaign as worked examples of the general procedure.

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
