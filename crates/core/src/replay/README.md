<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AISimulate Replay implementation

This module owns Dynamo-neutral virtual time, worker lifecycle, placement/scaling
composition and reports. The engine owns scheduling and cache allocation;
performance-model providers supply timing. User and API documentation lives in
[docs/replay](../../../../docs/replay/README.md).

- [Runtime extension invariants](../../../../docs/replay/extending.md)
- [Rust API](../../../../docs/replay/api/rust.md)
- [Native composition contract](../../../../docs/adapters/native-composition.md)
- [Feature and composition limits](../../../../docs/replay/features.md)

## File Map

- `src/replay/replayer.rs`
  Owns the canonical replay entrypoint and composition boundary.
- `src/replay/agg.rs`
  Shared offline cluster simulator for every aggregated replay.
- `src/replay/disagg.rs`
  Offline two-stage replay harness with separate prefill and decode pools.
- `src/replay/state.rs`
  Per-request state used by the aggregated and disaggregated runtimes.
- `src/replay/event.rs`
  `SimulationEvent`, `SimulationEventKind`, and worker-completion payload types
  used by both topology runtimes.
- `src/replay/components/`
  Admission, logical-worker, and Generalized Engine orchestration shared by the
  aggregated and disaggregated runtimes.
- `src/replay/core/`
  Neutral placement contracts and built-in round-robin policies.
- `src/replay/runtime_utils.rs`
  Shared helpers used by `agg.rs` and `disagg.rs`: event scheduling,
  `ReadyWorkerCompletions`, and `next_timestamp`.
- `src/replay/telemetry.rs`
  Serializable, policy-neutral telemetry snapshots and the optional observer
  contract.
- `src/replay/progress.rs`
  `ReplayProgress`, the indicatif-based progress bar used by the harnesses.
- `src/replay/report.rs`
  `TraceCollector` and the serializable `ReplayReport`.
- `src/replay/spec.rs`
  Canonical, serializable `ReplaySpec` and topology/provider descriptors.
