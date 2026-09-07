<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AISimulate Core

`aisimulate-core` is the single public Rust package for engine-neutral
inference simulation. It preserves three explicit implementation and API
layers:

- `aisimulate_core::engine` owns scheduling, native GPU KV accounting,
  preemption, timing, and attention-DP composition.
- `aisimulate_core::replay` owns deterministic virtual time, logical-worker
  lifecycle, placement and scaling composition, and replay reports.
- `aisimulate_core::perfmodel` owns the imported AIConfigurator latency,
  memory, and compiled-model implementation. Its `EngineConfig` and `engine`
  stay in this namespace so they cannot collide with the replay engine.

The crate root promotes the common `EngineConfig`, `ReplaySpec`, `Replayer`,
canonical `ReplayReport`, and timing-provider contracts. Dynamo-specific Router, Planner,
transport, and live-runtime adapters remain outside this crate.

## Source layout and upstream syncs

The former AIConfigurator Rust crate is kept as a stable mirror subtree under
`src/perfmodel/`. The crate root contains only the small compatibility aliases
needed by that imported source and the public namespace boundary. Keeping the
implementation together instead of interleaving it with replay makes future
AIConfigurator commits path-rewritable and reviewable while still producing a
single crates.io package.

The default feature set is Rust-only and does not depend on PyO3. The `python`
feature exposes registration hooks for the unified `aisimulate._runtime`
extension, and `embed-python` additionally enables PyO3 interpreter
initialization for standalone Rust applications that call into Python during
one-time model compilation.

## Scheduler fidelity

The replay engine models vLLM's attention-DP prefill cadence through
`EngineConfig::prefill_schedule_interval`. See the
[configuration and validation notes](../../docs/vllm-prefill-schedule-interval.md).
