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

## Generated concurrency requests

Programmatic replay callers can provide
`replay::loadgen::GeneratedRequests::new(request_count, factory)` through
`ReplayRuntimeInput::GeneratedRequests` instead of constructing every
`DirectRequest` up front. Set `ReplaySpec::max_in_flight` to the concurrency
limit. Replay calls the factory with the original zero-based request index
only when a slot becomes available, and applies the admission timestamp just
as it does for an eager request queue. The factory should generate one request
without retaining previous requests or capturing a prebuilt request list.

This keeps pending prompt storage independent of the total request count;
live requests, engine queues, and explicitly captured per-request output still
consume memory. The engine receives the same ordinary `DirectRequest` and
does not participate in source generation. Open-loop and existing trace inputs
retain their current behavior. Integrations such as Dynamo must opt into the
generated source after adopting a crate release containing this API.

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
