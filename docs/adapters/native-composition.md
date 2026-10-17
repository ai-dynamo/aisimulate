<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native Replay composition

`ReplayComposition` composes placement and optional scaling into
`Replayer<C>` without importing a serving framework into AISimulate core.
The built-in `RoundRobinComposition` has fixed workers. The unpublished
`aisimulate-python` wheel host adds `NativeDynamoReplayComposition`, calling
Dynamo's native policy crates pinned to revision
`f24c59b936ba23c933cc35d2f8c7f9ff71f54c3d`. The published core has no Dynamo
dependency.
Dynamo integrations may supply Planner callbacks through the same contracts.
These are Rust crate interfaces, not a stable dynamically loaded binary ABI.

## Standalone native routing

A serialized ReplaySpec selects `adapters.placement.provider: dynamo_kv_router`
with `config.affinity` and optional `config.router_config`. The native composition
forwards requests to Dynamo's `LocalScheduler` and `SessionAffinity`. Dynamo
owns queueing, capacity checks, selection, booking and affinity waits. AISimulate
forwards dispatch and completion events and drives a paused Tokio clock using
simulation timestamps; the shared Replayer owns the simulation. Python code is not invoked for
each routing decision. `routing_policy` reports native provider provenance and
per-pool evidence. Unsupported config combinations are errors.

Dynamo's compatibility runner may use `execute_native_router_replay` with optional
scaling/telemetry and prefill-estimator callbacks. Those hooks are not required
by the built-in AISimulate runner. Conversation affinity uses static worker pools.

## Composition construction

The trait is defined in
[`replayer.rs`](../../crates/core/src/replay/replayer.rs). Its associated types
are `Metadata`, `Observation`, `AggregatedPlacement` and
`DisaggregatedPlacement`. The placements operate on Replay request payloads
with the composition's admission metadata and observation batch.

| Method | Contract |
| --- | --- |
| `validate_spec(&self, &ReplaySpec) -> ReplayResult<()>` | Validate composition-specific provider descriptors/combinations; default accepts |
| `create_aggregated_placement(&mut self, dp_size, topology)` | Return one aggregated placement policy for the worker/rank topology |
| `create_disaggregated_placements(&mut self, prefill_dp_size, prefill_topology, decode_dp_size, decode_topology)` | Return separate prefill and decode policies; DP sizes may differ |
| `take_scaling_policy(&mut self)` | Transfer an optional `Box<dyn ReplayScalingPolicy>` into run ownership; default none |
| `set_determinism(&mut self, ReplayDeterminism)` | Respect explicitly requested canonical selection; normal runs remain unseeded |

Admission metadata and engine observations currently use `#[doc(hidden)]`
cross-crate integration traits. Their presence in the composition signature
requires version-aligned integrations; it does not turn all internal payloads
into stable user APIs. Public input/report contracts are in the
[Rust API reference](../replay/api/rust.md).

## PlacementPolicy

[`PlacementPolicy<Request>`](../../crates/core/src/replay/core/mod.rs) supplies:

- `place(request, metadata, session_id, now_ms) -> Result<PlacementEffects>`.
  Effects combine `Immediate(Placement)` or `Queued` with placements released
  by the same operation.
- `observe(observation, now_ms)`, `request_terminal(request_id, now_ms)` and
  `prefill_completed(request_id, now_ms)`, returning newly releasable placements.
- `cancel_pending(request_id) -> bool` and `pending_count()` for queued ownership.
- Optional `dispatch_committed`, `dispatch_aborted`, `advance_clock` and
  `next_wakeup_ms` hooks for tentative bindings and policy wakeups; see the
  [conversation lineage contract](../replay/agentic/conversation-lineage.md).
- `worker_ready`, `worker_draining`, `worker_removed`, and `topology_settled`
  lifecycle callbacks, also returning releasable placements.

`WorkerTopology` identifies a logical worker and its scheduler IDs.
`Placement` identifies the request and selected scheduler, reported overlap,
optional cache sample and placement replica. Reported overlap is a decision
observation, not actual engine cache reuse. Policies own pending placement
state; engines own admission, allocation and execution. Lifecycle callbacks
must not dispatch to retired ranks or release a request twice.

All callbacks run synchronously at Replay's virtual time. The runtime owns
the event queue and settles engine/transfer/admission work before telemetry
and scaling. A policy must not advance time or implement a parallel replay loop.
Fallible placement callbacks are classified as `ReplayError::Placement` at
the runtime boundary.

## ScalingPolicy

[`ReplayScalingPolicy`](../../crates/core/src/replay/scaling.rs) has:

```rust
fn initial_tick_ms(&mut self) -> anyhow::Result<f64>;
fn on_tick(&mut self, snapshot: ReplayScalingSnapshot)
    -> anyhow::Result<ReplayScalingDecision>;
fn capture_lifecycle_evidence(&self) -> bool; // default false
```

Times are absolute simulated milliseconds. A nonfinite initial tick disables
scaling. `on_tick` receives a settled snapshot with tick ordinal/time, recent
per-role FPM samples, traffic interval, and active/starting/draining worker IDs.
`ReplayScalingDecision` supplies optional prefill/decode targets and an optional
next absolute tick. Targets mean desired **active plus starting** workers;
draining workers are already leaving and do not count. An absent target
preserves capacity; no next tick stops recurring scaling. Aggregated replay
uses the decode target and ignores prefill target.

The policy is intentionally not `Send`: Replay is single-threaded and an
integration may hold Python state. `NoScaling` disables ticks with infinity.
Callback failures become `ReplayError::Scaling`. Resource quiescence remains
Replay's responsibility during worker drain and removal.

## Observations and telemetry

Compositions consume neutral engine observations only when needed. Round-robin
can skip KV event capture; the native routing composition converts observations
for Dynamo's KV index only when routing requires them. Offline P/D decode routing is overlap-blind and does not require
its own KV event stream. A requested deterministic selector uses
`ReplayDeterminism::selector_seed`, not an independently chosen seed.

<a id="kv-events"></a>

### KV events

Device (G1) events omit `tier`; a missing `tier` means `device`. A vLLM rank
with G2 also publishes G2 residency changes as `Stored`/`Removed` events with
`"tier": "host_pinned"`, as vLLM's offloading connector does. With
`dp_rank_local` these cover the rank's own cache. With `cluster_shared` they
cover the whole pool, starting with a snapshot when the rank joins. A child block can become resident before its parent, so each
`Stored` run carries the prompt index of its first block as `start_position`.
The join snapshot is ordered parent-first, but the live stream keeps landing
order, and a snapshot taken after a parent was evicted contains the children
without it. Consumers must place blocks by `start_position`, not by arrival.
Every rank of a shared pool retains its blocks' `tokens_hash`, so a block's
Router identity does not depend on which rank stored it.

`ReplayTelemetryObserver` is independent of scaling. It samples at a positive
finite cadence after workload settlement and before same-time scaling. It must
remain observational and cannot keep an otherwise deadlocked runtime alive.
Observer failures are typed `ReplayError::Telemetry`.

The optional `Request.session_id` is forwarded to G1 and G2
`StoredBlocks.session_id`, identifying the session whose request triggered the
store, not exclusive block ownership. G2 snapshots preserve that attribution;
removals remain hash-based.
