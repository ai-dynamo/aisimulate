<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native Replay composition

`ReplayComposition` composes placement and optional scaling into
`Replayer<C>` without importing a serving framework into AISimulate core.
The built-in `RoundRobinComposition` has fixed workers. Dynamo constructs its
own Router/Planner policies against the same neutral contracts.
These are Rust crate interfaces, not a stable dynamically loaded binary ABI.

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
can skip KV event capture; Dynamo performs Router-specific conversion outside
AISimulate core. Offline P/D decode routing is overlap-blind and does not require
its own KV event stream. A requested deterministic selector uses
`ReplayDeterminism::selector_seed`, not an independently chosen seed.

<a id="kv-events"></a>

### KV events

Engine `Request.session_id` and the `StoredBlocks.session_id` payload inside
`KvEventData::Stored` are optional metadata. Replay uses
`DirectRequest.replay_context.session_id` as its single session identity and
forwards it into the native engine request for aggregated execution and both
prefill/decode request legs. A session-only replay context may omit `authored_id`;
this does not manufacture a source request ID in reports. Legacy trace loaders
normalize authored session IDs into this context, including single-turn traces
with session reporting disabled. Whitespace is trimmed and blank IDs are absent.
Native G1 stores identify the request making blocks visible; they do not imply exclusive
ownership. Requests without an authored session stay unattributed, including
legacy traces that use synthetic row IDs for reporting. G2 residency events and
removals remain unattributed. Missing IDs deserialize as `None` and are omitted
from JSON output.
Rust struct literals constructing `Request` or `StoredBlocks` must supply
`session_id: None` when attribution is absent; JSON defaults do not make fields
omittable in Rust source.

Session IDs do not enter block hashes or change admission, scheduling, eviction,
or modeled timing. There is no new policy or configuration switch.

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
