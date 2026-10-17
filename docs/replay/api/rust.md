<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Rust Replay API

`aisimulate_core::replay` exposes native deterministic token replay. The native
`ReplaySpec` describes topology, engine/provider descriptors, requests and
report controls; it differs from Python Sweeper's `ReplaySpec`. The crate root
also re-exports common replay types.

## Construction and execution

The public entrypoints in
[`replayer.rs`](../../../crates/core/src/replay/replayer.rs) are:

```rust
Replayer::new(spec: ReplaySpec, factory: ReplayEngineFactory)
    -> ReplayResult<Replayer<RoundRobinComposition>>
Replayer::with_composition(spec: ReplaySpec, factory: ReplayEngineFactory, composition: C)
    -> ReplayResult<Replayer<C>>
Replayer::run(self) -> ReplayResult<ReplayReport>
```

These are signatures for reference. `C: ReplayComposition` supplies placement
and optional scaling; `RoundRobinComposition` is the built-in fixed-worker
composition. Construction validates the spec and composition. Running consumes
the Replayer so ownership, virtual time and caches remain within one execution.
`ReplayEngineFactory::default()` uses engine timing from the descriptor;
`with_timing_model` or `with_optional_role_timing_models` supplies explicit
`Arc<dyn TimingModel>` providers.

`run_engine_replay`, `run_engine_replay_with_timing`, and
`run_engine_replay_with_optional_role_timing` are exported convenience functions;
check their exact signatures in [engine construction](../../../crates/core/src/replay/engine.rs).
The [composition ABI](../../adapters/native-composition.md) describes policy
lifecycle and time units.

## Serializable spec

`CURRENT_REPLAY_SPEC_VERSION` is `1`. The
[spec module](../../../crates/core/src/replay/spec.rs) owns these public fields:

| Field | Contract |
| --- | --- |
| `version` | Exact supported integer version; omitted JSON defaults to current version |
| `topology` | `ReplayTopology::Aggregated` or `Disaggregated`; each pool starts with at least one worker |
| `engine` | Serializable native engine descriptor, not a Python backend deployment |
| `adapters` | Provider descriptors resolved by a composition; data only |
| `requests` | Unique IDs, validated arrival/input/output data and optional routing/token identity |
| `max_in_flight` | Positive source-side cap, or absent for timestamp admission |
| `max_sim_time_ms` | Optional soft virtual-time cutoff; events at the cutoff are processed |
| `record_per_request` | Defaults true for version-1 compatibility; controls retained request records |
| `sla` | Validated latency thresholds for goodput |

A request's optional materialized prompt tokens provide cache identity; an
adapter must not invent token content for KV-aware replay. `dp_rank` selects
aggregated/decode rank; `prefill_dp_rank` is P/D-only. `WorkerPoolSpec` defines
initial worker count and startup delay. Millisecond clocks, bandwidth-derived
handoff times and callback timestamps use simulated time, never host wall time.

A request may author `kv_hints: {"source_worker": W, "source_dp_rank": R}`, the
worker and attention-DP rank that hold a prefix of its prompt, as when a client
attaches a `kv.fetch` hint. It is omitted when absent. When replay starts, an
`R` outside the `dp_size` of the pool that places the request (aggregated, or
prefill in P/D) fails with `ReplayError::InvalidSpec`. The built-in round-robin
placement ignores a hint whose source worker is unknown, not yet ready, draining
or removed, and the P/D decode pool ignores every hint. Only a pool that
configures `kvcr` acts on hints ([KV cache](../engine/kv-cache.md#kvcr-peer-fetch));
elsewhere they change nothing. Engines receive the resolved source as
`engine::Request::kv_hint` (`KvHintSource { worker_id, dp_rank }`), which never
names the receiving rank.

`ReplayEngineConfig` has the aggregated rank/DP/TP/PCP descriptor and optional
prefill/decode `ReplayRoleConfig` objects. Engine and topology validators enforce
feature combinations beyond the generic spec validator. Grouped-cache, G2/G3,
Belady and AgentX restrictions are listed in [features](../features.md).

`ReplayEngineConfig` and `ReplayRoleConfig` carry
`prefill_context_parallel_size` for GPU accounting. It defaults to 1 when
omitted from JSON; Rust struct literals should set it or use `..Default::default()`.
It multiplies DP × TP for resource reports without creating additional attention-DP
schedulers. Decode context parallelism does not add physical GPUs.

The pinned Dynamo CI adapter is updated during its build to set this field to
1, matching its existing topology. That adapter does not expose PCP controls;
its performance comparison does not qualify nondefault PCP configurations.

## Reports and capture

`ReplayReport` is the native report with request counts, latency/throughput,
SLA goodput, optional per-request admission/routing records, cache and power
evidence. `ReplayTerminalStatus` is also exported as `RequestTerminalStatus`.
Native power diagnostics do not certify downstream Python adapter passthrough.

`with_tokenization(workers, base_ms, per_input_token_ms)` tokenizes each
request on the worker it was placed on before that worker's engine sees it.
Each worker of the entry role has `workers` first-come, first-served servers,
and a request takes `base_ms + per_input_token_ms × input tokens` on one of
them. Latency metrics keep the arrival time, and each per-request record gains
`tokenized_ms`. Both costs zero, the default, keep direct dispatch. Scaling is
rejected. See
[request tokenization](../engine/README.md#tokenization).

`with_capture_options(ReplayCaptureOptions)` controls capture and canonical
determinism. Canonical comparison uses the exported schema version, seed and
exclusion definitions rather than arbitrary JSON key order. The default
round-robin Replayer's `run_with_artifacts` supports fixed single-worker
aggregated replay with attention DP1 only; a single logical worker with DP2
is still rejected. Multiworker/P/D/scaling consumers use normal reports
and neutral observation contracts.

`with_telemetry_observer(interval_ms, observer)` requires a positive finite
interval and returns `ReplayResult<Self>`. The observer receives a baseline,
settled interval samples, and a final tail when needed. A sample coinciding
with a scaling tick observes settled pre-scaling state. Grouped cache byte
occupancy is available here even though the Python JSON runner does not expose
this observer API. Telemetry neither extends the time cap nor prevents deadlock.

## Errors and stability

`ReplayResult<T>` is `Result<T, ReplayError>`. Errors distinguish `InvalidSpec`,
`Engine`, `Placement`, `Scaling`, `Telemetry`, `Deadlock`, `ResourceLimited`, and
`Invariant`. Handle typed variants instead of parsing display messages.
`ResourceLimited` can represent exact-latency spill storage failure; it does
not produce a successful partial native report.

The authoritative exports are in
[`replay/mod.rs`](../../../crates/core/src/replay/mod.rs). Several cross-crate
integration seams are `pub` but marked `#[doc(hidden)]` (runtime input,
request payloads, engine observations and internal collectors). They are not
promoted here as stable end-user APIs. In-tree workload import and snapshot
APIs live under `replay::loadgen`; see [workloads](../workloads.md#agentic-load-controls) and
[agentic warmup](../agentic/warmup.md#seeded-request-boundary-snapshots) for their semantics. To build snapshots directly, call
`ValidatedAgenticGraph::prepare_snapshots(lanes, options)` with a positive lane
count and `AgenticSnapshotOptions`. It returns `Result<PreparedAgenticSnapshots>`
and fails for zero lanes or an empty corpus. Pass the prepared snapshots to
`WorkloadDriver::new_agentic_snapshots`.
