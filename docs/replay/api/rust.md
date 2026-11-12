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
`Arc<dyn TimingModel>` providers. Schedulers price each prefill batch with
`TimingModel::predict_prefill_totals_ms(batch_size, new_tokens, prefix_tokens)`:
the tokens the batch computes and the cached or earlier-chunk KV its requests
read. The default reduces these totals to the integer per-request means of
`predict_prefill_ms`; providers indexed by iteration totals override it.
A vLLM or TensorRT-LLM pass that computes prefill and decode steps together
is priced with `TimingModel::predict_mixed_pass_ms(prefill, batch_size, ...)`,
which takes the pass's `PrefillTotals` and otherwise the arguments of
`predict_decode_ms`. A provider that prices such a pass as one forward returns
`MixedPassMs`: the prefill latency and the latency the decode steps add, which
sum to its estimate of the pass. The default `None` prices the prefill with
`predict_prefill_totals_ms` and a separate decode pass with
`predict_decode_ms`. The built-in `ais` provider returns the pass only for
whole-model FPM interpolation, corrected as one mixed iteration when the model
is tuned. Any other timing it serves, including operation-level timing,
measured SD-round profiles and MTP drafts, returns `None`.
`TimingModel::predict_iteration_ms(&[RankWork])` comes first: it prices one
attention-DP group pass from every rank's `RankWork`, in DP-rank order with
idle ranks included, and returns `IterationMs`, the shared duration every rank
takes from its compute start under the prefill speedup. Schedulers ask only
while `decode_speedup_ratio` is 1 and some rank computes model work; otherwise,
and when the provider returns the default `None`, each rank prices its own
work with the phase methods above. The `ais` provider answers through the
canonical multi-rank `ForwardPassMetrics` query, so the phase methods serve
only the passes it declines; see
[engine replay timing](../../perf-model/configuration.md#engine-replay-timing).
`TimingModel::predict_vision_ms(&[VisionShape])`
prices one vision-encoder batch; the default returns `None`, and a rank with
image work then rejects it rather than treating the encoder as free. The `ais`
provider answers from the `EngineSpec` `vision` section compiled under
`encoder_parallel`.

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

`with_detokenization(workers, base_ms, per_output_ms)` detokenizes each pass's
outputs before the client receives them. Each worker that emits tokens has
`workers` servers, shared by its attention-DP ranks, and each request belongs
to one of them. A server takes the outputs of a pass that belong to it as one
job of `base_ms + per_output_ms × outputs`, after its earlier jobs. Token
times, latency metrics and request completion use the client's receive time.
Both costs zero, the default, keep every output at its pass end. Scaling and
agentic profiles are rejected. See
[response detokenization](../engine/README.md#detokenization).

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
occupancy is available here. Telemetry neither extends the time cap nor prevents deadlock.

`with_telemetry_capture(interval_ms)` retains native telemetry in
`ReplayReport.telemetry`, including preparation and profiling requests, periodic
samples, and agentic lifecycle events. It enables internal request capture while
preserving the caller's original `record_per_request` choice for
`ReplayReport.per_request`. Capture uses absolute virtual milliseconds and does
not move profiling/scaling boundaries. The capture and observer APIs are mutually
exclusive. See the [telemetry contract](../../getting-started/understand-results.md#replay-telemetry).
Downstream `ReplayReport` struct literals must include `telemetry: None` when
capture is absent, or preserve the field when rebuilding a captured report.

## Errors and stability

`ReplayResult<T>` is `Result<T, ReplayError>`. Errors distinguish `InvalidSpec`,
`Engine`, `Placement`, `Scaling`, `Telemetry`, `WorkloadSource`, `Deadlock`,
`ResourceLimited`, and `Invariant`. Handle typed variants instead of parsing display messages.
`ResourceLimited` can represent exact-latency spill storage failure; it does
not produce a successful partial native report.

The authoritative exports are in
[`replay/mod.rs`](../../../crates/core/src/replay/mod.rs). Several cross-crate
integration seams are `pub` but marked `#[doc(hidden)]` (runtime input,
request payloads, engine observations and internal collectors). They are not
promoted here as stable end-user APIs.

`ReplayWorkloadSource` and `Replayer::run_with_source(source)` are one such
seam. They let a reactive workload owned by another crate drive Replay through
the same admission queue as `WorkloadDriver`, on aggregated and disaggregated
topologies. Replay owns the clock: the next instant is the earliest of the next
engine event and the source's `next_wakeup_ms`, a request to be stepped then
that may produce nothing, so empty steps are legal. A step is one `pop_ready`
call, and Replay stamps each returned request's arrival as that instant. Each
instant `t` runs four phases:

1. **Wakeup step**, only if `t` is the source's wakeup or feedback awaits it:
   the source is stepped once, before engine work at `t` settles. Its
   submissions are scheduled arrivals, admitted exactly as native arrivals at
   `t`: Replay holds them and releases them where it releases native arrivals.
2. **Settle**: engine work due at `t` settles (pass ends, output tokens,
   terminals), delivered through `on_output_token` (with its instant),
   `on_causal_terminal` and `on_quiescent`.
3. **Reaction fixpoint**: the source is stepped again while a step left new
   feedback at `t`. These submissions are reactions.
4. **Pass starts**: idle workers start passes one at a time, each start taking
   the lowest-numbered ready worker (so ascending worker order); a worker's
   attention-DP ranks start together as its one pass, scheduled in ascending
   rank order. Native replay starts passes in this order, and so does a run
   with a source. Feedback from a pass start, such as an admission rejection,
   is delivered and the source stepped before the next worker starts, so a
   reaction joins every pass not yet started and none that has. On
   disaggregated topologies prefill workers start before decode workers.

An instant whose only activity is an empty wakeup step leaves engine state
untouched, so idle wakeups never change engine timing. Worker removal,
telemetry and scaling never run at an instant while delivered feedback still
awaits a step. Feedback that only emerges in a later drain pass at that instant
(a prefill rejection, say) can still bring a reaction after a telemetry or
scaling tick already ran at it, exactly where a native arrival known only at
that point would land; scheduled arrivals keep native tick ordering exactly.

Progress rules: after every step at `t` the next wakeup is `None` or finite and
strictly later than `t` (before the first step it must be finite and no earlier
than the first instant); a source not drained, with nothing in flight and no
wakeup, has stalled. Both fail the run with `WorkloadSource`, as do an error the
source returns, a reused request `uuid` and a rank out of range (ranks are
validated like spec ranks when the source submits the request). Steps at one
instant are bounded by `Replayer::with_max_source_steps_per_instant` (default
1,000,000); exceeding it, which only a reaction loop such as a rejected request
resubmitted at once can, fails with `ResourceLimited` naming the budget and the
instant. The run takes ownership of the source, which need not be `Send`. The
source owns its concurrency and duration, so `run_with_source` refuses
spec-authored requests or a second runtime input, `max_in_flight`,
`max_sim_time_ms`, Belady eviction, detokenization latency and
`ReplayDeterminism::CanonicalV1` with `InvalidSpec`.
Cancelling requests from the source is not part of this seam.

In-tree workload import and snapshot
APIs live under `replay::loadgen`; see [workloads](../workloads.md#agentic-load-controls) and
[agentic warmup](../agentic/warmup.md#seeded-request-boundary-snapshots) for their semantics. To build snapshots directly, call
`ValidatedAgenticGraph::prepare_snapshots(lanes, options)` with a positive lane
count and `AgenticSnapshotOptions`. It returns `Result<PreparedAgenticSnapshots>`
and fails for zero lanes or an empty corpus. Pass the prepared snapshots to
`WorkloadDriver::new_agentic_snapshots`.
