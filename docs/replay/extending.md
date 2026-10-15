<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Extending Replay

Keep simulation changes in the layer that owns the behavior. Public Python
runner and native placement/scaling extensions are described by the
[adapter ABI](../adapters/README.md); this page covers in-tree changes.

## Ownership and code paths

| Change | Source owner |
| --- | --- |
| Arrival, session or dependency semantics | [`loadgen/`](../../crates/core/src/replay/loadgen/) |
| Event ordering, virtual time and worker lifecycle | [`replayer.rs`](../../crates/core/src/replay/replayer.rs), [`agg.rs`](../../crates/core/src/replay/agg.rs), [`disagg.rs`](../../crates/core/src/replay/disagg.rs) |
| P/D reservations, source holds and transfer protocol | [`handoff.rs`](../../crates/core/src/replay/handoff.rs) and topology runtimes |
| Backend pass selection, allocation, preemption and cache | [`engine/`](../../crates/core/src/engine/) |
| Pass timing and memory estimation | [Performance model](../perf-model/extending.md) |
| Request metrics, aggregate distributions and artifacts | [`report.rs`](../../crates/core/src/replay/report.rs), [`artifact.rs`](../../crates/core/src/replay/artifact.rs), [`canonical.rs`](../../crates/core/src/replay/canonical.rs) |
| Public YAML and runner admission | [`config/`](../../python/aisimulate/src/aisimulate/config/), [`compiler.py`](../../python/aisimulate/src/aisimulate/compiler.py), [`runner.py`](../../python/aisimulate/src/aisimulate/runner.py) |

Do not create a second event loop for a new reporting mode or integration.
Native `Replayer` owns virtual time and both topology runtimes. Scheduler
changes belong in the engine; placement policy should not alter allocation or
pretend pending KV has arrived. A new Python capability must have matching native
validation because direct native callers bypass public configuration.

## Timestamp and lifecycle invariants

Same-time phases are engine completion, worker ready, transfer completion,
admission, telemetry, then scaling. Stable worker/pass/handoff identities break
ties before insertion order. Each topology's internal `step()` returns only
after the semantic timestamp reaches a fixed point and retains all state.
This is an internal seam, not a separate stable public stepping API.

A grouped attention-DP engine publishes completion at the slowest-rank
boundary, including idle ranks. P/D moves request ownership through its
coordinator; a client terminal does not imply that source holds, destination
reservations or transfer work have settled. Completion, cancellation and scale-in
must release each owned resource exactly once. Preserve these distinctions when
adding events, cache tiers or early exits.

The agentic driver applies feedback in immutable graph order: output progress,
causal terminals, then quiescence. Lanes count entire plays including background
children. Failures skip undispatched nodes; already-dispatched siblings become
terminal before client lane release. Server settlement may happen later.
Canonical failure is minimum `(causal_terminal_ms, graph_node_ordinal)`, not
status severity or callback order. HTTP/SSE delivery and client teardown latency
are outside the in-process timing model.

## Reporting and memory

Summary batch replay folds terminal requests after callbacks and spills exact
latency samples beyond 4096 values per distribution to anonymous temporary
files. Quantiles use the detailed report's rounded rank; counts and quantiles
remain exact. Means and standard deviations may differ at floating-point
roundoff due to accumulation order. Existing ITL/throughput sketches retain
their 0.1% relative quantile error. Filesystem failure returns
`ReplayError::ResourceLimited`, not a partial success report.

Detailed output keeps its ordered records without a request cap. Inputs, active
engine state, long agentic lifecycle evidence and capture options still consume
host memory. A generated concurrency source can defer prompt materialization
until admission; it does not bound active state or requested output capture.
Telemetry must remain observational: it neither keeps a deadlocked replay alive
nor advances beyond the configured virtual-time cap. Rust observers and Python
JSON reports have distinct supported fields.

## Validation

Use deterministic fixed timing to isolate scheduler, cache and lifecycle
changes. Cover both aggregated and P/D where supported, asymmetric attention-DP,
coincident events, preemption, cancellation, source/destination cleanup, and
invalid compositions. Test defaults as well as explicit controls so old inputs
retain their behavior. Compare ordinary reports, captured artifacts and repeated
canonical runs; a digest match alone does not validate an excluded metric.

For agentic changes, cover joins, overlapping/background children, source token
identity, snapshots, warm/cold cache, barrier failure, cutoff/grace and late
cleanup after lane reuse. Representative source-built checks include:

```bash
cargo test --locked -p aisimulate-core --test agentx_qualification --example qualify_weka
cargo test --locked -p aisimulate-core --lib agentic_pd_qualification
python/aisimulate/.venv/bin/pytest -q tests/test_unified_traffic_runtime.py tests/e2e/test_unified_cli_engine.py -k 'weka or agentx_replay or agentic_snapshot or agentic_warmup'
```

The optional `scripts/prediction_regression/qualify_agentx_replay.py` checks
revision-pinned published Weka samples through Python and CLI. Its default
scope is aggregated, turn-zero replay; use `--trace <local-weka-path>` for local
input. It is not a substitute for warmup/P/D/G2/profile composition tests or
hardware accuracy measurements. Rebuild the native extension after Rust changes;
on macOS follow the repository's pytest-timeout guidance.
