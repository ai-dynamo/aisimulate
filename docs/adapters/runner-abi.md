<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Runner ABI

Source contracts:
[`sweeper/replay.py`](../../python/aisimulate/src/aisimulate/sweeper/replay.py),
[stack discovery](../../python/aisimulate/src/aisimulate/stack.py), and the
[built-in runner](../../python/aisimulate/src/aisimulate/runner.py).

## Required interface

```python
class RunnerFactory(Protocol):
    def capabilities(self) -> RunnerCapabilities: ...
    def create(self, worker_id: int) -> Runner: ...

class Runner(Protocol):
    def run(
        self,
        spec: ReplaySpec,
        *,
        output_requirements: ReplayOutputRequirements | None = None,
    ) -> ReplayReport: ...
    def close(self) -> None: ...
```

A factory must be serializable for spawned workers. Each worker creates one
runner and reuses it across candidate replays; worker-local state must not leak
one candidate's requests or caches into another. `close()` releases resources.
The engine factory lazily imports the compiled runtime in its worker.

Optional factories are selected by entry-point name in
`aisimulate.runner_factories` (`--stack`). Resolution validates `capabilities`
and `create`; unavailable, duplicate or unloadable optional stacks raise
`StackNotFoundError`, `DuplicateStackError` or `StackLoadError`. A factory's
coarse capabilities are checked before runner creation; execution must still
validate finer feature combinations.

## Replay data contract

`REPLAY_SPEC_API_VERSION` is `1`. `ReplaySpec` contains:

| Field | Meaning |
| --- | --- |
| `backend_deployment` | Concrete backend/version, topology, role engine arguments, worker counts, parallel shape and exact estimator identities |
| `workload` | Concrete source/load/stop fields; strict data crossing the process boundary |
| `goal` | Evaluation/optimization requirements including optional SLA |
| `concurrency` | Closed-loop source cap or `None` for open-loop arrivals |
| `adapters` | Adapter-owned `AdapterReplaySpec` values |
| `api_version` | Must equal the runner's replay-spec version |
| `execution_mode` | `offline` by default; `online` only when advertised |

`runtime_hooks` flattens adapter hook tuples in adapter insertion order.
Each `RuntimeHookSpec` declares `(provider, kind, api_version, config)`.
Compatibility matches all three identity fields, with exact integer version
checks. Config/payload values must be strict JSON: finite numbers, strings,
booleans, null, lists and string-keyed dicts. `canonical_json` provides
stable ordering for serialization/cache keys; it does not license arbitrary
Python objects inside adapter payloads.

## Capabilities

`RunnerCapabilities` declares backend/topology pairs, execution modes, trace
formats and hook identities. Pair/trace declarations may use `"*"` where the
contract allows it. More specific flags cover disaggregated attention-DP,
AgentX backends/topologies/lanes/snapshots/warmup/profiles, offload/speculation,
analytical EPD, cached-prefix tokens, state/grouped cache and model controls.
Generic flag defaults are not qualification evidence for an optional stack.

`require_compatible(spec)` checks versions and coarse combinations before
execution. AgentX G2 additionally requires offline vLLM, a static single
aggregated worker or 1P1D, and DP1 on every role. G3 and agentic speculation
remain rejected. The [feature matrix](../replay/features.md) records the built-in
stack's actual boundary, including limitations stricter than a capability flag.

If every backend/topology pair fails Sweeper preflight, `RunnerIncompatibleError`
(subclass of `NoViableParallelConfig`) is raised before trials, with no
serialized `SweepResult`. Mixed incompatibility and model/memory/data failures
retain `NoViableParallelConfig` with diagnostics. The CLI reports configuration
errors as exit status 2. Candidate-specific build, replay, budget and timeout
failures are recorded as infeasible trials by Sweeper.

## Output and resource lifecycle

`ReplayReport` contains `metrics` and JSON `metadata`. Non-power metrics must
be finite numeric values; null is allowed only for designated power fields.
Report the normalized metric semantics used by scoring, and retain provenance
and limitations rather than substituting zero for unavailable measurements.

`ReplayOutputRequirements` defaults to summary output and can request raw,
per-request, telemetry, memory or timing diagnostics. A runner must reject
unsupported capture combinations. The Engine JSON runner rejects telemetry
capture even though native Rust observers support it.

An optional `estimate_host_resources(workload, *, concurrency=None)` factory
method participates in [resource admission](../reference/local-resources.md).
It must return `ResourceEstimate` API version 1 with nonnegative byte counts,
a positive known request count, and an optional positive peak estimate at least
as large as its lower bound. Unknown peak is not zero memory: guarded execution
admits at most one continuously supervised candidate when budgets permit.
Low-level `run()` callers own their supervision; the runner ABI itself does not
spawn a resource monitor or publish CLI artifacts.
