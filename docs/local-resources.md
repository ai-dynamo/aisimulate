<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Local execution resources

`aisimulate predict`, `aisimulate recommend`, and the public
`run_recommendation` Python API check host resources before preparing a run.
These controls describe the machine running AISimulate. Simulated GPU counts,
KV-cache memory fractions, traffic concurrency, and request counts retain their
original meaning.

```yaml
execution:
  resources:
    memory_limit_gib: auto
    cpu_limit: auto
    reserve_memory_gib: 2.0
    reserve_memory_fraction: 0.1
    available_memory_fraction: 0.5
```

Let A be available RAM and R the larger of 2 GiB and 10% of effective physical
RAM. The default memory budget is max(0, min(0.5 A, A - R)). Linux cgroup v1/v2
memory limits, current usage, ancestor limits, affinity and CPU quotas constrain
the host snapshot. Swap does not increase the budget. Automatic CPU selection
leaves one CPU free when more than one whole CPU is available. An explicit
positive `memory_limit_gib` or integer `cpu_limit` must fit the live host limits.
Unavailable resource probes stop execution with an explanation.

The plan reserves coordinator RSS plus 256 MiB and a 512 MiB baseline within
each candidate estimate. Recommendation execution parallelism is the minimum
of the requested parallelism, CPU allowance, and memory slots. Reducing it
preserves the suggestion batch size, trial budget and requested traffic.
One candidate must fit even when parallelism is one. Domain preflight uses the
largest requested load without enumerating its Cartesian product; if that
candidate cannot fit, the whole request is refused. This increment does not
silently prune the search space or return partial recommendations.

```sh
aisimulate recommend --stack dynamo --config sweep.yaml --dry-run \
  --output-dir resource-plan
```

`--dry-run` validates the core schema and writes `resource-plan.json` without
creating a runner or compiling adapters. It is not full backend/adapter
validation. The plan records the host snapshot, reserved memory, requested and
effective parallelism, allocation model, lower bound and estimated peak bytes.
A refused plan or execution exits with status 3 (`resource_limited`). A runtime
resource refusal aborts the sweep instead of observing a fictitious model score.
Use a new output directory or `--overwrite` to replace known outputs.

For a synthetic Dynamo workload with 64,512 concurrent requests, 100 requests
per load unit and 10,240 input tokens, the compatibility estimator calculates
6,451,200 requests and 264,241,152,000 bytes (246.09375 GiB) of eager input-token
vectors alone. A laptop budget rejects this request before request allocation.
This is an allocation calculation, not measured RSS or a diagnosis of an OS
panic. The regression tests use an allocation sentinel; they do not allocate
those vectors.

## Runner estimate contract and limitations

An optional factory method `estimate_host_resources(workload, *, concurrency)`
returns `aisimulate.resources.ResourceEstimate` with `api_version=1`:

- `allocation_model`: versioned implementation name;
- `request_count`: concrete count, or null when unresolved;
- `input_token_bytes` and `lower_bound_bytes`: nonnegative integer bytes;
- `estimated_peak_bytes`: positive integer bytes, or null when unqualified;
- `reason`: explanation of unresolved dimensions.

The built-in native-engine planning model accounts for eager session/hash
metadata, planned output tokens and active prompts. The Dynamo compatibility
model accounts for the older eager u32 input vectors and additional runtime
and output storage. A verified newer adapter can supply its own estimate;
AISimulate does not import Dynamo or assume that installing a core lazy-source
API automatically changes the adapter's allocation behavior.

An estimate is a planning heuristic, not a hard RSS limit. An unavoidable lower
bound can prove a candidate does not fit; it cannot prove that execution fits.
Supported JSON traces receive bounded metadata inspection (at most 16 MiB,
1024 files and 256 KiB per JSON document/record), including scalar token lengths,
hash expansion and cumulative delta/tool turns. Larger or unknown formats need
a runner estimate before admission. Fixed-capacity KV-relative recommendation
domains use a conservative token-capacity bound; other unresolved KV-relative
counts and unrecognized runner models remain unqualified and are refused. The
low-level Runner protocol itself remains an execution primitive.

Preflight checks complement the supervised execution below; neither is a proof
that an operating-system panic cannot occur.

## Supervised execution

The CLI now starts a supervised child before importing stack runtimes. The
public `run_recommendation` Python function uses the same process boundary;
custom factories/providers must support Python's spawn/pickle contract. Keep
script entrypoints behind `if __name__ == "__main__":` as with multiprocessing.
Low-level Runner and Sweeper objects remain explicit execution primitives.

`optimizer.parallelism` defaults to `auto`. Its historical suggestion batch
size stays 16; an existing integer remains both the suggestion batch size and
an upper limit on concurrent evaluations. The host budget can reduce execution
parallelism without changing suggestions, seeds, trial budgets or traffic.
Supervised sweep workers retire after each candidate to release retained runtime
memory, at the cost of repeated worker startup. Timeout recovery terminates,
escalates to kill and joins old workers before replacing the pool. Normal pool
shutdown also has a bounded grace period.

Worker environments set supported OpenMP, BLAS, Rayon, Polars and TensorFlow
thread limits to one before imports. Linux execution also inherits a restricted
CPU affinity set; macOS uses thread settings and reduced scheduling priority.
These are execution controls, not a portable hard CPU quota for arbitrary
third-party runtimes. Initialization is limited to 60 seconds by default via
`execution.resources.initialization_timeout_seconds`. Direct prediction shutdown
has a separate five-second `shutdown_timeout_seconds` limit.

The parent samples its RSS, owned descendant RSS and current host/container
headroom every 50 ms. A budget/headroom breach stops the run and terminates its
owned process group and observed descendants. The direct child is reaped before
returning; timeout replacement separately joins every pool worker. macOS RSS
polling remains best effort and can miss a fast allocation burst, so the
preallocation checks remain active. An estimate or watchdog is not proof that
an OS panic cannot occur.

The CLI writes `resource-runtime.json` with requested limits, the resolved
budget, observed peak RSS, terminal status and cleanup evidence. It also writes
`execution-events.jsonl`, preserving normalized requested configurations,
resource plans, started waves and completed-candidate records. A partial final
line is discarded after interruption. Resource failures exit 3; cancellation
exits 130, and supervisor timeouts exit 124. An interrupted run does not publish
normal completed-run recommendations. Its completed-candidate events are
partial evidence and must be read with the runtime status.

Checkpoints are bounded to 1 MiB per record and 64 MiB per run. CLI configuration
parsing is limited to 1 MiB. These bounds prevent diagnostics and parent-side
configuration handling from becoming a new unbounded allocation path. The SDK
bounds result transfer before parsing, exposes successful runtime evidence as
`result.execution_resources`, and raises `ResourceLimitError` with runtime and
bounded partial-event evidence in `.plan` on a memory refusal. Host evidence is
separate from the portable result JSON and its input fingerprint.
