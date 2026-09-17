<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Local execution resources

`aisimulate predict`, `aisimulate recommend`, and the public
`run_recommendation` Python API check host resources before preparing a run.

**No manual resource configuration is required.** AISimulate automatically
detects the local machine's available RAM and CPU capacity, reserves host
headroom, and limits parallel simulations to fit the estimated resource budget.
You can omit the entire `execution.resources` section. Set explicit limits only
when you want to override the automatic budget.

These controls describe the machine running AISimulate. Simulated GPU counts,
KV-cache memory fractions, traffic concurrency, and request counts retain their
original meaning.

Optional resource settings, with defaults shown:

```yaml
execution:
  resources:
    memory_limit_gb: auto
    cpu_limit: auto
    reserve_memory_gb: 1.0
    reserve_memory_fraction: 0.0
    available_memory_fraction: 0.9
    initialization_timeout_seconds: 60.0
    shutdown_timeout_seconds: 5.0
```

Memory settings use decimal GB: 1 GB is 1,000,000,000 bytes. Let A be available
RAM. The default host reserve is 1 GB, with no additional percentage-of-total-RAM
reserve. The default memory budget is max(0, min(0.9 A, A - 1 GB)); the separate
90%-of-available-RAM cap can leave more than 1 GB unused. If configured,
`reserve_memory_fraction` raises the reserve to the larger of `reserve_memory_gb`
and that fraction of effective physical RAM.

Linux cgroup v1/v2 memory limits, current usage, ancestor limits, affinity and CPU
quotas constrain the host snapshot. Swap does not increase the budget. Automatic
CPU selection leaves one CPU free when more than one whole CPU is available.
CPU capacity caps the number of parallel simulations; it does not estimate CPU
speed or simulation duration. An explicit positive `memory_limit_gb` or integer
`cpu_limit` must fit the live host limits. Unavailable resource probes stop
execution with an explanation.

The plan reserves coordinator RSS plus 256 MiB and a 512 MiB baseline within
each candidate estimate. Recommendation execution parallelism is the minimum
of the requested parallelism, CPU allowance, and memory slots. Reducing it
preserves the suggestion batch size, trial budget and requested traffic.
Admission checks each concrete candidate and the combined memory estimate of
its batch against fresh host headroom before starting workers. A large load in
the search domain does not block smaller candidates. If a batch cannot fit,
AISimulate reduces its parallelism; a candidate that cannot fit alone is recorded
as `resource_limited`, with no simulated metrics or score. It consumes a trial
and completes it as an optimizer rejection without a fabricated measurement.
Completed candidates remain available. Resource-limited candidates are counted
separately from `evaluated`; the selected recommendations cover only completed
evaluations, not every requested candidate.

Resource checks run automatically on every `predict` and `recommend` command.
A refused prediction exits with status 3 before creating its runner and writes
`resource-plan.json`, including the host snapshot, reserved memory, allocation
model, lower bound and estimated peak. Admitted work proceeds automatically;
there is no separate dry-run option.

Execution runs inside an owned subprocess tree. The supervisor fixes the memory
budget at startup, then samples total owned RSS and live host headroom every
50 ms, including initialization and output serialization. Runtime thread pools
are limited to one thread per worker. During recommendation evaluation, memory
pressure triggers worker cleanup and retries unfinished candidates at lower
parallelism, at most twice. Completed candidates are retained. Workers must be
reaped before another batch can start. Persistent pressure produces an explicit
`resource_limited` candidate and the sweep continues with other candidates.
A sudden jump past the supervisor limit stops the entire execution tree.

`resource-runtime.json` records observed peak RSS, effective budgets and the
termination outcome. `execution-events.jsonl` checkpoints completed candidates
and batch decisions so evidence survives a supervisor interruption. A completed
sweep writes `recommendation.json` and selected prediction files as usual, and
exits with status 3 if any candidates were resource-limited. If the entire tree
is stopped, the event log contains the completed subset; it is not a finalized
recommendation. The Python API returns partial sweep results for individual
candidate refusals, or raises `ResourceLimitError` with bounded partial events
when the supervisor stops the whole tree. Host evidence is available through
`result.execution_resources` and excluded from the portable result fingerprint.
Use a new output directory or `--overwrite` to replace known outputs.

The public Python recommendation API uses spawned processes. Factories and
providers must be pickleable, and script calls belong inside an
`if __name__ == "__main__":` guard. Initialization and shutdown deadlines are
configurable above; the optimizer's candidate timeout remains independent.

For a synthetic Dynamo workload with 64,512 concurrent requests, 100 requests
per load unit and 10,240 input tokens, the compatibility estimator calculates
6,451,200 requests and 264,241,152,000 bytes (about 264.24 GB) of eager input-token
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
Supported JSON and JSONL traces are inspected as a stream, including scalar
token lengths, hash expansion and cumulative delta/tool turns. Inspection does
not load complete documents or token arrays and has no total-file or record-size
cutoff. Unknown allocation models can run one candidate at a time under runtime
supervision when baseline headroom exists; this is explicitly an unqualified
estimate. Known lower bounds still reject impossible workloads before allocation.
The low-level Runner protocol itself remains an execution primitive.

RSS monitoring is best effort, not an operating-system memory sandbox. Allocations
can outpace polling, estimates can be conservative, and other programs can change
available RAM between samples. macOS offers no portable hard RSS cap; preallocation
checks remain necessary. A successful plan or watchdog test does not prove that
the original OS panic is resolved.
