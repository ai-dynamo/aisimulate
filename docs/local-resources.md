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
    memory_limit_gib: auto
    cpu_limit: auto
    reserve_memory_gib: 2.0
    reserve_memory_fraction: 0.1
    available_memory_fraction: 0.9
```

Let A be available RAM and R the larger of 2 GiB and 10% of effective physical
RAM. The default memory budget is max(0, min(0.9 A, A - R)). Linux cgroup v1/v2
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

Resource checks run automatically on every `predict` and `recommend` command.
If the workload cannot fit, AISimulate stops before creating a runner or
compiling adapters and exits with status 3 (`resource_limited`). It writes the
refused plan to `resource-plan.json` in the output directory, including the host
snapshot, reserved memory, requested and effective parallelism, allocation model,
lower bound and estimated peak bytes. Use a new output directory or `--overwrite`
to replace known outputs. An admitted workload proceeds to simulation. A runtime
resource refusal also exits with status 3 and aborts the sweep instead of
observing a fictitious model score.

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

This preflight increment does not enforce native thread counts, supervise RSS,
or contain allocations during adapter initialization and report serialization.
The companion worker-supervision change adds runtime protection. macOS offers
no portable hard RSS cap; preallocation checks remain necessary even with a
watchdog. Do not describe a successful resource plan as proof that the original
OS panic is resolved.
