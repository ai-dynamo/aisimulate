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

Normal package installation includes the required `psutil` and `ijson`
dependencies. Downstream images that install an AISimulate wheel with
`--no-deps` must explicitly install both dependencies before adopting a release
containing these controls.

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

The plan reserves coordinator RSS plus 256 MiB and a baseline within each
candidate estimate: 256 MiB for native Weka replay, 512 MiB for the other
built-in allocation models. Recommendation execution parallelism is the minimum
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
A prediction refused by preflight exits with status 3 before creating its runner and writes
`resource-plan.json`, including the host snapshot, reserved memory, allocation
model, lower bound and estimated peak. Refusals during earlier host discovery or
budget resolution also write the plan, with null for any unavailable host, budget
or workload estimate. Admitted work proceeds automatically;
there is no separate dry-run option.

The refusal message includes the estimated peak (or `unknown`), the unavoidable
lower bound, and the host budget. These are different quantities: a zero lower
bound does not mean the estimate that triggered rejection was zero.

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
termination outcome. Fields that could not be observed because resource discovery
failed are null. Errors loading configuration, applying overrides, or validating
the core schema preserve existing artifacts, including with `--overwrite`.
`execution-events.jsonl` checkpoints
completed candidates and batch decisions so evidence survives a supervisor interruption. A completed
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
not load complete documents or token arrays. There is no fixed file or record
size cutoff. Except for finite `agentic_mooncake` replay, the conservative storage
estimate is checked against live headroom before parsing, since a streaming
parser still holds individual scalar strings. Finite AgentX uses the array
accounting described below and requires runtime supervision.
Unknown allocation models can run one candidate at a time under runtime
supervision when baseline headroom exists; this is explicitly an unqualified
estimate. Known lower bounds still reject impossible workloads before allocation.
The low-level Runner protocol itself remains an execution primitive.

Direct `Sweeper` callers also receive bounded process-pool cleanup. A timeout
or orchestration failure stops and reaps owned workers and their observed
descendants before a replacement pool starts. Successful sweeps allow worker
finalizers a grace period, then stop workers that fail to exit. Cleanup failure
raises an error instead of starting replacement workers. This also applies to
custom factories without host-resource admission; it does not add memory
admission or RSS monitoring to those factories or supervise sequential Runner
execution.

RSS monitoring is best effort, not an operating-system memory sandbox. Allocations
can outpace polling, estimates can be conservative, and other programs can change
available RAM between samples. macOS offers no portable hard RSS cap; preallocation
checks remain necessary. A successful plan or watchdog test does not guarantee bounded peak RSS for
every workload and external adapter.

Finite `agentic_mooncake` replay loads one compact trace. `agentic_lanes` controls
active plays within that trace; it does not create a copy of the trace per lane.
Preflight records a lower bound from the maximum of three allocation requirements:
8 bytes per source hash ID during loading; 4 bytes per compact hash ID plus
4 bytes per planned output token in the prepared replay; and 4 bytes per token
in the largest expanded input prompt. These stages are not added together.
File bytes, lane count, and total logical input tokens do not multiply this bound.

Strings, graph metadata, temporary copies, active requests, caches, and report
storage are outside this lower bound. The full-run peak remains unknown
(`agentic-trace-unqualified-v1`). Such runs require the existing live supervisor
and execute one simulation at a time. All configured agentic lanes and simulated
workers are retained. Admission still checks baseline and host headroom and
rejects a lower bound that cannot fit. The monitor covers metadata inspection
and replay, but allocations can outpace its sampling; this is not an OOM guarantee.
Continuous profiles and other trace formats retain their existing accounting.

### Allocation-model provenance

The byte terms in [resources.py](../../python/aisimulate/src/aisimulate/resources.py)
are versioned admission policies. They do not change simulated operation latency,
energy, or GPU capacity. Their provenance and qualification are:

| Term | Basis | Qualification |
|---|---|---|
| Four bytes per token ID | The native [request protocol](../../crates/core/src/engine/protocol.rs) and [sequence storage](../../crates/core/src/engine/common/sequence.rs) use `Vec<u32>`. The Dynamo compatibility model explicitly assumes eager u32 prompts. | The width is concrete; applying it to an external adapter requires the stated eager-allocation assumption. |
| Engine session/token retention | The engine model reserves prompt, output, and session/hash bookkeeping from the declared request and turn counts. | The 32-byte token multiplier and 4,096-byte per-request term are conservative policy allowances, not measured object sizes. |
| Dynamo peak expansion | Twice the eager prompt bytes, plus 4,096 bytes per request and 16 bytes per output token. | These extra terms are policy headroom; they do not certify an adapter's peak RSS. |
| Trace expansion | Streamed field counts, hash block expansion, and cumulative delta/tool turns; 128 times file bytes, 32 bytes per counted token, and 65,536 bytes per counted request/turn. | Conservative policy allowances. The file-size guard also bounds parser scalar risk before metadata inspection. |
| Native Weka traces | Separate per-play import and replay phases; see below. | Uses the native importer/driver contracts, including real output arrays and synthesized hashes. External Dynamo adapters retain their existing compatibility model unless they supply their own estimator. |
| Process allowances | 256 MiB for native Weka, otherwise 512 MiB per worker; coordinator RSS plus 256 MiB. | Engineering reserves, not calibrated platform-specific measurements. |
| Finite AgentX arrays | Maximum of source `Vec<u64>` hash storage, prepared `Vec<u32>` hash/output storage, and the largest `Vec<u32>` input prompt. No file-size or lane multiplier. | Array widths are concrete. Other allocations are excluded: this is a lower bound, not a peak RSS estimate, and requires supervised serial execution. |
| Admission and recovery | The sum of candidate estimates must fit one live budget; observed pressure stops workers before bounded retry. | Budget/accounting invariants tested with bounded fixtures and real owned subprocesses. |

The resource tests exercise arithmetic boundaries, combined-wave accounting,
allocation sentinels, real process cleanup, and retained candidate outcomes.
These establish control-flow and accounting behavior; they do not qualify the
heuristic multipliers against every workload's measured peak RSS. Hardware
prediction accuracy and workload-specific RSS qualification remain separate.

### Native Weka allocation model

`weka-materialized-v1` streams the source metadata without loading hash or token
arrays. The source's `block_size` is authoritative; a configured block size is
an equality assertion, as in the native importer. Missing source hashes are
counted using `ceil(in / block_size)`, because the importer synthesizes them.
All request `out` lengths remain counted: the driver eagerly plans `u32` output
arrays even for requests that will later exceed the simulated context window.
Play summaries such as `totals`, `tool_tokens`, and `system_tokens` do not
represent additional replayed requests.

The planning peak is the 256 MiB process allowance plus the larger of these phases:

- **Import:** the largest single play's JSON bytes times 32, source or normalized
  hash count (whichever is larger) times 128, and request count times 32 KiB.
  The importer lowers and validates one play at a time, spools its rows to disk,
  and only then builds the complete corpus graph. The parser's byte count may
  include buffer lookahead. Before parsing, four times source-file bytes plus
  the process allowance checks the memory available for a single decoded scalar.
- **Replay:** 32 bytes per normalized hash, 32 bytes per planned output token,
  32 KiB per request, and 16 bytes per concurrently active input token. Weka's
  sequence and cross-stream completion dependencies bound active inputs by the
  maximum weighted overlap of recorded API intervals in each original request
  array; scopes are then summed. Equal starts are treated as concurrent, including
  missing/zero API durations. A possible detached preamble additionally reserves
  the largest main-stream prompt that its later recorded end could hide from
  a completion frontier. Scopes with hashless requests or possible end-order
  reversals within the importer's one-microsecond join tolerance reserve all
  input lengths. These exceptions cover cases where recorded interval overlap
  does not bound native concurrency. The native P/D success path releases its full
  original request before admitting requests unblocked by completion.

Ordinary finite lanes partition the corpus. Snapshot lanes cycle through source
plays, so selected full-play allocations are conservatively multiplied by
`ceil(lanes / plays)`, rather than multiplying the entire corpus by every lane.
Snapshots also retain an immutable corpus graph/rank/output context: 12 bytes
per normalized hash, four bytes per output token, and 2 KiB per request. Warmup
adds 32 KiB of evidence allowance per selected request and per warmup request
(ten per lane); its prompt buffers do not overlap profile buffers across the
quiescent preparation barrier. Continuing agentic profiles still have unknown
total peaks and require supervised serial execution because retained lifecycle
evidence grows beyond the finite corpus.

The four/eight-byte array element widths are concrete; the larger multipliers
and baseline are planning headroom for copies, identifiers, hash maps, KV state,
and reports, not exact object sizes or hard memory limits. The native sources
are [Weka ingestion](../../crates/core/src/replay/loadgen/weka.rs),
[snapshot preparation](../../crates/core/src/replay/loadgen/snapshot.rs), and
[workload execution](../../crates/core/src/replay/loadgen/driver.rs). The model
does not change simulated token lengths, lane counts, snapshots, or timing.
