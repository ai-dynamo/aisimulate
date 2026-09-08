---
name: aisimulate-replay-parity
description: Runs deterministic byte-parity and golden-point qualification campaigns for AI Simulate engine-only offline replay across round-robin vLLM, SGLang, and TRT-LLM configurations, including forced preemption, retraction, handoff, and liveness coverage. Use when validating replay refactors, engine scheduler or event changes, or topology changes against a pinned baseline revision; this skill does not execute performance gates, and the Dynamo KV replay skill covers Router, Planner, KVBM, and other Dynamo-owned composition behavior.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - aisimulate
    - offline-replay
    - engine
    - round-robin
    - parity
    - regression
---

# AI Simulate replay parity

Compare two revisions of AI Simulate engine-only offline replay using deterministic
virtual-time reports. Use the existing replay and test harnesses; extend them only when a
required semantic signal is unavailable. This version of the skill does not claim an
executable performance gate because current main lacks the required loop-only timing
boundary.

This campaign requires the built-in `RoundRobinComposition`: synchronous round-robin
placement and no scaling. It intentionally excludes injected KV Router, Planner, scaling,
KVBM, and other Dynamo-owned composition behavior. It also does not require a
same-timestamp event-only progress scenario. Validate that concern with a focused test
when a change specifically affects replay liveness or same-time settlement.

Before changing or qualifying replay behavior, read
`crates/core/src/replay/README.md` and `crates/core/src/replay/CLAUDE.md`.

## Required inputs

Resolve and record before running anything:

- immutable baseline and candidate commit SHAs;
- Rust toolchain, build profile, flags, and host characteristics;
- each artifact's exact Cargo features from the relevant manifests;
- trace or workload path, upstream repository and commit/blob when external, full-file
  SHA-256 checksum, and deterministic slice rule and checksum;
- engine, topology, concurrency, worker counts, DP/TP sizes, block sizes, and capacities;
- the canonical-report exclusion allowlist.

Do not compare against moving branches or reuse an artifact after changing its checkout.
Do not describe a configuration as "all features"; record explicit feature names. The
current `aisimulate-core/replay-bench` feature is empty and is not required for
`ReplayDeterminism::CanonicalV1`; do not assign it Dynamo's seeded-router meaning.

This skill ships a standalone Rust qualification runner, frozen engine-only configs, and
qualified 5,000-row golden points for a pinned main revision. Read
[engine-only golden-point seeds](references/engine-only-golden-points.md) before choosing
fixtures, running the runner, or claiming long-corpus coverage.

## Stage 1: Pin revisions and artifacts

1. Confirm the baseline is an ancestor or otherwise document the comparison relationship.
2. Create isolated checkouts for both revisions using the same host and Rust toolchain.
3. Apply any temporary determinism correction identically to both revisions. Keep it out
   of the measured semantic delta and record its patch checksum.
4. Use `scripts/build_runner.py` to copy the exact skill runner source and lockfile into
   separate build directories and point them at each checkout's `crates/core`. Record the
   identical runner-source checksum plus each checkout revision and binary checksum. The
   builder embeds the checkout revision into its binary; never accept a revision supplied
   by an editable replay config.
5. The runner accepts a frozen `ReplaySpec`, enables `ReplayDeterminism::CanonicalV1`, and
   emits `CanonicalReplayRecord` plus qualification counters. If it lacks a required
   correctness or lifecycle signal, add the narrowest reusable support before starting the
   campaign. Do not reconstruct canonical reports in shell.
6. Build one release artifact per revision with identical explicit features. Reuse that
   artifact across correctness rows unless a backend genuinely requires a distinct
   artifact; document any exception.
7. Copy artifacts to a temporary campaign directory. Record exact features, artifact
   SHA-256, artifact size, and `.text` size when the artifacts are comparable.
8. Return each checkout to its original branch after extracting immutable artifacts.

Never build while collecting replay samples.

## Campaign concurrency

Treat one engine/topology/frozen-configuration comparison row as the unit of node
placement. When scheduler capacity permits, allocate multiple nodes and assign independent
rows to them. The nodes do not need to be homogeneous because no individual comparison
crosses nodes. Keep the baseline, candidate, all determinism repetitions, and lifecycle
evidence for one row on the same node. Use the same prebuilt revision artifacts and inputs
throughout the campaign, and record node characteristics and CPU placement for every row.

Engine-only replay is normally single-core. Pin each process to one physical core after
confirming that assumption during preflight. If thread inspection shows additional runnable
workers, pin the process to a fixed CPU set large enough for them and keep the set size,
NUMA placement, and affinity identical between baseline and candidate.

For correctness, run baseline and candidate repetitions concurrently when the node has
sufficient resources. Pin concurrent processes to disjoint CPU sets, give them separate
output paths, and ensure they share no mutable state. Keep every repetition in a separate
process even when several repetitions run at the same time.

Multi-node and within-node correctness parallelism are campaign throughput optimizations;
they must not change workload concurrency.

## Stage 2: Establish deterministic reports

Require the harness to control every known entropy source:

- enable `ReplayDeterminism::CanonicalV1`;
- assign stable request UUIDs and preserve authored request correlation;
- use stable request ordinals and same-time event sequence numbers;
- keep the frozen request, session, and trajectory order;
- recursively sort JSON object keys;
- sort only explicitly unordered collections such as per-request records;
- preserve semantically ordered event and lifecycle arrays; and
- use exactly `CANONICAL_RESULT_EXCLUSIONS` from the pinned revision.

On current main that closed exclusion list is:

- `/summary/wall_time_ms`;
- `/summary/processed_tokens_per_s`;
- `/summary/processed_output_tokens_per_s`; and
- `/planner/html_report_path`.

The shared canonical schema retains the planner path for Dynamo compatibility. For the
built-in engine-only composition, emit `planner: null` and metadata that identifies
round-robin placement, no scaling, backend, topology, worker counts, DP/TP sizes, engine
configuration, input checksum, and slice rule. Do not claim that the planner exclusion
means a Planner was exercised.

This exclusion list is closed: include every other field, and stop for review before
adding another exclusion. Make semantic fields deterministic rather than dropping them.

Run baseline twice and candidate twice in separate processes. Each revision must produce
one unique canonical digest. This is an entropy-leak check, not a statistical trial. If a
revision is internally unstable, stop and diagnose it; do not increase repetitions and
average the outputs.

## Stage 3: Qualify a long interaction-heavy corpus

Prefer a fixed contiguous 5,000-request Mooncake window over many parity repetitions.
Preserve arrival order, session structure, and prefix locality. Record the starting offset,
request count, trace format, block size, and checksum. Do not randomly sample rows,
duplicate a shorter trace, or silently claim a 5,000-request campaign when fewer usable
requests exist.

The committed two-to-sixteen-row Mooncake fixtures listed in
[engine-only golden-point seeds](references/engine-only-golden-points.md) are quick
preflight inputs, not the authoritative long-corpus campaign.

Qualify and freeze one configuration per comparison row, or per explicitly named
configuration family when rows genuinely share every relevant parameter. For each frozen
configuration, prove every applicable path was exercised:

- round-robin distribution across all configured logical workers;
- immediate admission and engine-internal waiting under scheduler pressure;
- a small, bounded number of vLLM preemptions or SGLang retractions at the
  block-capacity edge;
- disaggregated prefill/decode handoff;
- prefix reuse and terminal cleanup;
- attention-DP rank/barrier behavior when selected; and
- trace-timestamp, concurrency, multi-turn, or agentic admission semantics selected by the
  input.

Use pressure evidence, per-request records, detailed artifacts, or lifecycle traces rather
than inferring paths from successful completion. Target one to three preemptions or
retractions per applicable configuration. Zero means the edge was not exercised; repeated
preempt/re-admit cycling, rapidly growing pressure counts, an effect-free zero-duration
pass, or failure to advance virtual time invalidates the fixture. Tune capacity or
concurrency minimally and identically for baseline and candidate within the row or family.
Back off rather than accepting a pressure flood. Never tune revisions separately.

### Start from engine-only seeds

Read [engine-only golden-point seeds](references/engine-only-golden-points.md) before
searching for capacity edges. Use the committed fixed-timing seeds to verify harness
wiring, determinism, worker identity, and handoff evidence. They are not substitutes for
qualifying internal-polynomial long-corpus configurations on the pinned baseline.

Do not import Dynamo's KV-aware capacities or expected counters as AI Simulate golden
points. Round-robin placement changes worker assignment, queue pressure, reuse, and
preemption/retraction behavior. Matching lifecycle counts do not waive an unstable
canonical digest; stop correctness and performance work until internal determinism and
cross-revision parity are established.

## Stage 4: Run byte parity

Run the frozen long corpus for this authoritative engine-only matrix:

| Engine semantics | Topology | Memory path | Placement | Scaling |
| --- | --- | --- | --- | --- |
| vLLM | Aggregated | Native engine KV | Round-robin | None |
| vLLM | Disaggregated, role DP=1 | Native engine KV | Round-robin per pool | None |
| vLLM | Disaggregated, prefill DP=2 / decode DP=4 | Native engine KV | Round-robin per pool | None |
| SGLang | Aggregated | Native engine KV | Round-robin | None |
| SGLang | Disaggregated, role DP=1 | Native engine KV | Round-robin per pool | None |
| SGLang | Disaggregated, prefill DP=2 / decode DP=4 | Native engine KV | Round-robin per pool | None |
| TRT-LLM guaranteed-no-evict | Aggregated | Native engine KV | Round-robin | None |
| TRT-LLM guaranteed-no-evict | Disaggregated, prefill DP=2 / decode DP=4 | Native engine KV | Round-robin per pool | None |

Use multiple logical workers in the aggregated and disaggregated rows so round-robin
distribution is observable. Add focused aggregated attention-DP rows whenever a change
touches grouped ranks, barriers, DP placement, per-rank FPM, or completion visibility.
When a pinned baseline predates disaggregated attention-DP or TRT-LLM disaggregated support,
record the new row as an intentional semantic exception and qualify it deterministically on the
candidate. Do not silently reduce its DP sizes or substitute an aggregated row.

For each row:

1. Produce canonical baseline and candidate outputs with the frozen configuration.
2. Compare their bytes or SHA-256 digests exactly.
3. Verify pressure, worker, handoff, reuse, and terminal evidence independently of the
   digest.
4. Delete matching full reports and retain their digests.
5. Preserve full reports and a focused diff only when outputs disagree.

## Stage 5: Classify semantic differences

Byte mismatch is a review gate, not an instruction to preserve incorrect behavior. Allow
an intentional mismatch only when the candidate is demonstrably more faithful to the
specified engine, scheduler, replay, topology, or handoff semantics.

For every proposed exception, record:

- the exact fields, requests, or lifecycle events that differ;
- the baseline behavior and why it is incorrect or less faithful;
- the candidate behavior and the semantic source of truth supporting it;
- why the difference is caused by the intended change rather than leaked entropy;
- a focused regression test that fails on the old behavior and passes on the correction;
- any downstream report or API compatibility impact; and
- the reviewer-visible disposition.

Use `PASS_WITH_SEMANTIC_EXCEPTIONS` only when every byte difference is covered by such a
record. Unexplained, incidental, or merely convenient differences fail. Do not patch the
candidate back to known-wrong behavior just to obtain identical bytes.

## Stage 6: Force rare lifecycles when needed

Use small deterministic fixtures only for required paths the long corpus cannot reliably
trigger:

- a single-worker scheduler-pressure fixture;
- a vLLM preemption edge targeting one to three preemptions and then completion;
- an SGLang retraction edge targeting one to three retractions and then completion;
- backend-specific prefill/decode handoff ordering;
- attention-DP slowest-rank completion and empty-rank barrier behavior;
- exact output-token plans, terminal cleanup, and capped virtual-time behavior; and
- balanced liveness at the effect-free zero-duration-pass boundary.

Assert bounded pressure, continued virtual-time progress, and the lifecycle itself. A
final-completion smoke test does not prove preemption, retraction, handoff, reuse, cleanup,
or barrier behavior occurred.

Injected scaling-policy fixtures may be focused compatibility tests when a change touches
the generic composition boundary, but they are outside the authoritative built-in
round-robin/no-scaling matrix.

## Stage 7: Performance follow-up (not implemented by this skill)

Stop after semantic qualification when using the current skill. Do not report a
performance pass, fail, or inconclusive result from its runner. A future performance
extension must first add the loop-only timing boundary and paired orchestration described
below; until then, route performance-regression work to a purpose-built harness.

The remaining requirements in this stage are the design contract for a future extension,
not executable instructions for the current skill. Once that extension exists, it must
measure every supported row from Stage 4 with its frozen configuration, reuse the release
artifacts from byte parity, and describe the result as engine-only round-robin replay-loop
parity.

The primary metric is replay execution time. Start its timer after trace normalization,
workload construction, and engine/runtime preparation, immediately before the prepared
runtime loop; stop it immediately after that loop returns and before collector
finalization or report aggregation. Emit this value as `replay_execution_ms`. Record
setup and end-to-end time separately as diagnostics.

Current main exposes `ReplayReport.throughput.wall_time_ms`, timed broadly from
`Replayer::run_inner` entry through report finalization. It is not the authoritative
replay-loop metric. If the campaign runner does not expose `replay_execution_ms`, extend
the harness before running the gated performance campaign; do not substitute
`wall_time_ms` or an outer shell timer.

Stage binaries and trace inputs on node-local storage and verify their checksums before
warmups. File transfer is campaign setup, not a sample. For each measured invocation, use
one iteration, a unique node-local timing output, and a fresh process. Do not emit
canonical or full reports during a gated performance invocation; lazy per-request capture,
report serialization, and in-process iteration state must not contaminate the metric.

For each row:

1. Run five warmups per arm, alternating arms for ten warmups total.
2. Generate and persist a fixed-seed, balanced 60-pair schedule with 30 baseline-first and
   30 candidate-first pairs in randomized order.
3. Collect all 60 measured pairs. Keep the two invocations in each pair adjacent and
   compute `r_i = candidate_replay_execution_ms / baseline_replay_execution_ms`
   regardless of run order.
4. Treat the ratios as independent and identically distributed, or otherwise
   exchangeable, only when the campaign can justify that sampling assumption; pair
   adjacency does not establish it. Predeclare thresholds for serial-dependence
   diagnostics, including lag autocorrelation and pair-order trends against elapsed time
   and available temperature or frequency telemetry, and record their results.
5. If diagnostics breach their thresholds or exchangeability cannot be justified, report
   `INCONCLUSIVE` or use a predeclared dependence-aware method. Do not apply the
   order-statistic gate.
6. Otherwise sort the 60 ratios. Conditional on the sampling assumption, use the 24th
   order statistic as the exact distribution-free one-sided 95% lower confidence bound
   for the population median ratio and the 37th order statistic as the corresponding
   upper bound. Do not use a Wald interval.
7. Pass when the upper bound is at most `1.05`.
8. Fail when the lower bound is greater than `1.05`.
9. Otherwise report `INCONCLUSIVE`; do not add samples adaptively and claim the original
   confidence level.

Never remove an observation because its value looks like an outlier. Retain every
attempted sample and its process status. A replay error, assertion failure, malformed
timing record, or other product failure fails or blocks the row; it is not a discardable
sample. Only a predeclared environmental condition, such as scheduler eviction,
node-health failure, affinity violation, or independently detected competing load, may
invalidate a sample. Invalidate both members of that pair, record the evidence and reason,
and rerun the complete pair with the same arm order. Never retry or replace a sample
silently. A paired bootstrap of log ratios may be reported as a secondary effect-size
diagnostic, but it does not decide the gate.

Also fail comparable release artifact or `.text` growth above 5% until the added
footprint is explained and narrowed.

### Investigate unacceptable overhead

When a statistically meaningful regression exceeds the accepted window:

1. Confirm baseline and candidate performed equivalent semantic work. Separate an accepted
   semantic correction from framework overhead when the correction intentionally adds
   work.
2. Inspect the diff and hot-path structure for obvious causes: event capture, request
   cloning, heap allocation, admission vectors, sorting, hashing, dynamic dispatch, lock
   traffic, or widened generic monomorphization.
3. If static analysis does not identify a convincing cause, profile the representative
   failing configuration with equivalent release/debug-symbol settings and inputs.
4. Compare self-time, call stacks, allocation-heavy paths, and new monomorphized
   functions. Attribute the regression to specific code before optimizing or requesting a
   waiver.
5. Repeat the paired performance gate after any fix.

Do not profile concurrently with builds or unrelated load.

## Stage 8: Decide and report

Use exactly one semantic result:

- `PASS`: all authoritative canonical outputs match;
- `PASS_WITH_SEMANTIC_EXCEPTIONS`: every mismatch is an evidenced improvement covered
  by a regression test; or
- `FAIL`: any unexplained mismatch, missing lifecycle evidence, or unstable revision.

If a separate supported performance harness was explicitly used, report performance
independently as pass, fail, or inconclusive. A semantic exception does not waive an
unexplained performance regression. Otherwise state that performance was not measured by
this skill.

The final report must include:

- revision SHAs and determinism-patch checksum, if any;
- trace checksum and deterministic slice specification;
- every artifact's exact feature manifest and checksum;
- each row's frozen configuration, node characteristics, CPU set, and NUMA placement;
- one row per engine/topology with canonical digests and lifecycle evidence;
- round-robin/no-scaling identity and the exact canonical exclusion allowlist;
- every semantic exception record;
- comparable artifact and `.text` sizes;
- skipped or unsupported coverage without overstating the result.

If a separate supported performance harness was used, additionally include its timing
scopes, persisted arm-order schedule, all attempted samples and invalidations, paired
ratios, sampling assumption, dependence diagnostics, conditional order-statistic
confidence bounds, and profiler findings.

## Stage 9: Clean up

Delete temporary full reports, traces, binaries, profiler captures, patches, and worktrees
after recording the required evidence. Retain full outputs only for unresolved mismatches
or performance investigations. Remove task-created Cargo targets when disk pressure
matters, but never delete unrelated caches, traces, artifacts, or worktrees.
