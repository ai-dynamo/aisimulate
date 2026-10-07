<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agentic cache warmup and profiling

The warmup run fills
the same native engine cache that will serve the measured suffix. Its input is
a prepared snapshot, its original prompt identity context, and the selected
engine configuration. Its output is retained KV state, an unchanged logical
snapshot frontier, and separate preparation evidence.

## Request inputs and outputs

| Stage | Input tokens | Output tokens | Completion condition |
| --- | --- | --- | --- |
| Primer | The complete original input of each live conversation's last historical request | Exactly one | Every selected primer succeeds |
| Warmup | Repeat that lane's primer inputs in deterministic conversation order; when there is no historical input, use its earliest retained request | Exactly one per request, ten requests per lane | Every lane completes ten successful warmup requests |
| Profile | Original retained requests, with their original planned outputs | Original output plan | The retained suffix finishes or reaches its configured time limit |

These input token IDs come from the prepared play context. They are never
normalized again as a shorter Weka request. Primer and warmup requests have
their own request IDs but retain the play, conversation, and cache identity.
Repeating a prefix does not append the warmup output token to the saved prompt.
The same context therefore supplies primer, warmup, and profile tokens.

Within a lane, preparation requests execute sequentially with no authored idle
delay. Lanes can prepare concurrently. A lane finishes its primers before its
ten warmup requests. No profile request can dispatch while any lane is still
preparing.

## The barrier

The barrier opens only after every preparation request has succeeded **and**
its server resources and outstanding native work have settled. For separate
prefill/decode (P/D) workers, this includes KV transfers, source holds,
destination reservations, and handoff cleanup in both pools. Client completion
and server settlement are recorded separately. This deliberately requires more
than a client terminal event and leaves no preparation request in flight at the
transition.

At the transition the runtime retains its engine, router, workers, KV cache,
and play identities. The saved frontier starts from this instant, preserving
remaining timers, joins, spawns, and speedup. Profile duration, request counts,
latencies, reuse ratios, and worker time exclude preparation. Request IDs retain
their original phase so late preparation events cannot become profile events.

The first unsuccessful preparation request stops further preparation admission.
Already issued requests drain, the barrier remains closed, and the report marks
preparation invalid with phase and request evidence. `predict` writes that
evidence and exits unsuccessfully; `recommend` excludes the failed candidate
from ranking. Preparation requires every primer and warmup request to succeed; there is no configurable failure threshold.
[Continuous agentic profiles](continuous-profiles.md) optionally add fixed-duration
lane recycling after this barrier.

## Public control and evidence

The opt-in control is `traffic.load.agentic_warmup: true`, alongside positive
`agentic_lanes` and `agentic_snapshot: {seed: ...}`. Omitting the control keeps
the existing cold snapshot behavior. The built-in Engine runner supports offline
vLLM and SGLang replay with either aggregated workers or separate prefill/decode
pools and HBM-only KV cache. vLLM also supports local or shared
[G2 host offload](../cache.md#agentic-g2) on a static single aggregated worker or 1P1D,
with attention DP1 on every role. Speculative decoding remains disabled. For example:

```yaml
traffic:
  source: {type: trace, format: weka, paths: [trace.json]}
  load:
    type: trace_timestamps
    agentic_lanes: 4
    agentic_snapshot: {seed: 42}
    agentic_warmup: true
```

For P/D, use `engine.mode: disaggregated` and configure `engine.workers.prefill`
and `engine.workers.decode` for the same target model; the traffic configuration
above is unchanged. Each primer and warmup is one logical request through the
native P/D handshake, not a separate replay on each pool. Its tokens and typed
play/conversation/cache identity survive that handshake. The barrier retains
both pools' caches for the profile suffix. Actual reuse still depends on native
placement, cache capacity, and eviction; preparing a prefix does not guarantee
it remains resident on every worker.

Weka, Agentic Mooncake, and agentic Dynamo trace inputs use this same offline
path. Reading a Dynamo trace does not require the Dynamo integration. Agentic
TensorRT-LLM, G3 offload, speculative decoding, and online P/D are rejected;
the built-in runner does not silently change the requested configuration.
Dynamo-owned routing and online integration require separate downstream
qualification.

`agentic_phases` records per-lane completion, request phase/source identity,
input length, cacheable complete-block tokens (`expected_full_block_tokens`),
actual first-admission reuse, terminal and settlement times, and barrier state.
The cacheable token count describes full prompt blocks, not expected admission
reuse. Native admission accounts for reuse according to the backend and block
size: in the qualification fixture with a resident 128-token input, vLLM with
64-token blocks reuses 64 tokens and SGLang with its default one-token blocks
reuses 127. Both leave work for the final input token. P/D reports retain both
prefill and decode admission records; their first-admission reuse comes from
prefill. Eviction and disabled caching can reduce reuse further. Router overlap
and transferred KV are not substitutes for actual admission evidence.

Measured per-request timestamps and `agentic_play_outcomes` terminal/settlement
times start at the barrier, whose absolute `profile_start_ms` is recorded.
All measured request records in a warmed run have `agentic_phase: profile`;
primer and warmup records live in the separate preparation ledger.
Preparation timestamps, `agentic_lifecycle`, and native runtime artifacts use
the absolute runtime clock. Profile `runtime_evidence` excludes preparation
records and restarts its ordinal/digest scope at the barrier; its event
timestamps retain the absolute runtime clock for correlation with artifacts.
G3 counters cover the profile time window while residency gauges describe the
preserved cache at the end of that window. Power and energy diagnostics include
only profile predictions; preparation resets their totals and provenance while
retaining timing-provider caches. `wall_time_ms` remains the host
execution time for the complete run, including preparation; it is not the
simulated profile duration. Scaling policy tick values retain their absolute
runtime-clock contract, and the first tick runs no earlier than the barrier.


## Seeded request-boundary snapshots

Opt into initial snapshots through the existing traffic load configuration:

```yaml
traffic:
  source:
    type: trace
    format: weka
    paths: [corpus]
  load:
    type: trace_timestamps
    agentic_lanes: 2
    agentic_snapshot:
      seed: 42
```

The seed is an unsigned 64-bit integer. The same corpus, lane count, and seed
reproduce each lane's source play, sampled cut, and play/cache identity. Initial
lanes take source plays in corpus order, wrapping when necessary. Each cut is
uniformly sampled between 25% and 75% of that play's first-to-last request-start
span; a zero-width span uses its single timestamp. Sampling uses original source
time. The configured `speedup` applies only to remaining execution timers.
Omitting `agentic_snapshot` preserves turn-zero replay. CLI users can set the
seed with `--set traffic.load.agentic_snapshot.seed=42`; prediction and
recommendation use the same field.

This is a request-boundary snapshot. Requests whose recorded start is strictly
before the cut are history, including requests whose recorded service interval
crosses the cut. Requests at or after the cut remain in the continuation.
Recorded service intervals provide dependency-timer provenance. Dynamo request
traces retain their existing first-request clock origin and preserve all source
intervals separately from completion-relative execution gates; legacy graph
serialization and digests are unchanged. The snapshot
does not estimate partial decode progress or restore a physical engine checkpoint.
For each continuing conversation with earlier history, its primer description
references the latest prior request's complete original input. No prompt is
renormalized after truncation, and no synthetic response is appended to a primer.

Rust consumers call `ValidatedAgenticGraph::prepare_snapshots` with
`AgenticSnapshotOptions`, inspect `PreparedAgenticSnapshots::snapshots`, and pass
the preparation to `WorkloadDriver::new_agentic_snapshots`. The retained
`AgenticReplayContext` can prepare further explicit play instances with fresh
ordinals. Every incarnation receives disjoint logical token identities and
request-instance identities; primer and profile prefix views share their play's
mapping. Identity capacity exhaustion fails instead of reusing an old range.
These APIs reuse the existing dependency executor and runtime feedback contract.

The native report's `agentic_snapshots` collection records source/graph identity,
seed, lane/play/cache identity, sampled cut, recorded request intervals, retained
frontier and remaining dependency timers, and primer descriptions. Default Python
results retain it in metadata; full native reports and CLI JSON preserve the same
evidence. Per-request agentic identities allow events to be attributed to their
original play even when a future phase reuses its lane.


## Reference and validation boundary

Snapshot behavior references NVIDIA AIPerf's
[`trajectory_source.py`](https://github.com/ai-dynamo/aiperf/blob/7db2ba37a62aa80c882bc90eaf61cc8073e2387b/src/aiperf/timing/trajectory_source.py)
and [`session_tree.py`](https://github.com/ai-dynamo/aiperf/blob/7db2ba37a62aa80c882bc90eaf61cc8073e2387b/src/aiperf/timing/session_tree.py)
at `7db2ba37a62aa80c882bc90eaf61cc8073e2387b`, licensed Apache-2.0 with NVIDIA copyright.
The original primer/warmup methodology reference is the
[InferenceX-app article](https://github.com/SemiAnalysisAI/InferenceX-app/blob/9bb7b13eb4985217a6282f340459fd5948613276/packages/app/src/components/datasets/agentx-methodology-article.tsx)
at `9bb7b13eb4985217a6282f340459fd5948613276`, whose repository is GPL-3.0.
A corroborating Apache-2.0 reference is the
[AgentX harness tutorial](https://github.com/SemiAnalysisAI/agentx-harness/blob/56a0cf70f4c0359454ee4bd15a17770b541a3e3e/docs/tutorials/agentx-mvp.md).
These references identify behavior, not a claim that website prose, figures, or code were copied.
The [retained source audit](../../../benchmarks/evidence/accuracy/replay-evidence.md)
records the original comparison and review boundary.

AISimulate uses its native dependency graph, a versioned BLAKE3-based snapshot
sample, and disjoint per-play token identities. Warmup repeats saved primer
inputs and preserves the frontier; the reference executor's warmup advances
its live trajectory. Matching preparation request counts does not establish
full AgentX parity. Qualification must cover exact inputs, one output token,
ten requests per lane, barrier ordering, failure handling, timer preservation,
cache reuse with caching enabled/disabled, and deterministic cold/warm runs.
The public reports remain `functional_only`, not hardware accuracy evidence.
