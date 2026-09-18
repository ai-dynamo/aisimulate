<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agentic cache warmup and profiling

This design extends the prepared snapshots from AIC-1811. The warmup run fills
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
from ranking. Failure thresholds and
fixed-duration recycling remain separate follow-up work.

## Public control and evidence

The opt-in control is `traffic.load.agentic_warmup: true`, alongside positive
`agentic_lanes` and `agentic_snapshot: {seed: ...}`. Omitting the control keeps
the existing cold snapshot behavior. The built-in Engine runner supports offline
vLLM and SGLang replay with either aggregated workers or separate prefill/decode
pools. Both use HBM-only KV cache with speculative decoding disabled. For example:

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
TensorRT-LLM, host offload, speculative decoding, and online P/D are rejected;
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

## Reference boundary

**Attribution review is pending; this work is not cleared for merge/release.**
The repository attribution gate also covers adapted behavior. The statement
that no upstream code was copied does not by itself satisfy that gate.

Source audit (2026-09-18):

- Original snapshot reference: NVIDIA AIPerf,
  [`src/aiperf/timing/trajectory_source.py`](https://github.com/ai-dynamo/aiperf/blob/7db2ba37a62aa80c882bc90eaf61cc8073e2387b/src/aiperf/timing/trajectory_source.py)
  at `7db2ba37a62aa80c882bc90eaf61cc8073e2387b`,
  [Apache-2.0](https://github.com/ai-dynamo/aiperf/blob/7db2ba37a62aa80c882bc90eaf61cc8073e2387b/LICENSE).
- Additional corroborating reference located during this review: SemiAnalysisAI
  AgentX harness,
  [`docs/tutorials/agentx-mvp.md`](https://github.com/SemiAnalysisAI/agentx-harness/blob/56a0cf70f4c0359454ee4bd15a17770b541a3e3e/docs/tutorials/agentx-mvp.md),
  at `56a0cf70f4c0359454ee4bd15a17770b541a3e3e`,
  [Apache-2.0](https://github.com/SemiAnalysisAI/agentx-harness/blob/56a0cf70f4c0359454ee4bd15a17770b541a3e3e/LICENSE).
  This is not a claim that this later audit reference was the original source.
- The original AgentX methodology website links to SemiAnalysisAI InferenceX-app.
  Its methodology article source is
  [`packages/app/src/components/datasets/agentx-methodology-article.tsx`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/9bb7b13eb4985217a6282f340459fd5948613276/packages/app/src/components/datasets/agentx-methodology-article.tsx)
  at `9bb7b13eb4985217a6282f340459fd5948613276`; the repository
  [license is GPL-3.0](https://github.com/SemiAnalysisAI/InferenceX-app/blob/9bb7b13eb4985217a6282f340459fd5948613276/LICENSE).
  No website code, figures, or prose were copied into this implementation.
  Maintainer/OSRB guidance is required on the behavioral-reference boundary;
  neither the harness license nor the dataset license resolves that question.

After that guidance, complete the derived-file inventory and canonical/packaged
notices as required. Byte-identical notice copies alone do not establish that
all required attribution is present. This audit does not grant an exemption or
relicense any source.

[AgentX methodology](https://inferencex.semianalysis.com/agentx/methodology)
defines one-output-token primers and ten additional warmup requests per lane.
The pinned [AIPerf implementation](https://github.com/ai-dynamo/aiperf/tree/7db2ba37a62aa80c882bc90eaf61cc8073e2387b/src/aiperf/timing)
provides request-boundary snapshots and primers; its optional duration-based
warmup advances trajectories. Repeating saved prefixes ten times is this
implementation's explicit choice for preserving the saved frontier. It is not
a claim of identical warmup payload selection in that AIPerf revision.

Qualification must assert exact input tokens, ten requests per lane, one output
token per request, barrier ordering, timer preservation, failure behavior, and
actual cache reuse with caching enabled and disabled. It must also retain cold
snapshot compatibility and deterministic repeated runs.

The initial offline P/D qualification covers the public Rust, Python, and CLI
paths, request identity through the handoff, and the same preparation/profile
barrier. Reports remain `functional_only`; these checks do not establish
prediction accuracy or full AgentX benchmark fidelity. Fixed-duration recycling,
cutoff behavior, and late events from recycled play instances still need joint
qualification under AIC-1813, AIC-1896, and AIC-1818. This initial boundary does
not complete all AIC-1895 acceptance work.
