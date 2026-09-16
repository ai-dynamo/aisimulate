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
its server resources and outstanding native cache offload work have settled. Client completion and server settlement are
recorded separately. This deliberately requires more than a client terminal
event and leaves no preparation request in flight at the transition.

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
the existing cold snapshot behavior. The initial public qualification covers
vLLM and SGLang aggregated Engine replay. For example:

```yaml
traffic:
  source: {type: trace, format: weka, path: trace.json}
  load:
    type: trace_timestamps
    agentic_lanes: 4
    agentic_snapshot: {seed: 42}
    agentic_warmup: true
```

Offline disaggregation is qualified
separately by AIC-1895.

`agentic_phases` records per-lane completion, request phase/source identity,
input length, expected cacheable tokens, actual first-admission reuse, terminal
and settlement times, and barrier state. Expected prefix size is not evidence
that tokens stayed resident: eviction and disabled caching can reduce realized
reuse. Router overlap is not an actual cache-hit measurement. Preparation timestamps
and runtime artifacts use the absolute runtime clock; measured per-request
timestamps start at the barrier, whose absolute `profile_start_ms` is recorded.
G3 counters cover the profile time window while residency gauges describe the
preserved cache at the end of that window. `wall_time_ms` remains the host
execution time for the complete run, including preparation; it is not the
simulated profile duration. Scaling policy tick values retain their absolute
runtime-clock contract, and the first tick runs no earlier than the barrier.

## Reference boundary

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
