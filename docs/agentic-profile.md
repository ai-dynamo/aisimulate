<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Continuous agentic profiles

`traffic.load.agentic_profile` keeps seeded agentic lanes occupied until an
admission deadline. Enable it with positive `agentic_lanes` and
`agentic_snapshot: {seed: ...}`. The existing finite snapshot replay remains the
default when this object is absent.

```yaml
traffic:
  source: {type: trace, format: weka, paths: [trace.jsonl]}
  load:
    type: trace_timestamps
    agentic_lanes: 4
    agentic_snapshot: {seed: 42}
    agentic_warmup: true
    agentic_profile:
      duration_seconds: 3600
      response_grace_seconds: 30
      cancel_drain_seconds: 10
      tree_idle_cap_seconds: 300
      global_idle_cap_seconds: 10
```

All five values have the defaults shown above, so `agentic_profile: {}` enables
that configuration. Duration and idle caps must be finite and positive. Response
and cancellation grace periods may be zero. Values whose millisecond deadlines
overflow are rejected. Do not combine the profile with
`traffic.stop.max_virtual_time_seconds` or the native `max_sim_time_ms` limit.
The same controls are preserved by prediction, recommendation, saved YAML,
Sweeper workloads, and the native JSON execution boundary.

The built-in engine supports offline vLLM and SGLang with aggregated or separate
prefill/decode workers, HBM-only KV cache, and speculative decoding disabled.
Weka, Agentic Mooncake, and agentic Dynamo traces share this path. Online execution,
other backends, and Dynamo-owned routing need their own qualified integration.

## Lanes and virtual time

The duration begins at the preparation barrier, or at simulation start when
warmup is disabled. Once a play and its descendants have reached their client
terminal states, its lane takes another complete play from a shared sequential
corpus cursor. The cursor wraps at the end of the corpus. New plays start at
turn zero with fresh request, conversation, play, and cache identities. Native
server cleanup from an earlier play may continue after its lane is reused.

Idle guards advance pending workload timers after a tree or the whole client
workload has no outstanding requests for its configured cap. Requests queued
for placement and requests between prefill and decode still count as outstanding.
The timer shift preserves relative delays and dependencies. It does not move
engine completions, KV transfers, or server cleanup events.

At the admission deadline, no new workload request or replacement play is
issued. Previously submitted requests may complete during the response grace
period. At its end, remaining client requests are canceled. Cancellation
acknowledgements have a separate bounded drain period, which ends early once
client requests are terminal. Already committed engine work retains its actual
timestamps; the report records any server work still unsettled at client
completion. The cancellation budget is not a GPU cleanup guarantee.
The supported offline runtimes acknowledge cancellation synchronously, so their
client drain ends immediately and `cancel_drain_timed_out` is false even when
the report records unsettled server work. The configured drain deadline is an
upper bound, not a minimum run duration.

## Results and limits

The `agentic_profile` report contains resolved options, barrier and deadline
timestamps, play/request counts, the corpus cursor, idle shifts, canceled and
never-issued requests, and remaining server work. Its timestamps use the
absolute runtime clock. Warmed per-request records use the existing
barrier-relative convention described in [agentic warmup](agentic-warmup.md).

Successful responses received during grace participate in the measured request
cohort. The observation interval starts with the earliest arrival among those
successful requests and ends with the latest successful response. It may differ from
the configured admission duration. Throughput uses the observed request interval.
With no successful requests, `observation_duration_ms` is `0` and
`successful_request_throughput` is `null`. A single successful request uses its
response time minus its arrival time; throughput is `null` only if that interval
is zero.
The report keeps the admission cutoff separately. A configured one-hour run can
have a request observation interval shorter or longer than one hour.

This feature retains #235's warmup policy: repeat saved primer inputs and keep
the original snapshot frontier. The referenced AgentX executor advances its live
trajectory during warmup. Initial snapshot sampling also retains AISimulate's
existing deterministic algorithm. These differences remain explicit; this is
functional qualification of profile control, not complete AgentX parity or
hardware performance accuracy. Long profiles retain lifecycle evidence and can
use more host memory than a finite corpus replay; runtime resource supervision
remains applicable.

## Reproducible smoke example

From the repository root, first install the current source as described in
[Use current source](installation.md#use-current-source):

```bash
uv sync --project python/aisimulate --extra dev
```

Then run:

```bash
python/aisimulate/.venv/bin/aisimulate predict \
  --config examples/cli/agentic-profile.yaml \
  --capture-per-request --format json --output-dir /tmp/agentic-profile
```

The example uses the repository's self-authored two-play trace and fixed timing
to exercise control flow without downloading a model or dataset. To change the
duration on the existing CLI, use
`--set traffic.load.agentic_profile.duration_seconds=3600`.

## Behavioral reference

The reference executor is SemiAnalysisAI's AIPerf fork
[`agentx-harness` at `754356e9a39acc6cc6afb242d123bb57c3fb6f75`](https://github.com/SemiAnalysisAI/agentx-harness/tree/754356e9a39acc6cc6afb242d123bb57c3fb6f75),
licensed [Apache-2.0](https://github.com/SemiAnalysisAI/agentx-harness/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/LICENSE).
Its timing modules are `src/aiperf/timing/phase/runner.py`,
`src/aiperf/timing/strategies/agentic_replay.py`, and
`src/aiperf/timing/replay_dependencies.py`; request observation duration is
defined in `src/aiperf/metrics/types/benchmark_duration_metric.py`.
InferenceX [`4ab85c1e33b66d6bd5a3087b3de5ba3e86cbbe80`](https://github.com/SemiAnalysisAI/InferenceX/tree/4ab85c1e33b66d6bd5a3087b3de5ba3e86cbbe80)
pins that executor under `utils/aiperf` and supplies the AgentX recipe in
`benchmarks/benchmark_lib.sh` and `benchmarks/runtime_settings.sh`.
The AISimulate implementation uses its native dependency graph and virtual event
loop; the tests and example workload are locally authored. This reference does
not resolve the inherited #235 attribution review.
