<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

<a id="traffic"></a>
<a id="11-traffic"></a>

# Replay workloads

Traffic is always expressed as three concepts:

```yaml
traffic:
  source: {}
  load: {}
  stop: {}
```

`source` defines requests or sessions, `load` defines when they begin, and `stop` defines when the run
ends. The load and stop unit follows the source type.

Omitting `traffic` is equivalent to this concrete default:

```yaml
traffic:
  source:
    type: synthetic
    input_tokens: 1024
    output_tokens: 128
  load:
    type: concurrency
    concurrency: 10
  stop:
    requests: 100
```

The default source contains independent requests with fixed input sequence length (ISL) and output
sequence length (OSL); it does not sample token-length distributions or create sessions. A supplied
`traffic` mapping does not merge recursively with this example. Once `traffic` is present, its normal
source, load, and stop validation applies.

The default run uses `100` requests at concurrency `10`, or 10 times the active concurrency, following
the current SA convention.

<a id="traffic-fields"></a>

## Traffic Fields

| Knob | Default | Default Range | Preset | Rules |
|---|---:|---|---|---|
| `traffic.source.type` | `synthetic` | `x` | `-` | `synthetic`, `synthetic-session`, or `trace`. |
| `traffic.source.input_tokens` | `1024` | `x` | `-` | Positive; `synthetic` only. |
| `traffic.source.output_tokens` | `128` | `x` | `-` | Positive; `synthetic` only. |
| `traffic.source.images` | Unset | `x` | `-` | Fixed positive `height`, `width`, `count` (default 1); synthetic analytical EPD only; requires `engine.workers.encoder`. |
| `traffic.source.new_input_tokens_per_turn` | `1024` | `x` | `-` | Positive; `synthetic-session` only. |
| `traffic.source.output_tokens_per_turn` | `128` | `x` | `-` | Positive; `synthetic-session` only. |
| `traffic.source.session.turns` | `4` | `x` | `-` | At least `2`. |
| `traffic.source.session.shared_prefix_ratio` | `0` | `x` | `-` | From `0` through `1`. |
| `traffic.source.session.prefix_groups` | `0` | `x` | `-` | Nonnegative; positive when prefix ratio is positive. |
| `traffic.source.session.inter_turn_delay_ms` | `0` | `x` | `-` | Nonnegative. |
| `traffic.source.paths` | Required for trace | `x` | `-` | One path except `dynamo`, which permits multiple. |
| `traffic.source.format` | `mooncake` | `x` | `-` | See [Trace Format Compatibility](#trace-format-compatibility). |
| `traffic.source.block_size` | `512`; embedded for `dynamo` and `weka` | `x` | `-` | Positive. For embedded formats, an explicit value is an equality assertion. |
| `traffic.source.nested_timestamp_basis` | `auto` | `x` | `-` | `auto`, `absolute`, or `relative`; Weka only. |
| `traffic.load.type` | `concurrency` | `x` | `-` | Synthetic: `concurrency`, `poisson`, `constant_rate`, or `kv_capacity_fraction`; trace: `trace_timestamps` or `concurrency`. |
| `traffic.load.concurrency` | `10` | `-` | `-` | Positive integer; explicit domains are allowed in `recommend`. |
| `traffic.load.requests_per_second` | `null` | `-` | `-` | Positive; synthetic request open-loop load only. |
| `traffic.load.sessions_per_second` | `null` | `-` | `-` | Positive; synthetic session open-loop load only. |
| `traffic.load.seed` | `42` | `x` | `-` | Nonnegative; `poisson` only. |
| `traffic.load.fraction` | `null` | `-` | `-` | Positive finite number; `kv_capacity_fraction` only and may exceed `1`. |
| `traffic.load.speedup` | `1` | `-` | `-` | Positive; trace timestamp load only. |
| `traffic.load.agentic_lanes` | `null` | `x` | `-` | Positive integer; `weka`, `agentic_mooncake`, or agentic `dynamo` timestamp replay only. |
| `traffic.load.agentic_snapshot` | `null` (unset) | `x` | `-` | Optional object `{seed: u64}`; required `seed` is an unsigned 64-bit integer (`0` through `2^64 - 1`). Requires `traffic.load.type: trace_timestamps` and positive `agentic_lanes`; supported formats are `weka`, `agentic_mooncake`, and agentic `dynamo`. Unset preserves turn-zero execution. See [snapshot](#agentic-snapshot). |
| `traffic.load.agentic_warmup` | `false` | `x` | `-` | Optional boolean; `true` requires `agentic_snapshot` and positive `agentic_lanes`. Primes caches before measurement; see [warmup](#agentic-warmup). |
| `traffic.load.agentic_profile` | `null` (unset) | `x` | `-` | Optional object; `{}` enables continuous lane replenishment with the defaults below. Requires `trace_timestamps`, positive `agentic_lanes`, and `agentic_snapshot`; cannot be combined with `traffic.stop.max_virtual_time_seconds`. Unset preserves finite replay. See [continuous profile](#agentic-profile). |
| `traffic.load.agentic_profile.duration_seconds` | `3600` when enabled | `x` | `-` | Positive finite admission duration, starting at the preparation barrier or simulation start without warmup. No new workload requests or replacement plays are issued after the deadline. |
| `traffic.load.agentic_profile.response_grace_seconds` | `30` when enabled | `x` | `-` | Nonnegative finite time for already submitted requests to respond after the admission deadline; remaining client requests are then canceled. |
| `traffic.load.agentic_profile.cancel_drain_seconds` | `10` when enabled | `x` | `-` | Nonnegative finite upper bound for cancellation acknowledgements. The supported offline runtimes acknowledge synchronously; this does not guarantee server/GPU cleanup. |
| `traffic.load.agentic_profile.tree_idle_cap_seconds` | `300` when enabled | `x` | `-` | Positive finite idle cap for advancing a play's pending workload timers when that play has no outstanding requests. |
| `traffic.load.agentic_profile.global_idle_cap_seconds` | `10` when enabled | `x` | `-` | Positive finite idle cap for advancing pending workload timers when the entire client workload has no outstanding requests. Engine completions and server cleanup keep their actual timestamps. |
| `traffic.stop.requests` | `100` for default traffic | `x` | `-` | Positive integer; 10× default concurrency; synthetic request source only. |
| `traffic.stop.requests_per_load_unit` | `null` | `x` | `-` | Positive; synthetic request source only. |
| `traffic.stop.sessions` | `null` | `x` | `-` | Positive integer; synthetic session source only. |
| `traffic.stop.sessions_per_load_unit` | `null` | `x` | `-` | Positive; synthetic session source only. |
| `traffic.stop.max_virtual_time_seconds` | `null` | `x` | `-` | Positive; supported trace formats only. |

<a id="synthetic-request-source"></a>

## Synthetic Request Source

```yaml
traffic:
  source:
    type: synthetic
    input_tokens: 1024
    output_tokens: 128
  load:
    type: poisson
    requests_per_second: 8
    seed: 42
  stop:
    requests: 100
```

`synthetic` generates independent single requests. It does not accept a `session` mapping.

<a id="synthetic-session-source"></a>

## Synthetic Session Source

```yaml
traffic:
  source:
    type: synthetic-session
    new_input_tokens_per_turn: 1024
    output_tokens_per_turn: 128
    session:
      turns: 4
      shared_prefix_ratio: 0.5
      prefix_groups: 16
      inter_turn_delay_ms: 1000
  load:
    type: constant_rate
    sessions_per_second: 8
  stop:
    sessions: 100
```

`synthetic-session` generates multi-turn sessions. It requires a `session` mapping, and requests
inside one session execute in turn order.

<a id="synthetic-load-and-stop"></a>

## Synthetic Load and Stop

Both synthetic source types support closed-loop concurrency, Poisson arrivals, constant-rate
arrivals, and recommendation-only KV-capacity-relative load as listed in the Traffic table.

The source-specific open-loop rate field is:

- `requests_per_second` for `source.type: synthetic`.
- `sessions_per_second` for `source.type: synthetic-session`.

The rate is a positive finite number. `concurrency` is a positive integer and counts independent
requests for `synthetic` or active sessions for `synthetic-session`. At most one turn of a session is
active at a time. `seed` is a nonnegative integer and defaults to `42`. `fraction` is greater than `0`
and finite; it has no upper bound. `fraction: 1` targets a concurrency whose estimated KV working set
equals the candidate's usable KV capacity. Values greater than `1` intentionally oversubscribe that
capacity. They remain valid because excess work can queue; they do not mean that the engine has more
physical KV memory.

The stop fields follow the source unit. The fixed-count field is a positive integer. The
load-relative field is
a positive number and resolves to `max(1, round(count_per_load_unit * load_unit))`. The load unit is
concurrency for `concurrency` and resolved `kv_capacity_fraction` traffic, requests per second for a
`synthetic` open-loop source, or sessions per second for a `synthetic-session` open-loop source.

For `synthetic-session`, a session with four turns contributes four requests but only one unit to the
load and stopping condition.

`sessions_per_second` controls the arrival rate of new sessions. For a multi-turn session, it schedules
the first turn; later turns follow that session's completion and `inter_turn_delay_ms` rules and do not
count as new load arrivals. `requests_per_second` schedules independent single requests.

In a recommendation input, source type, token fields, session shape, and stopping condition stay
concrete. Only `traffic.load` rows whose Default Range is not `x` can be search domains. A
`kv_capacity_fraction` recommendation is materialized as a concrete `concurrency` load in each
recommended prediction YAML.

<a id="trace-source"></a>

## Trace Source

```yaml
traffic:
  source:
    type: trace
    paths:
      - traces/requests.jsonl
    format: mooncake
    block_size: 512
  load:
    type: trace_timestamps
    speedup: 1.0
  stop:
    max_virtual_time_seconds: 300
```

Without `agentic_profile`, omitting `traffic.stop` for a trace runs to end of trace.
`max_virtual_time_seconds` is trace-only and cannot be combined with `agentic_profile` or used for
synthetic traffic. Trace source, format, and token/session content stay concrete in a recommendation
input; only a numeric trace-load field can be a domain.

`speedup: N` divides authored timing by `N`; for example, `2` replays the timing twice as fast. For
Mooncake session traces it scales both first-turn arrival timestamps and inter-turn delays. For
agentic traces it scales root-node timestamps and the combined dependency delay and tool wait.
`speedup` is deliberately rejected with `load.type: concurrency`: concurrency replaces authored
first-arrival pacing, and inter-turn or dependency delays remain unscaled.

<a id="trace-format-compatibility"></a>

## Trace Format Compatibility

| Format | JSONL Unit | Allowed Load | `speedup` | `max_virtual_time_seconds` | Other Constraints |
|---|---|---|---|---|---|
| `mooncake` | One request or session turn with a full prompt | `trace_timestamps`, `concurrency` | Timestamp load only | Supported | None specific to the format. |
| `mooncake-delta` | One session turn; follow-up input is only the new input delta | `trace_timestamps`, `concurrency` | Timestamp load only | Supported | Aggregated deployment only; `planner.policy` must be `disabled`. |
| `agentic_mooncake` | One request node in a dependency graph | `trace_timestamps` | Supported | Not supported; omit it | Offline aggregated or disaggregated vLLM/SGLang Engine replay; `planner.policy` must be `disabled`. |
| `weka` | A raw kv-cache-tester or published AgentX JSON/JSONL corpus; directories are traversed recursively and JSONL files may contain multiple plays | `trace_timestamps` | Supported | Not supported; omit it | Offline aggregated or disaggregated vLLM/SGLang Engine replay; source block size is embedded and the result is functionally qualified. |
| `applied_compute_agentic` | One complete session, expanded into `num_turns + 1` requests | `concurrency` | Not supported; omit it | Supported | Source rows have no first-turn timestamps. |
| `dynamo` standard trace | Native request-trace records, possibly across multiple files | `trace_timestamps`, `concurrency` | Timestamp load only | Supported | The embedded trace block size is authoritative. |
| `dynamo` agentic trace | Native agentic request-trace records, possibly across multiple files | `trace_timestamps` | Supported | Not supported; omit it | Offline aggregated or disaggregated vLLM/SGLang Engine replay; `planner.policy` must be `disabled`. |

The `dynamo` loader detects whether its records are standard or agentic and applies the corresponding
row above. If `traffic.source.block_size` is supplied for `dynamo` or `weka`, it must match the
embedded block size. For the other formats, `block_size` is the trace hash-block size used to
reconstruct prompts.

Agentic formats are qualified on the offline `engine` stack; see
[agentic load controls](#agentic-load-controls) for supported runtimes. P/D workers
must share one target model. Reading a `dynamo` trace does not require the Dynamo stack.

Weka is the AgentX source format. AISimulate converts a Weka corpus into
Agentic Mooncake and replays it natively; the Dynamo stack is not needed.

Weka producers use two timestamp conventions for requests inside a subagent:
raw kv-cache-tester traces are relative to the subagent marker, while published
AgentX traces are absolute. Set `traffic.source.nested_timestamp_basis` to
`relative` or `absolute` when you know which one applies. With `auto` (the
default), AISimulate picks `relative` for the whole corpus if any child request
is earlier than its subagent marker, and `absolute` otherwise. The chosen basis
is reported as `weka_nested_timestamp_basis`.

A Weka corpus can name several source models. Replay runs every request on
`engine.model` and records the source models and target under
`agentic_model_projection`.

<a id="agentic-load-controls"></a>

## Agentic Load Controls

Agentic traces (`weka`, `agentic_mooncake` and agentic `dynamo`) are made of
*plays*: dependency trees of requests from one agent session. Four
`traffic.load` fields control how plays are replayed. Each one requires the
previous:

```yaml
traffic:
  source: {type: trace, format: weka, paths: [trace.jsonl]}
  load:
    type: trace_timestamps
    agentic_lanes: 4              # concurrent plays
    agentic_snapshot: {seed: 42}  # start each lane mid-play
    agentic_warmup: true          # fill caches before measuring
    agentic_profile: {}           # recycle plays for a fixed duration
```

| Control | Without it | With it |
| --- | --- | --- |
| `agentic_lanes` | Every play starts at its authored timestamp. | At most `N` plays run at once. |
| `agentic_snapshot` | Each play starts at turn zero. | Each lane starts from a sampled point inside its play. |
| `agentic_warmup` | Measurement starts with cold caches. | Caches are primed before measurement starts. |
| `agentic_profile` | The run ends when the selected plays finish. | Lanes keep taking new plays until a deadline. |

Supported runtimes: offline `engine` stack, vLLM or SGLang, aggregated or P/D
workers, HBM-only KV cache, and no speculative decoding. vLLM additionally
supports [G2 host offload](engine/kv-cache.md#agentic-g2) on one aggregated
worker or 1P1D with `attention_data: 1`. TensorRT-LLM, G3 offload and online
P/D are rejected. Results are qualified `functional_only`, not hardware
accuracy. [Start an AgentX simulation](agentic/quickstart.md) walks through a
complete run.

<a id="agentic-lanes"></a>

### Lanes

`agentic_lanes: N` starts the first `N` plays in corpus order. When a play's
client work ends, its lane takes the next play from one shared queue. A play's
client work ends when all its requests complete, or, after a failure, when all
dispatched requests are terminal. Server cleanup, such as P/D source holds, can
continue after the lane moves on. Lanes limit whole plays, not the concurrent
child requests inside a play.

A child request becomes ready after its dependencies complete, plus the
recorded gap between them. Background children run without a join.

<a id="agentic-snapshot"></a>

### Snapshot

`agentic_snapshot: {seed: S}` starts each initial lane partway through its play.
The cut is sampled uniformly between 25% and 75% of the play's span of request
start times. Requests that started before the cut become history and are not
replayed; the rest are replayed with their original timers. The same corpus,
lane count and seed always give the same cuts. `S` is an unsigned 64-bit
integer.

A snapshot is cut at request boundaries. It does not model partial decode
progress or restore an engine checkpoint: without warmup, the caches start
empty.

<a id="agentic-warmup"></a>

### Warmup

`agentic_warmup: true` fills the caches before measurement, in three stages:

| Stage | Requests | Output |
| --- | --- | --- |
| Primer | For each conversation with history, its last historical request's full input | 1 token each |
| Warmup | 10 per lane, repeating that lane's primer inputs (or its first remaining request if it has no history) | 1 token each |
| Profile | The remaining requests of each play | Original output lengths |

Within a lane, preparation requests run one after another; lanes prepare in
parallel. Profiling starts at a barrier, after every preparation request has
succeeded and all its server work has settled, including P/D transfers. At the
barrier, worker caches and play identities are kept, and the measurement clock
starts. If any preparation request fails, the run stops before profiling:
`predict` exits with an error and `recommend` drops the candidate.

Warmup repeats the saved prefixes and does not advance the plays. Primed
prefixes can still be evicted or placed on another worker, so warmup does not
guarantee cache hits.

<a id="agentic-profile"></a>

### Continuous profile

`agentic_profile` keeps every lane busy until an admission deadline. When a
play finishes, its lane takes the next play from the corpus, wrapping at the
end. Replacement plays start at turn zero with fresh identities.

| Field | Default | Meaning |
| --- | --- | --- |
| `duration_seconds` | `3600` | Admission window, starting at the warmup barrier (or at time zero without warmup). No new requests or plays are issued after it. |
| `response_grace_seconds` | `30` | Time for already issued requests to finish after the deadline. Requests still running are then canceled. |
| `cancel_drain_seconds` | `10` | Upper bound for cancellation acknowledgements. Offline runtimes acknowledge immediately. |
| `tree_idle_cap_seconds` | `300` | Longest idle wait inside a play with no outstanding requests; longer authored waits are shortened. |
| `global_idle_cap_seconds` | `10` | Longest idle wait when no client request is outstanding. |

`agentic_profile: {}` uses all defaults. Durations and idle caps must be
positive; grace periods may be zero. The profile cannot be combined with
`traffic.stop.max_virtual_time_seconds`. Idle caps shorten client wait timers
only; engine work keeps its real timing.

Throughput uses the successful requests, including those that finish during
the grace period. The observed interval runs from their earliest arrival to
their latest response, so it can be shorter or longer than `duration_seconds`.

Long profiles keep lifecycle records for every play and can use much more host
memory than the trace itself. The CLI runs them under host-memory supervision;
a run that exceeds its budget stops with `resource_limited`.
[`examples/cli/agentic-profile.yaml`](../../examples/cli/agentic-profile.yaml)
is a small offline example with fixed timing:

```bash
aisimulate predict --config examples/cli/agentic-profile.yaml \
  --capture-per-request --format json --output-dir ./agentic-profile
```

<a id="agentic-results"></a>

### Agentic results

`prediction.json` adds these sections:

| Section | Contents |
| --- | --- |
| `agentic_snapshots` | Seed, sampled cut, history and remaining timers for each initial lane |
| `agentic_phases` | Preparation requests, barrier state and time (`profile_start_ms`) |
| `agentic_play_outcomes` | Each play's terminal status (`completed`, `failed` or `incomplete`) and times |
| `agentic_profile` | Resolved profile options, deadline, recycled plays, canceled and never-issued requests, unsettled server work |

With warmup, per-request records in `requests.jsonl` contain only profile
requests, and their timestamps start at the barrier. Preparation is excluded
from latency, throughput and reuse metrics. `wall_time_ms` is host time for the
whole run, including preparation.

To check cache reuse, use `first_admission_prefix_cache_reused_ratio` and the
per-request reuse fields described in [KV cache](engine/kv-cache.md#gpu-cache-g1).
Reuse is counted in whole blocks: a resident 128-token prefix reuses 64 tokens
with vLLM's 64-token blocks and 127 tokens with SGLang's 1-token pages.

<a id="agentic-references"></a>

### Behavior references

Snapshots follow NVIDIA AIPerf's
[`trajectory_source.py`](https://github.com/ai-dynamo/aiperf/blob/7db2ba37a62aa80c882bc90eaf61cc8073e2387b/src/aiperf/timing/trajectory_source.py)
and [`session_tree.py`](https://github.com/ai-dynamo/aiperf/blob/7db2ba37a62aa80c882bc90eaf61cc8073e2387b/src/aiperf/timing/session_tree.py)
(Apache-2.0). The one-token primers and ten warmups per lane follow the
[InferenceX-app methodology article](https://github.com/SemiAnalysisAI/InferenceX-app/blob/9bb7b13eb4985217a6282f340459fd5948613276/packages/app/src/components/datasets/agentx-methodology-article.tsx)
(GPL-3.0) and the
[AgentX harness tutorial](https://github.com/SemiAnalysisAI/agentx-harness/blob/56a0cf70f4c0359454ee4bd15a17770b541a3e3e/docs/tutorials/agentx-mvp.md)
(Apache-2.0). Continuous profiles follow the
[`agentx-harness` timing modules](https://github.com/SemiAnalysisAI/agentx-harness/tree/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/timing)
(Apache-2.0) as pinned by
[InferenceX](https://github.com/SemiAnalysisAI/InferenceX/tree/4ab85c1e33b66d6bd5a3087b3de5ba3e86cbbe80).
These sources describe the behavior; no code or prose was copied. The
implementation, tests and example workloads are written for AISimulate.

Known differences: AISimulate samples snapshots with its own deterministic
algorithm, and its warmup repeats saved prefixes instead of advancing the live
trajectory. This is not full AgentX parity. Source review notes are kept in the
[replay evidence record](../../benchmarks/evidence/accuracy/replay-evidence.md).

<a id="mooncake-and-mooncake-delta-jsonl"></a>

## Mooncake and Mooncake Delta JSONL

Both formats use the same row schema. Rows with the same `session_id` are turns in file order.

| Field | Required | Semantics |
|---|---|---|
| `request_id` | No | Request identity. |
| `session_id` | No | Groups rows into a session. An omitted value creates a one-row session whose per-request records use `request_<line>`; placement sees no session for it. |
| `input_length` or `input_tokens` | No | Input token count; defaults to the capacity represented by `hash_ids`. |
| `output_length` or `output_tokens` | Yes | Output token count. |
| `output_token_ids` | No | Exact output tokens; its length must equal the output token count. |
| `hash_ids` | Yes | Prompt hash blocks at `traffic.source.block_size`. |
| `timestamp` or `created_time` | Conditional | Virtual timestamp in milliseconds. Required for the first row of every session under `trace_timestamps`. |
| `delay` or `delay_ms` | No | Inter-turn delay in milliseconds. It must be omitted or zero on a session's first row; later rows use it instead of a timestamp difference. |
| `priority`, `strict_priority`, `policy_class` | No | Scheduling metadata. |

With `format: mooncake`, every row describes that turn's complete prompt. With
`format: mooncake-delta`, the first row describes the initial prompt, while each later row's input and
`hash_ids` describe only new input for that turn. Replay builds the next complete prompt by appending
the prior generated output and the new delta. Using `mooncake-delta` on a full-prompt trace would
double-count prior context.

<a id="agentic-mooncake-jsonl"></a>

## Agentic Mooncake JSONL

`agentic_mooncake` includes all Mooncake request fields above, but `request_id` is required, nonempty,
and unique. Each row is an independently schedulable request node rather than a turn inferred only
from session order. It adds these fields:

| Field | Default | Semantics |
|---|---:|---|
| `wait_for` | `[]` | Request IDs that must all complete first. Unknown IDs, self-dependencies, and cycles are rejected. |
| `delay` / `delay_ms` | `0` | Delay after the last dependency completes. |
| `tool_wait_ms` | `0` | Additional tool wait after dependencies; scheduling delay is `delay + tool_wait_ms`. |
| `timestamp` / `created_time` | `0` for roots | Ready time for a node with an empty `wait_for`; dependent-node timestamps do not control release. |
| `request_kind`, `branches`, `prefix_reset`, `tool_events` | Empty | Producer metadata accepted by the format. Scheduling is controlled by `wait_for`, and cache identity is carried by `hash_ids`; these metadata fields do not independently change replay behavior. |

A dependent node becomes ready after the latest request in `wait_for` completes, plus its `delay` and
`tool_wait_ms`. `speedup` therefore scales both authored root arrivals and those post-dependency waits.

<a id="applied-compute-agentic-jsonl"></a>

## Applied Compute Agentic JSONL

Each row is one session with `num_turns`, `input_prompt_length`, arrays
`assistant_response_length`, `tool_call_output_length`, and `tool_call_latency`, plus
`final_assistant_response_length`. Each array length must equal `num_turns`; tool latency is in
seconds. The row expands to `num_turns` assistant/tool turns plus one final assistant request. The
input grows cumulatively by each assistant response and tool output. Because the format has no
first-session arrival timestamps, it requires `load.type: concurrency` and rejects `speedup`.

> [!NOTE]
> `max_virtual_time_seconds` limits the total simulated virtual time of one prediction or recommendation
> candidate. It is not a separate processing-time limit for each path in `traffic.source.paths`, and it
> is not a real wall-clock timeout. The limit is a soft scheduling cutoff: events at the cutoff are
> processed, replay stops before the first event after the cutoff, and requests still in flight can be
> reported as incomplete. The reported duration can extend slightly past the cutoff while an already
> running engine pass finishes.




## Runtime admission and SDK mapping

Trace mode normalizes the first request or session to virtual time zero.
Concurrency replaces first-arrival pacing. An active multi-turn session holds
its slot through all turns and think time, so concurrency caps sessions, not
individual turns. Agentic lanes instead cap whole play instances and do not
cap concurrent child requests. Completion dependencies, tool delays, joins,
and background children determine which requests become ready.

The Python runner receives concrete `ReplaySpec.workload` plus `concurrency`;
native `ReplaySpec.max_in_flight` selects source-side concurrency. The SDK's
`Workload` uses fields such as `trace_path`, `isl`, `osl`, and `kv_load_ratio`,
which are distinct from this public YAML. See [Sweeper SDK](../sweeper/sdk.md)
for that mapping and [candidate-relative load](../sweeper/search-space.md)
for capacity-based search. Do not substitute SDK YAML for CLI YAML.
