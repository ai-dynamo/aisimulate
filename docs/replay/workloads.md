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
| `traffic.load.agentic_snapshot` | `null` (unset) | `x` | `-` | Optional object `{seed: u64}`; required `seed` is an unsigned 64-bit integer (`0` through `2^64 - 1`). Requires `traffic.load.type: trace_timestamps` and positive `agentic_lanes`; supported formats are `weka`, `agentic_mooncake`, and agentic `dynamo`. Unset preserves turn-zero execution. |
| `traffic.load.agentic_warmup` | `false` | `x` | `-` | Optional boolean; `true` requires `agentic_snapshot` and positive `agentic_lanes`. Physically primes the saved prefixes, completes ten warmup requests per lane, then profiles the saved suffix. Available on offline aggregated or disaggregated vLLM/SGLang Engine replay with HBM-only KV cache, or [qualified vLLM G2 configurations](engine/kv-cache.md#agentic-g2). Speculative decoding remains disabled. |
| `traffic.load.agentic_profile` | `null` (unset) | `x` | `-` | Optional object; `{}` enables continuous lane replenishment with the defaults below. Requires `trace_timestamps`, positive `agentic_lanes`, and `agentic_snapshot`; cannot be combined with `traffic.stop.max_virtual_time_seconds`. Unset preserves finite replay. See [continuous agentic profiles](agentic/continuous-profiles.md) for the full configuration, supported runtimes, and reporting semantics. |
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

Agentic Engine replay supports HBM-only KV cache on vLLM/SGLang and local or shared
[G2 host offload](engine/kv-cache.md#agentic-g2) on vLLM with a static single aggregated worker
or 1P1D, attention DP1 on every role, and no G3. Speculative decoding is disabled;
TensorRT-LLM is not qualified. P/D workers must share the same target
model. Online P/D fails validation. These functional replay guarantees do not
qualify the separate Dynamo runner, even when the input format is `dynamo`.

Weka is the public AgentX source format and AISimulate is its prediction entry point. AISimulate
deterministically lowers Weka into Agentic Mooncake v2, the versioned producer-neutral interchange
format, and then validates that lower IR as a `ValidatedAgenticGraph`, the runtime representation.
Dynamo is an optional integration and is not required to parse, convert, or predict a Weka corpus.
Two producer timestamp conventions exist: raw kv-cache-tester nested request timestamps are relative
to their subagent marker, while SemiAnalysis-published AgentX timestamps are root-trace absolute.
`nested_timestamp_basis` may select either convention explicitly. When omitted (or set to `auto`),
AISimulate scans every nested request in every JSON/JSONL row before lowering. If any child timestamp
is earlier than its subagent marker by more than the join epsilon, the complete corpus is interpreted
as relative; otherwise it is interpreted as absolute. This is one corpus-wide heuristic, never a
per-request rewrite. It cannot prove that a corpus is homogeneous: a malformed absolute request can
select relative for the entire corpus, while relative offsets that are all at or above their markers
can select absolute. Producers with ambiguous data should set the basis explicitly. Both conventions
lower uniformly to root-absolute canonical timestamps. The selected basis and whether it was inferred
heuristically or configured are logged; the resolved value is reported as
`weka_nested_timestamp_basis` and included in source identity.
The neutral importer accepts mixed source models and preserves each request's model label in graph
provenance and identity. Execution currently supports one target: before the graph enters
the model-neutral `WorkloadDriver`, AISimulate projects every request onto the one model configured by
`engine.model`. The report records the sorted source-model set, target model, and
`project_to_configured_target` policy under `agentic_model_projection`; per-node heterogeneous timing
models are not supported yet.
The lowering records a zero-based `source_play_ordinal` on every v2 row so materialized graphs retain
deterministic directory and JSONL order; missing ordinals remain valid for older v2 inputs, but an
ordered graph must provide one unique contiguous ordinal for every play.
Without `agentic_profile`, `agentic_lanes: N` starts the first N plays and replenishes each
free client lane from one shared queue in normalized graph order. Plays are not preassigned
to private lane queues, so a later play cannot overtake the next queued play when a lane finishes early.
The next play starts when the current play's client work ends: all authored requests complete on
success, or all dispatched requests become terminal after a failure skips undispatched work. Background requests remain part
of their play even without a parent join. P/D source holds and other server cleanup may outlive this
boundary; they still constrain engine admission and final drain, but do not delay client submission.
Omitting `agentic_lanes` preserves authored timestamp behavior. With `agentic_snapshot` and
`agentic_profile`, completed lanes take replacement plays from a shared sequential corpus cursor,
which wraps at the end of the corpus until the admission deadline. Replacement plays start at turn
zero with fresh request, conversation, play, and cache identities. This opt-in path supports offline
aggregated and P/D vLLM/SGLang Engine replay with HBM-only KV cache, and the
[qualified vLLM G2 configurations](engine/kv-cache.md#agentic-g2), with speculative decoding disabled.
It retains AISimulate's snapshot sampling and warmup frontier behavior; it does not establish complete
AgentX parity. See [continuous agentic profiles](agentic/continuous-profiles.md) for defaults, lifecycle and
idle controls, a runnable example, and the remaining limitations.

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
