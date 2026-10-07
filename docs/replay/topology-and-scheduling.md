<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Topology and scheduling

## Aggregated workers and attention-DP

One or more logical workers use the same aggregated runtime. For attention DP
above one, each worker owns a scheduler per rank. Placement retains a concrete
`(worker_id, dp_rank)` identity; replica accounting and scaling count logical
workers, not rank schedulers. Each ready rank selects its own pass, then the
worker completes at the maximum rank latency. Tokens, KV observations and
forward-pass timing become visible at this boundary. Empty ranks also wait,
so arrivals during an epoch cannot start early.

The event loop advances directly to the next arrival, engine completion,
transfer, readiness transition, or active telemetry/scaling tick. It settles
all work at that timestamp before advancing again. Same-time policy decisions
cannot expose compute or transferred KV before completion.

## Disaggregated prefill and decode

P/D uses separate worker pools and one logical clock. vLLM and TensorRT-LLM
use source-first handoff; SGLang uses destination-first handoff. TensorRT-LLM
uses `GUARANTEED_NO_EVICT` to reserve decode completion headroom while the
destination owns transferred prompt KV.

Attention-DP sizes may differ across pools. One request moves from a concrete
prefill `(worker, dp_rank)` to a concrete decode `(worker, dp_rank)`, while KV
transfer is one aggregate request-level event. Rank-wise layout conversion
and network contention are not modeled. Token-only vLLM P/D can use
[G2 host offload](cache.md); AgentX G2 has a narrower static 1P1D/DP1 boundary.

`dp_rank` selects the aggregated or decode rank. `prefill_dp_rank` optionally
selects the prefill rank; when omitted in P/D it falls back to `dp_rank`.
Each hint is validated against its role's independent DP size. SGLang roles
normalize chunked-prefill size by DP size and scale scheduling conservativeness
by `0.3` before constructing per-rank schedulers.

Prefill runs a hidden one-token bootstrap. On completion Replay applies its
KV observations, releases prefill placement state, and starts the original
request's decode path at the same logical time. The public request report is
decode-visible, so TTFT includes prefill queueing and compute. Source holds,
destination reservations and transfer completion remain distinct from a
client terminal. Decode admission with G2 prioritizes materialized handoff
requests and cannot overwrite G1 blocks still being stored to G2.

## KV transfer timing

`engine.kv_transfer` is valid only for disaggregated mode. Its fields are
concrete-only in both prediction and recommendation; search domains on these
fields are rejected.

| Field | Default | Semantics |
| --- | --- | --- |
| `bytes_per_token` | `auto` | Positive bytes per transferred token, derived from the prefill/source role's TP/PP/MoE shape when automatic. This link payload can differ from each worker's physical KV-cache bytes per token. |
| `bandwidth_gb_per_second` | `null` | Positive bandwidth when specified. `null` disables modeled transfer delay; it does not mean a zero-bandwidth link. |
| `timing_mode` | `destination_missing` | `full_prompt` charges for the full prompt KV footprint; `destination_missing` charges only the prompt KV absent from the selected decode worker. |

The transfer event participates in the same virtual-time loop as compute and
arrivals. Cache overlap and link bandwidth can therefore affect TTFT even
when prefill/decode compute timing is fixed. Rank-wise layout conversion and
network contention remain outside this request-level model.

## Placement and scaling

The engine stack uses round-robin placement and fixed workers. The
[Dynamo composition](dynamo.md) owns KV Router and Planner policy. Native
[composition contracts](../adapters/native-composition.md) govern lifecycle
notifications, request terminals and scaling ticks; Replay owns virtual time.
Dynamo's offline decode routing is overlap-blind, while prefill can publish KV
observations to its placement policy. This is an in-process model, not a live
networked Router or indexer.

## vLLM prefill scheduling cadence

vLLM can throttle prefill scheduling in data-parallel deployments so decode
steps remain balanced across ranks. Set the interval in an `aisimulate predict`
configuration under the scheduler for each affected worker role:

```yaml
engine:
  mode: aggregated
  model: nvidia/Kimi-K2.5-NVFP4
  hardware: b200_sxm
  backend: vllm
  workers:
    aggregated:
      parallelism:
        attention_data: 8
      scheduler:
        prefill_schedule_interval: 4
```

```bash
aisimulate predict --config prediction.yaml
```

The Python compiler lowers this field to the rank-level engine setting used by
the replay runtime. Lower-level Runner callers can set the same field directly:

```json
{
  "engine": {
    "dp_size": 4,
    "rank": {
      "backend": "vllm",
      "prefill_schedule_interval": 4
    }
  }
}
```

The default is `1`, which preserves the previous scheduling behavior. Values
above one take effect only for vLLM attention-DP groups. On a non-aligned group
step, local prefill work with more than one token remaining waits while decodes
continue. Connector loads, materialized requests, and requests with at most one
prefill token remaining can still advance. Throttling is temporarily released
when a non-preempting aligned step left queued requests due to scheduler
capacity.

SGLang has a separate
[`prefill_decode_interval`](#sglang-prefilldecode-interval), defaulting to zero.
Nondefault values of either field on the wrong backend fail validation.

The shared counter resets as soon as AISimulate observes that the full DP group
has drained, including after cancellation and internal-work transitions. vLLM
checks global unfinished state every 32 steps and may run a dummy tail before
resetting. AISimulate does not model that collective tail, so a request arriving
during the upstream tail can observe a different cadence phase.

This follows vLLM's `prefill_schedule_interval` scheduler behavior at commit
`e2fa28594f7baad142a426b0b6a2cfe2c79201c7`.


## SGLang prefill/decode interval

Set `prefill_decode_interval` under each affected worker's scheduler in an
`aisimulate predict` configuration:

```yaml
engine:
  mode: aggregated
  model: nvidia/Kimi-K2.5-NVFP4
  hardware: b200_sxm
  backend: sglang
  workers:
    aggregated:
      parallelism:
        attention_data: 8
      scheduler:
        prefill_decode_interval: 20
```

```bash
aisimulate predict --config prediction.yaml
```

Lower-level Runner specifications set `engine.rank.prefill_decode_interval`;
Rust callers set `EngineConfig::prefill_decode_interval`:

```json
{
  "engine": {
    "dp_size": 8,
    "rank": {
      "backend": "sglang",
      "prefill_decode_interval": 20
    }
  }
}
```

For disaggregated workers, configure the `prefill` and `decode` roles separately.
Each worker's attention-DP group owns its interval. The Python compiler forwards
the public field through Runner to the Rust scheduler; the execution spec retains
the effective value.

### Scheduling contract

The default is **0**, which preserves existing scheduling. A positive integer
`N` blocks prefill for exactly the next `N` scheduler rounds after an EXTEND
forward. For example, `N=2` produces this sequence when more prefill is queued:

| Round | Work eligible on a rank |
| --- | --- |
| 0 | Prefill; arm two blocked rounds |
| 1 | Decode, or idle if there is no running request |
| 2 | Decode, or idle if there is no running request |
| 3 | Prefill; arm another two blocked rounds |

The gate covers both new requests and every continuation chunk of a long prompt.
Running decode requests can advance while prefill waits. A scheduler round can
produce multiple speculative output tokens or execute no GPU work at all; the
counter advances once in either case. Pure decode, idle, and cache-only
zero-output completions do not rearm it. Materialized destination requests can
still enter decode without fresh prefill.

After all ranks select their work, AISimulate combines their EXTEND observations
and arms every rank in the worker's attention-DP group, including idle ranks.
Updating after the whole group prevents the result from depending on rank
iteration order. Each subsequent group round consumes one interval count.

With no running decode, blocked rounds have zero modeled GPU duration. They
still execute and drain the countdown, including after all requests complete.
Replay validates that this finite countdown decreases, allowing intervals above
its ordinary admission-convergence retry limit while retaining livelock checks
for impossible requests. No TTFT offset or artificial GPU time is added.

These interval-only idle rounds do not decay the decode admission ratio. When
the entire DP group is idle, the ratio resets as in upstream `on_idle`; a locally
idle rank preserves its ratio while a peer runs a forward. This prevents an idle
countdown from changing later admission decisions under KV capacity pressure.

### Compatibility and scope

`prefill_schedule_interval` remains vLLM's separate attention-DP cadence setting
with default **1**. It is not an alias for SGLang's field. Nondefault values on
the wrong backend fail validation instead of being silently ignored. Neutral
defaults (`prefill_schedule_interval: 1`, `prefill_decode_interval: 0`) remain
valid on every backend.

Custom Rust drivers that exhaustively match `SameTimestampRetry` must handle the
`Countdown { remaining }` variant. It reports the blocked rounds remaining
after a pass and allows bounded retries without advancing simulated time.

The scheduler capability is selected explicitly by this parameter.
`backend_version` continues to select performance data and does not infer whether
a particular SGLang release branch supports the upstream flag. The interval is
also distinct from SGLang's prefill delayer, whose rejection does not necessarily
block an in-progress chunk.

Existing modeling limits remain: mixed-chunk forwards, prefill-delayer policy,
CPU/collective overhead and optional idle sleeping are not modeled here. In the
speculative attention-DP path, upstream can force a peer to IDLE when another
rank prefills; AISimulate still selects peer decode independently in that same
round. The interval's subsequent blocked rounds are synchronized. The tests
verify this scheduling feature, not the cause or size of any silicon TTFT gap.

### Behavior source and validation

This is an independent implementation and original test suite based on SGLang's
scheduling behavior at immutable revision
`20621aa14bda7726a8a968f326198eac61717fef`:

- [Counter, gate and arming in scheduler.py](https://github.com/sgl-project/sglang/blob/20621aa14bda7726a8a968f326198eac61717fef/python/sglang/srt/managers/scheduler.py#L1211)
- [Prefill selection and post-sync arming](https://github.com/sgl-project/sglang/blob/20621aa14bda7726a8a968f326198eac61717fef/python/sglang/srt/managers/scheduler.py#L3201)
- [Global EXTEND reduction](https://github.com/sgl-project/sglang/blob/20621aa14bda7726a8a968f326198eac61717fef/python/sglang/srt/managers/scheduler_components/dp_attn.py#L194)
- [Chunk continuation and the separate delayer](https://github.com/sgl-project/sglang/blob/20621aa14bda7726a8a968f326198eac61717fef/python/sglang/srt/managers/schedule_policy.py#L1022)

Deterministic Rust tests cover defaults, exact round boundaries, successive
chunks, decode opportunities, speculation, asymmetric DP, idle tails, cache-only
completion, intervals above 1024, and impossible-request detection. Python tests
exercise public YAML validation, compilation, Runner forwarding and native
fixed-timing execution. Run Python tests against a freshly built extension from
the same worktree; on macOS pass `-p no:timeout`.

## Analytical AFD

AFD places attention and FFN/MoE work in separate A and F pools. The Engine
runner uses measured per-layer A, F, A→F and F→A times to construct deterministic
stage intervals. Optimistic cadence is `max(A, F, A_to_F + F_to_A)`, conservative
cadence is `max(A + A_to_F, F + F_to_A)`, and serial cadence sums all stages.
Pipeline fill and every microbatch-layer cadence contribute to the full pass.
A started pass is non-preemptive; completion effects stay hidden until its
modeled boundary, even when a caller wakes late.

Pure `afd` requires `phase: both`; `afd+pd` pairs one AFD phase with an ordinary
opposite-phase companion. Arrival and queue delay contribute to TTFT/E2E;
TPOT covers first-token to completion divided by `OSL - 1`, or zero for OSL1.
Replay requires concrete fixed synthetic ISL/OSL, `random_range_ratio: 1.0`,
absolute load and no trace. There is no scheduler-visible A/F KV capacity for
KV-relative load. AFD and EPD cannot be combined.

Successful AFD predictions write `afd-replay-spec.json` and
`afd-qualification.json`, with a SHA-256 link, complete model/topology provenance,
phase and companion checks, and GPU accounting. `native_deployment_supported:
false` and `launch.supported: false` identify the analytical path. These are
regression artifacts, not runnable backend manifests. See
[search-space contracts](../sweeper/search-space.md).

## Analytical EPD

EPD prepends an analytically estimated encoder pool to native language replay.
Visual tokens are derived from fixed image geometry and added to text input
exactly once. After language replay, the AIC overlay caps throughput at degraded
encoder capacity and adds raw encoder batch latency to mean TTFT and E2E.
Encoder backpressure does not change the language scheduling timeline.

The public result is marked `analytical_epd_overlay`. Duration is a rate-derived
accounting interval, and GPU-hours include encoder and language GPUs; original
language duration stays in provenance. Reports contain aggregate means, not EPD
percentiles, per-request goodput, raw traces or telemetry. Missing encoder power
is unavailable, never zero watts or total-deployment power.

This path requires fixed synthetic text/image inputs, fixed concurrency, static
workers and default op-level language timing. It rejects traces, sessions,
variable lengths, prefix-sharing, arrival-rate/KV-relative loads, whole-forward
FPM, adapters, online execution and per-request capture. Encoder CPU overhead,
embedding transfer, encoder queueing, and deployment generation are outside
this model. See [Sweeper search space](../sweeper/search-space.md) for candidate
and SLA restrictions. The imported model's provenance is
[AIC revision f8f2341](https://github.com/ai-dynamo/aiconfigurator/commit/f8f2341cb5761877bda694ab954cb6f5eff78fd4).

<a id="prompt-lookup-ngram-speculative-decoding"></a>

## Prompt-lookup (ngram) speculative decoding

Both `predict` and `recommend` accept an optional `engine.speculation` block.
For example, add this block under `engine` in a vLLM configuration:

```yaml
speculation:
  kind: ngram
  num_speculative_tokens: 3
  acceptance_rates: [0.8, 0.6, 0.4]
  seed: 42
```

Or override the same configuration from the command line:

```bash
aisimulate predict -c prediction.yaml \
  --set engine.backend=vllm \
  --set 'engine.speculation={kind: ngram, num_speculative_tokens: 3, acceptance_rates: [0.8, 0.6, 0.4], seed: 42}' \
  --output-dir ./ngram-prediction
```

The draft-token count is an integer from 1 to 5, matching the native Replay
sampler's current limit. Supply exactly one conditional acceptance probability
per draft token, each finite and in `[0, 1]`. Entry `i` is the probability of
accepting token `i` given that all preceding draft tokens were accepted.
These are workload assumptions: the example's expected progress per decode
round is `1 + 0.8 + 0.8*0.6 + 0.8*0.6*0.4 = 2.472` tokens. The seed is an
unsigned 64-bit integer (default `42`). Sampling stops at the first rejection
and clips the last burst to the remaining output length.

The existing ngram performance model prices target verification at draft count
plus one, with no draft network, draft weights, or draft KV cache. Replay uses
that iteration cost together with sampled accepted-token progress. It assumes
a lookup draft is available every decode round (`trigger_rate = 1`); actual
prompt/output token matching, mixed drafted/draftless rounds, and host lookup
latency are not modeled. Fixed/polynomial timing overrides still work, but their
decode latency is per verification round and does not estimate ngram costs.
Prompt lookup is separate from `kv_cache.prefix_caching` and AIC's `--prefix N`
cached-prompt assumption.

This interface supports offline engine-stack vLLM aggregated and disaggregated
language workers with operation-level timing. SGLang, TensorRT-LLM, FPM,
AFD/EPD, host/G3 offload, AgentX agentic execution, and Dynamo adapters are not
qualified for this option. MTP/EAGLE uses the separate `engine.nextn` interface described below. Omit `engine.speculation` to disable speculation.

Recommendation pins this block for every candidate; it does not search draft
length or acceptance. Saved prediction YAML retains the block for replay.
Backend deployment artifact generation rejects these candidates until the
ngram runtime flags are supported.

## Native MTP acceptance

The public `engine.nextn` control selects an MTP/EAGLE draft-token count from
0 through 5; nonzero values require an explicit `engine.nextn_accepted` in
`[0, nextn]`. These model inputs are distinct from the ngram specification and
cannot be combined with `engine.speculation`. They remain pinned during a
recommendation. The runner must advertise `supports_mtp_expected_acceptance`.

At native engine level, `aic_nextn` selects draft count and
`aic_nextn_accept_rates` supplies conditional probabilities as a comma-separated
string. `aic_mtp_seed` controls worker-local sampling. One verification forward
can emit up to `aic_nextn + 1` tokens; sampled progress is clipped to the request's
remaining output. Decode speedup must remain 1 because burst sampling already
models the acceleration. The timing query must price verification work rather
than ordinary single-token decode. See [performance-model configuration](../perf-model/configuration.md)
for estimator controls and matching support.

AgentX, grouped caches, G2/G3 offload and fine state-cache prefix matching reject
native speculation. An accepted estimator configuration does not qualify those
runtime combinations. Ngram's narrower backend and deployment-output limits
are stated in its own section above.
