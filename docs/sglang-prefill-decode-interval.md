<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SGLang prefill/decode interval

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

## Scheduling contract

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

## Compatibility and scope

`prefill_schedule_interval` remains vLLM's separate attention-DP cadence setting
with default **1**. It is not an alias for SGLang's field. Nondefault values on
the wrong backend now fail validation instead of being silently ignored. Neutral
defaults (`prefill_schedule_interval: 1`, `prefill_decode_interval: 0`) remain
valid on every backend.

Custom Rust drivers that exhaustively match `SameTimestampRetry` must handle the
new `Countdown { remaining }` variant. It reports the blocked rounds remaining
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

## Behavior source and validation

This is an independent implementation and original test suite based on SGLang's
scheduling behavior at immutable revision
`20621aa14bda7726a8a968f326198eac61717fef`:

- [Counter, gate and arming in scheduler.py](https://github.com/sgl-project/sglang/blob/20621aa14bda7726a8a968f326198eac61717fef/python/sglang/srt/managers/scheduler.py#L1211)
- [Prefill selection and post-sync arming](https://github.com/sgl-project/sglang/blob/20621aa14bda7726a8a968f326198eac61717fef/python/sglang/srt/managers/scheduler.py#L3201)
- [Global EXTEND reduction](https://github.com/sgl-project/sglang/blob/20621aa14bda7726a8a968f326198eac61717fef/python/sglang/srt/managers/scheduler_components/dp_attn.py#L194)
- [Chunk continuation and the separate delayer](https://github.com/sgl-project/sglang/blob/20621aa14bda7726a8a968f326198eac61717fef/python/sglang/srt/managers/schedule_policy.py#L1022)

The upstream feature was introduced in
[commit 6f69f927da9e5692bb4709821faecff6be9b5a8a](https://github.com/sgl-project/sglang/commit/6f69f927da9e5692bb4709821faecff6be9b5a8a)
([PR #35017](https://github.com/sgl-project/sglang/pull/35017)). These references
record behavior provenance; no upstream source or tests are copied or translated.

Deterministic Rust tests cover defaults, exact round boundaries, successive
chunks, decode opportunities, speculation, asymmetric DP, idle tails, cache-only
completion, intervals above 1024, and impossible-request detection. Python tests
exercise public YAML validation, compilation, Runner forwarding and native
fixed-timing execution. Run Python tests against a freshly built extension from
the same worktree; on macOS pass `-p no:timeout`.
