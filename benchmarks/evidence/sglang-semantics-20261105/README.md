<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
SPDX-License-Identifier: Apache-2.0
-->

# SGLang scheduler semantic comparison

This compares upstream AISimulate main `2effbae7688b7916e206125bed6c2d63a93bb756`
with the scheduler changes at `b8621b4af777292f55597ea16f6d526211f86daf`, using an identical native
scheduler probe on both revisions. The only changes to the base checkout were
the probe and its test-module declaration. The head measurement includes the
partial-restoration progress fix and precedes this evidence-only commit.
`provenance.json` records revisions and hashes; `base.json` and `head.json`
contain the full observations. All tokens and requests are synthetic.

The oracle is a set of hand-derived contracts from public SGLang v0.5.22 at
[`baa9befa381511cd4d56f83980951e5a45fffc6d`](https://github.com/sgl-project/sglang/tree/baa9befa381511cd4d56f83980951e5a45fffc6d).
We do not execute SGLang or compare GPU inference output. Fixed timing
(prefill 10 ms, decode 2 ms) isolates scheduler behavior; these are simulated
completion times, not wall-clock performance measurements.

## Results

| Contract and synthetic input | Upstream main | Changed scheduler |
| --- | --- | --- |
| P/D: capacity 4096, four 1000-token prompts, default 512-token reserve | Reserves all four; 96 tokens free | Reserves two; 2096 tokens free |
| Arrival: capacity 1008, page 16, prompt 991/992/1001, no outputs | 991/992 finish; 1001 remains queued after 16 passes | 991 finishes at 10 ms; 992/1001 reject at 0 ms |
| Generation: prompt 900, requested outputs 100, capacity 1008, page 16 | No admission within 128 passes | Clips to 79 outputs, completes at 166 ms |
| Computed KV: prompt 127, output 1, page 16 | Some modes cache 128 tokens | All four modes cache and reuse 112 tokens |
| Computed KV: prompt 100, output 28, page 16 | Caches 128 tokens | Caches and reuses 112 tokens |
| Computed KV: prompt 100, output 29, page 16 | Caches 128 tokens | Caches and reuses 128 tokens |
| Non-speculative overlap, each output length above | One extra 2 ms decode tick | Finishing row is filtered before another forward |
| P/D decode caching, default settings, prompt 127, output 1 | 112 cached tokens | No cached tokens; all 2048 pool tokens free |
| LPM queue: active 24-token continuation, then warm 32-token follower; chunk 8 | Follower finishes first: IDs 3 at 60 ms, 2 at 80 ms | Continuation finishes first: IDs 2 at 70 ms, 3 at 80 ms |
| Ratio set to 0.3: three blocked passes, then cancel/drain | Decays on every pass, ending at about 0.295987 | Stays 0.3 while blocked; resets to 0.7 after drain |

Completion cases span overlap on/off, speculation on/off, and output lengths
1/28/29 (12 combinations). Every accepted head execution completes exactly
once with its expected output count; rejection cases emit one terminal and no
outputs. Cached-prefix probes are followed by another request and check actual
admission reuse, not just the radix lookup. All measured head execution cases
drain, with finite times, valid ratios, and zero unexpected pressure events.
The base's bounded queues are observations at the stated limits, not claims of
permanent deadlock.

## Independent expectations

- **P/D reserve:** each held 1000-token prompt protects a 512-token decode
  reserve, so `floor(4096 / (1000 + 512)) = 2` reservations fit. The reserve
  constrains admission but is not physically allocated with the prompt.
  See [`decode.py`](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/disaggregation/decode.py).
- **Arrival limit:** `min(capacity - 1 - 5, capacity - page) = 992`; inputs
  at or above the limit reject. **Generation limit:** take the minimum of
  the requested length, `max_req_len - input - 1`, and
  `capacity - ceil(input / page) * page - page - 1`. With these inputs this
  is `min(100, 106, 79) = 79`. See
  [`scheduler.py`](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/managers/scheduler.py)
  (`init_req_max_new_tokens`) and
  [`allocator/base.py`](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/mem_cache/allocator/base.py#L88-L105).
- **Computed KV:** the last sampled output has not been forwarded. Cacheable
  tokens are `floor((prompt + outputs - 1) / page) * page`. Ordinary completion
  takes `10 + 2 * (outputs - 1)` ms. With two always-accepted draft tokens,
  it takes `10 + 2 * ceil((outputs - 1) / 3)` ms, plus one observed speculative
  tail tick under overlap in this model. See
  [`schedule_batch.py`](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/managers/schedule_batch.py)
  and [`batch_result_processor.py`](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/managers/scheduler_components/batch_result_processor.py).
- **Queue and ratio:** a chunk continuation keeps priority over a new prefix
  hit; blocked idle passes preserve the ratio and a fully drained scheduler
  resets it. See
  [`schedule_policy.py`](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/managers/schedule_policy.py)
  and [`new_token_ratio_tracker.py`](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/managers/scheduler_components/new_token_ratio_tracker.py).

`verify.py` evaluates these formulas separately from the Rust implementation
and checks the measured anomalies and completion accounting. Attribution,
license and adaptation details are in the root `THIRD_PARTY_NOTICES.md` and its
identical packaged copy.

## Reproduce

On each revision, use the same
`crates/core/src/engine/scheduler/sglang/semantic_evidence.rs` from this evidence
commit. On the base, add `#[cfg(test)] mod semantic_evidence;` to the adjacent
`mod.rs`; do not copy any production scheduler code.

From that checkout's root, with its normal Rust toolchain:

```sh
AISIM_SEMANTIC_REPORT=/absolute/path/report.json \
  cargo test -p aisim-core --lib scheduler_semantic_matrix -- --ignored
```

Compare the two outputs with the project Python environment:

```sh
python/aisimulate/.venv/bin/python \
  benchmarks/evidence/sglang-semantics-20261105/verify.py \
  --base /absolute/path/base.json --head /absolute/path/head.json
```

Omitting the arguments validates the committed reports. The probe is ignored
by default because it writes an explicitly selected report and is intended for
manual comparisons; ordinary scheduler regressions remain in the normal suite.

The separate `partially_restored_decode_requests_make_progress` regression
uses native destination reservation/activation: four 8-token prompts request
176 outputs each in a 256-token pool with reserve 16. It fails the bounded
drain assertion on pre-repair PR head `145c852dde45a0ff9e7d30ea81efd7c907071eb7`.
After the repair, all four finish exactly once with all outputs, and all 256
KV tokens are released. That regression tests restoration pressure; the matrix
above compares the wider scheduler changes with upstream main.

This evidence does not measure real-model accuracy, EOS behavior, peer KVCR,
HiCache policies, or speculative P/D first-token transfer. Integrated HiCache
and prior CPU-profile evidence live in `sglang-fidelity-20261105`; their recorded
revisions remain the authority for those results.
