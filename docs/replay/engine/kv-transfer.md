<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# P/D KV transfer

In `mode: disaggregated`, each request is prefilled by a `prefill` worker and
then decoded by a `decode` worker. `engine.kv_transfer` sets how long moving the
prompt's KV cache between them takes. Add it to a disaggregated configuration
such as the [P/D example](README.md#a-complete-pd-example):

```yaml
engine:
  mode: disaggregated
  kv_transfer:
    bytes_per_token: auto
    bandwidth_gb_per_second: 50
    timing_mode: destination_missing
```

Omitting `kv_transfer` is the same as `{}`: the transfer happens with no delay.
A native ReplaySpec or Sweeper search space whose prefill rank sets a positive
`kv_transfer_bandwidth` without `kv_transfer_bytes_per_token` is rejected; the
YAML above always resolves the bytes.

## Transfer fields

| Knob | Default | Recommend | Rules |
|---|---|---|---|
| `engine.kv_transfer.bytes_per_token` | `auto` | fixed | Positive integer or `auto`. Bytes moved per prompt token. `auto` derives it from the model and the prefill role's TP/PP/MoE shape. It can differ from either worker's `kv_cache.bytes_per_token`. With `state_cache`, `auto` uses the prefill role's resolved `kv_cache.bytes_per_token`. |
| `engine.kv_transfer.bandwidth_gb_per_second` | `null` | fixed | Positive, in decimal GB/s. `null` means no transfer delay, not a zero-bandwidth link. |
| `engine.kv_transfer.timing_mode` | `destination_missing` | fixed | `destination_missing` charges only the prompt KV the chosen decode worker does not already hold in its cache. `full_prompt` charges the whole prompt. |
| `engine.kv_transfer.prefill_backoff_tokens` | `0` | fixed | Nonnegative prompt suffix length left for decode-side computation. vLLM P/D only. Set from the pinned connector contract; both roles receive the same value. A prompt must be longer than a positive backoff. |
| `engine.kv_transfer.decode_recompute_tokens` | `0` | fixed | Minimum prompt suffix recomputed on decode, capped at prompt length. vLLM P/D only. The effective suffix is the larger of this value and `prefill_backoff_tokens`; it does not shorten source computation or transfer. With recurrent `state_cache`, it cannot exceed the source backoff. |

All fields accept only concrete values; `recommend` does not search them.
`kv_transfer` is rejected in `aggregated` and `afd` modes.

Transfer time in seconds is

```text
(transferred_tokens × bytes_per_token + state_bytes) / (bandwidth_gb_per_second × 10^9)
```

where `transferred_tokens` follows `timing_mode`. With
`destination_missing`, a decode worker that already caches a shared prefix
receives the request sooner, so prefix reuse on the decode side can lower TTFT
even when compute time does not change. `state_bytes` is the prefill role's
`state_cache.bytes_per_request` for
[hybrid-KV models](kv-cache.md#state-cache), plus any automatically resolved
DSpark draft KV described below. It is 0 when neither applies.

<a id="handoff"></a>

## Request flow

1. The prefill worker runs the prompt (minus `prefill_backoff_tokens`) and
   produces a source-side sample. Replay does not count that sample as client
   output.
2. The request is handed to a decode worker at the same virtual time. KV
   transfer starts, and the prefill worker's copy stays reserved until the
   transfer finishes.
3. After the transfer completes, the decode worker admits the original request,
   computes any remaining prompt suffix, and generates all requested client
   output tokens.

The reported TTFT includes prefill queueing, prefill compute, the transfer,
decode admission and computation through the first client-visible token.
Transfer events share one virtual clock with compute and arrivals.

Backends order the handoff differently:

| Backend | Order |
| --- | --- |
| vLLM, TensorRT-LLM | Source-first: the prefill side finishes and then the decode side reserves space. |
| SGLang | Destination-first: the decode side reserves space before the transfer. |

TensorRT-LLM decode workers use `GUARANTEED_NO_EVICT` scheduling, which reserves
room for the whole output while transferred KV is held.

The two roles may use different `attention_data` sizes. A request moves from one
concrete prefill rank to one concrete decode rank. A trace request's
`prefill_dp_rank` chooses the prefill rank; if omitted, its `dp_rank` is used
for both roles.

## Limitations

- The transfer is a single request-level event. Per-layer streaming, KV
  layout conversion between different TP shapes, and network contention
  between concurrent transfers are not modeled.
- Bandwidth applies to each transfer independently.
- vLLM P/D can be combined with G2 host offload; see
  [KV cache](kv-cache.md#host-offload-g2). G3 offload is aggregated-only.

## DSpark state

With DSpark and automatic transfer geometry, the source prepares draft context
KV during prefill. Handoff adds its rank-local footprint once:
`min(prompt_tokens, draft_window) × draft_bytes_per_token` for windowed drafts,
or the full context for full-attention drafts. Target-only prefix hits do not
prove a draft-cache hit, so the full draft window is charged conservatively.
The decode worker reuses the transferred prefix and prepares only its configured
recomputed suffix. A numeric
`bytes_per_token` is an explicit total-transfer override and must include any
draft state; automatic draft bytes are not added to it.

This models a connector that transfers prepared draft KV alongside target KV.
Frameworks that rebuild draft state or transfer auxiliary hidden states require
a different handoff contract and are not qualified by this approximation.
The controlled 1P1D validation must establish the connector's actual contract.

For the campaign's pinned vLLM revision
`ac9126e58aa7bbab1856ba6593ba4d5003fea516`, NIXL registers target and draft KV
groups. Its DSpark prefill backoff is **1 for Kimi-K3's recurrent state** and
**0 for V4/V4.1**. Configure Kimi with `prefill_backoff_tokens: 1`. Separately,
the destination scheduler recomputes a prompt suffix to recover output logits
or bounded decoder state: set `decode_recompute_tokens: 1` for V4/Kimi and
`decode_recompute_tokens: 128` for V4.1 with decoder replay enabled. Replay does
not infer a connector version from a model name. The transferred recurrent
state and reusable blocks cover only the source-completed prefix. A block
containing a suffix not computed on the source stays private until decode
finishes it, so another request cannot reuse unfinished KV. Fully transferred
blocks can remain reusable even when decode recomputes them for logits.
Source preparation and transfer use the shortened
prompt. Destination recomputation can extend farther back for stateless models,
but never before the received recurrent-state boundary. These are separate
controls because a transferred token can still require recomputation for logits.
For short prompts, destination recomputation covers the entire prompt.
For `destination_missing`, the target-KV payload excludes that suffix even
when its destination block is newly allocated. Without speculation, the
suffix forward produces the first output directly; ongoing decoders in a
mixed step still contribute their own decode cost.

These settings model prompt boundaries, not a complete connector simulator.
Physical block transfer padding,
heterogeneous-TP redistribution and transport overlap still require matched
runtime evidence.
