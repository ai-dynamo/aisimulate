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
YAML above always resolves the bytes. A rank with `kv_cache_groups` is the
exception. It takes the bytes from its groups, and it refuses
`kv_transfer_bytes_per_token`. The YAML leaves that field out for such a rank.
A native ReplaySpec needs identical `kv_cache_groups` on its prefill and decode
roles.

## Transfer fields

| Knob | Default | Recommend | Rules |
|---|---|---|---|
| `engine.kv_transfer.bytes_per_token` | `auto` | fixed | Positive integer or `auto`. Bytes moved per prompt token. `auto` derives it from the model and the prefill role's TP/PP/MoE shape. It can differ from either worker's `kv_cache.bytes_per_token`. With `state_cache`, `auto` uses the prefill role's resolved `kv_cache.bytes_per_token`. A grouped FPM profile ignores this field and must leave it at `auto`. The transfer then moves what vLLM moves for each cache group. |
| `engine.kv_transfer.bandwidth_gb_per_second` | `null` | fixed | Positive, in decimal GB/s. Bandwidth of each prefill rank's sending link and each decode rank's receiving link; concurrent transfers [share them](#link-sharing). `null` means no transfer delay, not a zero-bandwidth link. |
| `engine.kv_transfer.timing_mode` | `destination_missing` | fixed | `destination_missing` charges only the prompt KV the chosen decode worker does not already hold in its cache. `full_prompt` charges the whole prompt. |

All fields accept only concrete values; `recommend` does not search them.
`kv_transfer` is rejected in `aggregated` and `afd` modes.

A transfer alone on its links, with both roles at the same bandwidth, takes, in seconds,

```text
(transferred_tokens × bytes_per_token + state_bytes) / (bandwidth_gb_per_second × 10^9)
```

where `transferred_tokens` follows `timing_mode`. With
`destination_missing`, a decode worker that already caches a shared prefix
receives the request sooner, so prefix reuse on the decode side can lower TTFT
even when compute time does not change. `state_bytes` is the prefill role's
`state_cache.bytes_per_request` for
[hybrid-KV models](kv-cache.md#state-cache), and 0 otherwise.

For a [grouped profile](kv-cache.md#other-cache-modes) the transferred bytes
are `full_blocks × bytes_per_full_block + fixed_bytes`. `full_blocks` is the
number of full-attention blocks of the transferred prompt, a partly filled last
block included. When the prefill stops one token short, the transferred prompt
leaves out the last prompt token, as the [request flow](#handoff) shows.
`fixed_bytes` holds one block of each state group and the window blocks of
each group with a `sliding_window`. Such a group moves only the blocks of the
transferred prompt inside the window of the next token. The decode worker
allocates only these blocks. With `T` transferred tokens, that is
`cdiv(T, block) - floor(max(0, T - window + 1) / block)` blocks. `block` is the
group's own `block_size_tokens`. The bytes of one block come from the profile's
`transfer_bytes_per_block`. `bytes_per_full_block` adds them up over the
full-attention groups. With `destination_missing`, a decode worker that already
caches a prefix of the prompt does not receive the full-attention blocks of
that prefix again. The state moves with every transfer that moves at least one
token. These are the bytes that vLLM reads for the request, not the bytes on
the wire. On vLLM 0.31.0 the tail ring of GLM-5.3-Flash can read a whole
attention row, and no version's wire overhead is modeled.

Replay differs from vLLM in these known ways for a grouped profile:

- When the decode worker already caches every full-attention block of the
  transferred prompt, vLLM moves nothing, but replay still charges
  `fixed_bytes`. Only a prefill that stops one token short can reach this case.
  It needs a prompt of `N` tokens whose first `N - 1` tokens fill whole
  full-attention blocks, all cached on the decode worker. If the decode worker
  lacks the state of those tokens, vLLM computes the prompt again from the hit
  that every group agrees on. Replay computes one token.
- `fixed_bytes` charges every block of a sliding-window group inside the
  window, whatever the decode worker caches. vLLM moves only the window blocks
  that the decode worker does not cache. GLM-5.3-Flash has no sliding-window
  group.

The decode-side differences under
[grouped caches](kv-cache.md#other-cache-modes) can also change the
transferred blocks.

<a id="link-sharing"></a>

### Link sharing

Each prefill rank sends over one link and each decode rank receives over one
link, both at `bandwidth_gb_per_second`. Transfers in flight at the same time
share these links max-min fairly: a link's bandwidth is split equally among
its transfers, and bandwidth a transfer cannot use because its other link is
the tighter one goes to that link's remaining transfers.

For example, two 1-second transfers from different prefill ranks into one
decode rank both finish after 2 seconds. Sent to two decode ranks, they both
finish after 1 second. Rates are recomputed whenever a transfer starts,
finishes or is canceled, so a transfer can speed up or slow down mid-flight.

A transfer keeps its share until its bytes arrive. Canceling a request releases
its share once the cancellation completes on both sides.

A rank is one attention-data-parallel rank of a worker. YAML gives both roles
the same bandwidth. A native ReplaySpec can set the roles'
`kv_transfer_bandwidth` separately; the slower link then limits even a transfer
alone on its links. A decode role that omits it leaves receiving links
unlimited, so only the prefill ranks' sending links are shared.

<a id="handoff"></a>

## Request flow

1. The prefill worker runs the prompt. A grouped profile with recurrent state
   layers stops the prefill one token short, as vLLM does. The prefill of a
   one-token prompt computes that token, and the decode worker computes it
   again.
2. The request is handed to a decode worker at the same virtual time. KV
   transfer starts, and the prefill worker's copy stays reserved until the
   transfer finishes.
3. After the transfer completes, the decode worker admits the request. Its
   first step produces the first token. On vLLM, that step computes the last
   prompt token. Later steps generate the remaining tokens.

The reported TTFT includes prefill queueing, prefill compute, the transfer, the
decode worker's queue and its first step.
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

- The transfer is a single request-level event. Per-layer streaming and KV
  layout conversion between different TP shapes are not modeled.
- Links are per rank. Ranks that share a NIC or a switch uplink do not
  contend beyond their own links.
- vLLM P/D can be combined with G2 host offload; see
  [KV cache](kv-cache.md#host-offload-g2). G3 offload is aggregated-only.
