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
| `engine.kv_transfer.bandwidth_gb_per_second` | `null` | fixed | Positive, in decimal GB/s. Bandwidth of each prefill rank's sending link and each decode rank's receiving link; concurrent transfers [share them](#link-sharing). `null` means no transfer delay, not a zero-bandwidth link. |
| `engine.kv_transfer.timing_mode` | `destination_missing` | fixed | `destination_missing` charges prompt KV absent from the chosen decode worker's cache; SGLang bills whole uncached pages, including unused tail slots. `full_prompt` charges the logical prompt length without page rounding. |

All fields accept only concrete values; `recommend` does not search them.
`kv_transfer` is rejected in `aggregated` and `afd` modes.

A transfer alone on its links, with both roles at the same bandwidth, takes, in seconds,

```text
(transferred_tokens × bytes_per_token + state_bytes) / (bandwidth_gb_per_second × 10^9)
```

where `transferred_tokens` follows `timing_mode`. For SGLang,
`destination_missing` uses the number of uncached physical prompt pages multiplied
by the decode worker's `kv_cache.block_size`. This includes unused slots in the
last page. `full_prompt` uses the original logical prompt length, so
`destination_missing` can cost more than `full_prompt` even without a cache hit.

For example, a 10-token prompt with SGLang page size 4 occupies three pages.
With 16 bytes per token, bandwidth `0.000001` GB/s, no state bytes and no other
transfers, `full_prompt` bills 10 token slots (160 ms), while a cold
`destination_missing` transfer bills 12 slots (192 ms). If the decode cache
already holds one reusable four-token page, two pages remain: 8 slots (128 ms).
A shared six-token prefix still reuses only one complete page in this example.

Within `destination_missing`, a decode worker that already caches a shared prefix
receives the request sooner, so prefix reuse on the decode side can lower TTFT
even when compute time does not change. `state_bytes` is the prefill role's
`state_cache.bytes_per_request` for
[hybrid-KV models](kv-cache.md#state-cache), and 0 otherwise.

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

1. The prefill worker runs the prompt and produces the first token.
2. The request is handed to a decode worker at the same virtual time. KV
   transfer starts, and the prefill worker's copy stays reserved until the
   transfer finishes.
3. After the transfer completes, the decode worker admits the request and
   generates the remaining tokens.

The reported TTFT includes prefill queueing, prefill compute and the transfer.
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
