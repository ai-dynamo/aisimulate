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

## Transfer fields

| Knob | Default | Recommend | Rules |
|---|---|---|---|
| `engine.kv_transfer.bytes_per_token` | `auto` | fixed | Positive integer or `auto`. Bytes moved per prompt token. `auto` derives it from the model and the prefill role's TP/PP/MoE shape. It can differ from either worker's `kv_cache.bytes_per_token`. |
| `engine.kv_transfer.bandwidth_gb_per_second` | `null` | fixed | Positive, in decimal GB/s. `null` means no transfer delay, not a zero-bandwidth link. |
| `engine.kv_transfer.timing_mode` | `destination_missing` | fixed | `destination_missing` charges only the prompt KV the chosen decode worker does not already hold in its cache. `full_prompt` charges the whole prompt. |

All fields accept only concrete values; `recommend` does not search them.
`kv_transfer` is rejected in `aggregated` and `afd` modes.

Transfer time in seconds is

```text
transferred_tokens × bytes_per_token / (bandwidth_gb_per_second × 10^9)
```

where `transferred_tokens` follows `timing_mode`. With
`destination_missing`, a decode worker that already caches a shared prefix
receives the request sooner, so prefix reuse on the decode side can lower TTFT
even when compute time does not change.

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

- The transfer is a single request-level event. Per-layer streaming, KV
  layout conversion between different TP shapes, and network contention
  between concurrent transfers are not modeled.
- Bandwidth applies to each transfer independently.
- vLLM P/D can be combined with G2 host offload; see
  [KV cache](kv-cache.md#host-offload-g2). G3 offload is aggregated-only.
