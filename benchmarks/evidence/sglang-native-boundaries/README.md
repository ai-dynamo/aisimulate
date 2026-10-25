<!--
SPDX-License-Identifier: Apache-2.0
-->

# Selected SGLang native behavior boundaries

These synthetic behavior comparisons used native SGLang 0.5.22 at
`baa9befa381511cd4d56f83980951e5a45fffc6d` and AISimulate at
`a0a35e70937a1f680bbb30ca7404cc9cb9b65e4e` on October 22–23, 2026.
The fix is based on main at `aeb05094a243dc0a054731ac11e63d1af49b6e89`.
Both sides used 128-token pages, a 4096-token KV pool per attention-DP rank,
a 4096-token prefill chunk and schedule conservativeness 0.3. Native TP16 /
attention-DP2 requests were explicitly routed to DP0. CPU regressions isolate
that rank; they do not model inter-rank performance.

| Boundary | Native observation | Original simulator observation | Regression |
| --- | --- | --- | --- |
| Prefill cap 64, input 128, output 1, ample KV | EXTEND128 completes | No admission within a 1000-pass driver bound | `native_chunked_prefill_progresses_above_the_input_cap` |
| Cache A2048, then disjoint B2560, then A2048 again; prefill cap 16384 | B evicts A's whole 2048-token leaf; final A reuses 0 | B first evicts only 512; final A reuses 1536 | `native_lru_pressure_does_not_reuse_a_partially_evicted_leaf` |
| Two queued disjoint inputs1792/output288, prefill cap16384, ignore_eos=true | Second request waits for the first to finish; 0 retractions | Second request admitted on next pass; 1 later retraction | `native_fixed_output_reservation_keeps_the_second_request_queued` |

The fixed-output regression also exercises `ignore_eos=false` and preserves
the original discounted reservation and its pressure outcome. The public
Replay regression round-trips both settings through JSON and forwards them
through the engine adapter. Legacy JSON without the field defaults to false.
Belady keeps page-wise tail eviction; whole-leaf LRU does not replace that policy.

Native arrival order and effective conservativeness were checked; earlier
simulator runs at conservativeness 1.0 were excluded. The original simulator
used **fixed 1ms per forward**, solely to drive scheduling. No TTFT, throughput,
whole-model accuracy percentage or production-frequency claim follows from
these boundary tests. Native per-request page-overhead accounting and other
version differences still need separate parity work; this change does not
claim whole-scheduler equivalence.

The numerical assertions come from these controlled token/page geometries,
not a learned timing model. Tests generate fresh synthetic integer token
sequences, with no production prompts, outputs, model weights or credentials.

Related work: [native radix-core integration #363](https://github.com/ai-dynamo/aisimulate/pull/363)
also proposes whole-leaf LRU as part of a larger cache replacement. This patch
aligns the current implementation without adding a native-tree dependency.

## Review follow-up: input-cap crossing

The native input-cap rejection before admission is explicitly guarded by
`rem_chunk_tokens is None`. With chunking enabled, native selection admits
the candidate and `budget_state()` stops further selection after the cap is
exhausted. This also applies when the batch already contains work: cap256 and
inputs128/256/128 select the first two inputs (384 tokens), then stop. Adding
an empty-batch-only exception would disagree with the pinned native version.

`probe_input_cap.py` executes `_select_prefill_admission` and `budget_state`
directly from the immutable native Git blob, with sufficient-capacity and
tile gates supplied by a CPU harness. It is a source-level control check,
not a new full-server measurement. It includes a non-chunked negative control.

```sh
python benchmarks/evidence/sglang-native-boundaries/probe_input_cap.py \
  --sglang-checkout /path/to/sglang
```

The expected source SHA-256 is
`03ba021fdd225cc4e7d0d5eb822a725fe6e3907e40157ec6461390f23b7f2cb9`.
Source and license: [SGLang schedule_policy.py](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/managers/schedule_policy.py),
Copyright SGLang Team and contributors, Apache-2.0. The harness is a modified
behavioral adaptation; native serving implementation is not vendored.

## Review follow-up: disaggregated consumers

`disagg_adapter_preserves_fixed_output_reservation_at_the_decode_consumer`
passes both flag values through the production `direct_to_native` conversion
and a real SGLang **decode worker**, asserting the second admission and final
retraction count. It exercises fresh-KV admission, which a decode worker also
uses after retraction. This is not a mock or a field-only assertion.

`disaggregated_replay_accepts_fixed_output_flags_through_both_handoff_roles`
additionally round-trips the authored JSON and runs complete destination-first
PD handoffs for both values, checking routing, output lengths and retractions.
PD destination reservation currently reserves physical prompt pages without
the new-token discount. Both values therefore have the same initial destination
admission in that fixture. The test does not invent a flag-sensitive destination
policy or claim to fix the separately documented native PD headroom gap.

Mutation check: replacing `ignore_eos: request.ignore_eos` with
`ignore_eos: false` in `direct_to_native` makes the real-consumer regression
fail on its second-admission assertion. The production conversion was restored
before rerunning the normal suite. This verifies that the regression detects
metadata loss at that adapter rather than merely exercising the unchanged
physical destination-reservation behavior.

## Dynamo adoption dependency

The repository-local `scripts/performance/build_simulation_dynamo.py` bridge
for pinned Dynamo `def3b79` now forwards the core-owned `DirectRequest.ignore_eos`
into native `Request`. Its current-API regression checks the generated conversion;
incompatible types, visibility and later source drift still fail before any write.
This CI build bridge does not update Dynamo's SGLang HTTP parser or released pin.

At [Dynamo 123d667](https://github.com/ai-dynamo/dynamo/tree/123d667df9d8f85a7859c52dba5761135d4d0452),
`Cargo.toml` pins `aisimulate-core =0.13.0-dev.202610160000000073`, which does
not contain this new field. This AISimulate change does not silently upgrade
that dependency. Its eventual release/pin update must include both adapters:

| Consumer | Required change when the dependency advances |
| --- | --- |
| `lib/mocker/servers/sglang/src/server_request.rs` | Retain `sampling_params.ignore_eos.unwrap_or(false)` in `PreparedRequest`; copy it explicitly into `DirectRequest.ignore_eos`. The existing EOS-stopping check alone does not preserve reservation metadata. |
| `lib/mocker/src/grouped_scheduler/compatibility.rs` | Forward `DirectRequest.ignore_eos` into native `engine::Request`; update other newly required request members when adopting the newer core. |

The companion server-entrypoint regression should submit otherwise identical
requests with omitted/false/true flags, verify the parsed flag reaches the
core conversion, then compare admission/retraction on the real consumer.
For decode-role PD handoffs, test metadata preservation independently of the
physical destination-reservation policy, as above. A downstream patch that
uses the field before upgrading the exact core pin would not compile; that
cross-repository release work remains explicitly pending rather than being
reported as completed by this PR.
