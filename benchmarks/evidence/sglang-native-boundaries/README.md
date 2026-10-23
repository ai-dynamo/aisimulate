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
