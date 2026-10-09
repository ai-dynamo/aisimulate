<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Replay qualification evidence

Source revision: `acaca5d1`. These are historical, scoped qualification records,
not general support claims. Source documents and full provenance remain
recoverable at the pinned revision.

## G2 scoped validation

## Example

[`examples/cli/shared-g2-predict.yaml`](../../../examples/cli/shared-g2-predict.yaml)
runs 64 requests sharing an 896-token prefix on two replicas with attention
DP 2 and fixed compute timing. With `cluster_shared` the reuse ratio is 0.861
and one 1,024-block pool holds 568 blocks; overriding `scope=dp_rank_local`
gives 0.820, because each DP rank must first compute the prefix itself. Mean
TTFT rises from 20 ms to 30.6 ms in shared scope: fixed compute time does not
shrink with reuse, while restores add H2D time.

## GPU parity

Private G2 (`dp_rank_local`) was compared with native vLLM 0.25.1 serving
MiniMax-M2.5 with TP2 and attention DP 2 on four B200 GPUs, BF16 KV and
32-token blocks. Each arm ran the same 5,000 Mooncake requests on a fixed
schedule. Each DP engine had 8,192 G1 blocks; the G2 arm added 16,384 G2 blocks
per DP engine. The simulator used the existing vLLM 0.24.0 timing profiles and
the saved GPU inputs: prompt token IDs, ingress times and DP routes.

| Arm | Metric | GPU | Simulator | Absolute relative error |
| --- | --- | ---: | ---: | ---: |
| G1 only | G1 reuse | 25.7605% | 25.6863% | 0.29% |
| | Mean TTFT | 770.268 s | 666.276 s | 13.50% |
| | Mean TPOT | 69.514 ms | 67.556 ms | 2.82% |
| G1 + G2 | G1 reuse | 26.1932% | 25.9547% | 0.91% |
| | G2 reuse | 7.1087% | 6.9254% | 2.58% |
| | Combined reuse | 33.3019% | 32.8801% | 1.27% |
| | Mean TTFT | 755.304 s | 636.553 s | 15.72% |
| | Mean TPOT | 67.305 ms | 63.793 ms | 5.22% |

- GPU tier reuse is the run's delta of `vllm:prompt_tokens_by_source_total`
  (`local_cache_hit` is G1, `external_kv_transfer` is G2) over one common
  denominator of 46,542,292 prompt tokens. Simulator reuse is attributed to the
  tier a request's prefix came from at first admission.
- Errors are single-run aggregate relative errors, not repeated-run MAPE.
- The replay set `bytes_per_token: 253952`, the per-token KV footprint of a
  whole DP engine across its two attention-TP ranks. That is twice the 126,976
  bytes `auto` estimates for one TP shard, so the replay's 32 GB/s D2H and H2D
  limits apply to the whole-engine footprint.
- The G1-only arm recorded 74 native preemptions (42 on DP0, 32 on DP1)
  against 82 simulator readmissions. With G2, native preemptions rose about
  elevenfold to 830 (387 on DP0, 443 on DP1), while the simulator still recorded
  82. The counters are defined differently, and their contribution to the timing
  error was not isolated.
- Event-by-event scheduler parity is not claimed. `cluster_shared` has no native
  reference.


## vLLM prefill cadence qualification

This follows vLLM's `prefill_schedule_interval` scheduler behavior at commit
`e2fa28594f7baad142a426b0b6a2cfe2c79201c7`.

## Validation

Kimi-K2.5 NVFP4, ISL 8192, and OSL 1024 replay measurements show that interval
4 materially closes the TPOT gap on both B200 and B300. Interval 1 reproduces
the previous AISimulate baseline exactly.

### B200, attention DP 8 / MoE EP 8

| Concurrency | Silicon TPOT | Interval 1 | Gap | Interval 4 | Gap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 89.548 ms | 173.787 ms | +94.1% | 96.173 ms | +7.4% |
| 1024 | 143.230 ms | 248.305 ms | +73.4% | 147.367 ms | +2.9% |

Interval 4 also moves output throughput to within 6.0% of silicon at concurrency
512 and within 2.3% at concurrency 1024.

### B300, attention DP 4 / MoE EP 4

| Concurrency | Silicon TPOT | Interval 1 | Gap | Interval 4 | Gap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 48.845 ms | 67.754 ms | +38.7% | 57.043 ms | +16.8% |
| 256 | 69.027 ms | 100.508 ms | +45.6% | 75.998 ms | +10.1% |
| 512 | 109.110 ms | 153.701 ms | +40.9% | 115.119 ms | +5.5% |

Interval 4 moves per-GPU output throughput gaps from -27.2%, -30.3%, and
-27.7% to -14.0%, -8.8%, and -4.8%, respectively. TTFT gaps improve from
-20.1%, -21.3%, and -23.4% to -15.0%, -13.3%, and -10.5%. The remaining gap
is largest at low concurrency, so cadence is the dominant effect but not the
only source of error for this case.

## Belady simulator comparison

Mooncake validation used a Llama-3.1-8B/H200 timing profile and 4,096 cache blocks
of 64 tokens, after smoke sweeps confirmed eviction losses and the load knee.
All full-trace comparisons completed 23,608 requests and 4,299,817 output tokens.
Loaded 1/2/4-worker **simulated** throughput gains were 2.01–2.11% for vLLM and
0.39–0.77% for SGLang, with less prefill work; TRT-LLM gained 3.01% on 1,000 requests.
Arrival-limited throughput stayed unchanged. A SGLang smoke case lost 0.165%
despite better reuse because attention batch/context costs increased; maximum
TTFT also worsened in one full run. Better reuse does not guarantee faster serving
or better tails. Local simulator wall time rose about 3.4%/7.2% for vLLM/SGLang.
The [validation record in PR #256](https://github.com/ai-dynamo/aisimulate/pull/256)
contains the revisions, trace checksums, configuration, commands, and paired results.


## Reference boundary

The sources below identify the methods and request-boundary semantics used as
references. Shared requirements alone do not establish that implementation code,
tests, or documentation were copied or adapted. The file-level comparison below
supports treating this use as a methodological reference with an AISimulate
implementation.

Source audit (updated 2026-09-21):

- Original snapshot reference: NVIDIA AIPerf,
  [`src/aiperf/timing/trajectory_source.py`](https://github.com/ai-dynamo/aiperf/blob/7db2ba37a62aa80c882bc90eaf61cc8073e2387b/src/aiperf/timing/trajectory_source.py)
  and its [`session_tree.py`](https://github.com/ai-dynamo/aiperf/blob/7db2ba37a62aa80c882bc90eaf61cc8073e2387b/src/aiperf/timing/session_tree.py)
  at `7db2ba37a62aa80c882bc90eaf61cc8073e2387b`,
  [Apache-2.0](https://github.com/ai-dynamo/aiperf/blob/7db2ba37a62aa80c882bc90eaf61cc8073e2387b/LICENSE).
  Both source files carry NVIDIA copyright notices.
- Additional corroborating reference located during this review: SemiAnalysisAI
  AgentX harness,
  [`docs/tutorials/agentx-mvp.md`](https://github.com/SemiAnalysisAI/agentx-harness/blob/56a0cf70f4c0359454ee4bd15a17770b541a3e3e/docs/tutorials/agentx-mvp.md),
  at `56a0cf70f4c0359454ee4bd15a17770b541a3e3e`,
  [Apache-2.0](https://github.com/SemiAnalysisAI/agentx-harness/blob/56a0cf70f4c0359454ee4bd15a17770b541a3e3e/LICENSE).
  This is not a claim that this later audit reference was the original source.
- The original AgentX methodology website links to SemiAnalysisAI InferenceX-app.
  Its methodology article source is
  [`packages/app/src/components/datasets/agentx-methodology-article.tsx`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/9bb7b13eb4985217a6282f340459fd5948613276/packages/app/src/components/datasets/agentx-methodology-article.tsx)
  at `9bb7b13eb4985217a6282f340459fd5948613276`; the repository
  [license is GPL-3.0](https://github.com/SemiAnalysisAI/InferenceX-app/blob/9bb7b13eb4985217a6282f340459fd5948613276/LICENSE).
  The referenced content describes one-output-token primers, ten additional
  warmup requests per lane, and a preparation/profile boundary. The comparison
  did not identify website source code, figures, or prose copied into the
  reviewed implementation. The later harness reference does not replace this
  original source or its license.

Implementation comparison:

- [`snapshot.rs`](../../../crates/core/src/replay/loadgen/snapshot.rs), inherited from
  #207, follows AIPerf's request-start boundary and historical-prefix semantics.
  It operates on AISimulate's validated dependency graph, uses a versioned
  BLAKE3-based sample keyed by graph/play/lane identity, and allocates checked
  per-play token ranges. The referenced AIPerf code operates on its Python
  session trees and samples with its RNG. This is a separate source comparison
  from the AgentX article.
- [`phase.rs`](../../../crates/core/src/replay/loadgen/phase.rs) implements the primer
  and warmup requirements using the prepared original-node inputs, per-lane
  queues, distinct request identities, and native completion/settlement
  feedback. It repeats saved prefixes and preserves the frontier; AIPerf's
  optional duration-based warmup advances trajectories. The payload selection
  policies therefore differ.
- [`driver.rs`](../../../crates/core/src/replay/loadgen/driver.rs) and the native
  [aggregated](../../../crates/core/src/replay/agg.rs) and
  [P/D](../../../crates/core/src/replay/disagg.rs) paths integrate preparation
  with AISimulate's existing executor, handoff, cache, and measurement state.
  Their regression tests exercise these local interfaces and simulator timing;
  matching the method's request counts is not itself evidence of copied tests.

**Review status:** the [attribution finding](https://github.com/ai-dynamo/aisimulate/pull/235#discussion_r4042517941)
remains open pending reviewer/maintainer acceptance of this reference scope.
The earlier statement that adapted behavior automatically triggers the gate
was too broad: [AGENTS.md](../../../AGENTS.md) addresses copied, adapted, translated,
or substantially derived code or other content. This comparison does not claim
a formal clean-room process or grant a license exemption. No notice entry is
added for the proposed methodology-only treatment. If the review identifies
specific copied or adapted content, record its file scope and applicable
attribution in both canonical and packaged notices before closing the finding.
The notice-equality check alone does not resolve it.

Qualification must assert exact input tokens, ten requests per lane, one output
token per request, barrier ordering, timer preservation, failure behavior, and
actual cache reuse with caching enabled and disabled. It must also retain cold
snapshot compatibility and deterministic repeated runs.

The initial offline P/D qualification covers the public Rust, Python, and CLI
paths, request identity through the handoff, and the same preparation/profile
barrier. Reports remain `functional_only`; these checks do not establish
prediction accuracy or full AgentX benchmark fidelity. Fixed-duration recycling,
cutoff behavior, and late events from recycled play instances still need joint
qualification under AIC-1813, AIC-1896, and AIC-1818. This initial boundary does
not complete all AIC-1895 acceptance work.

## AgentX G2 qualification provenance



AISimulate #381 opens and qualifies AgentX G2 through the existing Engine
runner and native JSON entrypoint. It adds no execution bridge, package version
API or consumer build plumbing.

Dynamo's G2 transport, event handling and routing are implemented separately in
[Dynamo #15647](https://github.com/ai-dynamo/dynamo/pull/15647). AgentX-specific
Dynamo integration and qualification belong to
[Dynamo #15626](https://github.com/ai-dynamo/dynamo/pull/15626), based on that PR.
Use that consumer's dependency requirements and installation instructions once
its AgentX support is qualified. AISimulate's Engine acceptance results do not
qualify a Dynamo installation.
