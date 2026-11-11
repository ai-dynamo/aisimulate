<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Replay feature support

This matrix describes the built-in offline `engine` stack. A capability is
supported only within its stated composition; individual supported features
are not automatically supported together. Optional stacks advertise their own
[runner capabilities](../adapters/runner-abi.md). Timing/profile availability
is a separate [performance-model question](../perf-model/support-matrix.md).

| Feature | Backend / execution | Topology and workload | Boundaries |
| --- | --- | --- | --- |
| Native token scheduling | vLLM, SGLang, TensorRT-LLM | Aggregated or P/D; synthetic and compatible traces | Engine-specific admission/cache behavior; no real network or host-runtime overhead |
| Attention-DP | All three native backends | Aggregated and independently sized P/D pools | Rank-local scheduling and G1; one grouped pass completes at the maximum rank time |
| Native MTP | Supported engine/model path | Token or qualified agentic replay with explicit acceptance | Separate from ngram; canonical MTP supports scoped G2 below; no grouped-cache/G3/sub-block state-cache combination; agentic accuracy is functional-only |
| [Registered speculative schemes](engine/speculation.md#registered-schemes) | Offline vLLM/SGLang, subject to scheme/model constraints | Prediction, aggregated or P/D; token or qualified agentic replay | Existing linear 1–5-token executor; fixed KV capacity; default timing with supported OP cost; only MTP supports scoped G2; no trees, grouped-cache/G3/sub-block state-cache combination or generic-scheme recommendation |
| Ngram prompt lookup | Offline vLLM | Aggregated or P/D, op-level or explicit timing; HBM-only agentic replay | Fixed draft/acceptance assumption; no lookup CPU cost, adapters or deployment generation |
| Sessions | Native token replay | Synthetic or compatible trace sessions | Concurrency counts sessions through think time; see trace-format restrictions |
| Prefix reuse | Native backend cache | G1 with prefix caching enabled | Actual reuse differs from Router overlap; complete block and backend admission rules apply; SGLang P/D decode reuse is opt-in through `decode_radix_cache`, without speculation |
| [Request cache-salt isolation](engine/kv-cache.md#sglang-request-cache-salt) | Native SGLang | Aggregated or P/D; native requests and inline replay | Different salts keep distinct KV identities and retained storage in shared memory pools; events preserve the namespace; built-in trace schemas remain unsalted |
| Grouped FPM cache | vLLM | Cold aggregated, PP1/CP1 | HBM only, no speculation, prefix caching off, no scalar block capacity, no P/D |
| G2 host offload | vLLM | Aggregated and token-only P/D, any attention-DP size | Prefix caching; MTP only for aggregated DP1 local G2, uniform shared FullAttention; no other speculation/Belady/state-cache offload; otherwise local or layout-compatible shared pool; an aggregated worker’s local pool can also follow vLLM 0.32 connector profiles (no G3 or KVCR) |
| [Local HiCache](engine/kv-cache.md#sglang-hicache) | SGLang | Aggregated prediction and native offline replay; local host pool per attention-DP rank | Prefix caching; write-through, selective write-through or write-back; no P/D, recommendation, Belady, external storage or peer KVCR; excluded from HBM-only serving qualification |
| Agentic G2 | vLLM | One aggregated worker, several aggregated workers each with local G2, or P/D with any number of workers per role, attention DP1 on every role | Static pools; canonical MTP only for a single aggregated worker with local G2 and prefix caching; no other speculation/G3; without speculation, either G2 scope works with one aggregated worker or with P/D |
| G3 offload | vLLM | Aggregated, attention DP1; fixed or scaled pools | Requires G2; no speculation, AgentX, P/D, recommendation or hardware integration |
| KVCR peer fetch | vLLM | Aggregated with any attention-DP size, or P/D prefill ranks; fixed or scaled pools | Requires private G2 and hints from a placement policy or ReplaySpec requests (CLI traffic has none); no G3, decode-role KVCR or recommendation; idle under AgentX |
| Best-effort Belady | All three native backends | Fixed aggregated, DP1, complete open-loop input trace | Native descriptor; prefix caching required; rejects closed-loop/generated/agentic/delta inputs, scaling and offload |
| Agentic dependency graph | vLLM, SGLang | Offline aggregated or P/D; Weka, Agentic Mooncake and agentic Dynamo traces | One target model for P/D; speculation uses HBM or the canonical MTP/local G2 composition; no agentic TensorRT-LLM; HBM or qualified vLLM G2 |
| Seeded snapshots and warmup | Same agentic boundary | Timestamp load, positive lanes, unsigned 64-bit seed | Request-boundary history, not a physical engine checkpoint; warmup preserves snapshot frontier |
| Continuous agentic profiles | Same agentic boundary | Snapshots plus profile duration; warmup optional | No simultaneous virtual-time cap; grace can finish with unsettled server work; live host-memory supervision required |
| AIPerf metric selection | Native offline Engine runner including AFD, predict and recommend CLI, SDK | TTFT, request latency, per-request ITL and user TPS; avg, p50, p75, p90, p95 and p99 from the native summary | Success-only request population; nearest-rank percentiles; optional strict constraints; no analytical EPD or failure-adjusted metrics. See [definitions](../sweeper/optimization-goals.md#aiperf-metric-selectors). |
| Replay telemetry | Rust Replayer, Engine Python runner, CLI | Settled virtual-time samples; opt-in all-phase request/lifecycle capture | Offline vLLM/SGLang aggregated or P/D; consumer stacks have separate adapters |
| Detailed native artifacts | RoundRobin Replayer | Fixed single aggregated worker with attention DP1 | No multiworker/scaling/P/D or shared G2 artifact claim; normal reports have broader scope |
| Router and Planner | Optional Dynamo stack | Consumer-supported topologies/workloads | Separate installation, configuration ABI and downstream qualification |
| AFD pipeline timing with continuous replay | SGLang | Aggregated AFD or P/D with AFD on either or both roles; synthetic, trace and prefix-cached traffic | No KV-relative load, encoder composition or native deployment renderer |
| Analytical EPD | Offline engine analytical overlay | Encoder + aggregate or P/D language workers; Qwen3-VL and Kimi K3 towers only | Fixed synthetic lengths/images/concurrency; default op-level timing; aggregate means only, no adapters/per-request capture |
| Native SGLang VL replay | SGLang; op-level timing for the vision tower | Aggregated worker or P/D prefill worker; fixed synthetic images; PP1, attention DP1 | Frontend stages from a host cost table per serving environment; no decode-side frontend, GPU image processor, video or output-side costs |
| Native encoder pools | SGLang `--encoder-only` servers ahead of a `--language-only` worker | Encoder + aggregate or P/D SGLang language workers; Qwen3-VL towers only (Kimi K3 refused at the image geometry, Llama 4 at the encoder builder); fixed synthetic images; static pools | CPU preprocessing extrapolated from the host table; no adaptive local encoding, GPU-direct transfer or encoder prefix cache |

## Validation and evidence

The authoritative gates are
[`RunnerCapabilities.require_compatible`](../../python/aisimulate/src/aisimulate/sweeper/replay.py),
[`EngineReplayRunner`](../../python/aisimulate/src/aisimulate/runner.py),
[public engine validation](../../python/aisimulate/src/aisimulate/config/engine.py),
and the native [engine](../../crates/core/src/engine/) and
[Replay](../../crates/core/src/replay/) validators. Native validation remains
necessary when callers bypass public YAML.

Qualification includes fixed-timing lifecycle tests, Python/native report
checks and CLI artifact checks. AgentX qualification does not establish full
AgentX benchmark parity or measured model accuracy. The
[retained replay evidence](../../benchmarks/evidence/accuracy/replay-evidence.md)
keeps scoped GPU/simulator comparisons, revisions and caveats separate from
feature acceptance. See [engine](engine/README.md),
[KV cache](engine/kv-cache.md), and [traffic-format compatibility](workloads.md#trace-format-compatibility)
for the detailed constraints.
