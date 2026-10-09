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
| Native MTP | Supported engine/model path | Token replay with explicit acceptance | Separate from ngram; no AgentX/grouped-cache/G2/G3/fine state-cache combination |
| Ngram prompt lookup | Offline vLLM | Aggregated or P/D, op-level or explicit timing | Fixed draft/acceptance assumption; no lookup CPU cost, AgentX, adapters or deployment generation |
| Sessions | Native token replay | Synthetic or compatible trace sessions | Concurrency counts sessions through think time; see trace-format restrictions |
| Prefix reuse | Native backend cache | G1 with prefix caching enabled | Actual reuse differs from Router overlap; complete block and backend admission rules apply |
| Grouped FPM cache | vLLM | Cold aggregated, PP1/CP1 | HBM only, no speculation, prefix caching off, no scalar block capacity, no P/D |
| G2 host offload | vLLM | Aggregated and token-only P/D, any attention-DP size | Prefix caching, no native MTP/Belady/state-cache offload; local or layout-compatible shared pool |
| Agentic G2 | vLLM | One aggregated worker or 1P1D, attention DP1 on every role | Static pools, no speculation/G3; each role may omit G2 or use local/shared scope |
| G3 offload | vLLM | Aggregated, attention DP1; fixed or scaled pools | Requires G2; no speculation, AgentX, P/D, recommendation or hardware integration |
| Best-effort Belady | All three native backends | Fixed aggregated, DP1, complete open-loop input trace | Native descriptor; prefix caching required; rejects closed-loop/generated/agentic/delta inputs, scaling and offload |
| Agentic dependency graph | vLLM, SGLang | Offline aggregated or P/D; Weka, Agentic Mooncake and agentic Dynamo traces | One target model for P/D; no speculation or agentic TensorRT-LLM; HBM or qualified vLLM G2 |
| Seeded snapshots and warmup | Same agentic boundary | Timestamp load, positive lanes, unsigned 64-bit seed | Request-boundary history, not a physical engine checkpoint; warmup preserves snapshot frontier |
| Continuous agentic profiles | Same agentic boundary | Snapshots plus profile duration; warmup optional | No simultaneous virtual-time cap; grace can finish with unsettled server work; live host-memory supervision required |
| Native telemetry observer | Rust Replayer | Settled virtual-time samples | Engine Python JSON runner rejects telemetry capture; consumer stacks have separate adapters |
| Detailed native artifacts | RoundRobin Replayer | Fixed single aggregated worker with attention DP1 | No multiworker/scaling/P/D or shared G2 artifact claim; normal reports have broader scope |
| Router and Planner | Optional Dynamo stack | Consumer-supported topologies/workloads | Separate installation, configuration ABI and downstream qualification |
| Analytical AFD | vLLM, SGLang, TensorRT-LLM model paths | `afd` or `afd+pd`; fixed-length synthetic requests | No traces, random lengths, KV-relative load or native deployment renderer |
| Analytical EPD | Offline engine analytical overlay | Encoder + aggregate or P/D language workers | Fixed synthetic lengths/images/concurrency; default op-level timing; aggregate means only, no adapters/per-request capture |
| Native SGLang VL replay | SGLang; op-level timing for the vision tower | Aggregated worker or P/D prefill worker; fixed synthetic images; PP1, attention DP1 | Frontend stages from a host cost table per serving environment; no decode-side frontend, GPU image processor, video or output-side costs |
| Native encoder pools | SGLang `--encoder-only` servers ahead of a `--language-only` worker | Encoder + aggregate or P/D SGLang language workers; fixed synthetic images; static pools | CPU preprocessing extrapolated from the host table; no adaptive local encoding, GPU-direct transfer or encoder prefix cache |

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
