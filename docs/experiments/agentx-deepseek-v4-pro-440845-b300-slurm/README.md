<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AgentX 440845: DeepSeek-V4-Pro on one B300 Slurm node

This experiment targets [AgentX point 440845](https://inferencex.semianalysis.com/inference/agentic/440845):
DeepSeek-V4-Pro FP4 with MTP, attention DP8 and DRAM HiCache at concurrency 32.
It is a hardware reproduction for subsequent AISimulate validation.
**Preparation only: no local B300 measurement or parity result exists yet.**
See [preparation log](preparation-2026-09-09.md) for checkpoint download status,
Computelab resource discovery and remaining work.

## Reference identity

The configuration below was checked on September 9, 2026 against the
[published benchmark API](https://inferencex.semianalysis.com/api/v1/benchmarks?model=DeepSeek-V4-Pro),
[server log](https://inferencex.semianalysis.com/api/v1/server-log?id=440845&file=results%2Fserver.log),
and [client log](https://inferencex.semianalysis.com/api/v1/server-log?id=440845&file=results%2Fbenchmark.log).

The associated [InferenceX run, attempt 2](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/33145139961/attempts/2)
uses commit `fb85931b1edec09f9498509835a8c814bebe3c65`.
Consult its [B300 DSv4 launcher](https://github.com/SemiAnalysisAI/InferenceX/blob/fb85931b1edec09f9498509835a8c814bebe3c65/benchmarks/single_node/agentic/dsv4_fp4_b300_sglang_mtp.sh)
and [thinking chat template](https://github.com/SemiAnalysisAI/InferenceX/blob/fb85931b1edec09f9498509835a8c814bebe3c65/benchmarks/single_node/chat_templates/deepseek_v4_thinking.jinja)
when preparing execution; neither is vendored here.

| Item | Reference configuration |
| --- | --- |
| Model | `deepseek-ai/DeepSeek-V4-Pro`; FP4 routed experts detected by the server |
| Checkpoint revision selected for download | `b5968e9190ef611bbf34a7229255be88a0e937c1` |
| Hardware / topology | One B300 node, 8 GPUs; aggregated serving |
| Parallelism | TP8, EP8, attention DP8, PP1 |
| Image reported by benchmark API | `lmsysorg/sglang:nightly-dev-cu13-20260827-20621aa1` |
| SGLang build commit | `20621aa14bda7726a8a968f326198eac61717fef` |
| Router | `sglang-router` 0.3.2, DP-aware consistent hashing; correlation-based routing |
| KV / attention | FP8 E4M3, `dsv4` attention, page size 256, SWA/full ratio 0.075 |
| GPU memory / requests | Static fraction 0.93; max running requests 64 |
| Prefill | Global chunk 65536, normalized by DP8 to 8192 per rank; max-prefill tokens 16384 |
| Decode CUDA graphs | Maximum batch size 544; prefill graphs disabled by runtime |
| MoE | `megamoe`; FP4 activations and MXF4 kind enabled through environment variables; token cap 8320 per rank |
| Speculation | EAGLE/MTP, 3 steps, top-k 1, 4 draft tokens, draft from the same model |
| Synthetic acceptance | Length 2.49, `match-expected`, `real-draft-token` |
| HiCache | Enabled; ratio 3; `write_back`, `direct`, `page_first_direct`; no storage backend |
| Load | AgentX, 32 replay lanes, 3600-second profiling target |
| Dataset | `semianalysisai/cc-traces-weka-062126`, revision `23f152f6f0f9399a85901b89a6458def0ef16729` |
| Client behavior | `ignore_eos=true`, first-turn-prefix cache busting, system-idle-gap cap 10 seconds |

The selected model revision appears in the reference client's config/tokenizer
HTTP requests. The server loaded a local directory and reported `revision=None`;
this does not prove the exact revision of its weight files. Preserve this distinction
when reporting reproduction fidelity. The client dependency revision and full replay
arguments still need to be pinned from the reference workflow before execution.

The API image includes `cu13`, while the server's `SGLANG_IMAGE_TAG` omits it.
Resolve and record the actual image digest before launch. This experiment must not
silently inherit the GLM-5.2 Dynamo image or its no-HiCache, c4, TP8/EP1 settings.
Synthetic speculative acceptance makes this a performance experiment, not a model
quality evaluation.

## Published comparison targets

| Metric | Reference |
| --- | ---: |
| Total tokens/s/GPU | 17480.41899 |
| Output tokens/s/GPU | 114.98434 |
| TTFT p50 / p90, seconds | 1.83555 / 7.72655 |
| ITL p90, milliseconds | 19.5 |
| Request latency p90, seconds | 42.79698 |
| Completed requests | 3410 |
| Exported duration, seconds | 3628.90598 |
| GPU cache hit fraction | 0.958711019592267 |
| Reported KV cache pool, tokens | 10646528 |
| Reported allocated CPU DRAM, GB | 2849 |

The API also contains `num_prefill_gpu=64` and `num_decode_gpu=64`. These fields
conflict with the single-node TP8/EP8 launch and with total throughput divided by
per-GPU throughput, which gives eight. Do not use those fields to request 64 GPUs
or normalize this point's measurements.

Total throughput includes reused prompt tokens. Preserve the exported duration,
request errors, cancellations, output-length checks and warmup separation; do not
reinterpret the reported throughput as uncached GPU compute throughput.

## Computelab constraints and next steps

SC-01 exposes both DGX B300 (8 GPUs, 256 logical CPUs) and B300 NVL8
(8 GPUs, 224 logical CPUs), each with approximately 2 TB host RAM. The September 9
query found no unreserved idle eight-GPU node for `hongkuanz`.

Host memory needs explicit validation. The reference reports 2849 GB CPU DRAM,
and its launcher describes ratio 3 at static fraction 0.93 as a host tier near
2 TB on a roughly 3 TB node. For DSv4 this launcher uses the host/device ratio,
not `TOTAL_CPU_DRAM_GB`, to determine cache capacity; it states that `--hicache-size`
is unsupported. A 2 TB Computelab node may therefore need a smaller ratio to leave
room for the engine, client, router and page cache. Measure the actual allocation;
any reduced host cache is a documented variant, not exact memory-capacity parity.

1. Finish the pinned checkpoint download and verify all index-referenced shards.
2. Pin the runtime digest, client revision, chat template and complete replay settings.
3. Obtain one complete B300 allocation and record hardware, driver, power limits,
   topology, host memory and effective Slurm resources.
4. Validate weight loading, host cache sizing, DP-aware routing and a short replay.
5. Run the measured c32 experiment with separate warmup and complete raw artifacts;
   compare against the targets above and record every configuration difference.

The existing [GLM-5.2 B200 baseline](../agentx-glm-5.2-440958-b200-slurm/README.md)
and [GLM-5.2 GB200 HiCache experiment](../agentx-glm-5.2-440082-gb200-hicache/README.md)
remain separate experiments. Directory names include model and AgentX ID so results
cannot be confused across reference points.
