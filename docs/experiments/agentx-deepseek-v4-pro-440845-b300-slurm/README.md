<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AgentX 440845: DeepSeek-V4-Pro on one B300 Slurm node

This experiment targets [AgentX point 440845](https://inferencex.semianalysis.com/inference/agentic/440845):
DeepSeek-V4-Pro FP4 with MTP, attention DP8 and DRAM HiCache at concurrency 32.
It is a hardware reproduction for subsequent AISimulate validation.
**Execution authorized: FPM-off, then FPM-on with raw capture; both runs disable
HiCache. No local B300 measurement or parity result exists yet.**
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

## Planned local runtime: shared FPM-fixed x86 image

Use the **same linux/amd64 FPM-fixed image as the B200 GLM AgentX job
`4207957`**, with FPM disabled for the first run and enabled explicitly with recording for
the second run. Its submission pins this
multi-architecture index:

```text
nvcr.io/nvidian/dynamo-dev/sglang-agentx@sha256:9fb6f18c1b224a5651f4482cdc20efc27b2916cd14f92173da3c349b8412f308
```

The linux/amd64 child is:

```text
nvcr.io/nvidian/dynamo-dev/sglang-agentx@sha256:f856a45537f82e1900ea7607edcbaa7f77fbb2e70220eae522d1d50d0046727e
```

See the [shared FPM image build record](../agentx-glm-5.2-440082-gb200-hicache/fpm-image-2026-09-09.md).
It backports SGLang PR #38711 at `ed18d64951b93ac252d921ee037ce0d3327eda1f`,
including timing fixes and immutable decode-length statistics, onto engine base
`71de97b264b04dcd514cf904003028aefe9775c8`. It also includes the FPM disk recorder
and isolated AIPerf environment. The production backport adds no D2H or timing
synchronization; performance overhead remains to be measured.

The reference image in the table above describes the published AgentX run,
not our planned runtime. Its API tag includes `cu13`, while its server log tag
omits it. Our shared FPM image is an intentional runtime difference from that
reference. Preserve DSv4's c32, TP8/EP8 and attention DP8 settings;
do not inherit the GLM baseline's model-specific launch arguments.

DSv4/MegaMoE/HiCache GPU compatibility still needs smoke validation with this
image. CPU CLI preflight found that this newer engine removed the reference
`--prefill-decode-interval 20` option. Both local runs omit it and retain
`--enable-prefill-delayer`; no equivalence between the removed interval and the
current scheduling policy is assumed. Record any required image change explicitly. Synthetic speculative
acceptance makes this a performance experiment, not a model quality evaluation.

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

The user selected **HiCache disabled in both cases**, given no CPU cache hits
in the reference point. This also removes the reference host-tier memory burden
from the approximately 2 TB Computelab nodes. Preserve GPU radix caching and
measure GPU hit rate and cache capacity; absence of reference CPU hits alone
does not prove that disabling HiCache is performance-neutral.

## Execution protocol

[campaign.py](campaign.py) runs two cases in order in one exclusive B300 node
allocation, using one immutable image and the same model, client and settings:

1. **off**: FPM disabled, no recorder; 3600 seconds of measured AgentX c32.
2. Stop the engine and router, then launch fresh instances and repeat the same
   smoke/warmup procedure. GPU KV starts fresh; compilation caches can be reused.
3. **on**: FPM enabled, [record_fpm.py](record_fpm.py) subscribes to all eight
   native SGLang DP-rank IPC endpoints; another 3600-second measurement.
4. Validate rank coverage, finite timing/length statistics, and counter gaps;
   [compare.py](compare.py) compares throughput, TTFT, ITL, request latency,
   request counts, cache hits and validity evidence.

Both cases use native SGLang serving and the DP-aware SGLang router inside the
shared image. This preserves the reference routing path while collecting native
SGLang FPM directly; no Dynamo frontend is added to this pair. The runtime-only
thinking chat template comes from the pinned reference commit, with its license
and provenance retained on scratch; it is not vendored in this directory.

The recorder writes buffered JSONL on node-local disk throughout startup, smoke,
warmup and measurement. At shutdown it flushes and copies the complete capture
and SHA256 to scratch. Keep AIPerf phase timestamps to select the measured window.
Counter gaps remain visible diagnostics; the recorder does not invent missing
samples. The GPU time comes from the repaired engine instrumentation.

[submit.sh](submit.sh) requests eight GPUs, one exclusive node and a four-hour
limit, covering both measurements plus loading, compilation and warmup. Submission
is gated on the pinned checkpoint's successful shard verification and the CPU
image/CLI preflight. Hardware identity is checked before model loading.

Artifacts live at `/home/scratch.hongkuanz_gpu/agentx-dsv4-results/job-<job-id>/`:
`off/` and `on/` each retain commands, environment, server/router/client logs,
AIPerf raw exports and lifecycle timestamps. `on/fpm.jsonl`, `on/fpm.sha256` and
`on/fpm-validation.json` retain the FPM capture and diagnostics. The root contains
hardware/topology, pinned model/image identity and `comparison.json`/`.md`.

This is one ordered off/on pair. Startup effects and different closed-loop
request sets limit causal overhead claims. Compare this pair first, then compare
against point 440845 separately with the no-HiCache and runtime differences clear.

The existing [GLM-5.2 B200 baseline](../agentx-glm-5.2-440958-b200-slurm/README.md)
and [GLM-5.2 GB200 HiCache experiment](../agentx-glm-5.2-440082-gb200-hicache/README.md)
remain separate experiments. Directory names include model and AgentX ID so results
cannot be confused across reference points.
