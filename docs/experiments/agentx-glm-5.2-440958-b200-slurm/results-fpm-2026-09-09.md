<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# B200 FPM recording results — September 9, 2026

Completed one FPM-on AgentX c4 run, compared with the existing
[FPM-off baseline](results-2026-09-08.md), job `4187044`. No new FPM-off run
was performed, per user direction. Job `4208956` completed on `umb-b200-041`
with exit `0:0`; the allocation was released and the monitor stopped.

## Configuration and validity

The model and serving settings follow the [baseline runbook](README.md):
8 B200 GPUs, GLM-5.2 NVFP4 revision `53e0691e21895a3863a606dfd12910c69eba94ab`,
TP8/EP1, FP8 KV, no HiCache, static memory fraction 0.83, 8192-token prefill
chunks, maximum 8 running requests, CUDA graph maximum batch size 8, and
EAGLE steps3/topk1/draft4 with simulated acceptance length 2.99.

This run uses the [FPM-fixed image](../agentx-glm-5.2-440082-gb200-hicache/fpm-image-2026-09-09.md):

```text
nvcr.io/nvidian/dynamo-dev/sglang-agentx@sha256:9fb6f18c1b224a5651f4482cdc20efc27b2916cd14f92173da3c349b8412f308
```

Its amd64 manifest is `sha256:f856a45537f82e1900ea7607edcbaa7f77fbb2e70220eae522d1d50d0046727e`.
It backports [SGLang PR #38711](https://github.com/sgl-project/sglang/pull/38711)
at `ed18d64951b93ac252d921ee037ce0d3327eda1f` onto SGLang
`71de97b264b04dcd514cf904003028aefe9775c8`; it is not a full PR-head engine upgrade.
AIPerf uses the pinned SemiAnalysis revision `754356e9a39acc6cc6afb242d123bb57c3fb6f75`
in the image's isolated venv. The workload remains all 393 trajectories, seed42,
c4, and the same warmup, trace-idle caps and cache-busting settings.

Warmup completed 44 requests with zero errors/cancellations. Profiling started
at **17:06:12.297 PDT**, stopped sending at **18:06:12.298**, and completed its
30-second drain at **18:06:42.297**. All **635 requests** completed, with zero
errors/cancellations. The client reports `submission_valid=true`; TTFT coverage
was 99.9344%, and ITL coverage was 100%. The overall allocation lasted 1:17:52,
including startup, preparation, warmup, measurement, export and cleanup.

## Historical performance comparison

| Metric | FPM-off, job 4187044 | FPM-on, job 4208956 | Relative change |
| --- | ---: | ---: | ---: |
| Total tokens/s/GPU | 4166.115976 | 4134.242919 | -0.765% |
| Output tokens/s/GPU | 28.464868 | 28.370522 | -0.331% |
| TTFT p50, seconds | 0.192347 | 0.202279 | +5.164% |
| TTFT p90, seconds | 0.690483 | 0.667699 | -3.300% |
| ITL p50, milliseconds | 2.849541 | 2.850977 | +0.050% |
| ITL p90, milliseconds | 3.826451 | 3.730968 | -2.495% |
| Request latency p90, seconds | 11.610088 | 11.543507 | -0.573% |
| Completed requests | 638 | 635 | -0.470% |

These are the exported client metrics; aggregate throughput is divided by eight.
No throughput denominator is silently renormalized to remove the drain.
There is no clear aggregate regression in this comparison, but it **does not
establish zero FPM overhead**: the nodes, runtime patch, resolved client
dependencies and completed closed-loop request sets differ. Client dependencies
are preserved in this run's `client-freeze.txt`; the historical directory did
not contain that file when checked. The c4 workload is not a saturation test.

## FPM capture and download

The recorder wrote compact buffered JSONL locally, then copied it to shared
scratch before cleanup. Validation and an independent local scan found:

- 230,772 records, including 228,056 decode records; 123,378,989 bytes (~117.66 MiB).
- One `(worker_id, dp_rank)` stream: `("2609536487445137822", 0)`.
- Counter IDs 0–230771, with zero observed gaps, non-increasing counters or rejected messages.
- Valid decode request counts, positive decode KV sums and nonnegative finite timing.

The capture includes smoke, warmup, profiling and post-profiling idle heartbeats,
not only the measured hour. Counter continuity is a diagnostic, not a guarantee
against publisher-side loss or missing pre-subscription events.

SHA256, verified after download:

```text
0c6e52190021247e095f518c8413a4d8afc2b9c38aab8bbe67fd1ab0a9e1c8bf
```

The FPM file is already on the SSH workstation `hzhou-workstation`. Run this
on your own computer (use your usual SSH host alias if different):

```bash
scp hongkuanz@hzhou-workstation:/home/hongkuanz/Experiments/agentx-fpm-ab-20260909/job-4208956/on/fpm.jsonl ./fpm-4208956.jsonl
```

All raw AIPerf exports, worker/frontend/recorder logs and the FPM capture remain
on shared scratch at `/home/scratch.hongkuanz_gpu/agentx-results/job-4208956/on/`.
A downloaded copy is at `/home/hongkuanz/Experiments/agentx-fpm-ab-20260909/job-4208956/on/`;
the structured comparison is `/home/hongkuanz/Experiments/agentx-fpm-ab-20260909/result-4208956.json`.
Large raw artifacts are deliberately not committed to Git.

## Incidents and interpretation limits

1. Initial job `4207957` failed before container startup: Enroot 3.5.0 rejected
   the Docker-style `@sha256` URI. It expects `:sha256:<digest>`. The retry used
   the existing digest-pinned amd64 squashfs cache; the embedded FPM patch hash
   matched the published build. No benchmark data came from the failed attempt.
2. Optional persistent dataset mmap caching reported home-filesystem `ENOSPC`.
   The actual 6.37GB runtime mmap had already been built successfully, warmup
   and profiling continued, and FPM kept growing. No shared data was deleted.
3. A recoverable CUDA allocator OOM warning occurred during warmup; subsequent
   prefill/decode continued without request failures.
4. Prometheus export reported counter resets for `dynamo_component_response_bytes`,
   `sglang:cuda_graph_passes`, and `sglang:realtime_tokens`. No corresponding
   worker restart or FPM stream reset was found. The exact telemetry mechanism
   remains unresolved; retain the warnings and use caution with derived server
   counter statistics. The performance table above uses client-side metrics.
5. The 30-second grace timeout occurred with all 635 requests complete and zero
   cancellations. Pending join/child-state cleanup warnings were retained, as
   in the baseline. The worker's final `Killed` line followed `COMPLETE` and
   belongs to the script's bounded teardown, not a failure during measurement.
