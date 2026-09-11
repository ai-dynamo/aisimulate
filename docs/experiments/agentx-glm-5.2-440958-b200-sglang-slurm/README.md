<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GLM-5.2 / AgentX 440958 — B200 reproduction and results

Completed historical FPM-off job `4187044` on `umb-b200-260` and one FPM-on
job `4208956` on `umb-b200-041`. Both used eight B200 GPUs and HiCache off.
Allocations are released. This is a historical comparison, not a same-node pair.

## Reproduce

| Setting | Value |
| --- | --- |
| Model | nvidia/GLM-5.2-NVFP4, revision `53e0691e21895a3863a606dfd12910c69eba94ab` |
| Hardware | 8×B200, 183359 MiB/GPU, 1000W, driver 610.57.04 |
| Serving | Dynamo frontend, TP8/EP1, PP1, no attention DP |
| KV / memory | FP8 E4M3, GPU radix cache, HiCache off, static fraction 0.83 |
| Prefill / scheduler | chunk and max-prefill 8192; max running 8; graph max batch 8 |
| Speculation | EAGLE/native MTP, steps3/topk1/draft4; simulated acceptance 2.99 |
| Client | SemiAnalysisAI/aiperf `754356e9a39acc6cc6afb242d123bb57c3fb6f75` |
| Workload | 393 Weka trajectories, c4, seed42, 3600s measurement; 10 warmup requests/lane |

The exact warmup, cache-busting and idle-cap arguments are in the launchers.
Synthetic acceptance is for performance, not model-quality evaluation.

Use the published FPM-fixed image:
`nvcr.io/nvidian/dynamo-dev/sglang-agentx@sha256:9fb6f18c1b224a5651f4482cdc20efc27b2916cd14f92173da3c349b8412f308`.
Its amd64 child is `sha256:f856a45537f82e1900ea7607edcbaa7f77fbb2e70220eae522d1d50d0046727e`.
This backports SGLang PR #38711 at `ed18d64951b93ac252d921ee037ce0d3327eda1f`
onto `71de97b264b04dcd514cf904003028aefe9775c8`, rather than upgrading the whole engine.
It includes AIPerf in `/opt/agentx-aiperf` and the recorder in `/opt/fpm`.
Engine and client Python environments remain separate.

1. On Computelab, verify your account/partitions and scratch capacity before submission.
   Reuse the pinned checkpoint: 47 safetensors shards, about 432.9 GiB. Preserve
   HF cache blobs backing snapshot symlinks; do not redownload existing weights.
2. Reuse `/home/scratch.hongkuanz_gpu/images/sglang-agentx-fpm-f856a455-amd64.sqsh`.
   To import elsewhere, run inside a compute allocation:
   `enroot import -o /your/scratch/fpm.sqsh docker://nvcr.io#nvidian/dynamo-dev/sglang-agentx:sha256:f856a45537f82e1900ea7607edcbaa7f77fbb2e70220eae522d1d50d0046727e`.
   Enroot3.5.0 requires this digest syntax, not Docker's `@sha256`.
3. Stage the recording launchers and submit from the directory containing this README:

```bash
ssh hongkuanz@computelab-sc-01 'mkdir -p /home/scratch.hongkuanz_gpu/agentx-fpm-ab-20260909'
scp run-fpm.sh campaign.sh submit.sh check_fpm.py provenance.txt \
  hongkuanz@computelab-sc-01:/home/scratch.hongkuanz_gpu/agentx-fpm-ab-20260909/
ssh hongkuanz@computelab-sc-01 \
  'sbatch /home/scratch.hongkuanz_gpu/agentx-fpm-ab-20260909/submit.sh'
```

Adapt account/partition/scratch paths to your allocation. [submit.sh](submit.sh)
requests one node, eight GPUs, 112 CPUs, 1T RAM and 2.5 hours. [campaign.sh](campaign.sh)
verifies hardware/checkpoint and runs **only FPM-on** through [run-fpm.sh](run-fpm.sh).
The historical unmodified FPM-off launcher remains [run.sh](run.sh); it uses
the original nightly `cc84ea52fc8e66fb61a54692a30b2dcfa4d9b2349aa39d565039657cba8c26bb`
and installs its client separately. Do not call a new run an exact historical reproduction
without checking dependency differences.

The worker and frontend use shared file discovery, TCP and ZMQ, not separate
in-memory discovery. Results use fresh job-specific directories and refuse overwrite.
Recording is buffered locally, validated by [check_fpm.py](check_fpm.py), then
copied to scratch before bounded process cleanup. Verify final exports and
`sacct` exit status plus an empty `squeue` entry.

## Table 1: SA versus FPM-off and FPM-on

SA is [AgentX 440958](https://inferencex.semianalysis.com/inference/agentic/440958),
checked against the [published API](https://inferencex.semianalysis.com/api/v1/benchmarks?model=GLM-5.2).
All columns use eight B200 GPUs, TP8/EP1 and c4; SA enables HiCache.
Changes use unrounded client metrics: `100 * (new / baseline - 1)`.
Throughput includes cached prompt tokens and retains each exported drain denominator.

| Metric | SA 440958 | Ours FPM off | Ours FPM on | Off vs SA | On vs off |
| --- | ---: | ---: | ---: | ---: | ---: |
| Total throughput, tokens/s | 31,157.915 | 33,328.928 | 33,073.943 | +6.97% | -0.77% |
| Total throughput, tokens/s/GPU | 3,894.739 | 4,166.116 | 4,134.243 | +6.97% | -0.77% |
| Output throughput, tokens/s | 220.016 | 227.719 | 226.964 | +3.50% | -0.33% |
| Output throughput, tokens/s/GPU | 27.502 | 28.465 | 28.371 | +3.50% | -0.33% |
| TTFT mean, s | 0.628 | 0.352 | 0.358 | -43.88% | +1.66% |
| TTFT p50, s | 0.389 | 0.192 | 0.202 | -50.52% | +5.16% |
| TTFT p90, s | 1.328 | 0.690 | 0.668 | -47.99% | -3.30% |
| TTFT p95, s | 1.524 | 1.085 | 1.057 | -28.81% | -2.59% |
| ITL mean, ms | 3.520 | 3.334 | 3.227 | -5.27% | -3.22% |
| ITL p50, ms | 3.220 | 2.850 | 2.851 | -11.50% | +0.05% |
| ITL p90, ms | 3.850 | 3.826 | 3.731 | -0.61% | -2.50% |
| ITL p95, ms | 4.100 | 4.418 | 4.421 | +7.76% | +0.07% |
| Request latency mean, s | 4.993 | 4.291 | 4.300 | -14.06% | +0.23% |
| Request latency p50, s | 2.002 | 1.685 | 1.637 | -15.83% | -2.90% |
| Request latency p90, s | 13.882 | 11.610 | 11.544 | -16.37% | -0.57% |
| Request latency p95, s | 19.763 | 16.122 | 16.694 | -18.42% | +3.55% |
| Completed measured requests | 608.000 | 638.000 | 635.000 | +4.93% | -0.47% |
| Mean input tokens/request | 183,323.122 | 188,334.517 | 187,770.929 | +2.73% | -0.30% |
| Mean output tokens/request | 1,303.707 | 1,295.643 | 1,297.449 | -0.62% | +0.14% |
| Exported window including drain, s | 3,602.716 | 3,630.001 | 3,630.000 | +0.76% | ≈0.00% |
| Measured request errors | Not reported | 0 | 0 | — | — |
| Profiling cancellations | Not reported | 0 | 0 | — | — |
| Output-length mismatches | Not reported | 0 | 0 | — | — |
| Submission valid | Not reported | true | true | — | — |
| Response-reported prompt reuse | Not reported | 97.715% | 97.700% | — | -0.015 percentage points |
| Server-reported GPU cache hit | 98.215% | Not used | Not used (telemetry caveat) | — | — |

## Table 2: FPM details

Counts cover the full capture, including smoke, warmup, profiling and idle heartbeats.

| FPM detail | Value | Scope |
| --- | ---: | --- |
| Total records | 230,772 | One DP-rank stream |
| Active iteration records | 229,017 | At least one scheduled request |
| Pure prefill | 961 | Prefill > 0; decode = 0 |
| Pure decode | 228,056 | Decode > 0; prefill = 0 |
| Mixed | 0 | Both counts > 0 |
| No scheduled requests | 1,755 | All time=0; excluded from active records |
| DP ranks covered | 1 (rank0) | TP8 without attention DP, not seven missing ranks |
| Active records with positive timing | 229,017 / 229,017 | Full capture |
| Rejected / counter gaps / non-increasing | 0 / 0 / 0 | Counters 0–230771 |
| Raw JSONL size | 123,378,989 bytes | 123.38 MB; 117.66 MiB |
| gzip size | Not generated | Raw capture retained |

The recorder diagnostics are not a guarantee against publisher-side loss.
FPM `wall_time` is the span between existing GPU timing boundaries, not full
iteration latency or client TTFT/ITL.

## Results, artifacts and limitations

FPM-on warmup: 44 successful requests. Profiling: 635 successes, zero errors/cancellations,
`submission_valid=true`; TTFT/ITL coverage 99.9344%/100%. Measured sending ran
17:06:12–18:06:12 PDT on September 9; drain ended 18:06:42. Job 4208956 completed 0:0.
The baseline had 638 successes, zero errors/cancellations and valid coverage.

Output throughput changes -0.33%, total throughput -0.77%, ITL p50 +0.05%.
These historical, different-node runs do not establish zero FPM overhead:
runtime patch, resolved client dependencies and closed-loop request sets differ.
SA also uses an older runtime, no Dynamo frontend and a smaller GPU KV pool
(1,704,256 versus our observed 1,961,024 tokens). c4 is not a saturation test.
Do not attribute the TTFT improvement solely to disabling HiCache.

Optional persistent dataset-cache writes failed for lack of home space, but the
complete runtime mmap was built and both benchmarks finished. The on run had a
recoverable warmup allocator warning. Prometheus counter-reset warnings were not
corroborated by worker/FPM stream resets; their cause remains unresolved, so do not
use affected server counter deltas to prove performance. The table uses client exports.
Both runs retain drain/pending-join cleanup caveats; no measured requests were cancelled.

Raw results:
`/home/scratch.hongkuanz_gpu/agentx-results/job-4187044/` and
`/home/scratch.hongkuanz_gpu/agentx-results/job-4208956/on/`.
The recording and all exports are also downloaded to the SSH workstation.
Run on your own computer (substitute your usual SSH alias if needed):

```bash
scp hongkuanz@hzhou-workstation:/home/hongkuanz/Experiments/agentx-fpm-ab-20260909/job-4208956/on/fpm.jsonl ./fpm-4208956.jsonl
sha256sum fpm-4208956.jsonl
```

Expected SHA256:
`0c6e52190021247e095f518c8413a4d8afc2b9c38aab8bbe67fd1ab0a9e1c8bf`.
Raw traces stay outside Git; final machine-readable results remain on scratch.
