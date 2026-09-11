<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AgentX DSv4 rerun after the fixed-workload FPM sweep

**Completed:** job `4228907` ran both 600-second diagnostic cases on
`umb-b300-dp-192`, completed 0:0 and released its allocation. The completed
[fixed 8K/1K experiment](../../sglang-fpm-fixed-8k1k/README.md) used `umb-b300-dp-142`. The fixed sweep
showed small throughput gaps; this rerun checks whether AgentX reproduces its
prior gap and whether request context/cache distributions explain it.

The workload is [AgentX 440845](https://inferencex.semianalysis.com/inference/agentic/440845):
DeepSeek-V4-Pro, c32, all 393 Weka trajectories, seed 42, TP8/EP8, attention DP8, PP1,
HiCacheoff, synthetic EAGLE/native-MTP acceptance 2.49. Both runs use the same
FPM-fixed amd64 image digest
`sha256:f856a45537f82e1900ea7607edcbaa7f77fbb2e70220eae522d1d50d0046727e`.

Run off then on, each with fresh engine/KV and equivalent warmup, then 600 seconds of
measured sending, per the request for shorter diagnostic iterations. FPM-on records all eight rank streams. Serving/router settings
match the original AgentX experiment, including `--max-running-requests 64`; this differs
from the fixed sweep's 256-slot pool. The goal is to reproduce the AgentX comparison,
not treat the fixed workload and replay as otherwise identical configurations.

Dataset setup uses `AIPERF_DATASET_WEKA_PARALLEL_WORKERS=1` to avoid the previously
verified blocked-SIGCHLD forkserver cleanup bug. The client documents identical
reconstruction semantics for serial/parallel paths. Its content-addressed mmap
cache is located at `/home/scratch.hongkuanz_gpu/agentx-dsv4-rerun-mmap-cache-20260910/`
to avoid small-home ENOSPC. Both cases use the same setup settings; these do not
change the engine or the authored trace. Startup wait is bounded at 3600 seconds.

Files are staged at `/home/scratch.hongkuanz_gpu/agentx-dsv4-rerun-20260910/` with
the pinned thinking template/license and existing image-squashfs checksum.
`submit.sh` requests one exclusive 8×B300 node for at most 2.5 hours. Raw results:
`/home/scratch.hongkuanz_gpu/agentx-dsv4-rerun-results/job-4228907/`.

After both runs, inspect final client validity/errors/cancellations and FPM
integrity, then compare matched source-request keys and distributions of total
prompt length, uncached prompt tokens and prefix reuse. Keep unmatched/recycled
requests visible. Stratified replay evidence can support a workload-effect
hypothesis but does not by itself establish causality.

Configuration provenance remains SemiAnalysisAI/InferenceX
`fb85931b1edec09f9498509835a8c814bebe3c65`,
`benchmarks/single_node/agentic/dsv4_fp4_b300_sglang_mtp.sh`, Apache-2.0.
The original thinking template is a runtime dependency, not vendored here.

## Short diagnostic window

The user requested approximately 10-minute measurements to accelerate iteration.
Both off/on cases use `--benchmark-duration 600 --unsafe-override`: the pinned
AgentX scenario normally enforces a 900-second minimum. Consequently these runs
are explicitly diagnostic, and `submission_valid=false` is expected from that
protocol override. Request errors, cancellations, coverage and FPM validity still
need independent checks. Keep warmup separate, and do not compare these as formal
one-hour leaderboard submissions. A short run may miss late cache-pressure or
eviction effects.

## Queue and initial request-level evidence

The initial node 142 constraint was removed because its running allocation was
expected to last until 17:40 PDT. The candidate pool stays within B300 NVL8;
the AgentX off/on pair still runs on one node. If the selected node differs from
the fixed sweep, keep that hardware change explicit in cross-workload comparisons.

Analysis of the original one-hour pair4209414 finds 3388 shared unique source
requests. Across all successful measured requests, average uncached input was
4657 tokens off versus 4864 on (+4.4%). Thus the original throughput gap includes
an observed workload difference. Matched-source context/cache strata are preserved
in [original4209414 request analysis](original-4209414-request-analysis.json).
This is observational evidence, not a causal attribution of the gap to cache hits.

## NVLink failure and retry

Attempt 4227233 on `umb-b300-dp-148` failed during the first warmup prefill,
with `uncorrectable NVLink error detected during the execution` from DSv4's
`main_norm_rope.cuh:430` on all eight scheduler ranks. HTTP/metrics processes
remained alive while the GPU schedulers exited, leaving all 34 warmup requests
waiting. No measured phase started; this is not an FPM performance result.

The primary task saved the server tracebacks and an NVIDIA status dump, cancelled
that allocation, and submitted retry 4228907 with node 148 excluded. The launcher
now detects scheduler exceptions incrementally in server logs and fails promptly
rather than relying only on the surviving HTTP parent. Model/image/workload
settings remain the same. Failed-attempt evidence stays in the job 4227233 root.

## Completed short AgentX comparison

Both cases used the same node/image and separate warmup. These 600-second runs
used `--unsafe-override`, so both report `submission_valid=false` and the canonical
profiling-duration coverage check was skipped. This is expected for the explicitly
requested short diagnostic, not a claim of a valid leaderboard submission.

| Metric | FPM off | FPM on | On/off change |
| --- | ---: | ---: | ---: |
| Total tokens/s/GPU | 14120.22 | 14249.01 | +0.91% |
| Output tokens/s/GPU | 96.280 | 97.051 | +0.80% |
| TTFT p50, ms | 1565.22 | 1459.96 | -6.73% |
| TTFT p90, ms | 3616.66 | 3314.50 | -8.35% |
| ITL p90, ms | 29.265 | 31.750 | +8.49% |
| Completed measured requests | 800 | 802 | +0.25% |
| Measured request errors | 0 | 0 | — |
| Profiling cancellations | 5 | 4 | — |
| Mean prompt tokens/request | 89754.56 | 90347.98 | +0.66% |
| Mean uncached prompt tokens/request | 5337.92 | 5342.89 | +0.09% |
| Token-weighted prompt reuse | 94.0528% | 94.0863% | +0.0336 percentage points |

The exported windows are approximately 640 seconds each: 600 seconds sending plus
30 seconds drain and 10 seconds cancellation-credit handling. Retain those original
throughput denominators. Warmup exports report 353/352 successes and 1/2 empty-content
response errors, respectively; phase progress counters did not expose those errors.

### Request-level checks

Source-request matching uses trace/kind/outer/inner/conversation/turn fields,
excludes duplicate keys and preserves unmatched requests. For the near-equal-work
subset, output length must match exactly and both prompt and uncached-token counts
must differ by no more than 4 tokens. Exact-match results are retained in JSON too.

| Matched population | Requests | Median paired TTFT change | Median paired ITL change |
| --- | ---: | ---: | ---: |
| Shared unique source requests | 786 | -5.03% | -0.78% |
| Near-equal prompt/cache work and equal output length | 776 | -5.26% | -0.75% |
| Exact prompt/cache token counts | 165 | -2.87% | -4.27% |

Within the near-equal-work subset, 128K–256K prompts (87 requests) have median
TTFT/ITL changes of -2.58%/-1.06%; >256K prompts (27 requests) have -2.21%/-0.18%.
There is no consistent slowdown concentrated in long prompts in this short rerun.
The aggregate ITL p90 increase describes a tail-distribution change, not a uniform
per-request slowdown; the matched median moves in the opposite direction.

### Interpretation

The fixed 8K/1K sweep observed throughput gaps below 0.3% in magnitude across all
three concurrency levels. This short AgentX pair did not reproduce the original
one-hour throughput regression: output throughput increased 0.8%, while latency
percentiles moved in both directions. The original one-hour on case had 4.4% more
uncached prompt work per request; the short pair differs by only 0.09% in that
measure. Request/cache work and scheduler timing therefore need to be controlled
before interpreting aggregate differences as FPM overhead.

These observations do not establish that long prefill or prefix hits intrinsically
amplify FPM cost. Nor do they establish zero overhead or a performance benefit from
FPM. The short window may miss later cache pressure/evictions, and it has only one
ordered pair. Fixed-workload and AgentX runs also used different NVL8 nodes, as
recorded above. The FPM statistics implementation iterates over requests/queue
entries and reads lengths; it does not scan every prompt token just because the
context is longer. Scheduling and timing interactions remain possible.

### FPM and artifacts

The new capture has 136741 records across 8 DP ranks, 106509 active records and
104116 decode records. Validation found 0 invalid records, 0 observed counter gaps
and 0 resets. Raw size: 75076664 bytes; gzip: 5412306 bytes. Raw SHA256:
`3c306bc64aa3f8d7c160408725c597b48c101d0d7221a49a7401ba264eae4715`.

[Aggregate comparison](job-4228907-comparison.json),
[request/context/cache analysis](job-4228907-request-analysis.json), and
[FPM validation](job-4228907-fpm-validation.json) retain the full evidence.
Raw client and FPM files are under
`/home/scratch.hongkuanz_gpu/agentx-dsv4-rerun-results/job-4228907/`.

```bash
scp computelab-sc-01:/home/scratch.hongkuanz_gpu/agentx-dsv4-rerun-results/job-4228907/on/fpm.jsonl.gz .
gzip -dc fpm.jsonl.gz | sha256sum
```
