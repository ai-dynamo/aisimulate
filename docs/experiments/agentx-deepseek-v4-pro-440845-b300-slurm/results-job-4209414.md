<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DSv4 AgentX 440845: FPM off/on comparison

Both cases use one B300 node, the same FPM-fixed image, c32 and HiCache disabled.

## Table 1: SA versus our FPM-off and FPM-on performance

SA values are the recorded [AgentX 440845](https://inferencex.semianalysis.com/inference/agentic/440845)
row from the [published benchmark API](https://inferencex.semianalysis.com/api/v1/benchmarks?model=DeepSeek-V4-Pro),
retrieved September 9, 2026. Our columns use final measured-phase client exports.
All three configurations use eight B300 GPUs, TP8/EP8, attention DP8 and c32;
SA enables HiCache, while both of our cases disable it and use the shared
FPM-fixed runtime described in [the runbook](README.md).

Changes are `100 * (new / baseline - 1)` using unrounded values. Higher
throughput and lower latency are preferable. Missing reference fields remain
unreported; the two cache-hit aggregations are deliberately shown separately.

| Metric | SA 440845 | Ours FPM off | Ours FPM on | Off vs SA | On vs off |
| --- | ---: | ---: | ---: | ---: | ---: |
| Total throughput, tokens/s | 139,843.35 | 141,041.93 | 140,120.42 | +0.86% | -0.65% |
| Total throughput, tokens/s/GPU | 17,480.42 | 17,630.24 | 17,515.05 | +0.86% | -0.65% |
| Output throughput, tokens/s | 919.875 | 929.159 | 924.580 | +1.01% | -0.49% |
| Output throughput, tokens/s/GPU | 114.984 | 116.145 | 115.573 | +1.01% | -0.49% |
| TTFT mean, s | 3.642 | 1.909 | 2.021 | -47.58% | +5.84% |
| TTFT p50, s | 1.836 | 1.399 | 1.429 | -23.80% | +2.13% |
| TTFT p90, s | 7.727 | 3.380 | 3.468 | -56.26% | +2.62% |
| TTFT p95, s | 13.308 | 4.254 | 4.674 | -68.04% | +9.88% |
| ITL mean, ms | 13.940 | 17.104 | 17.310 | +22.70% | +1.20% |
| ITL p50, ms | 12.890 | 13.155 | 13.208 | +2.06% | +0.41% |
| ITL p90, ms | 19.500 | 23.711 | 23.589 | +21.59% | -0.52% |
| ITL p95, ms | 21.120 | 29.514 | 29.354 | +39.74% | -0.54% |
| Request latency mean, s | 17.961 | 17.204 | 17.463 | -4.21% | +1.51% |
| Request latency p50, s | 8.784 | 7.877 | 8.014 | -10.33% | +1.74% |
| Request latency p90, s | 42.797 | 40.984 | 42.679 | -4.24% | +4.14% |
| Request latency p95, s | 65.066 | 65.079 | 66.381 | +0.02% | +2.00% |
| Completed measured requests | 3,410 | 3,423 | 3,399 | +0.38% | -0.70% |
| Mean input tokens/request | 147,841.71 | 148,996.19 | 149,066.32 | +0.78% | +0.05% |
| Mean output tokens/request | 978.93 | 988.06 | 990.14 | +0.93% | +0.21% |
| Exported window including drain, s | 3,628.906 | 3,640.000 | 3,640.001 | +0.31% | +0.00% |
| Measured request errors | Not reported in cited API row | 0 | 0 | — | — |
| Profiling cancellations | Not reported in cited API row | 3 | 4 | — | — |
| Output-length mismatches | Not reported in cited API row | 0 | 0 | — | — |
| Submission valid | Not reported in cited API row | true | true | — | — |
| Response-reported prompt reuse | Not reported in cited API row | 96.874% | 96.737% | — | -0.137 percentage points |
| Server-reported GPU cache hit fraction | 95.871% | Not used (see telemetry caveat) | Not used (see telemetry caveat) | — | — |

Validity and collection:

- Submission valid: off=True, on=True.
- Profiling cancellations: off=3, on=4.
- Off warmup export: 353 successful and one empty-content response error,
  although the phase progress log counted all 354 as completed with zero errors.
  On warmup exported 354 successful requests and omitted the error-count metric.

One ordered pair, off then on; warmed compilation cache and fresh engine/KV plus warmup per case. Closed-loop requests may differ; no statistical overhead claim.

FPM rank coverage, counter gaps, request validity and cancellations are retained
in [the machine-readable comparison](job-4209414-comparison.json).

## Table 2: FPM details

Counts cover the **complete capture**, including startup, smoke, warmup,
measurement and drain. Each DP rank's record counts separately; these are not
deduplicated global iterations. Classification uses the scheduled-request
prefill/decode counts, not the timing value. No-request records can include idle
heartbeats and ranks with no scheduled work; they are not counted as active
iterations. [Machine-readable type counts](job-4209414-fpm-iteration-counts.json)
include the per-rank breakdown.

| FPM detail | Value | Definition / scope |
| --- | ---: | --- |
| Total records | 808,721 | All eight ranks and all capture phases |
| Active iteration records | 628,149 | At least one scheduled prefill or decode request |
| Pure prefill iterations | 5,670 | Prefill request count > 0; decode count = 0 |
| Pure decode iterations | 622,479 | Decode request count > 0; prefill count = 0 |
| Mixed iterations | 0 | Both prefill and decode request counts > 0 |
| No-scheduled-request records | 180,572 | Both counts = 0; excluded from active iterations |
| DP ranks covered | 8 (0–7) | Every rank has active and decode records |
| Active records with positive GPU timing | 628,149 / 628,149 | Checked across the complete capture |
| Invalid records | 0 | Schema, wire counter, timing and decode-length checks |
| Observed counter gaps / resets | 0 / 0 | Checked separately for each rank |
| Raw JSONL size | 445,560,968 bytes (445.56 MB; 424.92 MiB) | `on/fpm.jsonl` |
| gzip size | 32,900,290 bytes (32.90 MB; 31.38 MiB) | `on/fpm.jsonl.gz` |
| Compression ratio | 13.54:1 | Raw size divided by gzip size |

Raw SHA256, verified against the local copy:
`bf0f277dbd74a8276e0ed8021a882a199a0a394f18132fded80c3e6fa2df2614`.
The [timing and checksum audit](job-4209414-fpm-audit.json) separately includes
per-rank timing statistics for the 3600-second measured window using recorder
receive timestamps; Table 2's counts and sizes cover the full file.

Download the compressed capture from Computelab, then verify its decompressed
contents against the SHA256 above:

```bash
scp computelab-sc-01:/home/scratch.hongkuanz_gpu/agentx-dsv4-results/job-4209414/on/fpm.jsonl.gz .
gzip -dc fpm.jsonl.gz | sha256sum
```

Both raw and compressed copies, `fpm.sha256`, `fpm-validation.json` and
`fpm-audit.json` remain in that scratch `on/` directory.

## Collection limitations

Both measured exports report `submission_valid=true` and no request errors or
output-length mismatches. The runs reached the sending duration, then used the
30-second grace period and a further 10-second cancellation-credit timeout;
profiling cancellations were three and four. Preserve these exported windows
instead of renormalizing throughput to exactly 3600 seconds.

Each setup needed the guarded forkserver recovery documented in the preparation
log. The on-case recovery occurred promptly after reconstruction; no serving or
client source changed. Optional persistent mmap-cache population failed with
ENOSPC, but the complete 6386608130-byte runtime mmap was built and replay ran.

The on-case Prometheus export reported counter-reset warnings across several
series. Their precise cause is unresolved; affected Prometheus delta/rate
statistics should not be used to substantiate this performance comparison.
The table uses client request exports, and native FPM independently has continuous
per-rank counters with no resets. Preserve the warnings in `on/client.log`.

## Comparison with the published point

[AgentX 440845](https://inferencex.semianalysis.com/inference/agentic/440845)
reports 17480.42 total tokens/s/GPU, 114.984 output tokens/s/GPU, TTFT p90
7.727 seconds and ITL p90 19.5 milliseconds. Our FPM-off baseline has similar
aggregate throughput, lower TTFT and approximately 21.6% higher p90 ITL.
This is not strict parity: our runtime differs, HiCache is disabled and the
reference prefill/decode interval option was removed in our newer engine.
Do not attribute the TTFT difference solely to HiCache or the sub-1% paired
throughput difference solely to FPM from this single ordered pair.

Job: `4209414`, node `umb-b300-dp-127`: the allocation and campaign step
completed 0:0. Diagnostic step `.16` is the deliberately timed-out strace and
`.25` is a failed host-path probe before locating the capture inside the container;
neither is a benchmark failure.

<details>
<summary>Full Slurm accounting, including monitoring and diagnostic steps</summary>

```text
4209414|COMPLETED|0:0|umb-b300-dp-127|03:14:09
4209414.batch|COMPLETED|0:0|umb-b300-dp-127|03:14:09
4209414.extern|COMPLETED|0:0|umb-b300-dp-127|03:14:09
4209414.0|COMPLETED|0:0|umb-b300-dp-127|03:12:03
4209414.1|COMPLETED|0:0|umb-b300-dp-127|00:00:06
4209414.2|COMPLETED|0:0|umb-b300-dp-127|00:00:05
4209414.3|COMPLETED|0:0|umb-b300-dp-127|00:00:06
4209414.4|COMPLETED|0:0|umb-b300-dp-127|00:00:05
4209414.5|COMPLETED|0:0|umb-b300-dp-127|00:00:06
4209414.6|COMPLETED|0:0|umb-b300-dp-127|00:00:00
4209414.7|COMPLETED|0:0|umb-b300-dp-127|00:00:00
4209414.8|COMPLETED|0:0|umb-b300-dp-127|00:00:06
4209414.9|COMPLETED|0:0|umb-b300-dp-127|00:00:01
4209414.10|COMPLETED|0:0|umb-b300-dp-127|00:00:06
4209414.11|COMPLETED|0:0|umb-b300-dp-127|00:00:06
4209414.12|COMPLETED|0:0|umb-b300-dp-127|00:00:06
4209414.13|COMPLETED|0:0|umb-b300-dp-127|00:00:00
4209414.14|COMPLETED|0:0|umb-b300-dp-127|00:00:00
4209414.15|COMPLETED|0:0|umb-b300-dp-127|00:00:06
4209414.16|FAILED|124:0|umb-b300-dp-127|00:00:06
4209414.17|COMPLETED|0:0|umb-b300-dp-127|00:00:06
4209414.18|COMPLETED|0:0|umb-b300-dp-127|00:00:01
4209414.19|COMPLETED|0:0|umb-b300-dp-127|00:00:01
4209414.20|COMPLETED|0:0|umb-b300-dp-127|00:00:07
4209414.21|COMPLETED|0:0|umb-b300-dp-127|00:00:01
4209414.22|COMPLETED|0:0|umb-b300-dp-127|00:00:06
4209414.23|COMPLETED|0:0|umb-b300-dp-127|00:00:01
4209414.24|COMPLETED|0:0|umb-b300-dp-127|00:00:07
4209414.25|FAILED|1:0|umb-b300-dp-127|00:00:00
4209414.26|COMPLETED|0:0|umb-b300-dp-127|00:00:06
4209414.27|COMPLETED|0:0|umb-b300-dp-127|00:00:06
4209414.28|COMPLETED|0:0|umb-b300-dp-127|00:00:00
4209414.29|COMPLETED|0:0|umb-b300-dp-127|00:00:00
4209414.30|COMPLETED|0:0|umb-b300-dp-127|00:00:00
4209414.31|COMPLETED|0:0|umb-b300-dp-127|00:00:00
4209414.32|COMPLETED|0:0|umb-b300-dp-127|00:00:01
4209414.33|COMPLETED|0:0|umb-b300-dp-127|00:00:01
4209414.34|COMPLETED|0:0|umb-b300-dp-127|00:00:06
4209414.35|COMPLETED|0:0|umb-b300-dp-127|00:00:07
4209414.36|COMPLETED|0:0|umb-b300-dp-127|00:00:06
4209414.37|COMPLETED|0:0|umb-b300-dp-127|00:00:07
4209414.38|COMPLETED|0:0|umb-b300-dp-127|00:00:00
4209414.39|COMPLETED|0:0|umb-b300-dp-127|00:00:01
4209414.40|COMPLETED|0:0|umb-b300-dp-127|00:00:07
4209414.41|COMPLETED|0:0|umb-b300-dp-127|00:00:03
4209414.42|COMPLETED|0:0|umb-b300-dp-127|00:00:06
4209414.43|COMPLETED|0:0|umb-b300-dp-127|00:00:06
4209414.44|COMPLETED|0:0|umb-b300-dp-127|00:00:07
4209414.45|COMPLETED|0:0|umb-b300-dp-127|00:00:00
4209414.46|COMPLETED|0:0|umb-b300-dp-127|00:00:08
4209414.47|COMPLETED|0:0|umb-b300-dp-127|00:00:01
4209414.48|COMPLETED|0:0|umb-b300-dp-127|00:00:06
4209414.49|COMPLETED|0:0|umb-b300-dp-127|00:00:07
4209414.50|COMPLETED|0:0|umb-b300-dp-127|00:00:06
4209414.51|COMPLETED|0:0|umb-b300-dp-127|00:00:00
4209414.52|COMPLETED|0:0|umb-b300-dp-127|00:00:06
4209414.53|COMPLETED|0:0|umb-b300-dp-127|00:00:01
4209414.54|COMPLETED|0:0|umb-b300-dp-127|00:00:07
4209414.55|COMPLETED|0:0|umb-b300-dp-127|00:00:00
4209414.56|COMPLETED|0:0|umb-b300-dp-127|00:00:00
4209414.57|COMPLETED|0:0|umb-b300-dp-127|00:00:07
4209414.58|COMPLETED|0:0|umb-b300-dp-127|00:00:06
4209414.59|COMPLETED|0:0|umb-b300-dp-127|00:00:07
4209414.60|COMPLETED|0:0|umb-b300-dp-127|00:00:07
4209414.61|COMPLETED|0:0|umb-b300-dp-127|00:00:01
4209414.62|COMPLETED|0:0|umb-b300-dp-127|00:00:07
4209414.63|COMPLETED|0:0|umb-b300-dp-127|00:00:00
4209414.64|COMPLETED|0:0|umb-b300-dp-127|00:00:01
4209414.65|COMPLETED|0:0|umb-b300-dp-127|00:00:06
4209414.66|COMPLETED|0:0|umb-b300-dp-127|00:00:06
```

</details>

Raw artifacts: `/home/scratch.hongkuanz_gpu/agentx-dsv4-results/job-4209414/`.

No active allocation remained in squeue when this report was collected.

Campaign status:

```json
{
  "status": "complete",
  "completed_at_ns": 1789014599091525744,
  "cases": [
    "off",
    "on"
  ]
}
```
