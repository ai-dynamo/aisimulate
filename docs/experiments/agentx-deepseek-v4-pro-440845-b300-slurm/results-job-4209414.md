<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DSv4 AgentX 440845: FPM off/on comparison

Both cases use one B300 node, the same FPM-fixed image, c32 and HiCache disabled.

| Metric | FPM off | FPM on | On/off change |
| --- | ---: | ---: | ---: |
| Total tokens/s/GPU | 17630.24 | 17515.05 | -0.65% |
| Output tokens/s/GPU | 116.145 | 115.573 | -0.49% |
| TTFT p50, seconds | 1.399 | 1.429 | +2.13% |
| TTFT p90, seconds | 3.380 | 3.468 | +2.62% |
| ITL p90, milliseconds | 23.711 | 23.589 | -0.52% |
| Request latency p90, seconds | 40.984 | 42.679 | +4.14% |
| Completed requests | 3423 | 3399 | -0.70% |
| Response-reported prompt reuse | 96.874% | 96.737% | -0.14% relative |

Validity and collection:

- Submission valid: off=True, on=True.
- Profiling cancellations: off=3, on=4.
- Off warmup export: 353 successful and one empty-content response error,
  although the phase progress log counted all 354 as completed with zero errors.
  On warmup exported 354 successful requests and omitted the error-count metric.

One ordered pair, off then on; warmed compilation cache and fresh engine/KV plus warmup per case. Closed-loop requests may differ; no statistical overhead claim.

FPM rank coverage, counter gaps, request validity and cancellations are retained
in [the machine-readable comparison](job-4209414-comparison.json).

## Native FPM evidence

The full capture contains **808721 records**, including 628149 active records
and 622479 decode records. All eight DP ranks have active/decode coverage;
validation found **zero invalid records, zero counter gaps and zero resets**.
A full local audit also confirmed positive GPU timing for every active record.

- Raw: 445560968 bytes; gzip: 32900290 bytes.
- Raw SHA256: `bf0f277dbd74a8276e0ed8021a882a199a0a394f18132fded80c3e6fa2df2614`.
- [Timing and checksum audit](job-4209414-fpm-audit.json) includes per-rank
  statistics for the 3600-second measured window using recorder receive times.
  The complete raw capture also includes setup, smoke, warmup and drain.

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
