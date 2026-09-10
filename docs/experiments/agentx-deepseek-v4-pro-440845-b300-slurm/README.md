<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4-Pro / AgentX 440845 — B300 reproduction and results

Completed job `4209414` on `umb-b300-dp-127`: one ordered FPM-off/on pair,
same eight-GPU node and fixed runtime, c32, HiCache disabled. Campaign and
allocation completed 0:0 and the allocation is released.

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
when reporting reproduction fidelity. Our client is pinned to SemiAnalysisAI/aiperf `754356e9a39acc6cc6afb242d123bb57c3fb6f75`; exact arguments are in campaign.py.


## Reproduce

Use the shared FPM-fixed amd64 manifest
`nvcr.io/nvidian/dynamo-dev/sglang-agentx@sha256:f856a45537f82e1900ea7607edcbaa7f77fbb2e70220eae522d1d50d0046727e`.
It backports SGLang PR#38711 at `ed18d64951b93ac252d921ee037ce0d3327eda1f`
onto `71de97b264b04dcd514cf904003028aefe9775c8`, with isolated AIPerf in
`/opt/agentx-aiperf`. This differs from SA's runtime. The local pair disables
HiCache and omits the removed `--prefill-decode-interval 20` option, retaining
prefill delayer; no equivalence to the old scheduling policy is assumed.
Synthetic acceptance is a performance setting, not a quality test.

1. Reuse the verified model snapshot under
   `/home/scratch.hongkuanz_gpu/models/hub/models--deepseek-ai--DeepSeek-V4-Pro/snapshots/b5968e9190ef611bbf34a7229255be88a0e937c1`.
   Verification reports are in `agentx-dsv4-pro-440845-checkpoint/` on that scratch.
   Required:91 files,64 safetensors shards,864739856856 total bytes,
   `download-result.json` status=complete with no errors, and `manifest.json`.
   Preserve HF backing blobs. Downloads/builds must run inside compute allocations.
2. Reuse `images/sglang-agentx-fpm-f856a455-amd64.sqsh` on scratch.
   Its SHA256 is `2a75f79d921933731c2220845e8680ae25c50af2a52addd31ac07bd3e3048987`.
   Enroot3.5 uses `docker://nvcr.io#nvidian/dynamo-dev/sglang-agentx:sha256:f856a45537f82e1900ea7607edcbaa7f77fbb2e70220eae522d1d50d0046727e`
   when importing a new cache inside a compute allocation.
3. Stage [campaign.py](campaign.py), [record_fpm.py](record_fpm.py),
   [compare.py](compare.py), [submit.sh](submit.sh) and [provenance.txt](provenance.txt)
   in `/home/scratch.hongkuanz_gpu/agentx-dsv4-440845-ab-20260909/`.
   Preserve that bundle's `image-squashfs.sha256` (matching the cache above) and
   `deepseek_v4_thinking.jinja` from the pinned InferenceX revision linked above,
   with its license/provenance. Neither template nor weights are vendored here.
   Adapt scratch/account/partition paths for other users.
4. Discover an available eight-GPU B300 partition, then submit:

```bash
ssh hongkuanz@computelab-sc-01 \
  'sbatch /home/scratch.hongkuanz_gpu/agentx-dsv4-440845-ab-20260909/submit.sh'
```

The script requests eight GPUs,112 CPUs, an exclusive node, all host RAM and four
hours, with a termination warning three minutes before the limit. The campaign
checks hardware, image imports and checkpoint validation before serving.
It starts native SGLang plus DP-aware consistent-hashing SGLang router (not a
Dynamo frontend), then runs off followed by on, each with fresh engine/KV,
the same warmup and3600s measurement. Compilation caches may be reused.
The recorder subscribes to all eight rank-local IPC endpoints and copies buffered
capture to scratch before teardown. The comparison script runs after the pair.

### Dataset setup safeguard

The completed pair needed manual recovery before each measured phase:
dataset reconstruction finished, but executor cleanup stalled with its forkserver
waiting for children. This is not automatically repaired by campaign.py.
Before any intervention, verify the exact owned Slurm cgroup, completed393/393
reconstruction, stack waiting in pool termination/join, blocked/pending SIGCHLD,
and all sixteen reconstruction children already zombies. Only then was that
specific forkserver terminated to allow assembly to resume; never kill a generic
Python/forkserver process or interrupt unfinished reconstruction. If the checks
do not hold, stop and diagnose rather than applying this workaround.
No engine/client source or serving setting was changed during that recovery.

## Table 1: SA versus our FPM-off and FPM-on performance

SA values are the recorded [AgentX 440845](https://inferencex.semianalysis.com/inference/agentic/440845)
row from the [published benchmark API](https://inferencex.semianalysis.com/api/v1/benchmarks?model=DeepSeek-V4-Pro),
retrieved September 9, 2026. Our columns use final measured-phase client exports.
All three configurations use eight B300 GPUs, TP8/EP8, attention DP8 and c32;
SA enables HiCache, while both of our cases disable it and use the shared
FPM-fixed runtime described above.

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

Each setup needed the guarded forkserver recovery described above. No serving or
client source changed. Optional persistent mmap-cache population failed with
ENOSPC, but the complete 6386608130-byte runtime mmap was built and replay ran.

The on-case Prometheus export reported counter-reset warnings across several
series. Their precise cause is unresolved; affected Prometheus delta/rate
statistics should not be used to substantiate this performance comparison.
The table uses client request exports, and native FPM independently has continuous
per-rank counters with no resets. Preserve the warnings in `on/client.log`.


## Saved results and completion checks

Raw artifacts: `/home/scratch.hongkuanz_gpu/agentx-dsv4-results/job-4209414/`.
Retain off/on client exports, commands/environment, engine/router logs and
phase timestamps; on also retains FPM JSONL/gzip, checksum and validation.
Machine-readable final results are linked in the tables above; hardware identity
is [job-4209414-hardware.csv](job-4209414-hardware.csv), and final campaign state is
[job-4209414-campaign-result.json](job-4209414-campaign-result.json).
Check request validity, rank coverage, final campaign exit and absence from
squeue before declaring a reproduction complete. Diagnostic probes are not
benchmark cases; their exit codes do not override the successful campaign result.
