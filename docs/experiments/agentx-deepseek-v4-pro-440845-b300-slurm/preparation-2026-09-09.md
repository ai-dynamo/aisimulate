<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4-Pro / AgentX 440845 preparation: September 9, 2026

## Scope and status

Prepare the checkpoint on `/home/scratch.hongkuanz_gpu` and establish a separate
B300 experiment record for [AgentX 440845](https://inferencex.semianalysis.com/inference/agentic/440845).
The user subsequently authorized one FPM-off run followed by one FPM-on run,
with FPM raw data retained and results compared. Both cases disable HiCache.

## Checkpoint

Download `deepseek-ai/DeepSeek-V4-Pro` at revision
`b5968e9190ef611bbf34a7229255be88a0e937c1`. This revision is visible in the
reference client's config/tokenizer fetches. The reference server's local weight
revision was not recorded, so exact weight identity is not yet established.

**Download started; completion and shard validation are pending.**
The preparation agent confirmed the following on September 9, 2026:

| Item | Value |
| --- | --- |
| Running preparation job | `4208387` |
| Node | `lego-c2-qs-25` |
| Requested resources | 4 CPUs, 16 GiB RAM, 0 GPUs |
| Expected snapshot | 91 files, including 64 safetensors shards |
| Expected total bytes | 864739856856 (approximately 805.35 GiB) |
| Transfer | Four HTTP download workers; Xet disabled to bound memory |

Snapshot destination:

```text
/home/scratch.hongkuanz_gpu/models/hub/models--deepseek-ai--DeepSeek-V4-Pro/snapshots/b5968e9190ef611bbf34a7229255be88a0e937c1
```

Evidence paths:

```text
/home/scratch.hongkuanz_gpu/agentx-dsv4-pro-440845-checkpoint/manifest.json
/home/scratch.hongkuanz_gpu/dsv4-download-4208387.log
/home/scratch.hongkuanz_gpu/agentx-dsv4-pro-440845-checkpoint/download-result.json
```

The manifest exists; `download-result.json` is the planned terminal success/error
report, not evidence of success at this update. The snapshot uses the HF cache
layout: preserve backing blobs when moving it.

An initial frontend submission failed because the Slurm CLI filter could not
import `pydantic`. Preparation job `4208366` on `computelab-armbuild-3` then
failed in TaskProlog before downloading. The replacement job above reached the
transfer phase. Unrelated jobs were preserved.

Download completion requires every safetensors index entry to resolve to a
complete shard, file sizes to match the pinned Hub metadata, and a durable
verification report. A running job or existing directory alone is insufficient.
The preparation allocation must exit after validation; keep checkpoint files and
small verification artifacts in scratch.

## B300 availability snapshot

Read-only CDB, `sinfo` and `scontrol` queries on SC-01 found complete eight-GPU
DGX B300 and B300 NVL8 nodes, with approximately 2 TB host RAM. No unreserved
idle full node was available to `hongkuanz` at query time.

- `umb-b300-001/003/005/025/026`: allocated DGX B300 nodes.
- `umb-b300-083` through `089`: reserved under `ALP--COM-5529`; the authorized
  users did not include `hongkuanz`.
- `umb-b300-dp-148`: MIXED/PLANNED, but all eight GPUs allocated.
- `umb-b300-dp-185`: MIXED/PLANNED, five GPUs allocated, only three unallocated.

These are point-in-time observations; re-query before submitting a benchmark.
No B300 allocation was submitted for checkpoint preparation.

## Reproduction checks still open

- Verify checkpoint completion and release the CPU preparation allocation.
- Use the same FPM-fixed linux/amd64 image as B200 GLM job `4207957`, as pinned
  in the [runtime plan](README.md#planned-local-runtime-shared-fpm-fixed-x86-image).
  Validate DSv4 GPU compatibility and align exact AIPerf revision/settings.
- Obtain a complete B300 node and validate interconnect and actual GPU identity.
- Disable HiCache in both runs per user instruction; retain GPU radix caching
  and record measured GPU capacity and hit rates. This differs from the reference.
- Preserve the reference's DP-aware SGLang router and thinking chat template;
  record any later Dynamo substitution as a separate configuration difference.
- Run smoke/warmup and the c32 measured phase; no local performance result exists.

## Repository organization

The GLM-5.2 directories were renamed to include model and reference point:

- `agentx-glm-5.2-440958-b200-slurm`
- `agentx-glm-5.2-440082-gb200-hicache`

All 28 existing experiment files were checked against the preceding commit.
Only Markdown path references changed; launcher/configuration/result contents
were preserved. The root README, cross-links and both third-party notice copies
were updated to the new paths. See the [experiment index](../README.md).

## Automatic execution handoff

The local controller bundle is
`/home/hongkuanz/Experiments/agentx-dsv4-440845-ab-20260909/`.
Its dispatcher waits for the exact checkpoint revision, 91 verified files,
64 shards and no verification errors, plus successful image preflight, before
submitting the pair. State, submission ID and controller logs are retained there.
A failed preparation blocks submission. The controller collects the terminal
Slurm state and small result artifacts, then publishes a scoped GitHub result
record; the large raw client and FPM files remain on scratch. Scheduling still
depends on an available B300 node.
