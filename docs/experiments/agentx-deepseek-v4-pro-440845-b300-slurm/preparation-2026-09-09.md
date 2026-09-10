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

**Download complete and verified at approximately 17:21 PDT on September 9.**
The preparation agent confirmed the following on September 9, 2026:

| Item | Value |
| --- | --- |
| Completed preparation job | `4208387`; allocation released after validation |
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

`download-result.json` now reports `status=complete`, 91 verified files,
64 shards and `errors=[]`, with all file sizes matching the pinned Hub metadata.
`cleanup-result.json` verifies the allocation is absent from `squeue`; the download
step completed 0:0, followed by the cleanup watcher cancelling the allocation. The snapshot uses the HF cache
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

- Checkpoint verification and CPU allocation cleanup are complete.
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

## Preflight and dispatch update

- CPU-only x86 checks on `2u1g-b650-1035` completed: job `4208710` verified
  image FPM invariants and engine/router/client CLI startup; job `4208757`
  parsed the exact engine flags and exercised the native recorder against all
  eight SGLang publisher endpoints. Both exited 0:0 and released allocations.
- The newer image does not accept the reference `--prefill-decode-interval 20`.
  Both cases omit that removed setting and retain prefill delayer. This is an
  explicit scheduling difference from the published reference, shared by the pair.
- Native FPM validation passed rank coverage and decode/timing checks. Synthetic
  negative checks reject missing ranks and invalid timings; counter gaps are
  counted. Comparing identical historical AIPerf exports produced zero deltas.
- The pinned amd64 squashfs cache is
  `/home/scratch.hongkuanz_gpu/images/sglang-agentx-fpm-f856a455-amd64.sqsh`,
  SHA256 `2a75f79d921933731c2220845e8680ae25c50af2a52addd31ac07bd3e3048987`.
  Image preparation on ARM CPU job `4208645` produced the cache but could not
  execute its NVIDIA hook there; execution checks therefore used the native
  x86 jobs above. Failed image-authentication attempts were not GPU runs.
- The dispatcher is running locally with `dispatch-state.json`, `dispatch.log`
  and `dispatch.pid` in the controller bundle. After verified checkpoint completion it submitted B300 job `4209414`,
  which is pending with reason `Priority` at approximately 17:22 PDT.
- Slurm `sbatch --test-only` accepted the resource request. Its September 9
  estimate was September 10 at approximately 04:10 PDT; this is only a scheduling
  estimate, not a reservation or an actual submitted job ID.
- Both cases use UTC for saved timestamps. A Slurm termination signal three
  minutes before the allocation limit gives cleanup time; FPM is flushed and
  copied before slow engine teardown.

## Submitted campaign

Actual B300 job: **4209414**, submitted automatically after the checkpoint and
image gates passed. The initial state was `PENDING (Priority)`. The protocol
remains FPM-off then FPM-on, 3600 measured seconds each, both with HiCache off.
The raw output root will be
`/home/scratch.hongkuanz_gpu/agentx-dsv4-results/job-4209414/`.
The local submission record is
`/home/hongkuanz/Experiments/agentx-dsv4-440845-ab-20260909/submission.json`.
A GPT-5.6 Luna agent monitors the controller, queue, execution and artifact
validation; the primary task remains waiting and handles actionable failures.
No B300 performance result is available at submission time.
