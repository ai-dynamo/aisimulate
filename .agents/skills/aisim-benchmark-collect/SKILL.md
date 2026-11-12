---
name: aisim-benchmark-collect
description: Collect op, FPM or end-to-end AISim measurements on Slurm or Kubernetes and record their GPU time from the scheduler's job record. Use after a plan identifies missing or stale cells and the user has authorized GPU use.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags: [aisim, benchmark, collection, slurm, kubernetes, gpu-time]
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Collect and record GPU time

Stage 2 of [`aisim-benchmark`](../aisim-benchmark/SKILL.md). Only for cells the plan
marked `collect` or `recollect_or_reevaluate`, within the user's authorization.

## Collect

| Layer | How |
| --- | --- |
| op | Collector under `python/aisim/collector/`; read its README and the collector rules in `AGENTS.md` first |
| FPM | `aisim onboard` flow and [FPM self-service](../../../docs/perf-model/fpm-self-service/README.md) |
| E2E | `aisim-eval` collection guidance and its InferenceX recipes |

Run through `slurm-run` or `k8s-run`. Smoke first (small shape, check that rank, step and timing
fields exist), then the sweep. Save the job ID or Job/Pod names immediately; every job feeds GPU time.

Before the sweep, answer each [guidelines](../../../docs/perf-model/collector/README.md#benchmark-guidelines)
row in the run's manifest: observed (not requested) settings, workload, warmup count and cache state,
timed region, timed ranks. Keep raw tool output unmodified with its hash, command and image digest.

## Record GPU time (automatic, from the job record)

After the job reaches a terminal state, save the scheduler output and run the helper; it calls
`gpu_time_from_sacct` or `gpu_time_from_kubernetes` and prints the `GpuTime` plus findings at
archival and qualification level:

```bash
S=.agents/skills/aisim-benchmark-collect/scripts/record_gpu_time.py
sacct -j <ids> --parsable2 -o JobID,JobName,State,Submit,Start,End,ElapsedRaw,Elapsed,NNodes,AllocTRES > sacct.txt
python/aisim/.venv/bin/python $S slurm --sacct sacct.txt --gpu-type <observed type>
kubectl get job <name> -o json > job.json; kubectl get pods -l job-name=<name> -o json > pods.json
python/aisim/.venv/bin/python $S k8s --job job.json --pods pods.json --gpu-type <observed type>
```

- Include every job of the campaign (failed and requeued ones become `failed` / `retried`
  attempts, not part of the result). Queue wait is excluded by construction.
- A parser error (unfinished job, unknown state, GPU-count mismatch) means the time is not final;
  fix the cause or leave `gpu_time` unknown. Never type seconds by hand to fill a gap.
- `--gpu-type` comes from node inventory or `nvidia-smi` in the job log. Without it the type stays null.
- If no job record exists (laptop, other scheduler), `GpuTime(authority="declared", ...)` is allowed
  only with values the user stated, and it warns at qualification. Otherwise `GpuTime.unknown()`.
- Phase split (setup, warmup, measure, teardown) is optional; record it only from timestamps in the logs.

Store the output beside the run's manifest. Handoff: run directory, job IDs, `gpu_time`, findings,
failed cells against the frozen matrix. Collection success does not mean coverage or accuracy.
