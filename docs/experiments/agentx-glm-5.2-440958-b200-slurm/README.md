<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AgentX GLM-5.2 on one B200 Slurm node

This is a hardware measurement runbook for later AISimulate validation, not an
AISimulate simulation command. It records the September 8, 2026 experiment against
[AgentX point 440958](https://inferencex.semianalysis.com/inference/agentic/440958).
The one-hour run is complete. See [results and comparison](results-2026-09-08.md)
for measured metrics, comparability limits and verified allocation cleanup.

September 9 update: the [FPM-fixed multi-architecture image build log](../agentx-glm-5.2-440082-gb200-hicache/fpm-image-2026-09-09.md)
records the new image shared with GB200 Kubernetes. The FPM-on/off performance
comparison is still pending; the baseline and cached image documented below
have not been replaced.

## Pinned configuration

| Item | Experiment |
| --- | --- |
| Hardware | One node, 8 B200 GPUs, 183359 MiB per GPU, 1000 W power limit |
| Driver | 610.57.04 |
| Dynamo | 1.5.0.dev20260908, commit `946accea5edfd778f5120a3096b082e54e5bce2b` |
| SGLang / FlashInfer | 0.5.18 / 0.6.17 |
| Model | `nvidia/GLM-5.2-NVFP4`, revision `53e0691e21895a3863a606dfd12910c69eba94ab` |
| Parallelism | One aggregated worker, TP8, EP1, no attention DP |
| Cache | FP8 E4M3 KV, GPU radix cache; HiCache/G2 offload disabled |
| Memory / prefill | Static fraction 0.83; chunk and max-prefill tokens 8192 |
| Scheduler | Max running requests 8; CUDA graph max batch size 8 |
| Speculation | EAGLE, 3 steps, top-k 1, 4 draft tokens; simulated acceptance length 2.99 |
| Load | AgentX scenario, concurrency 4, 3600-second measured phase, seed 42 |
| Dataset | `semianalysis_cc_traces_weka_062126`, all 393 trajectories |
| Client | [SemiAnalysisAI/aiperf](https://github.com/SemiAnalysisAI/aiperf/tree/754356e9a39acc6cc6afb242d123bb57c3fb6f75), commit `754356e9a39acc6cc6afb242d123bb57c3fb6f75` |

Image (linux/amd64):

```text
nvcr.io/nvidia/ai-dynamo/sglang-runtime-nightly@sha256:cc84ea52fc8e66fb61a54692a30b2dcfa4d9b2349aa39d565039657cba8c26bb
```

Use the digest, not a moving nightly tag. This is a performance experiment with
synthetic speculative acceptance, **not a text-quality evaluation**. The local
launcher is authored for this run; upstream client code is installed as a dependency,
not vendored here. Client transitive dependencies are not fully locked: archive
`pip freeze` alongside results when repeating the experiment.

## 1. Reuse the checkpoint and image cache

On Computelab, connect using your own Unix account to `computelab-sc-01`.
You need cluster access, an eligible Slurm account/partition, registry access, and
authorized model access. Use your own writable scratch allocation, not another
user's directory. Put these files there before the timed benchmark:

```text
<scratch>/
  images/dynamo-sglang-nightly-20260908-cc84ea52-amd64.sqsh
  models/hub/models--nvidia--GLM-5.2-NVFP4/snapshots/
    53e0691e21895a3863a606dfd12910c69eba94ab/
  agentx-run-scripts/run.sh
```

The checkpoint contains 47 safetensors shards (~432.90 GiB); the cached squashfs
image is ~28 GiB. HF snapshot files may be symlinks: preserve their backing `blobs/`
or materialize them when copying. Check `df -h`, `du -sh`, the safetensors index,
and that every referenced shard exists before allocating expensive GPUs. Allow
additional space for the client environment, dataset (~6.4 GB runtime mmap), logs,
and temporary image import files; the 461 GiB model-plus-image total is not a safe
overall storage budget.

Prefer an existing verified cache; the launcher does not download model weights.
If the image is missing, import it inside a separate preparation allocation with
Enroot, using the digest above and the site's existing registry credentials:

```bash
enroot import -o /path/to/your/scratch/images/dynamo-sglang-nightly-20260908-cc84ea52-amd64.sqsh \
  docker://nvcr.io/nvidia/ai-dynamo/sglang-runtime-nightly@sha256:cc84ea52fc8e66fb61a54692a30b2dcfa4d9b2349aa39d565039657cba8c26bb
```

The original cached file SHA256 was
`19d81fdeb293ef3837c88b07ddd3b8a879ed1c65ac606de53ec632ad1ffbf7c0`.
Use that to verify copies of that file; independently rebuilding squashfs may
produce different bytes. Do not build images or download datasets on login nodes.

## 2. Submit one allocation

Copy [run.sh](run.sh) into `<scratch>/agentx-run-scripts/run.sh` and save the following
as `submit.sh`, replacing the scratch path and using an account/partition you can
access. The partition below is the one used in this experiment; discover current
B200 availability with CDB/Slurm before submission. No `srt-slurm` is required.

```bash
#!/bin/bash
#SBATCH --account=aifm
#SBATCH --qos=batch-short
#SBATCH --partition=b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=1T
#SBATCH --gpus=8
#SBATCH --time=04:00:00
#SBATCH --job-name=agentx-glm52-c4
set -euo pipefail
SCRATCH_ROOT=/path/to/your/scratch
srun --ntasks=1 \
  --container-image="$SCRATCH_ROOT/images/dynamo-sglang-nightly-20260908-cc84ea52-amd64.sqsh" \
  --container-mounts="$SCRATCH_ROOT:/scratch" \
  --container-remap-root \
  bash /scratch/agentx-run-scripts/run.sh
```

Submit with an absolute log path (the directory must already exist):

```bash
sbatch --output=/path/to/your/scratch/agentx-b200-%j.log submit.sh
squeue -j <returned-job-id>
```

The recorded allocation requested `aifm`; the scheduler resolved it to
`aifm_fallback`. Record the effective account, resources and time limit with
`scontrol show job <job-id>` rather than assuming the request was unchanged.

## 3. What the launcher does

Read [run.sh](run.sh) for the complete, copyable worker and AIPerf commands. It:

1. Creates a separate client venv on scratch and checks that the pinned fork supports
   `--warmup-requests-per-lane`.
2. Starts the Dynamo frontend on port 8000 and one SGLang TP8 worker with metrics on
   9090. Both processes share file discovery, TCP requests and ZMQ events; no etcd
   or NATS is needed. Do not replace shared file discovery with separate in-memory
   discovery backends for these two processes.
3. Waits for model registration, sends a small chat smoke request, then runs AIPerf.
4. Replays all 393 trajectories at c4, starting at ratios 0.25–0.75, warming up with
   10 requests per lane and allowing a 1800-second warmup grace period. Trace/system
   idle gaps are capped at 300/10 seconds, with first-turn-prefix cache busting,
   streaming, `ignore_eos:true` and server token counts.
5. Saves raw artifacts and stops its worker/frontend when the script exits.

Use a fresh job for a fresh run: the launcher refuses to overwrite an existing
`/scratch/agentx-results/job-<job-id>`. It uses a job-specific discovery directory
and client venv; these are small reproducibility safeguards added to the original
launcher. Fixed ports assume one run per node. No inference settings were changed.

The original startup needed retries before the final launcher worked. The measured
run was started as an overlapping step in the existing Pyxis container:

```bash
srun --jobid=<existing-job-id> --overlap --ntasks=1 \
  enroot start --root --mount /path/to/your/scratch:/scratch \
  pyxis_<existing-job-id>.0 bash /scratch/agentx-run-scripts/run.sh
```

That is a recovery record, not the default launch procedure. Only reuse an existing
container after verifying no old worker/frontend remains; otherwise ports and GPUs
conflict. The cleaned-up single-submit wrapper has syntax validation, but has not
yet been rerun end to end on another allocation.

## 4. Monitor and preserve evidence

Outputs are under `<scratch>/agentx-results/job-<job-id>/`:

- `run.log`: hardware/software identity and benchmark lifecycle.
- `frontend.log`, `worker.log`: startup, cache capacity, errors and engine activity.
- `smoke.json`: endpoint smoke response.
- `aiperf/`: raw request records, summary and server metric artifacts.

Watch the batch log during client installation, then `run.log` and `worker.log`.
Running/Ready alone does not prove the replay is progressing: check that request
records keep growing and that worker logs show no failures. Preserve warmup versus
measurement phase distinctions, errors, cancellations, exact commands, package
versions and model revision. Record `pip freeze` from the job's client venv before
archiving results. Do not include credentials in an evidence bundle.

For the original run, warmup completed 44 requests with 0 errors in ~74 seconds.
The 3600-second measurement began at **16:26:58 PDT on September 8, 2026**;
393 trajectories contained 98,827 source requests. This is not a claim that all
source requests complete during the one-hour time window. Budget startup, loading,
compilation, dataset preparation, warmup and post-measurement drain/export in
addition to the measured hour. Four hours was the allocation limit, not measured
inference time.

## 5. Compare with AgentX, without claiming exact parity

Reference values recorded for point 440958:

| Metric | AgentX reference |
| --- | ---: |
| Total tokens/s/GPU | 3894.73934 |
| Output tokens/s/GPU | 27.50196 |
| TTFT p50 / p90 (s) | 0.38874 / 1.3275 |
| Request latency p90 (s) | 13.88218 |
| ITL p90 (s) | 0.00385 |

Use the completed measured-phase export, not warmup or an intermediate JSONL count.
Divide aggregate throughput by **8**, convert latency units explicitly, and report
relative delta as `100 * (measured / reference - 1)`. Inspect the pinned fork's
export schema before extracting values; missing fields must not become zeros.
Check successful/failed/cancelled requests, achieved concurrency, total/output
throughput, TTFT, ITL, request latency and GPU/CPU cache hit rates. Do not equate a
percentile of reciprocal ITL with the reciprocal of the same ITL percentile.

Important differences from the published reference:

- This run adds a Dynamo frontend and uses a newer pinned Dynamo/SGLang nightly.
- HiCache is disabled here. The reference enabled it, although its recorded CPU
  hit fraction at c4 was only ~0.000184 (0.0184%); GPU hit fraction was ~0.98215.
- Observed GPU KV capacity was 1,961,024 tokens versus the reference's 1,704,256.
- This is one run, with simulated speculative acceptance, not a repeated-run
  statistical parity result or a quality benchmark.

## 6. Pitfalls and cleanup

- Upstream `ai-dynamo/aiperf` used in an initial attempt lacked
  `--warmup-requests-per-lane`; use the exact SemiAnalysis fork revision above.
- The small home filesystem filled when pip used its default cache. The launcher
  directs pip cache to scratch and runtime caches to `/tmp`; check both filesystems.
- Optional persistent dataset mmap caching reported `ENOSPC` in the original run;
  runtime mmap still worked. Check free space and continued replay progress rather
  than silently treating every cache error as harmless.
- A bare `wait` in an EXIT trap hung on the `tee` process substitution. The launcher
  waits only for its explicit worker/frontend PIDs.
- Do not leave the allocation running after collecting results. From the login
  node, inspect `squeue -j <your-job-id>`. If the batch shell is still alive after
  completion or failure, cancel **only that verified job** with
  `scancel <your-job-id>`, then verify `squeue -h -j <your-job-id>` prints no row.
  Keep the checkpoint, cached image and raw results; no broad scratch cleanup is
  part of this procedure.
