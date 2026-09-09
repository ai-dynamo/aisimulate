<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Staged GB200 validation, September 9

Stage 1 was submitted at 10:12:57 PDT on September 9, 2026. The initial live state
was gang Pending, with no GPUs allocated. This is a deployment record, **not a
successful transport or benchmark result**. Stage 2 has not been authorized by
the validation gate yet.

At 10:20 PDT stage 1 obtained all 12 GPUs: Prefill leader `grdk`, Prefill follower
`tqtb`, and Decode/frontend `gjn2`. The ComputeDomain reported all three nodes
Ready in the same clique. Before HTTP testing, a separate bounded 64 MiB DRAM
NIXL WRITE from each Prefill node to Decode passed full byte verification.
UCX protocol logs show `rc_mlx5` zero-copy, with no TCP transport enabled. The two
client-observed transfer times were 25.9 and 30.3 ms; these are sanity probes,
not calibrated bandwidth measurements. NIXL per-transfer telemetry was unavailable
in this image, and Linux RDMA netdev counters did not account for verbs traffic.
Actual inference and HiCache reload acceptance were still pending at this update.

Both configurations stay in `hzhou`, use the existing read-only shared model PVC,
and store new results on `agentx-gb200-results-hyperdisk`. Previous attempt folders
are not overwritten. The image digest and checkpoint revision remain those in
[FORMAL.md](FORMAL.md).

## Topologies and acceptance

- Stage 1: DEP8 Prefill across two nodes, one TP4 Decode on a third node, 12 GPUs.
  Prefill retains 135 GiB HiCache per DP rank. Decode has no HiCache, retains
  disabled radix cache and the documented decode FPM environment workaround.
- Stage 2: the same Prefill plus three independent TP4 Decode replicas, 20 GPUs
  on five nodes. This is a three-decode variant, not parity with reference 440082's
  four-decode deployment. Formal AIPerf remains c48, 393 trajectories, 3600 seconds,
  seed 42, ten warmups per lane, and the reference idle caps and trajectory ratios.

Stage 1 first waits for P and D `/live` and the frontend model registry. It then
pauses at `AWAITING_INTERCONNECT.json`: an operator must verify the actual
allocated pods' RDMA interfaces, GID index, locked-memory limit, ComputeDomain,
and UCX/NIXL initialization before writing `INTERCONNECT_OK.json` in its private
results folder. Do not write that marker based only on manifest settings.

The bounded suite sends exact token-ID prompts, verifies returned token counts
and actual `prefill_dp_rank`, and saves individual JSON responses:

- A 4096-token request to each of eight Prefill DP ranks.
- DP0 prompts of 32768, 131072, 262144, 524288, 786432, and 996579 tokens.
- Revisit the earlier 524288-token prefix after eviction pressure.
- Concurrency 2, 4, and 8 with separate 131072-token prefixes.

There are 29 requests. The longest input requires at least 997120 GPU KV token
slots with the suite's 512-token safety allowance; insufficient reported capacity
fails explicitly rather than clipping prompts or increasing model context.
Each request has a 300-second client deadline. `HTTP_SUITE_PASS.json` is necessary,
but actual transport counters/logs and HiCache offload/reload evidence must also
be reviewed before `ACCEPTED.json` and stage 2. Enabled HiCache alone is not a hit.

## GCP transport corrections

Following the cluster's [GCP NIXL guide](https://dl.gitlab-master-pages.nvidia.com/ai-dynamo/ops-docs/engineer/dynamo_nixl_deployment_guide/),
each worker requests all four RDMA resources and declares `eth0` plus `rdma0..3`.
Live GKE admission also adds the corresponding `.IP` resources. `UCX_TLS` is
`cuda_ipc,cuda_copy,rc`, with GID index 3, IPC_LOCK and unlimited memlock. There is
no TCP bulk-data fallback. NVLink allocator settings and the shared ComputeDomain
are retained; all GPU workers stay in the ordinary w0e pool's same NVLink clique.
No Pinedrift toleration or cluster-scoped write is introduced.

The reference bootstrap timeout, waiting timeout, and heartbeat-failure limit are
all 100000. This prevents the old 300-second Prefill timeout from preceding
Decode admission; a separate bounded experiment deadline prevents indefinite
GPU retention. No engine source is patched.

## Deployment and independent cleanup

Run from this directory. Set the intended context explicitly for every kubectl
command; the examples omit it only for readability.

```bash
python3 render-staged.py --stage validate --part support > /tmp/stage-support.yaml
python3 render-staged.py --stage validate --part workload > /tmp/stage-workload.yaml
kubectl -n hzhou apply --dry-run=server -f /tmp/stage-support.yaml
kubectl -n hzhou apply --dry-run=server -f /tmp/stage-workload.yaml
kubectl -n hzhou apply -f /tmp/stage-support.yaml
kubectl -n hzhou logs job/agentx-stage1-guard-0909-v2
# Verify ARMED before applying GPU resources:
kubectl -n hzhou apply -f /tmp/stage-workload.yaml
```

The independent CPU-only guard requires no results disk and starts before the GPU
workload. Stage 1 has a 5400-second deadline from the first assigned GPU pod;
stage 2 has 14400 seconds. Both have a 12-hour maximum queue lifetime. The guard
tracks resource UIDs and deletes only this DGD and ComputeDomain, never PVCs.
The runner also saves snapshots and cleans up after success or failure. A
cluster/API outage can still delay cleanup; monitor the guard status ConfigMap.

Stage 1 results: `/results/agentx-gb200-20260909-stage1-rdma-v1`.
Stage 2 results: `/results/agentx-gb200-20260909-stage2-rdma-v1`.
Exclusive start markers reject accidental reruns after pod restarts.

Only after reviewing stage 1 and confirming its GPU resources are gone, render
`--stage formal`, arm the stage 2 guard, and apply its workload. Do not apply the
combined generated manifests blindly, or reuse these run directories for a new
attempt. Preserve failure evidence and use a new explicitly named attempt.

`rdma_probe.py` can be streamed through `kubectl exec -i` into existing experiment
pods: run `python3 -u - server` on Decode, then `python3 -u - client DECODE_POD_IP
PATTERN_BYTE` on each Prefill node. The server binds port 18997 for at most 180
seconds. It registers only a 64 MiB CPU buffer and never allocates GPU memory.
Use `UCX_LOG_LEVEL=info UCX_PROTO_INFO=y` for actual protocol-selection evidence;
save both client and server output. Do not run against another user's pod.
The formal runner checks every expected Decode replica directly and scrapes all
three Decode metrics endpoints, not a service that might reach only one replica.
