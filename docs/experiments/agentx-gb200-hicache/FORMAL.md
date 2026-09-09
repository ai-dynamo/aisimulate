<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Queued one-hour AgentX run: 12-GPU variant, c48

This run retains the validated [smoke engine topology](KUBERNETES.md) but uses
the published AgentX replay workload: 393 Weka trajectories, concurrency 48,
3600-second measurement, seed 42, trajectory start ratios 0.25–0.75,
10 warmup requests per lane, 1800-second warmup grace, trace/system idle caps
300/10 seconds and `first_turn_prefix` cache busting.

**This is not parity with 440082**: prefill remains DEP8 on two GB200 nodes,
but decode has one TP4 engine instead of four. The offered c48 load is deliberately
not divided by four; engine request limits can queue that load. No concurrency
reduction or trace filtering is silently applied. Decode retains the documented
FPM-off workaround; prefill retains FPM and HiCache 135 GB per DP rank.

## Files and submission

- [formal-runner.py](formal-runner.py): independent frontend-sidecar runner.
- [render-formal.py](render-formal.py): derives the formal manifest from the
  validated `deploy.yaml` and embeds the runner in a namespaced ConfigMap.
- [formal-deploy.yaml](formal-deploy.yaml): generated manifest to apply.

```bash
python3 render-formal.py > formal-deploy.yaml
kubectl --context nv-prd-dgxc.teleport.sh-dynamo-gcp-dev-02 \
  apply --dry-run=server -n hzhou -f formal-deploy.yaml
kubectl --context nv-prd-dgxc.teleport.sh-dynamo-gcp-dev-02 \
  apply -n hzhou -f formal-deploy.yaml
```

The manifest references the existing `aiperf-paper-rig-nonpreempting` PriorityClass
(value 100, `Never`). It does not create or modify a PriorityClass. The cluster's
admission rejects `preemptionPolicy: Never` without a compatible PriorityClass.
All worker pods select ordinary pool `customer-gpu-w0e` and its existing NVLink
clique; no hard-coded three-node reservation or Pinedrift toleration remains.
If GPUs are unavailable, the workload stays queued without preempting others.

## Durable results and one-shot behavior

The frontend pod has a separate `aiperf` container. Before starting it requires
successful `/live` responses from both prefill and decode and the exact model
in frontend `/v1/models`. It does not mistake a listening HTTP process for model
readiness. Warmup precedes the one-hour measurement.

Results PVC: **`hzhou/agentx-gb200-results-hyperdisk`**, 50 GiB, dynamically
provisioned using the existing `jegu-hyperdisk-balanced-rwo` StorageClass.
The GB200 GCP `a4x-highgpu-4g` instance cannot attach `standard-rwo`'s pd-balanced
disk; do not substitute that default class. No static cluster PV is authored.

Data lives at `/results/agentx-gb200-20260908-c48/`, containing:

- `STARTED.json`, followed by `COMPLETE.json`, `FAILED.json` or `INTERRUPTED.json`;
- `command.json`, `client-config.yaml`, `aiperf-console.log`;
- `aiperf/` raw records, summaries and server metrics;
- worker logs and `pods.json` snapshots.

`STARTED.json` is created with exclusive creation on persistent storage. A pod
restart after that marker exists **does not start another replay or overwrite
the run**. It marks an unfinished attempt interrupted and releases the named
workload. An operator must inspect the artifacts and choose a new run ID before
retrying. Do not remove a marker simply to force a rerun over old results.

The shared `shared-model-cache` PVC remains read-only. Only small tokenizer/config
files are copied into the result directory; **no GLM weights are downloaded or
copied**. HF dataset download is enabled with its cache on this private result
PVC. The full Weka dataset is distinct from model weights. Client server-metric
discovery is disabled; only explicit endpoints in `hzhou` are scraped.

The sidecar has a namespace-scoped Role: read pods/logs and discovery objects,
and delete only the exact named DGD and ComputeDomain. It has no node, PV,
StorageClass, other-namespace or cluster-operator write permission. After benchmark
success/failure (hard run-process ceiling three hours), it captures logs, flushes
results, then requests deletion of those two resources to release GPUs. Namespace,
result PVC, model PVC and configuration remain. A new mount of the result PVC
can retrieve artifacts after the frontend disappears.

## Initial submission status

Submitted September 8 around 18:40 PDT. GPU pods scheduled without preemption:
prefill on `lv8c`/`ss59`, decode on `5rzq`; frontend on `24wk`.
The initial result PVC used `standard-rwo`, was provisioned but never attached,
and frontend could not start because GCP rejected pd-balanced on a4x.
The manifest is corrected to Hyperdisk; applying that correction and replacing
the unstarted frontend requires renewal of the expired Teleport session.
At this checkpoint **the benchmark has not started**. No success claim is made.

After login renewal, apply the current manifest, replace the old unstarted
frontend pod, and delete only the failed, empty initial `agentx-gb200-results`
PVC after its old consumer has gone. Preserve `agentx-gb200-results-hyperdisk`.
Check the sidecar logs and confirm all 393 trajectories are loaded before
interpreting the measurement.
