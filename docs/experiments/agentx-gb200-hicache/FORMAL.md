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
- [prepare-client.py](prepare-client.py): uses the pinned AIPerf CLI resolver to
  generate and validate a complete configuration envelope before executing it.
- [stage-client-data.py](stage-client-data.py): stages tokenizer assets in a
  canonical private HF cache plus the public dataset; never copies model weights.
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

Current attempt data lives at `/results/agentx-gb200-20260908-c48-attempt4/`, containing:

- `STARTED.json`, followed by `COMPLETE.json`, `FAILED.json` or `INTERRUPTED.json`;
- `command.json`, `client-config.yaml`, `aiperf-console.log`;
- `aiperf/` raw records, summaries and server metrics;
- worker logs (including previous-container logs after restarts), `pods.json`
  snapshots and namespace events.

`STARTED.json` is created with exclusive creation on persistent storage. A pod
restart after that marker exists **does not start another replay or overwrite
the run**. It marks an unfinished attempt interrupted and releases the named
workload. An operator must inspect the artifacts and choose a new run ID before
retrying. Do not remove a marker simply to force a rerun over old results.

The shared `shared-model-cache` PVC remains read-only. Only small tokenizer/config
files are copied into `/results/hf/hub` using the canonical model/revision layout;
**no GLM weights are downloaded or copied**. A separate staging subprocess runs
with offline environment variables **removed**, caches the public Weka dataset,
and exits. The actual AIPerf process then uses offline mode with model repo ID
and revision, not a local tokenizer path. The full Weka dataset is distinct from
model weights. Client server-metric
discovery is disabled; only explicit endpoints in `hzhou` are scraped.

The frontend/sidecar ServiceAccount has a namespace-scoped Role: read pods/logs
and discovery objects, create/patch/update DynamoWorkerMetadata for frontend
router registration, and delete only the exact named DGD and ComputeDomain.
It has no node, PV,
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
After login renewal at 19:29 PDT, Hyperdisk was successfully attached and the
old unstarted frontend and empty `agentx-gb200-results` PVC were removed. No
model or experiment-result data was deleted. Frontend router registration also
required namespace-local DynamoWorkerMetadata patch permission, now included.

The first client attempt failed at 19:32 before sending any requests: passing a
partial `server_metrics` YAML switched this AIPerf fork into envelope mode, where
`benchmark.datasets` and `benchmark.phases` are required. Its `FAILED.json`,
command and logs remain under `/results/agentx-gb200-20260908-c48/`. Automatic
cleanup released the GPU workload after this setup failure.

The corrected helper resolves the full CLI into a complete envelope, disables
cluster-wide server-metric discovery, validates it, and then executes that file.
Both `aiperf config validate` and the actual profile config loader/plan builder
passed in a no-GPU preflight pod using the exact runtime image. Attempt 2 was
submitted at 19:38, preserving the failed attempt rather than removing its marker.
The public dataset was prefetched onto the private result PVC while engines
loaded: all **393 trajectories** are cached. This is dataset staging, not replay
or model-weight download.

The runner resolves the prefill **leader** pod for readiness and metrics. The
generic prefill Service can also select the nonleader multinode pod, whose system
metrics endpoint did not respond in this test. Explicit leader selection avoids
that ambiguity without changing engine parallelism or routing.

Attempt 2 also failed before any inference requests, at tokenizer configuration:
this fork's `_is_offline_mode()` calls `bool(os.environ.get(...))`, so the string
`"0"` is treated as enabled. It therefore tried to resolve a local tokenizer
directory as a Hub repo ID. Its failed attempt directory is retained. Attempt 3
uses the canonical private HF cache, repo ID and pinned revision in true offline
mode; staging removes those variables rather than setting `"0"`.

Before rescheduling attempt 3, a no-GPU AIPerf preflight completed actual tokenizer
configuration, loaded all 393 traces with zero context exclusions, reconstructed
9843 conversations / 98827 turns, finalized the 6.368 GB mmap and reached timing
setup. It then failed at the deliberately closed `localhost:9` diagnostic target,
as expected; it never called a GPU server and is not a performance result. Those
diagnostic artifacts are separately named `client-preflight-attempt3`. Attempt 3
was scheduled at 19:55 PDT. The formal command remains c48 and 3600 seconds.

The frontend's HTTP container is capped at 4 CPUs; the client container requests
4 CPUs/24 GiB and is capped at 8 CPUs/48 GiB, to bound CPU use on its shared node.

Attempt 3 passed dataset configuration and entered real warmup at **20:05:35 PDT**
with 531 expanded warmup requests; 51 were sent. Kubelet then killed decode at
20:05:50 and prefill shortly afterward: the operator's default `/live` liveness
probe had `failureThreshold: 1` and a five-second period. Long prefill delayed
the inference canary, producing HTTP 503 and a liveness-triggered SIGTERM. Events
explicitly report `Container main failed liveness probe, will be restarted`.
There was no CUDA/FPM exception preceding that termination. The client aborted
warmup after request failures; **the one-hour profiling phase did not start**.

The formal variant now retains the HTTP `/live` **startup** probe as its real
model-readiness gate, then uses TCP9090 liveness/readiness checks with three
failures tolerated. This avoids treating a delayed canary under load as a dead
process. The sidecar still requires both HTTP worker checks and the visible
frontend model before its first request. Only this DGD's probes change; the
cluster operator, engine source and c48 load remain unchanged.

Attempt 4 was submitted at 20:09 PDT with those probes; the actual generated Pod
specs were checked. By 20:12, other workloads occupied the ordinary w0e pool:
only three fragmented GPUs remained and no whole node was free. The four pods
therefore remain gang-queued rather than preempting another workload. Once
capacity is available, initialization, warmup and one-hour profiling start
automatically. Earlier attempt directories remain intact on the result PVC.
