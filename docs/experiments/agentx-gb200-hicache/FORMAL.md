<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Queued one-hour AgentX run: 12-GPU variant, c48

## Final outcome: failed during warmup

Recovered and verified on September 9, 2026. **Attempt 5 did not reach the
one-hour profiling phase.** The runner's three-hour subprocess timeout fired and
`FAILED.json` was persisted. GPU cleanup ran; live checks found no remaining DGD,
worker pods, ComputeDomain or ResourceClaims. Both result/model PVCs were retained.
No new attempt was started during result recovery.

| Milestone / count | Verified value |
| --- | --- |
| Preparation started | September 8, 21:10:20 PDT |
| Warmup started | September 8, 21:14:44 PDT |
| Hard timeout / failure marker | September 9, 00:10:23 PDT |
| Requests sent | 178 |
| Returned records | 121, all warmup |
| Returned without error | 105 |
| Returned with error | 16 |
| Still unreturned at timeout | 57 |
| Formal profiling records | 0 |

The console continued printing `errors=0`, but the record JSONL contains **16
`InvalidInferenceResultError` entries** with no output content. Use the per-record
errors, not that progress counter, when interpreting this run. No final aggregate
or server-metrics export was produced before the hard timeout; this is not a
valid AgentX performance comparison.

The strongest failure evidence is repeated PD timeout propagation:

- Prefill logged 300-second `KVPoll.Bootstrapping` timeouts for 73 unique rooms.
- Decode logged 900-second `KVPoll.WaitingForInput` timeouts for 16 unique rooms;
  all 16 also appear among prefill's failed bootstrap rooms.
- For example, room `1250160255957218402` failed on prefill at 21:19:56 PDT,
  then on decode at 21:35:46 PDT. Decode warned that KV transfer completion was
  not received after bootstrapping.
- The final saved pod snapshot had zero container restarts. This failure was
  not the earlier liveness-probe termination.

These logs establish bootstrap/transfer timeout failure, not a proven permanent
Gloo deadlock or GPU hardware failure. The current configuration uses the default
300-second prefill bootstrap timeout and a 900-second decode waiting timeout;
these were not fully aligned with the much longer timeouts in the reference
launcher. A follow-up should investigate the single-decode admission queue and
timeout ordering, then error propagation, before changing engine code or c48.
No such changes or reruns were made during this recovery.

Structured results and checksums: [formal-result-20260909.json](formal-result-20260909.json).
The complete 11 MB attempt directory is retained on the private PVC and copied to
`/home/hongkuanz/Projects/aisimulate-agentx-results/gb200-20260908/attempt5/`.
An archive is adjacent at `attempt5.tar.gz`. The temporary recovery pod mounted
only that result PVC read-only, requested no GPUs, and was deleted after copying.

## Intended experiment

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

Current attempt data lives at `/results/agentx-gb200-20260908-c48-attempt5/`, containing:

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
were gang-queued rather than preempting another workload. Capacity became available
at 20:14 and initialization resumed. Earlier attempt directories remain intact
on the result PVC.

### Startup loader fallback

On newly selected decode nodes, the default loader stalled after scanning all
47 shards. A stack sample showed its main thread waiting in
`deepseek_weight_loader.py:444` (`as_completed`) while 32 loader threads were in
MoE tensor device-copy/scale-loading calls. A separate tiny CUDA probe succeeded.
Recreating only decode on another eligible node produced similar symptoms.
The diagnostic stack/logs are retained in the attempt4 directory. An ephemeral
diagnostic container used SYS_PTRACE only in that owned decode pod's process
namespace; it exited successfully and was removed with that pod.

The formal manifest therefore uses the image's existing
**`--load-format runai_streamer`** for P and D. This loader is explicitly supported
for prequantized ModelOpt models and synchronously consumes its streamed tensors,
avoiding the default loader's asynchronous CPU-to-GPU copy pool. RunAI's native
import and actual model/draft loading passed. There is no engine-source patch or
new checkpoint: the same shared, read-only snapshot, quantization and serving
parallelism remain. This is an initialization-path difference from the reference
and must be recorded when reproducing the run.

### NVLink KV allocation

The initial manifest omitted two settings from the reference launcher:
`SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True` and `MC_FORCE_MNNVL=1`. They are now
restored, along with the reference's `MC_TE_METRIC`, `NVSHMEM_REMOTE_TRANSPORT`
and thinking/reasoning environment. Despite its name, the custom-pool setting
controls the KV allocator independently of selecting NIXL/UCX as the transfer
backend. It allocates NVLink-compatible buffers; enabling UCX MNNVL support alone
does not make the default allocations suitable for that path.

Attempt 4 reached warmup but its one-token snapshot primers took 125–200 seconds
and decode's pod Ethernet received hundreds of GB. This strongly indicated bulk
KV network fallback. It was manually interrupted before profiling and has an
explicit `INVALID.json` plus network evidence; **do not treat it as a valid
NVLink performance result**.

With the reference allocator settings restored, a separate pre-warmup transport
check on attempt 5 completed a cross-node 7181-input-token / 1-output-token
request. Decode eth0 RX increased by only **153,534 bytes** (332,770 → 486,304),
instead of carrying the large KV payload. `transport-check.json` on the result
PVC records this evidence. The first-request 9.94-second latency is a cold
transport/setup diagnostic, not a steady-state performance number. The formal
warmup and profiling data remain separate from this check.

### Interim observation (21:26 PDT, superseded by final outcome above)

Attempt 5 entered warmup at 21:14:44. It completed 43 of the 531-request warmup
budget with zero reported errors; eight of the 51 initial primers remained
in flight, with no new completion for several minutes. The one-hour profiling
phase **had not started**. Pods remained Ready with zero restarts.

Prefill DP0 reported 235,776 occupied host-cache tokens (capacity 2,966,784/rank),
so G2 storage was actually being used; this alone does not prove G2 reload hits.
Decode's usable KV capacity was 997,184 tokens versus the model's 1,048,576
architectural context limit. The remaining requests need further admission/
transfer investigation rather than a silent concurrency reduction.

Read-only stack samples are preserved in the attempt5 directory:
`decode-scheduler-stack.txt` and `decode-other-ranks-stack.txt`. TP0 was sampled
in metadata-gated Gloo `all_reduce`, TP1/TP3 in request broadcast, and TP2 in
decode preallocation code. These are diagnostic observations, not a proven
root cause; the samples were not an atomic all-rank capture.

Teleport access expired at 21:29, preventing further live observation until
renewed. The in-pod runner continues independently. Its 10,800-second AIPerf
process timeout is a hard bound; the 1800-second warmup **grace** must not be
interpreted as a confirmed total warmup deadline. Completion, failure and GPU
release were not yet verified at that time; the September 9 recovery and final
outcome above provide the verified terminal state.
