<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Three-node GB200 HiCache smoke test

This is a **12-GPU topology variant**, not a reproduction of the throughput of
[AgentX 440082](https://inferencex.semianalysis.com/inference/agentic/440082).
The published point has one eight-GPU prefill engine and four four-GPU decode
engines, with 48 replay lanes. This variant keeps the prefill engine and reduces
decode to **one** replica. The short synthetic AIPerf test below checks serving;
it does not reproduce the AgentX trace distribution, lane timing or cache pressure.

## Completed run: September 8, 2026

With the decode FPM workaround below, DGD readiness and chat passed, followed by
**32/32 AIPerf requests, zero errors, in 30.0945 seconds**. This is an un-warmed
synthetic smoke, not an AgentX throughput measurement. The first request burst
had much higher TTFT than subsequent requests; do not use its aggregate metrics
as a steady-state performance result.

| Smoke metric | Value |
| --- | --- |
| Maintained HTTP concurrency | 4 |
| Actual input / output tokens per request | 1036 / 128 |
| TTFT p50 / p90 | 1.481 / 12.591 seconds |
| Request latency p50 | 2.013 seconds |
| ITL p50 | 4.192 ms |
| Aggregate output throughput, all 12 GPUs | 136.09 tokens/s |

The 1024-token synthetic prompt becomes 1036 input tokens after chat formatting.
Structured results are in [smoke-result-20260908.json](smoke-result-20260908.json).
Original AIPerf artifacts, full worker logs, startup arguments, resource snapshot
and metrics are preserved on the operator workstation under
`/tmp/hzhou-gb200-debug-20260908/`; the result JSON records the archive SHA-256.

HiCache actually initialized on all eight prefill DP ranks. In this newer nightly,
the 135 GB pool packs 78 target KV layers plus one MTP layer, giving **2,966,784
host tokens per rank**, plus 30.94 GB/rank for the packed DSA indexer. This differs
from the older reference's separately allocated draft cache. Prefill GPU capacity
was 1,100,608 tokens/rank; decode capacity was 997,312 tokens for its TP4 engine.
The prefill leader's exported DP0–3 `hicache_host_used_tokens` gauges were all
zero after this workload. **G2 eviction/reload was not exercised or validated.**

NIXL initialized its UCX backend and end-to-end P-to-D requests completed.
ComputeDomain was Ready on all three nodes, IMEX `channel0` was visible and
intra-node topology showed NV18. No standalone pairwise NIXL benchmark ran, and
the precise data-plane choice was not instrumented: this does not independently
prove that every transfer used NVLink rather than an available fallback. The
generic interconnect probe also flagged absent pod RDMA/GDRCopy devices; this
manifest targets MNNVL rather than provisioning RDMA interfaces.

The AIPerf fork's offline resolver requires a **repo ID plus revision**, not a
local snapshot directory: use `--tokenizer nvidia/GLM-5.2-NVFP4`,
`--tokenizer-revision 53e0691e21895a3863a606dfd12910c69eba94ab` and
`HF_HUB_CACHE=/model-cache`. The first client attempt failed before sending any
requests when a local directory was passed in offline mode; the corrected Job
loads the tokenizer entirely from the existing read-only cache.

Cleanup was verified at **18:16 PDT**: no pods, DGD, ComputeDomain or ResourceClaims
remained in `hzhou`; all three selected nodes again had zero scheduled GPU
requests, releasing the test's **12 GPUs**. The namespace, image-pull secret and
auto-generated shared PVC were retained. No model/cache data was deleted.

## Scope and fixed inputs

- Context: `nv-prd-dgxc.teleport.sh-dynamo-gcp-dev-02`.
- Namespace: `hzhou`; all explicit resource writes are in this namespace,
  except creation/labeling of the user-requested namespace itself.
- Image: the ARM64 digest in [README.md](README.md), extending the B200 Slurm
  nightly and containing `/opt/agentx-aiperf/bin/aiperf` at the pinned AgentX fork.
- Model: `nvidia/GLM-5.2-NVFP4`, snapshot
  `53e0691e21895a3863a606dfd12910c69eba94ab`, identical to the B200 experiment.
  Preflight found all 47 shards, totaling 464,823,042,096 bytes.
- Shared storage: namespace-local `shared-model-cache` PVC. The cluster's existing
  Kyverno `generate-shared-model-cache` policy creates the PVC/PV automatically
  after namespace creation. We did not create or edit a PV, StorageClass or policy.
  This is the existing shared Lustre filesystem, **not a new 36 TB allocation**.
  Both PVC references and volume mounts are read-only; HF offline mode is enabled.
- Worker scratch/compiler caches are container-local, not the shared cache.
- No node taints/labels, other namespaces, cluster RBAC, CRDs, operator deployments,
  global scheduling settings or other users' resources are modified.

The manifest pins the three ordinary-pool nodes that were free at the start of
this run. These names are **not reservations**. Before another run, verify capacity
again and update all three files consistently if placement changes. Do not add
Pinedrift tolerations or use preemption to make a test fit.

| Role | Node suffix | GPUs | CPU request | Memory request / limit |
| --- | --- | --- | --- | --- |
| Prefill rank 0/1 | `lv8c`, `ss59` | 4 each | 64 each | 800 / 820 GiB each |
| Decode | `tqtb` | 4 | 64 | 400 / 600 GiB |

The three nodes belong to the same NVLink clique. A namespace-local ComputeDomain
provides IMEX channel claims to all worker pods. Existing cluster controllers
perform their normal reconciliation; no controller configuration is changed.

## Engine parameters

The [DGD](deploy.yaml) uses one multinode prefill component (`nodeCount: 2`),
TP8/attention-DP8/EP8, and one TP4 decode component. Grove/operator inject the
prefill rendezvous address and node ranks; do not manually run two independent
prefill engines. `replicas: 1` refers to the entire two-node prefill engine.

Prefill has HiCache 135 GB **per DP rank**, write-back, direct I/O and
`page_first_direct`. Reference host allocations include additional DSA indexer and
draft caches, approximately 671 GB per node before process/model overhead; this
is why prefill requests 800 GiB, rather than 135 GiB. Decode has no HiCache and
radix caching is disabled.

The reference launcher passes prefill `--chunked-prefill-size 65536`; SGLang DP8
divides that into an effective 8192 per DP rank. Do not copy the post-normalization
8192 from `ServerArgs` into the CLI and divide it again. Similarly, the reference
omits `--schedule-conservativeness`, which becomes 0.3 after speculative adjustment.
Decode uses speculative EAGLE steps 2 / draft tokens 3 and synthetic acceptance
length 2.5 (`match-expected`, `real-draft-token`), so this is a performance smoke
configuration, **not a model quality evaluation**.

Port numbers and Kubernetes discovery differ from Slurm. The `nsa-*` reference
flags are expressed as current `dsa-*` flags. The initial version omits explicit
multithread model-loader configuration; loading speed is not a benchmark result.
Observe the nightly's actual normalized startup arguments before interpreting
performance. Same visible flags do not guarantee identical kernel/runtime behavior.

### Nightly decode FPM workaround

The first endpoint smoke on September 8 reached prefill and decode but crashed
SGLang's speculative disaggregated decode scheduler while emitting forward-pass
metrics. The operator automatically sets `DYN_FORWARDPASS_METRIC_PORT=20380`,
which opts the engine into FPM. The first failure was:

```text
scheduler.py:3952 process_batch_result
  self.metrics_reporter._emit_forward_pass_metrics(batch, result)
metrics_reporter.py:1040 _emit_forward_pass_metrics
  scheduled_requests=self._build_scheduled_request_metrics(batch)
metrics_reporter.py:291 _build_scheduled_request_metrics
  for sl in batch.seq_lens_cpu:
TypeError: 'NoneType' object is not iterable
```

The DGD therefore removes that environment variable **only from decode's engine
process** using `env -u DYN_FORWARDPASS_METRIC_PORT python3 -m dynamo.sglang`.
Normal engine metrics remain enabled. No image source, operator or cluster
configuration is patched. A successful smoke with this workaround must not be
reported as validation of FPM-enabled speculative disaggregated decode.

The original complete failure log is preserved at
`/tmp/hzhou-gb200-debug-20260908/decode-fpm-failure-full.log` on the operator
workstation. Subsequent cross-configuration FPM-on/off campaigns should account
for this failure before enabling FPM. The manifest image digest fixes the exact
failing engine build.

Changing the DGD advances its worker namespace hash. Both prefill and decode
must be recreated under that generation; leaving an old prefill process alive
with a new decode process would split their discovery namespaces. Only this
test's named pods/PodCliqueScalingGroup were recreated.

## Run

Follow the Teleport login/SSH callback tunnel instructions used for cluster work.
Always make the context explicit when operating beside other clusters:

```bash
KCTX=nv-prd-dgxc.teleport.sh-dynamo-gcp-dev-02
kubectl --context "$KCTX" get namespace hzhou
kubectl --context "$KCTX" get pvc shared-model-cache -n hzhou
kubectl --context "$KCTX" get nodes -o wide
kubectl --context "$KCTX" get pods -A -o wide
```

Create `hzhou` only if absent; do not overwrite a pre-existing namespace or its
workloads. Wait until the auto-generated PVC is Bound. Provision `nvcr-secret`
using your own nvcr.io credentials in this namespace without putting credentials
in checked-in manifests. No operator installation is required here.

Before allocating GPUs, verify the model snapshot above using a no-GPU pod with
a read-only mount. Do not start a large model download when the cache is missing.
Then server-validate and apply:

```bash
kubectl --context "$KCTX" apply --dry-run=server -n hzhou -f deploy.yaml
kubectl --context "$KCTX" apply -n hzhou -f deploy.yaml
kubectl --context "$KCTX" get dgd,pods -n hzhou -o wide
kubectl --context "$KCTX" get computedomains -n hzhou
kubectl --context "$KCTX" wait -n hzhou \
  --for=condition=Ready dgd/agentx-glm52-hicache --timeout=1800s
```

Check each worker's startup log, GPU topology, IMEX device and NIXL/UCX transport
before trusting disaggregated metrics. Readiness alone does not prove NVLink KV
transfer. An interconnect probe that cannot find a tool is inconclusive.

Port-forward and send one short OpenAI-compatible streaming chat request:

```bash
kubectl --context "$KCTX" port-forward -n hzhou \
  svc/agentx-glm52-hicache-frontend 18000:8000
curl -fsS http://127.0.0.1:18000/v1/models
curl -fsS http://127.0.0.1:18000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nvidia/GLM-5.2-NVFP4","messages":[{"role":"user","content":"Say hello briefly."}],"max_tokens":32,"stream":true}'
```

Only after successful readiness/chat checks, run [aiperf-smoke.yaml](aiperf-smoke.yaml):

```bash
kubectl --context "$KCTX" apply -n hzhou -f aiperf-smoke.yaml
kubectl --context "$KCTX" logs -n hzhou -f job/agentx-glm52-aiperf-smoke
```

This runs **32 synthetic requests, HTTP concurrency 4, input length 1024 and output
length 128**. Unlike AgentX's replay scenario, this is ordinary maintained request
concurrency, not four replay lanes containing trace idle gaps. Short requests
are not expected to induce G2 pressure. Report HiCache allocation independently
from measured G2 hit/offload/reload activity; “enabled” does not mean “exercised.”

The Job writes only an `emptyDir` and stays alive for five minutes after AIPerf
completion so artifacts can be copied out. Retrieve them immediately (replace
`<aiperf-pod>` with the pod selected by `job-name=agentx-glm52-aiperf-smoke`):

```bash
kubectl --context "$KCTX" cp -n hzhou \
  <aiperf-pod>:/artifacts/aiperf.tar.gz ./aiperf.tar.gz
```

## Cleanup

Save runtime versions, normalized engine arguments, pod/node placement, GPU
snapshots, worker logs, relevant metrics and AIPerf artifacts first. Then delete
only this test's named GPU workload and ComputeDomain:

```bash
kubectl --context "$KCTX" delete -n hzhou job agentx-glm52-aiperf-smoke
kubectl --context "$KCTX" delete -n hzhou dgd agentx-glm52-hicache
kubectl --context "$KCTX" delete -n hzhou computedomain agentx-glm52-cd
kubectl --context "$KCTX" get pods -n hzhou -o wide
```

Confirm no remaining GPU pods for this deployment. Keep the namespace and
auto-generated shared PVC intact. Never delete the shared model directory or
another user's PVC to clean this experiment.

## Sources and attribution

The DGD structure is adapted and modified from the Apache-2.0 Dynamo recipe
[`recipes/glm-5.2/sglang/disagg-b200-agentic/deploy.yaml`](https://github.com/ai-dynamo/dynamo/blob/f7612301f01bc3ef557cc8f38687d559b0d54b39/recipes/glm-5.2/sglang/disagg-b200-agentic/deploy.yaml),
immutable commit `f7612301f01bc3ef557cc8f38687d559b0d54b39`, copyright NVIDIA.
See the repository-root `THIRD_PARTY_NOTICES.md` for canonical attribution.

AgentX parameter facts were checked against the public point's
[launcher log](https://inferencex.semianalysis.com/api/v1/server-log?id=440082&file=sweep_23982.log),
[prefill log](https://inferencex.semianalysis.com/api/v1/server-log?id=440082&file=watchtower-navy-cn01_prefill_w0.out)
and [decode log](https://inferencex.semianalysis.com/api/v1/server-log?id=440082&file=watchtower-navy-cn03_decode_w0.out).
Reference run: `32207758126`, attempt 2, workflow head
`6130a8b5b671be6fc9695e8d159197f2cd275482`. No benchmark source is vendored.
