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
- Worker scratch/compiler caches are container-local `/tmp`, not the shared cache.
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
