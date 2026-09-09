<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GB200 AgentX image and HiCache preparation

Status: built and published on September 8, 2026; single-GPU ARM64/H100 smoke test
passed. The three-node GB200 Kubernetes variant is documented in
[KUBERNETES.md](KUBERNETES.md), with [DGD](deploy.yaml) and
[short AIPerf Job](aiperf-smoke.yaml). Cache-pressure offload/reload is not
validated by the short smoke workload; consult the run status there before
interpreting any results.

Published image:

```text
nvcr.io/nvidian/dynamo-dev/sglang-agentx:hzhou-0908-01-arm64
nvcr.io/nvidian/dynamo-dev/sglang-agentx@sha256:a0199d04c53a54dfa6ea3d37eb7d559afb55047a1b290bb783c441b63fb72688
```

The [Dockerfile](Dockerfile) extends the exact multi-architecture nightly index
used in the [B200 Slurm experiment](../agentx-b200-slurm/README.md), installing the
same pinned SemiAnalysis AIPerf fork in an isolated venv. Client dependency
resolution cannot upgrade SGLang's Torch, Transformers or other engine packages.
Transitive client versions are resolved at build time and recorded in
`/opt/agentx-aiperf/freeze.txt`; use the resulting image digest for subsequent runs.
The client is a dependency, not vendored source.

Registry inspection on September 8, 2026 confirmed:

| Manifest | SHA256 |
| --- | --- |
| Shared nightly index | `cc84ea52fc8e66fb61a54692a30b2dcfa4d9b2349aa39d565039657cba8c26bb` |
| B200 linux/amd64 child | `c679edf42a24a3d6f80a50b9150d0bd4160fc51ca5c8df60f5ec933e597ed022` |
| GB200 linux/arm64 child | `ad5ca4cb23a18d6065bfd2a6b723257ec0a1be25e6c58c3af14ef6aa137a1e0a` |

The base ARM64 image uses user `dynamo` and the NVIDIA entrypoint. Both are
preserved. This is not an attempt to run the B200 amd64 squashfs on GB200.

## Build and publish

Use an authorized native ARM64 builder with sufficient disk for the large base
image and build cache. The current workstation advertises only amd64/386 build
platforms and had 62 GiB free; it is not a validated builder for this recipe.
Do not install privileged emulation handlers or consume a shared GPU node merely
to work around that without arranging an appropriate builder.

Run from this directory on the selected builder, with registry authentication
already configured. The destination follows the Dynamo developer-image convention:

```bash
IMAGE=nvcr.io/nvidian/dynamo-dev/sglang-agentx:hzhou-0908-01-arm64
docker buildx build --platform linux/arm64 --progress plain \
  --metadata-file build-metadata.json --tag "$IMAGE" --push .
skopeo inspect --override-arch arm64 "docker://$IMAGE"
```

The example tag above is now published: use a new tag for another build. Record the output digest;
do not put credentials in build args, Dockerfiles or committed metadata. The build
checks architecture, client dependency consistency, the AgentX warmup CLI option
and engine package versions. It does not validate CUDA/NIXL or inference.

Client command inside the image:

```bash
/opt/agentx-aiperf/bin/aiperf profile --help
```

Run workers with the original `python3 -m dynamo.sglang`, and frontend with
`python3 -m dynamo.frontend`. Do not globally activate the client venv for workers.
Prefer separate client and engine pods even if they share this image. Model weights
and replay data belong on mounted storage, not in image layers.

## HiCache target: AgentX 440082

Reference: [point 440082](https://inferencex.semianalysis.com/inference/agentic/440082)
and its [published server log files](https://inferencex.semianalysis.com/api/v1/server-log-files?id=440082).
This is a deployment configuration, not an image-wide environment switch:

| Role | Reference topology | Cache configuration |
| --- | --- | --- |
| Prefill | 1 worker, TP8 / attention DP8 / EP8, across 2 GB200 nodes | HiCache enabled, 135 GB per DP rank |
| Decode | 4 workers, each TP4 / DP1 / EP1, one GB200 node each | No HiCache; GPU radix cache disabled |
| Load | Concurrency 48 | Use pinned AgentX client, not the B200 c4 setting |

Each node has 4 GPUs: full reference topology requires **6 nodes / 24 GPUs**.
Confirm current free capacity, NVLink/ComputeDomain topology and host memory before
preparing a deployable DGD. Do not silently substitute a smaller topology and label
it the same reference point.

Prefill cache argument fragment (append to the model/parallelism configuration):

```bash
--enable-hierarchical-cache \
--hicache-size 135 \
--hicache-write-policy write_back \
--hicache-io-backend direct \
--hicache-mem-layout page_first_direct
```

Decode cache argument fragment:

```bash
--disable-radix-cache
```

Do not pass `--enable-hierarchical-cache` to decode. No shared HiCache storage
backend was configured in the reference. Verify these flags against the ARM64
nightly's actual engine help and startup log before launch; this document is not
a complete, GPU-validated worker command or DGD.

The 135 GB setting is **per DP rank**, not per node or per deployment. Reference
logs reported ~135 GB KV + 30.94 GB DSA indexer + 1.73 GB draft cache per rank:
~167.67 GB/rank, ~671 GB/node with four ranks, before weights, processes and other
host-memory overhead. Reserve additional host RAM; a 135 GB pod memory limit is
not sufficient. The explicit size takes precedence over the default cache ratio.

There are eight independent rank-local G2 caches. This configuration does not
provide automatic G2-to-G2 sharing; prefill-to-decode NIXL transfer is a separate
mechanism. Check actual host allocations and transport readiness at bring-up.

## Completed ARM64 smoke test

Computelab job `4188457` used `ipp1-3396` (aarch64, one allocated H100 PCIe,
driver 595.58.03), with 32 CPUs, 128 GiB host RAM and a two-hour limit. The image
was built natively with `docker buildx build --platform linux/arm64 --load`, then
tested before `docker push`. Only the allocated GPU UUID was exposed to Docker.

[smoke.sh](smoke.sh) exercises the original runtime and isolated AIPerf client in
the same container. Provide a writable node-local directory mounted at
`/run/agentx`; copy the script there with mode 0644. Run only inside your Slurm
allocation, replacing the GPU UUID with the one assigned to your job:

```bash
docker run --name agentx-arm-smoke-<job-id> \
  --gpus device=<allocated-gpu-uuid> --ipc host \
  --mount type=bind,source=/tmp/<your-job-directory>,target=/run/agentx \
  nvcr.io/nvidian/dynamo-dev/sglang-agentx@sha256:a0199d04c53a54dfa6ea3d37eb7d559afb55047a1b290bb783c441b63fb72688 \
  bash /run/agentx/smoke.sh
```

Validated results:

- Native aarch64 execution and one-GPU CUDA matrix multiplication: passed.
- AIPerf dependency check and `--warmup-requests-per-lane` availability: passed.
- Engine packages unchanged: Dynamo 1.5.0.dev20260908, SGLang 0.5.18,
  FlashInfer 0.6.17.
- Qwen3-0.6B BF16, TP1, Triton attention, 8192-token GPU KV pool and
  2 GB host HiCache (`write_back`, `direct`, `page_first_direct`): started.
- Dynamo frontend chat: 11 input tokens, 32 output tokens.
- AIPerf: 8 streaming requests, concurrency 2, all completed; summary
  `request_count.avg=8`, `error_summary=[]`; container exit code 0, no OOM kill.

The first H100 attempt used SGLang's default FA3 backend and failed because this
ARM64 base image lacks the FA3 extension. The smoke script explicitly selects
Triton. This workaround is specific to the H100 small-model test; it is not a
change to the planned GB200 GLM backend. Initial script mount permissions also
required mode 0644 for the non-root `dynamo` container user.

This is a service/installation smoke test: eight short requests do **not** prove
that KV was evicted to G2 and restored. Do not interpret it as a GB200 performance
result or a validation of GLM's DSA/indexer host caches.

Original build/push logs, client package freeze, successful smoke artifacts and
the FA3 failure archive are preserved on shared scratch:
`/home/scratch.hongkuanz_gpu/agentx-arm-build-4188457/`.
The successful artifact archive excludes downloaded model weights and compiler
caches. Preserve results before deleting your named container and releasing your
allocation; never prune unrelated Docker images or another user's scratch.
