<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GB200 AgentX image and HiCache preparation

Status: build recipe prepared; the ARM64 image has **not yet been built, pushed or
GPU-validated**. This does not allocate nodes or deploy a workload.

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

Use a new tag if that tag has already been published. Record the output digest;
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
