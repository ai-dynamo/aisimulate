<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GLM-5.3-Flash operator databases

This systems root holds the per-operator tables measured for GLM-5.3-Flash
(`zai-org/GLM-5.3-Flash`, `nvidia/GLM-5.3-Flash-NVFP4`) on GB300 with the
model-pinned collector runtimes in `collector/framework_manifest.yaml`. It is
separate from `systems/data/` so the GLM runtime versions never become the
fleet default or `next` version for other models (DeepSeek-V4.1 profile
precedent).

| Backend | Version directory | Runtime |
| --- | --- | --- |
| vLLM | `0.30.0+glm53tail.eb4704514fdf` | `vllm/vllm-openai:v0.30.0` (arm64 child `sha256:4864d466…`) plus the glm53tail PYTHONPATH overlay |
| SGLang | `0.5.20` | `lmsysorg/sglang:v0.5.20` (arm64 child `sha256:b0d8718a…`) |
| NCCL | `2.30.7` | nccl-tests against the `libnccl.so.2` both images load |

`gb300.yaml` is the fleet GB300 system file with `nccl_version: '2.30.7'`.
Every version directory has a `collection_meta.yaml` with the collector
reference, case-plan hash, row count and data SHA-256 per table.

Tables: `gemm_perf`, `moe_perf`, `custom_allreduce_perf` (TP2/TP4),
`computescale_perf`/`scale_matrix_perf`, and `nccl_perf` (TP2/TP4). Other GLM
tables (KDA, mHC, GLM attention module) are added by their own collection
workstreams under the same root.

Select it with `systems_paths=[<this directory>]`, the exact
`backend_version` above, `database_mode="SILICON"`, `shared_layer=False` and
`strict_provenance=True`.
