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
| vLLM | `0.31.0` | `vllm/vllm-openai:v0.31.0` (arm64 child `sha256:3f7dd5b7…`), stock, no overlay; the model-pinned collector runtime. Ops tables gemm, moe, kda (TP1/2/4 shards), mhc, custom_allreduce, quantize collected 2026-10-08/09 (jobs 896352-896354); the GLM attention table is staged by its own workstream |
| vLLM | `0.30.0+glm53tail.eb4704514fdf` | retired runtime (`vllm/vllm-openai:v0.30.0` plus the glm53tail PYTHONPATH overlay); its Ops tables were removed when the `0.31.0` tables were staged; only the GLM attention table, owned by its own workstream, may remain here |
| SGLang | `0.5.20` | `lmsysorg/sglang:v0.5.20` (arm64 child `sha256:b0d8718a…`) |
| NCCL | `2.30.7` | nccl-tests against the `libnccl.so.2` both images load (unchanged in `vllm/vllm-openai:v0.31.0`) |

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
