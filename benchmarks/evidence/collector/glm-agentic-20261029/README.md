<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GLM agentic DSA decode measurements

This collection adds 220 measured rows to the B300 / SGLang 0.5.20
`sparse_attention/dsa_generation_module_perf.parquet` table. It covers the
per-rank attention shapes used for TP8 GLM agentic replay. The recorded model
identity remains `nvidia/GLM-5.2-NVFP4`; these rows are not relabeled as GLM-5.3.
The separate `glm53_attention` family is unchanged.

## Coverage and measurement

- 110 rows include the DSA indexer; 110 reuse its selected indices through
  `dsa_generation_module_skip_indexer`. The two operation names and kernel
  labels remain distinct.
- Each operation covers batches `4, 8, 16, 32, 52` and past-KV lengths
  `1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
  32768, 65536, 131072, 262144, 393216, 524288, 655360, 720896`.
- Geometry: 8 local attention heads, target TP8, one decode query per request,
  FP8 KV cache, BF16 attention compute and projection GEMMs, page size 64,
  index top-k 2048, TRTLLM DSA decode backend.
- Each job uses one B300 GPU to measure a local attention module. TP8 describes
  the local shape; this is not an eight-GPU end-to-end measurement. The model's
  NVFP4 name does not change the recorded BF16 attention GEMM precision.
- Latency is in milliseconds. All 220 shapes used exact-shape CUDA graphs,
  including batch 52. Indexer-inclusive latency spans 0.0355–0.8941 ms;
  indexer-reuse latency spans 0.0237–0.0496 ms.

## Provenance

- Collector commit:
  [`3d41164034624251209588ee0284694e86a13bd8`](https://github.com/ai-dynamo/aisimulate/commit/3d41164034624251209588ee0284694e86a13bd8).
- Immutable [shard plan](https://github.com/ai-dynamo/aisimulate/blob/3d41164034624251209588ee0284694e86a13bd8/python/aisimulate/collector/sglang/agentx_dsa_shards.yaml).
- Runtime: SGLang 0.5.20, source commit
  `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`, image
  `lmsysorg/sglang:v0.5.20-cu130@sha256:06e4f2ed21afde4ff513cda65070124e727ba23ccaeff7712b8c40e1097d611f`.
  The collector sidecar retains its original normalized image tag and the
  same exact digest; the launch tag above records the CUDA 13.0 variant.
- Successful parent [pipeline 77726488](https://gitlab-master.nvidia.com/dl/ai-dynamo/aisim-auto-collector/-/pipelines/77726488)
  and child [pipeline 77726600](https://gitlab-master.nvidia.com/dl/ai-dynamo/aisim-auto-collector/-/pipelines/77726600).
- Indexer-inclusive [job 517782462](https://gitlab-master.nvidia.com/dl/ai-dynamo/aisim-auto-collector/-/jobs/517782462),
  Slurm 135871; indexer-reuse [job 517782464](https://gitlab-master.nvidia.com/dl/ai-dynamo/aisim-auto-collector/-/jobs/517782464),
  Slurm 135872. Both finished successfully with zero collector errors.

## Import and validation

The packaged table is the exact concatenation of the two successful 110-row
artifacts. Schema-v2 `collection_meta.yaml` preserves both original collection
events, including their collector and case-plan hashes. No measurements from
the earlier failed or canceled attempts are included.

[`manifest.json`](manifest.json) records the source artifact hashes, retained
checkpoint IDs, exact grid, Arrow schema, packaged hashes and anomaly checks.
Each checkpoint records one completed module task that emitted 110 rows;
completed tasks and measured rows are different counts.

Validation checked the full expected grid, source-row equality, all 220 graph
timing log entries, both zero-error summaries, sidecar schema, and absence of
nulls, duplicate physical keys, nonfinite latency or nonpositive latency.
The native table views load all 110 rows per operation with shared-layer reuse
disabled and preserve every latency value. The backend census adds the two
observed operation/precision slices.
At base commit `62519bfd26ee9d9c4cc9b53aba81fb30426f8a3d`, this versioned DSA
table did not exist. Existing 0.5.14 and 0.5.21 data remain unchanged.

Repository checks, run from `python/aisimulate` in its development environment:

```bash
python tools/perf_database/check_collector_data.py
python tools/perf_database/backend_facts.py --check
python -m pytest -q -p no:timeout tests/unit/collector/test_provenance.py
```

All seven collector-data rules passed, the backend census matched, and all 47
provenance tests passed. Local checks used the existing alignment environment's
native extension; no native rebuild or new GPU collection was required.

These checks establish collection coverage and data integrity. They do not
establish full-model serving accuracy, prefill coverage, or cross-version reuse.
