<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4 SGLang attention collection

The collector measures the complete SGLang CSA/HCA `self_attn` module for
DeepSeek-V4 Flash and Pro: projections, norm/RoPE/cache work, compression,
indexing/top-k, attention, and output projection. A module row is not an isolated
attention-kernel row. V4.1 has a separate [model and execution contract](../models/deepseek-v41.md).

## Tables and collection

The four phase/kind tables are `dsv4_csa_context_module_perf.parquet`,
`dsv4_hca_context_module_perf.parquet`, `dsv4_csa_generation_module_perf.parquet`,
and `dsv4_hca_generation_module_perf.parquet`, under the `sparse_attention` family.
The collector stages CSV-formatted `.txt` output and finalizes Parquet with
provenance. Inspect the current SGLang registry and
[`collect_dsv4_attn.py`](../../../python/aisimulate/collector/sglang/collect_dsv4_attn.py)
for runtime and case support.

From `python/aisimulate/`, inspect the retained plan before a GPU run:

```bash
python3 collector/collect.py --backend sglang \
  --model-path sgl-project/DeepSeek-V4-Flash-FP8 \
  --ops dsv4_csa_context_module dsv4_hca_context_module \
        dsv4_csa_generation_module dsv4_hca_generation_module --plan-only
```

The worker's outer task identifies attention kind, TP, GEMM type, and batch;
valid sequence and context-prefix points are swept inside that task. Use the
current model/base case files as the grid source rather than copying a historical
list. Limits apply to total tokens, KV memory, rotary bounds, and runtime setup.
Native FP4-expert checkpoints and converted FP8 checkpoints can have different
hardware eligibility even when timing an attention module.

## Precision and physical identity

Rows separate `compute_dtype`, `kv_cache_dtype`, and `gemm_type`. KV storage
precision is not an attention-arithmetic override. Persist rank-local heads and
real TP, deriving native heads as `num_heads * tp_size` for this family's genuine
TP sweeps. Flash and Pro have different native heads and must not merge even
when their local heads match.

The context lookup keys precision, native heads, local heads, compression ratio,
prefix (`step`), extend length, and batch. Generation uses its total context
coordinate under the matching precision/geometry. Sparse calibration tables
retain their own native-head keys; see [head-axis rules](data-format.md#head-axis-keying).

## CSA top-k correction

Dummy module inputs can produce degenerate flat scores. The runtime may use a
non-flat top-k distribution, so the database applies a measured top-k delta
after module lookup. Calibration must match selector geometry: Flash and Pro
have different `index_topk`. Never subtract Flash calibration from Pro because
the module's local heads happen to match. The correction is admitted only at an
exact native-geometry match; missing calibration leaves that module path
uncorrected with a diagnostic instead of borrowing an incompatible delta.

Validate module boundary, native/local geometry, precision, prefix handling,
finite latency/energy, and positive/negative calibration matches. Exercise both
Python model lowering and native consumer queries after finalization. Successful
collection does not qualify every context-parallel shape or end-to-end serving
configuration.
