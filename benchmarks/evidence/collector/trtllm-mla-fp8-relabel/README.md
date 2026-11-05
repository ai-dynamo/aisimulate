<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# TRT-LLM MLA context tables: FP8 compute label

`summary.json` describes the 26 TRT-LLM MLA context tables (`context_mla_perf.parquet`
and `mla_context_module_perf.parquet`, releases 1.3.0rc20 and 1.3.0rc29, every shipped
system) that PR #676 relabels. For each table it records the row count before and after,
the column list, the cells that changed per column, and the `(kv_cache_dtype, mla_dtype)`
row counts before and after.

What changed: only the `mla_dtype` column. Every row whose `kv_cache_dtype` is `fp8` now
says `mla_dtype=fp8` instead of `bfloat16` (51,187 cells over 102,406 rows). Row counts,
schemas, keys, shapes, units and latencies are unchanged. The reason is in
`collector/trtllm/collect_mla_module.py`: TRT-LLM's MLA context attention quantizes Q/K/V
to FP8 itself when the KV cache is FP8 on SM90/100/103/120, so those rows always timed FP8
compute, and the collector now labels new rows the same way.

Consumers: `aisimulate_core.sdk.models.helpers.trtllm_mla_runs_fp8` resolves a TRT-LLM
bfloat16 FMHA request on an FP8 KV cache to fp8 on those SMs, and the Task's data-driven
FMHA check reads these tables' fp8 slices
(`tests/unit/sdk/task_v2/test_task_config.py::test_large_ep_trtllm_context_fmha_capability_uses_granular_table`
keeps fp8 on gb200 because the relabelled granular table has the slice).

Regenerate the summary from a checkout with `git show <base>:<path>` for each table:
compare the base and head frames with pandas, count rows, differing cells per column
and the `(kv_cache_dtype, mla_dtype)` groups.
