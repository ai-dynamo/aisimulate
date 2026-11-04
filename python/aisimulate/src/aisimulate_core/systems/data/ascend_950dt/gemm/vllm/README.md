<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# SYNTHETIC staging data — do not use for any prediction or accuracy claim

`0.30.0rc2/gemm_perf.parquet` is **synthetic M4 staging data**, not a
measurement. Every row carries `latency = 1.0` ms exactly and the device
column reads `Ascend950DT (synthetic M4 staging)`.

This note sits in the backend directory, not in the version directory, on
purpose: the loader's strict-provenance check treats *every* non-dot file in a
version directory as a data table needing a `collection_meta.yaml` entry
(`_version_dir_data_filenames`), so a README there fails
`AIC_STRICT_PROVENANCE=1`. Version directories stay data-only; the synthetic
marking inside `0.30.0rc2/` is the constant latency, the device column and the
sidecar's comment banner.

## Why it exists

*Consumption verification*: it proves that an `ascend_950dt` gemm table is
queryable end to end through the FPE/SDK paths (perf database lookup, generic
`VLLMBackend` resolution, support-matrix rows, `op_backend_facts.yaml`
coverage) **before** the device window — so the first real collection
only has to produce data, not also debug the consumer chain.

## How it was produced

Through the collector's real finalization path, never by hand: the staging
helper calls `collector.helper.log_perf` with the exact argument shape of
`collector/vllm/collect_gemm_npu.py` (so the staging CSV header is the frozen
collector schema) and then `collector.helper.finalize_perf_files` (the same
zstd-parquet finalization `collect.py` uses). The six `(m, n, k)` shapes are
real members of the `vllm_npu` gemm case grid in
`collector/cases/base_ops/gemm.yaml`.

`collection_meta.yaml` is a deliberately status-only schema-v1 sidecar: it
records no `collector_ref` / `collector_hash` / `case_plan_hash`, because
those attest a campaign on a real device that has not happened.

## M5 replacement plan

M5 collects bf16 eager gemm on the target node and **replaces the whole `0.30.0rc2/` directory**: delete
`gemm_perf.parquet`, this README and the status-only sidecar, then publish the
collector's own output plus the `collection_meta.yaml` that `collect.py`
writes. Do not merge real rows on top of these (the staging script finalizes
with `merge_existing=False` for the same reason). The facts row in
`collector/op_backend_facts.yaml` and the generated support matrices are
regenerated from the real data in that same change.
