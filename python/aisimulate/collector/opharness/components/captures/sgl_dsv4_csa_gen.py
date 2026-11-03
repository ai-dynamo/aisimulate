# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""sglang DSV4 CSA generation capture: one planned cell at the serving record's isl (DeepSeek-V4-Flash,
FP8 checkpoint -> fp8_block projection GEMMs, KV auto = the fp8_e4m3 pool, TP1, batch 1, 4096 tokens).
The sglang DSV4 module collector runs each row in a subprocess; this calls its in-process entry with the
plan's row for that cell."""

import sys
from pathlib import Path

sys.argv = ["x"]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # components/: op_smoke.planned_row
from collector.helper import get_sm_version
from collector.sglang.collect_dsv4_attn import _kv_dtype_db_to_sglang, run_dsv4_mla_module
from op_smoke import planned_row

_seq, batch, tp, kv, _compute, gemm, model, kind, _backend, points = planned_row(
    "sglang",
    "dsv4_csa_generation_module",
    get_sm_version(),
    model_path="sgl-project/DeepSeek-V4-Flash-FP8",
    seq_len=4096,
    batch_size=1,
    tp_size=1,
    gemm_type="fp8_block",
)
run_dsv4_mla_module(
    model_path=model,
    mode="generation",
    attn_kind=kind,
    batch_size=batch,
    seq_lens_by_prefix={prefix: list(seq_lens) for prefix, seq_lens in points},
    kv_cache_dtype=_kv_dtype_db_to_sglang(kv),
    gemm_type=gemm,
    tp_size=tp,
    output_path="/tmp",
    perf_filename_prefix="dsv4",
    device="cuda:0",
)
