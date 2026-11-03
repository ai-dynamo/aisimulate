# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""sglang DSA module capture, BF16 KV, DECODE: batch 1 at the serving record's KV length 4096."""

import sys
from pathlib import Path

sys.argv = ["x"]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # components/: op_smoke.planned_row
from collector.helper import get_sm_version
from collector.sglang.collect_mla_module import run_mla_module
from op_smoke import planned_row

# The plan row run_mla_module_worker receives, for this one cell on this SM (the FP8-KV
# backend the plan pins differs between Hopper and Blackwell).
_seq, _batch, heads, kv, compute, gemm, model, attn_type, backend, tp, prefill_backend, points = planned_row(
    "sglang",
    "dsa_generation_module",
    get_sm_version(),
    model_path="deepseek-ai/DeepSeek-V3.2",
    compute_dtype="bfloat16",
    gemm_type="fp8_block",
    num_heads=128,
    batch_size=1,
    kv_cache_dtype="bfloat16",
    seq_len=4096,
)
run_mla_module(
    attn_type,
    heads,
    model,
    kv,
    compute,
    gemm,
    is_prefill=False,
    gpu_id=0,
    output_path="/tmp",
    attention_backend=backend,
    target_tp_size=tp,
    dsa_prefill_backend=prefill_backend,
    points=points,
)
