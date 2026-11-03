# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""sglang GLM-5 DSA attention, module level, FP8 KV, CONTEXT (batch 1, isl 4096, prefix 0).
On Blackwell sglang resolves DSA kv auto to fp8_e4m3 and serves GLM-5 on the TRT-LLM FMHA
token-sparse cubins (B200 2026-10-04); the kernel-level glm5_dsa_attn_module bench runs BF16
FlashMLA sparse_attn_fwd. The plan records GLM-5's DSA module rows under the BF16 GLM-5.2
checkpoint at its 64 native heads (TP1), so that is the cell captured here."""

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
    "dsa_context_module",
    get_sm_version(),
    model_path="zai-org/GLM-5.2",
    kv_cache_dtype="fp8",
    compute_dtype="bfloat16",
    gemm_type="bfloat16",
    num_heads=64,
    batch_size=1,
    seq_len=4096,
    prefix_len=0,
)
run_mla_module(
    attn_type,
    heads,
    model,
    kv,
    compute,
    gemm,
    is_prefill=True,
    gpu_id=0,
    output_path="/tmp",
    attention_backend=backend,
    target_tp_size=tp,
    dsa_prefill_backend=prefill_backend,
    points=points,
)
