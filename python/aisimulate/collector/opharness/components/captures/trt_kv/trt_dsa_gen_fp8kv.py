# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""trtllm FP8-KV single-cell capture: dsa_generation_module, the planned cell at the serving record's isl
(deepseek-ai/DeepSeek-V3.2, 128 heads (TP1), FP8 KV, BF16 compute, fp8_block projections,
batch 1, 4096 tokens, prefix 0).
The plan times TRT-LLM DSA modules through batch rows; run_planned restricts the row to this one point.
TRT-LLM serves the DSA FP8 KV cache only in the SM100 family's trtllm-gen sparse MLA kernels,
so the plan leaves these cells out on SM90 (collector/skips.py) and the capture is graded on SM100."""

import sys
from pathlib import Path

GRADED_SMS = (100,)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # components/: op_smoke.run_planned
from op_smoke import run_planned

run_planned(
    "trtllm",
    "dsa_generation_module",
    model_path="deepseek-ai/DeepSeek-V3.2",
    seq_len=4096,
    batch_size=1,
    num_heads=128,
    kv_cache_dtype="fp8",
    compute_dtype="bfloat16",
    gemm_type="fp8_block",
    prefix_len=0,
)
