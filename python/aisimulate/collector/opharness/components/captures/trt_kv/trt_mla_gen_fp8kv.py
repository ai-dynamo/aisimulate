# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""trtllm FP8-KV single-cell capture: mla_generation_module, the planned cell at the serving record's isl
(deepseek-ai/DeepSeek-V3, 128 heads (TP1), FP8 KV, BF16 compute, fp8_block projections,
batch 1, 4096 tokens, prefix 0).
The plan times TRT-LLM MLA modules through batch rows; --select restricts the row to this one point."""

import runpy
import sys
from pathlib import Path

sys.argv = [
    "op_smoke.py",
    "--backend",
    "trtllm",
    "--op",
    "mla_generation_module",
    "--select",
    "model_path=deepseek-ai/DeepSeek-V3,seq_len=4096,batch_size=1,num_heads=128,kv_cache_dtype=fp8,compute_dtype=bfloat16,gemm_type=fp8_block,prefix_len=0",
    "--out-dir",
    "/tmp/smk",
]
runpy.run_path(str(Path(__file__).resolve().parents[2] / "op_smoke.py"), run_name="__main__")
