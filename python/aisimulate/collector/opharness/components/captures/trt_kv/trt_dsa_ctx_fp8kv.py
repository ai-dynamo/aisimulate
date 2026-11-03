# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""trtllm FP8-KV single-cell capture: dsa_context_module, the planned cell at the serving record's isl
(deepseek-ai/DeepSeek-V3.2, 128 heads (TP1), FP8 KV, BF16 compute, fp8_block projections,
batch 1, 4096 tokens, prefix 0).
The plan times TRT-LLM DSA modules through batch rows; --select restricts the row to this one point.
TRT-LLM serves the DSA FP8 KV cache only in the SM100 family's trtllm-gen sparse MLA kernels,
so the plan leaves these cells out on SM90 (collector/skips.py) and the capture is graded on SM100."""

import runpy
import sys
from pathlib import Path

GRADED_SMS = (100,)

sys.argv = [
    "op_smoke.py",
    "--backend",
    "trtllm",
    "--op",
    "dsa_context_module",
    "--select",
    "model_path=deepseek-ai/DeepSeek-V3.2,seq_len=4096,batch_size=1,num_heads=128,kv_cache_dtype=fp8,compute_dtype=bfloat16,gemm_type=fp8_block,prefix_len=0",
    "--out-dir",
    "/tmp/smk",
]
runpy.run_path(str(Path(__file__).resolve().parents[2] / "op_smoke.py"), run_name="__main__")
