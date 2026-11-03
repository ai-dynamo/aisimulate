# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

sys.argv = ["x"]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # components/: op_smoke.run_planned
from op_smoke import run_planned

# MiniMax-M3 MSA context, isl 4096, 64 heads (tp1 shard), bf16
run_planned(
    "vllm",
    "msa_context_module",
    seq_len=4096,
    batch_size=1,
    num_heads=64,
    kv_cache_dtype="bfloat16",
    compute_dtype="bfloat16",
    gemm_type="bfloat16",
    model_path="MiniMaxAI/MiniMax-M3",
    prefix_len=0,
)
