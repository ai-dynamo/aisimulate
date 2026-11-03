# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

sys.argv = ["x"]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # components/: op_smoke.run_planned
from op_smoke import run_planned

# Llama-3.1-8B context cell (b=1, isl/kv len 4096, 32/8 heads, hd 128, no window, bf16 KV, TRTLLM backend): serving ref
# meta-llama/Meta-Llama-3.1-8B. Recipe was missing from the repo (B200 handoff 2026-10-04: trtllm gates "MISSING
# CAPTURE").
run_planned(
    "trtllm",
    "attention_context",
    batch_size=1,
    input_len=4096,
    num_heads=32,
    num_kv_heads=8,
    head_dim=128,
    window_size=0,
    kv_cache_dtype="bfloat16",
    attn_dtype="bfloat16",
)
