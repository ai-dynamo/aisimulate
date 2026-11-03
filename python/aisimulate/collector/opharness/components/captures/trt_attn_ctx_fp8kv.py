# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

sys.argv = ["x"]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # components/: op_smoke.run_planned
from op_smoke import run_planned

# Llama-3.1-8B context cell (b=1, isl 4096, 32/8 heads, hd 128, no window, TRTLLM backend), fp8 KV cache with BF16
# context FMHA: serving ref meta-llama/Meta-Llama-3.1-8B with kv_cache_config.dtype fp8. The collector's other fp8
# precision case (fp8 KV + fp8 context FMHA) runs the e4m3 FMHA kernel, which the fp8-KV serving config does not select
# on sm89 (L40 probe: serving prefill = fmha_v2_flash_attention_bf16_..._sm89, collector e4m3 variant diverged), so the
# gate captures the BF16-compute combo. Recipe was missing from the repo (B200 handoff 2026-10-04).
run_planned(
    "trtllm",
    "attention_context",
    batch_size=1,
    input_len=4096,
    num_heads=32,
    num_kv_heads=8,
    head_dim=128,
    window_size=0,
    kv_cache_dtype="fp8",
    attn_dtype="bfloat16",
)
