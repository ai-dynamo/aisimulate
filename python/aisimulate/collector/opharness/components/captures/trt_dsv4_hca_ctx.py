# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""trtllm DSV4 HCA context capture: one planned cell at the serving record's isl (DeepSeek-V4-Flash,
FP8 checkpoint -> fp8_block projection GEMMs, TP1, batch 1, 4096 tokens, prefix 0). The TRT-LLM DSV4
collector runs in-process; the plan's row carries the FP8 KV cache, which collect_dsv4_attn serves as the
packed fp8_ds_mla pool on SM90 and as the plain FP8 pool elsewhere, as serving does."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # components/: op_smoke.run_planned
from op_smoke import run_planned

run_planned(
    "trtllm",
    "dsv4_hca_context_module",
    model_path="sgl-project/DeepSeek-V4-Flash-FP8",
    seq_len=4096,
    batch_size=1,
    tp_size=1,
    gemm_type="fp8_block",
    prefix_len=0,
)
