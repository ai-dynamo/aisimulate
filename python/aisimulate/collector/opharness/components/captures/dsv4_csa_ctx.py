# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys

sys.argv = ["x"]
from collector.vllm.collect_dsv4_attn import run_dsv4_attn_worker

run_dsv4_attn_worker(
    4096,
    1,
    1,
    "fp8",
    "bfloat16",
    "fp8_block",
    "sgl-project/DeepSeek-V4-Flash-FP8",
    "csa",
    None,
    0,
    perf_filename="/tmp/dsv4_csa_context_perf.txt",
    device="cuda:0",
)
