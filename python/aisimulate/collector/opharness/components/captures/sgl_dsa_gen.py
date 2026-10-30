# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys

sys.argv = ["x"]
from collector.sglang.collect_mla_module import run_mla_module

run_mla_module(
    "dsa",
    128,
    "deepseek-ai/DeepSeek-V3.2",
    "bfloat16",
    "bfloat16",
    "fp8_block",
    is_prefill=False,
    gpu_id=0,
    output_path="/tmp",
    batch_size_filter=1,
)
