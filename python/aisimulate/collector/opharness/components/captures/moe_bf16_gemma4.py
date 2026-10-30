# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys

sys.argv = ["x"]
from collector.vllm.collect_moe import run_moe_torch

run_moe_torch(
    "bfloat16",
    [1, 4096],
    2816,
    704,
    8,
    128,
    1,
    1,
    "google/gemma-4-26B-A4B",
    "balanced",
    0.0,
    perf_filename="/tmp/moe_perf.txt",
    device="cuda:0",
)
