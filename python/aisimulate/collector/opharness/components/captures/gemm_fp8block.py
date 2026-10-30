# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys

sys.argv = ["x"]
from collector.vllm.collect_gemm import run_gemm

# DeepSeek-V3.2 fp8 shape family (down-proj n=7168): serving ref DSV3.2 fp8 (deepgemm)
run_gemm("fp8_block", 4096, 7168, 16384, perf_filename="/tmp/gemm_perf.txt", device="cuda:0")
