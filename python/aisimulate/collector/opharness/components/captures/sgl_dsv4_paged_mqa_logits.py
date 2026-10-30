# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys

sys.argv = ["x"]
from collector.sglang.deepseekv4_sparse_modules import run_dsv4_sparse_kernel_worker

run_dsv4_sparse_kernel_worker(
    "sgl-project/DeepSeek-V4-Flash-FP8",
    "paged_mqa_logits",
    1,
    perf_filename="/tmp/sgl_dsv4_paged_mqa_logits_perf.txt",
    device="cuda:0",
)
