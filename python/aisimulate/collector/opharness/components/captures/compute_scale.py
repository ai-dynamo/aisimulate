# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys

sys.argv = ["x"]
from collector.vllm.collect_computescale import run_computescale

run_computescale(
    4096,
    7168,
    perf_filename="/tmp/computescale_perf.txt",
    extra_perf_filenames=("/tmp/scale_matrix_perf.txt",),
    device="cuda:0",
)  # registry extra_perf_filenames (signature changed with the two-table producer)
