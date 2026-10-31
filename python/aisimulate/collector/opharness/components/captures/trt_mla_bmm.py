# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys

sys.argv = ["x"]
from collector.trtllm.collect_mla_bmm import run_mla_gen_post, run_mla_gen_pre

run_mla_gen_pre(1, 128, "bfloat16", 2, 10, perf_filename="/tmp/mla_bmm_perf.txt", device="cuda:0")
run_mla_gen_post(1, 128, "bfloat16", 2, 10, perf_filename="/tmp/mla_bmm_perf.txt", device="cuda:0")
