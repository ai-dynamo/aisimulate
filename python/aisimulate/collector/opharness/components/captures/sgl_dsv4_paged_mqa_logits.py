# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""sglang DeepSeek-V4 paged MQA-logits kernel (the CSA indexer's decode scorer), one planned decode
cell: DeepSeek-V4-Flash, TP1, batch 1, one new token over 4096 cached."""

import sys
from pathlib import Path

sys.argv = ["x"]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # components/: op_smoke.planned_row
from collector.helper import get_sm_version
from collector.sglang.deepseekv4_sparse_modules import run_dsv4_sparse_kernel_worker
from op_smoke import planned_row

row = planned_row(
    "sglang",
    "dsv4_paged_mqa_logits_module",
    get_sm_version(),
    model_path="sgl-project/DeepSeek-V4-Flash-FP8",
    tp_size=1,
    batch_size=1,
    seq_len=1,
    prefix_len=4096,
)
run_dsv4_sparse_kernel_worker(*row, perf_filename="/tmp/sgl_dsv4_paged_mqa_logits_perf.txt", device="cuda:0")
