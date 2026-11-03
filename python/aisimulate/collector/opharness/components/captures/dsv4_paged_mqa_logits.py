# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

sys.argv = ["x"]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # components/: op_smoke.run_planned
from op_smoke import run_planned

run_planned(
    "vllm",
    "dsv4_paged_mqa_logits_module",
    batch_size=1,
    seq_len=4096,
    prefix_len=0,
    tp_size=1,
    kernel="paged_mqa_logits",
    model_path="sgl-project/DeepSeek-V4-Flash-FP8",
)
