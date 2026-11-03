# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""sglang GLM-5 DSA sparse attention kernel (BF16 FlashMLA sparse_attn_fwd), one planned cell."""

import sys
from pathlib import Path

sys.argv = ["x"]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # components/: op_smoke.planned_row
from collector.helper import get_sm_version
from collector.sglang.glm5_dsa_sparse_modules import run_glm5_dsa_sparse_kernel_worker
from op_smoke import planned_row

# One context cell (batch 1, seq 4096, prefix 0). The plan picks the checkpoint per SM: the
# BF16 GLM-5 on Hopper, the NVFP4 GLM-5.2 on Blackwell (collector/skips.py).
row = planned_row("sglang", "glm5_dsa_attn_module", get_sm_version(), batch_size=1, seq_len=4096, prefix_len=0)
run_glm5_dsa_sparse_kernel_worker(*row, perf_filename="/tmp/sgl_glm5_dsa_attn_perf.txt", device="cuda:0")
