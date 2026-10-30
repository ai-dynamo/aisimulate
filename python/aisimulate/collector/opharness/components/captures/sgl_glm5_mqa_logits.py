# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys

sys.argv = ["x"]
# SM90 GLM DSA representative = zai-org/GLM-5 (glm5_dsa_sparse_modules._selected_glm5_models)
from collector.sglang.glm5_dsa_sparse_modules import run_glm5_dsa_sparse_kernel_worker

run_glm5_dsa_sparse_kernel_worker(
    "zai-org/GLM-5", "mqa", 1, perf_filename="/tmp/sgl_glm5_mqa_logits_perf.txt", device="cuda:0"
)
