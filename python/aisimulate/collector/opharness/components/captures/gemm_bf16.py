# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

sys.argv = ["x"]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # components/: op_smoke.run_planned
from op_smoke import run_planned

# Llama-3.1-8B shape (4096x4096, k=4096): serving ref Llama-3.1-8B (cublas)
run_planned("vllm", "gemm", gemm_type="bfloat16", m=4096, n=4096, k=4096)
