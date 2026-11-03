# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

sys.argv = ["x"]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # components/: op_smoke.run_planned
from op_smoke import run_planned

# DeepSeek-V3.2 fp8 shape family (down-proj n=7168): serving ref DSV3.2 fp8 (deepgemm)
run_planned("vllm", "gemm", gemm_type="fp8_block", m=4096, n=7168, k=16384)
