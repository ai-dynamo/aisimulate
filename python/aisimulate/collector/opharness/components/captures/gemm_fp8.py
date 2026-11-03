# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

sys.argv = ["x"]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # components/: op_smoke.run_planned
from op_smoke import run_planned

# per-tensor fp8 (Qwen3-32B-FP8-Static-PerTensor shape 5120): serving ref that model
run_planned("vllm", "gemm", gemm_type="fp8", m=4096, n=5120, k=5120)
