# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

sys.argv = ["x"]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # components/: op_smoke.run_planned
from op_smoke import run_planned

# Qwen3-VL-2B vision tower: hidden 1024 / 16 heads -> head_dim 64; one 1024-token image
run_planned("vllm", "encoder_attention", batch_size=1, seq_len=1024, num_heads=16, head_dim=64)
