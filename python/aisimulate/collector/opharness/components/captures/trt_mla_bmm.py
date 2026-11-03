# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

sys.argv = ["x"]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # components/: op_smoke.run_planned
from op_smoke import run_planned

run_planned("trtllm", "mla_bmm_gen_pre", num_tokens=1, num_heads=128, dtype="bfloat16", num_warmups=2, num_runs=10)
run_planned("trtllm", "mla_bmm_gen_post", num_tokens=1, num_heads=128, dtype="bfloat16", num_warmups=2, num_runs=10)
