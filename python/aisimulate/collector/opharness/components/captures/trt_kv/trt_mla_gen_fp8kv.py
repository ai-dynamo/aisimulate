# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""trtllm fp8-KV single-cell capture: mla_generation_module
(--case-prefix "[4096, 1, 128, 'fp8', 'bfloat16', 'fp8_block'")."""

import runpy
import sys

sys.argv = [
    "op_smoke.py",
    "--backend",
    "trtllm",
    "--op",
    "mla_generation_module",
    "--case-prefix",
    "[4096, 1, 128, 'fp8', 'bfloat16', 'fp8_block'",
    "--cases",
    "1",
    "--out-dir",
    "/tmp/smk",
]
runpy.run_path("/work/ais/python/aisimulate/collector/opharness/components/op_smoke.py", run_name="__main__")
