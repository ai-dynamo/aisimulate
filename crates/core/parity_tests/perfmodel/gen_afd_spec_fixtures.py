#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate source-ordered AFD pool specs for native Rust tests."""

import json
from pathlib import Path

import aisimulate_core
from aisimulate_core.sdk.engine import compile_engine


def main():
    output = Path(__file__).parent / "goldens" / "afd_specs"
    for name, system, role in [
        ("moe_pure_h200_a_sglang", "h200_sxm", "a"),
        ("moe_pure_h200_f_sglang", "h200_sxm", "f"),
        ("moe_hetero_b200_f_sglang", "b200_sxm", "f"),
    ]:
        captured = []
        original = aisimulate_core.engine_spec_bincode_from_json

        def capture(spec):
            captured.append(spec)
            return original(spec)

        aisimulate_core.engine_spec_bincode_from_json = capture
        try:
            compile_engine(
                "Qwen/Qwen3-30B-A3B",
                system,
                "sglang",
                "0.5.14",
                tp_size=1 if role == "a" else 8,
                moe_tp_size=1,
                moe_ep_size=1 if role == "a" else 8,
                afd_pool_role=role,
            )
        finally:
            aisimulate_core.engine_spec_bincode_from_json = original
        (output / (name + ".json")).write_text(json.dumps(json.loads(captured[-1]), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
