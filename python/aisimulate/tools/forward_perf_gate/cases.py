# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed advisory forward-prediction benchmark matrix."""

from __future__ import annotations

from tools.prediction_regression_gate import grid

SYSTEM = "b200_sxm"
BACKEND = "vllm"
BACKEND_VERSION = "0.24.0"
DATABASE_MODES = ("SILICON", "EMPIRICAL")

MODELS = (
    {
        "model_id": "qwen3-32b",
        "model_path": "Qwen/Qwen3-32B",
        "tp_size": 4,
        "pp_size": 1,
        "attention_dp_size": 1,
        "moe_tp_size": 1,
        "moe_ep_size": 1,
    },
    {
        "model_id": "qwen3-235b-a22b",
        "model_path": "Qwen/Qwen3-235B-A22B",
        "tp_size": 8,
        "pp_size": 1,
        "attention_dp_size": 1,
        "moe_tp_size": 1,
        "moe_ep_size": 8,
    },
)


def expand_cases() -> list[dict]:
    result: list[dict] = []
    points = [("context", batch_size, isl, grid.CTX_OSL) for batch_size, isl in grid.PREFILL_POINTS] + [
        ("generation", batch_size, isl, grid.GEN_OSL) for batch_size, isl in grid.DECODE_POINTS
    ]
    for model in MODELS:
        for database_mode in DATABASE_MODES:
            for phase, batch_size, isl, osl in points:
                case_id = f"{model['model_id']}/{database_mode.lower()}/{phase}/bs{batch_size}-isl{isl}"
                result.append(
                    {
                        "case_id": case_id,
                        **model,
                        "system_name": SYSTEM,
                        "backend_name": BACKEND,
                        "backend_version": BACKEND_VERSION,
                        "database_mode": database_mode,
                        "phase": phase,
                        "batch_size": batch_size,
                        "isl": isl,
                        "osl": osl,
                        "prefix": 0,
                        "stride": grid.STRIDE,
                    }
                )
    return result
