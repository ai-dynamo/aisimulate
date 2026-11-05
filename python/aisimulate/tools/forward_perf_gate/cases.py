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

DEEPSEEK_V3_2 = {"model_id": "deepseek-v3.2", "model_path": "deepseek-ai/DeepSeek-V3.2"}
QWEN3_5 = {"model_id": "qwen3.5-397b-a17b", "model_path": "Qwen/Qwen3.5-397B-A17B"}

ADDITIONAL_MODELS = (
    DEEPSEEK_V3_2,
    {"model_id": "deepseek-v4-flash", "model_path": "deepseek-ai/DeepSeek-V4-Flash"},
    QWEN3_5,
    {"model_id": "gpt-oss-120b", "model_path": "openai/gpt-oss-120b"},
    {
        "model_id": "nemotron-3-super-120b-fp8",
        "model_path": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
        "system_name": "h100_sxm",
    },
)

# (phase, batch size, input length, cached prefix length).
ADDITIONAL_MODEL_POINTS = (
    ("context", 1, 1024, 0),
    ("context", 1, 32768, 0),
    ("generation", 1, 1024, 0),
    ("generation", 128, 1024, 0),
)
PREFIX_POINTS = (("context", 1, 8192, 4096), ("context", 1, 8192, 7168))
ATTENTION_DP_POINTS = (("generation", 32, 1024, 0), ("generation", 8, 32768, 0))
SGLANG_POINTS = (("context", 1, 8192, 0), ("generation", 32, 1024, 0))


def _silicon_cases(model: dict, points: tuple[tuple[str, int, int, int], ...]) -> list[dict]:
    profile = {
        "system_name": SYSTEM,
        "backend_name": BACKEND,
        "backend_version": BACKEND_VERSION,
        "tp_size": 8,
        "pp_size": 1,
        "attention_dp_size": 1,
        "moe_tp_size": 1,
        "moe_ep_size": 8,
        **model,
    }
    # Profile labels separate report cells. The worker also groups by explicit
    # configuration fields; the prefix suffix separates otherwise equal groups.
    profile["model_id"] = (
        f"{profile['model_id']}-{profile['system_name']}-{profile['backend_name']}-{profile['backend_version']}"
        f"-tp{profile['tp_size']}-pp{profile['pp_size']}-adp{profile['attention_dp_size']}"
        f"-mtp{profile['moe_tp_size']}-ep{profile['moe_ep_size']}"
    )
    return [
        {
            **profile,
            "case_id": f"{profile['model_id']}/silicon/{phase}/bs{batch_size}-isl{isl}-prefix{prefix}",
            "database_mode": "SILICON",
            "phase": phase,
            "batch_size": batch_size,
            "isl": isl,
            "osl": grid.CTX_OSL if phase == "context" else grid.GEN_OSL,
            "prefix": prefix,
            "stride": grid.STRIDE,
        }
        for phase, batch_size, isl, prefix in points
    ]


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

    for model in ADDITIONAL_MODELS:
        result.extend(_silicon_cases(model, ADDITIONAL_MODEL_POINTS))
    for model in MODELS:
        result.extend(_silicon_cases({**model, "model_id": f"{model['model_id']}-prefix"}, PREFIX_POINTS))
    result.extend(_silicon_cases({**DEEPSEEK_V3_2, "tp_size": 1, "attention_dp_size": 8}, ATTENTION_DP_POINTS))
    result.extend(
        _silicon_cases(
            {**QWEN3_5, "backend_name": "sglang", "backend_version": "0.5.14"},
            SGLANG_POINTS,
        )
    )
    return result
