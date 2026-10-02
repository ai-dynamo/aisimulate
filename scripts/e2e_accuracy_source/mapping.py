# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

HARDWARE_TO_SYSTEM: dict[str, str] = {
    "h100": "h100_sxm",
    "h200": "h200_sxm",
    "b200": "b200_sxm",
    "b300": "b300_sxm",
    "gb200": "gb200",
    "gb300": "gb300",
}


MOE_MODELS: frozenset[str] = frozenset(
    {
        "minimaxm2.5",
        "minimaxm2.7",
        "dsr1",
        "kimik2.5",
        "kimik2.6",
        "kimik3",
        "qwen3.5",
        "gptoss120b",
        "dsv4",
        "glm5",
        "glm5.1",
        "glm5.2",
        "minimaxm3",
    }
)
