# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate and present producer-owned power summaries without estimating power."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real

POWER_FIELDS = ("power_w", "power_coverage")


def normalize_power_summary(summary: Mapping[str, object]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for name in POWER_FIELDS:
        value = summary.get(name)
        if value is None:
            result[name] = None
            continue
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number or null")
        result[name] = float(value)
    watts, coverage = result["power_w"], result["power_coverage"]
    if coverage is not None and not 0.0 <= coverage <= 1.0:
        raise ValueError("power_coverage must be within [0, 1]")
    if watts is not None and (watts <= 0.0 or coverage is None or coverage < 0.9):
        raise ValueError("numeric power_w must be positive and requires power_coverage >= 0.9")
    return result


def power_unavailable_reason(summary: Mapping[str, object]) -> str:
    coverage = summary.get("power_coverage")
    if coverage is None:
        return "energy provider or topology unsupported"
    if coverage < 0.9:
        return "insufficient energy coverage"
    return "publication conditions not met"


def power_metadata(summary: Mapping[str, object]) -> dict[str, object]:
    watts, coverage = summary.get("power_w"), summary.get("power_coverage")
    metadata = {
        "source": "modeled" if coverage is not None else "unavailable",
        "scope": "active_forward_pass_per_gpu",
        "power_w_unit": "W",
        "coverage_gate": 0.9,
        "publication_status": "available"
        if watts is not None
        else "withheld"
        if coverage is not None
        else "unavailable",
    }
    if watts is None:
        metadata["unavailable_reason"] = power_unavailable_reason(summary)
    return metadata


def format_power_summary(summary: Mapping[str, object]) -> str:
    values = normalize_power_summary(summary)
    watts, coverage = values["power_w"], values["power_coverage"]
    unavailable = f"unavailable ({power_unavailable_reason(values)})"
    power_text = f"{watts:.4g}W" if watts is not None else unavailable
    coverage_text = f"{coverage:.2%}" if coverage is not None else unavailable
    return f"power_w={power_text} power_coverage={coverage_text}"
