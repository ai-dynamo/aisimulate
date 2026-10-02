# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable

from .schema import DropRecord, SiliconRow

NVIDIA_HARDWARE = frozenset({"h100", "h200", "b200", "b300", "gb200", "gb300"})


FRAMEWORK_TO_AIC_BACKEND: dict[str, str] = {
    "trt": "trtllm",
    "vllm": "vllm",
    "sglang": "sglang",
    "dynamo-trt": "trtllm",
    "dynamo-vllm": "vllm",
    "dynamo-sglang": "sglang",
}


def _row_drop_reason(row: SiliconRow) -> str | None:
    if row.hardware not in NVIDIA_HARDWARE:
        return f"filter:hardware (got {row.hardware!r}, not in NVIDIA set)"
    if row.framework not in FRAMEWORK_TO_AIC_BACKEND:
        return f"filter:framework (got {row.framework!r}, not in supported set)"
    if row.benchmark_type != "single_turn":
        return f"filter:benchmark_type (got {row.benchmark_type!r}, expected 'single_turn')"
    if row.spec_method == "mtp":
        return "filter:spec_method=mtp (deferred to v2)"
    if row.metrics is None:  # error rows are stripped at dedupe; this is a belt-and-braces guard
        return "filter:no-metrics"
    return None


def apply_filter_rules(rows: Iterable[SiliconRow]) -> tuple[list[SiliconRow], list[DropRecord]]:
    """Drop rows that fail any pre-mapping filter rule. Pure / no I/O."""
    kept: list[SiliconRow] = []
    drops: list[DropRecord] = []
    for r in rows:
        reason = _row_drop_reason(r)
        if reason is None:
            kept.append(r)
            continue
        stage = reason.split(" ", 1)[0]
        drops.append(
            DropRecord(
                stage=stage,
                reason=reason,
                config_id=r.config_id,
                isl=r.isl,
                osl=r.osl,
                conc=r.conc,
            )
        )
    return kept, drops
