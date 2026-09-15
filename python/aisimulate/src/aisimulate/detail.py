# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Views of execution-owned evidence for the unified prediction CLI.

No estimate is recomputed here. Memory describes the initial rank capacity estimate before
native adjustments; operation timings and energy describe the replay itself.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .replay.reporting import format_power_diagnostics

DETAIL_SECTIONS = ("summary", "memory", "time", "energy", "source")


def parse_detail_sections(value: str) -> tuple[str, ...]:
    tokens = {part.strip() for part in value.split(",")}
    unknown = tokens - {*DETAIL_SECTIONS, "all"}
    if unknown:
        raise ValueError(
            f"unknown or empty detail section {sorted(unknown)!r}; choose from {', '.join(DETAIL_SECTIONS)}, all"
        )
    return tuple(section for section in DETAIL_SECTIONS if section in tokens or "all" in tokens)


def _unavailable(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "unavailable_reason": reason}


def prediction_summary(native: dict[str, Any]) -> dict[str, Any]:
    summary = native.get("summary", native)
    if not isinstance(summary, dict):
        raise ValueError("prediction summary must be a mapping")
    return {
        key: value
        for key, value in summary.items()
        if key not in {"power_diagnostics", "memory_diagnostics", "details", "per_request"}
    }


def build_prediction_details(native: dict[str, Any], sections: tuple[str, ...]) -> dict[str, Any]:
    summary = prediction_summary(native)
    power = native.get("power_diagnostics")
    if power is not None and not isinstance(power, dict):
        raise ValueError("power_diagnostics must be a mapping")
    phases = power.get("phases", []) if power else []
    if not isinstance(phases, list) or any(not isinstance(phase, dict) for phase in phases):
        raise ValueError("power diagnostic phases must be mappings")
    for phase in phases:
        operations = phase.get("operations", [])
        if not isinstance(operations, list) or any(not isinstance(op, dict) for op in operations):
            raise ValueError("power diagnostic operations must be mappings")
    reason = (power or {}).get("unavailable_reason") or (
        "selected runner or timing provider did not export replay operation evidence"
    )
    result: dict[str, Any] = {}
    for section in sections:
        if section == "summary":
            result[section] = {"status": "available", "scope": "serving_workload", "metrics": summary}
        elif section == "memory":
            memory = native.get("memory_diagnostics")
            if memory is not None and not isinstance(memory, dict):
                raise ValueError("memory_diagnostics must be a mapping")
            if memory:
                if any(not isinstance(role, dict) for role in memory.values()):
                    raise ValueError("memory diagnostic roles must be mappings")
                statuses = {role.get("status", "unavailable") for role in memory.values()}
                result[section] = {
                    "status": "available"
                    if statuses == {"available"}
                    else ("partial" if "available" in statuses else "unavailable"),
                    "scope": "capacity_estimate_per_rank",
                    "roles": deepcopy(memory),
                }
            else:
                result[section] = _unavailable(
                    "selected runner or topology did not export its memory capacity calculation"
                )
        elif section == "energy":
            # Preserve the native gate and omission semantics without deriving watts.
            result[section] = deepcopy(power) if power else _unavailable(reason)
        elif section == "time":
            result[section] = {
                "status": "available" if phases else "partial",
                "scope": (power or {}).get("scope", "unspecified"),
                "latency_unit": "ms",
                "serving_metrics": {
                    key: value for key, value in summary.items() if key.endswith("_ms") and key != "wall_time_ms"
                },
                "sol": _unavailable("the replay provider does not export matched SOL operation evidence"),
                "phases": [
                    {
                        "name": phase["name"],
                        "latency_ms": phase["latency_ms"],
                        "operations": [
                            {"name": op["name"], "latency_ms": op["latency_ms"]} for op in phase.get("operations", [])
                        ],
                    }
                    for phase in phases
                ],
            }
            if not phases:
                result[section]["unavailable_reason"] = reason
        elif section == "source":
            result[section] = {
                "status": "available" if phases else "unavailable",
                "scope": (power or {}).get("scope", "unspecified"),
                "phases": [
                    {
                        **{key: phase[key] for key in ("name", "source", "source_kind") if key in phase},
                        "operations": [
                            {
                                key: op[key]
                                for key in ("name", "source", "source_kind", "status", "uncovered_reason")
                                if key in op
                            }
                            for op in phase.get("operations", [])
                        ],
                    }
                    for phase in phases
                ],
            }
            if not phases:
                result[section]["unavailable_reason"] = reason
        else:
            raise ValueError(f"unknown detail section {section!r}")
    return {"schema_version": "1.0", "sections": result}


def format_prediction_details(details: dict[str, Any], *, top_n: int) -> str:
    lines: list[str] = []
    for name, section in details["sections"].items():
        lines.append(f"Detail: {name}")
        if name == "energy" and "publication_status" in section:
            if section.get("scope") != "active_forward_pass_per_gpu":
                lines.append(f"  scope: {section.get('scope', 'unspecified')}")
                lines.append(
                    "  This provider scope has no energy table renderer; inspect prediction.json for full evidence."
                )
                continue
            lines.append(format_power_diagnostics(section, top_n=top_n))
            continue
        if "scope" in section:
            lines.append(f"  scope: {section['scope']}")
        if "status" in section:
            lines.append(f"  status: {section['status']}")
        if "unavailable_reason" in section:
            lines.append(f"  unavailable: {section['unavailable_reason']}")
        for key, value in section.get("metrics", section.get("serving_metrics", {})).items():
            if isinstance(value, (str, int, float)):
                lines.append(f"  {key}: {value}")
        if "sol" in section:
            lines.append(f"  SOL unavailable: {section['sol']['unavailable_reason']}")
        for role, memory in section.get("roles", {}).items():
            lines.append(f"  {role}: {memory.get('status', 'unavailable')}")
            if "unavailable_reason" in memory:
                lines.append(f"    {memory['unavailable_reason']}")
            for key, value in memory.items():
                if key.endswith(("_bytes", "_tokens")) or key in {"source", "stage", "estimated_num_gpu_blocks"}:
                    lines.append(f"    {key}: {value}")
            for key, value in (memory.get("memory_breakdown") or {}).items():
                lines.append(f"    {key}: {value}")
        for phase in section.get("phases", []):
            label = str(phase["name"])
            if "latency_ms" in phase:
                label += f": {phase['latency_ms']:.6g} ms"
            lines.append(f"  {label}")
            operations = sorted(
                phase.get("operations", []),
                key=lambda op: (-op.get("latency_ms", 0), op["name"]),
            )
            for op in operations[:top_n]:
                if name == "time":
                    lines.append(f"    {op['name']}: {op['latency_ms']:.6g} ms")
                else:
                    lines.append(
                        f"    {op['name']}: {op.get('source', 'unavailable')} ({op.get('source_kind', 'missing')})"
                    )
            if len(operations) > top_n:
                lines.append(f"    ... {len(operations) - top_n} more operations in prediction.json")
        lines.append("")
    return "\n".join(lines).rstrip()
