# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Present existing serving metrics and initial memory estimates without recomputing them."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .power import normalize_power_summary

DETAIL_SECTIONS = ("summary", "memory", "time", "energy")
# ReplayReport's serving latency distributions and its trajectory median alias.
# Durations of the replay or simulator are not request/trajectory latency.
SERVING_LATENCY_METRICS = frozenset(
    f"{stat}_{metric}_ms"
    for metric in ("ttft", "ttst", "tpot", "itl", "e2e_latency", "trajectory_e2e_latency")
    for stat in ("mean", "min", "max", "median", "p75", "p90", "p95", "p99", "std")
) | {"p50_trajectory_e2e_latency_ms"}


def parse_detail_sections(value: str) -> tuple[str, ...]:
    tokens = {part.strip() for part in value.split(",")}
    unknown = tokens - {*DETAIL_SECTIONS, "all"}
    if unknown:
        raise ValueError(
            f"unsupported or empty detail section {sorted(unknown)!r}; choose from {', '.join(DETAIL_SECTIONS)}, all"
        )
    return tuple(section for section in DETAIL_SECTIONS if section in tokens or "all" in tokens)


def prediction_summary(native: dict[str, Any]) -> dict[str, Any]:
    summary = native.get("summary", native)
    if not isinstance(summary, dict):
        raise ValueError("prediction summary must be a mapping")
    return {
        key: value
        for key, value in summary.items()
        if key not in {"memory_diagnostics", "power_diagnostics", "details"}
    }


def _normalize_energy_publication(diagnostics: dict[str, Any]) -> dict[str, Any]:
    power = normalize_power_summary(diagnostics)
    status = diagnostics.get("publication_status")
    if status not in ("available", "withheld", "unsupported", "not_observed", "missing"):
        raise ValueError("energy publication_status must identify a supported publication state")
    if (status == "available") != (power["power_w"] is not None):
        raise ValueError("energy publication_status must be available exactly when power_w is numeric")
    coverage = power["power_coverage"]
    if status == "withheld" and (coverage is None or coverage >= 0.9):
        raise ValueError("withheld publication_status requires numeric power_coverage below 0.9")
    if status == "missing" and (coverage is None or coverage < 0.9):
        raise ValueError("missing publication_status requires power_coverage >= 0.9")
    if status == "unsupported" and coverage is not None:
        raise ValueError("unsupported publication_status requires null power_coverage")
    if status == "not_observed" and coverage not in (None, 0.0):
        raise ValueError("not_observed publication_status requires zero or null power_coverage")
    return {**diagnostics, **power}


def energy_diagnostics(native: dict[str, Any]) -> dict[str, Any]:
    diagnostics = native.get("power_diagnostics")
    if isinstance(diagnostics, dict):
        result = _normalize_energy_publication(deepcopy(diagnostics))
        phases = result.get("phases", [])
        if not isinstance(phases, list) or any(not isinstance(phase, dict) for phase in phases):
            raise ValueError("energy phases must be a list of phase records")
        result["phases"] = [_normalize_energy_publication(phase) for phase in phases]
        if any(phase["power_coverage"] is None for phase in result["phases"]):
            raise ValueError("phase publication_status requires numeric power_coverage")
        return result
    return {
        "schema_version": "1.0",
        "scope": "active_forward_pass_per_gpu",
        "publication_status": "unsupported",
        "power_w": None,
        "power_coverage": None,
        "coverage_gate": 0.9,
        "phases": [],
        "unavailable_reason": (
            "selected runner did not export typed timing-energy evidence; "
            "energy details are supported by --stack engine with op-level timing on supported topologies; "
            "downstream Dynamo adapter export is not qualified"
        ),
    }


def build_prediction_details(native: dict[str, Any], sections: tuple[str, ...]) -> dict[str, Any]:
    summary = prediction_summary(native)
    result: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    for name in sections:
        if name == "summary":
            result[name] = {"status": "available", "scope": "serving_workload", "metrics": summary}
        elif name == "memory":
            memory = native.get("memory_diagnostics", {})
            if not isinstance(memory, dict) or any(not isinstance(role, dict) for role in memory.values()):
                raise ValueError("memory diagnostics must map roles to their capacity estimates")
            available = [role for role in memory.values() if role.get("status") == "available"]
            if not available:
                reasons = [f"{role}: {value['unavailable_reason']}" for role, value in memory.items()]
                skipped[name] = (
                    "; ".join(reasons) or "selected runner or topology did not export a memory capacity estimate"
                )
                continue
            result[name] = {
                "status": "available" if len(available) == len(memory) else "partial",
                "scope": "capacity_estimate_per_rank",
                "roles": deepcopy(memory),
            }
        elif name == "energy":
            diagnostics = energy_diagnostics(native)
            result[name] = {
                "status": diagnostics["publication_status"],
                "scope": "active_forward_pass_per_gpu",
                "diagnostics": diagnostics,
            }
        elif name == "time":
            metrics = {key: value for key, value in summary.items() if key in SERVING_LATENCY_METRICS}
            if not metrics:
                skipped[name] = "selected runner did not export serving timing metrics in milliseconds"
                continue
            result[name] = {
                "status": "available",
                "scope": "serving_workload",
                "latency_unit": "ms",
                "serving_metrics": metrics,
            }
        else:
            raise ValueError(f"unsupported detail section {name!r}")
    return {"schema_version": "1.0", "sections": result, "skipped": skipped}


def format_prediction_details(details: dict[str, Any], *, energy_top_n: int = 12) -> str:
    lines: list[str] = []
    for name, section in details["sections"].items():
        if name == "energy":
            from .replay.reporting import format_power_diagnostics

            lines.extend(["Detail: energy", format_power_diagnostics(section["diagnostics"], top_n=energy_top_n), ""])
            continue
        lines.extend([f"Detail: {name}", f"  scope: {section['scope']}", f"  status: {section['status']}"])
        for key, value in section.get("metrics", section.get("serving_metrics", {})).items():
            if isinstance(value, (str, int, float)):
                lines.append(f"  {key}: {value}")
        for role, memory in section.get("roles", {}).items():
            lines.append(f"  {role}: {memory['status']}")
            if "unavailable_reason" in memory:
                lines.append(f"    {memory['unavailable_reason']}")
            for key, value in memory.items():
                if key.endswith(("_bytes", "_tokens")) or key in {"source", "stage", "estimated_num_gpu_blocks"}:
                    lines.append(f"    {key}: {value}")
            for key, value in (memory.get("memory_breakdown") or {}).items():
                lines.append(f"    {key}: {value}")
        lines.append("")
    for name, reason in details["skipped"].items():
        lines.append(f"Skipped {name}: {reason}")
    return "\n".join(lines).rstrip()
