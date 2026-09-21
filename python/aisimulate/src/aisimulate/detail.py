# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Present existing serving metrics and initial memory estimates without recomputing them."""

from __future__ import annotations

from copy import deepcopy
from sys import float_info
from typing import Any

from .power import normalize_power_summary

DETAIL_SECTIONS = ("summary", "memory", "time", "energy", "source")
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
        if key not in {"memory_diagnostics", "power_diagnostics", "performance_diagnostics", "details"}
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


def _validate_performance_diagnostics(record: dict[str, Any]) -> None:
    """Check the runner's diagnostic contract before indexing or formatting it."""

    def require(valid: bool, path: str, expected: str) -> None:
        if not valid:
            raise ValueError(f"{path} must be {expected}")

    def fields(value: Any, path: str, required: set[str], optional: frozenset[str] | set[str] = frozenset()) -> None:
        require(isinstance(value, dict), path, "a mapping")
        for key in sorted(required - value.keys()):
            require(False, f"{path}.{key}", "present")
        for key in sorted(value.keys() - required - optional, key=str):
            require(False, f"{path}.{key}", "a supported field")

    def string(value: Any, path: str) -> None:
        require(isinstance(value, str) and bool(value.strip()), path, "a nonempty string")

    def number(value: Any, path: str) -> None:
        require(
            type(value) in (int, float) and value >= 0 and value <= float_info.max,
            path,
            "a finite nonnegative number",
        )

    def array(value: Any, path: str) -> None:
        require(isinstance(value, list), path, "a list")

    def sol(value: dict[str, Any], path: str) -> None:
        if value["sol"] is None:
            string(value["sol_unavailable_reason"], f"{path}.sol_unavailable_reason")
        else:
            fields(value["sol"], f"{path}.sol", {"latency_ms", "math_ms", "memory_ms"})
            for key, duration in value["sol"].items():
                number(duration, f"{path}.sol.{key}")
            require(value["sol_unavailable_reason"] is None, f"{path}.sol_unavailable_reason", "null with SOL data")

    path = "performance_diagnostics"
    fields(record, path, {"status", "scope", "latency_unit", "phases"}, {"unavailable_reason"})
    require(record["status"] in ("available", "unavailable", "not_observed"), f"{path}.status", "a supported state")
    require(record["scope"] == "accumulated_active_forward_pass_per_gpu", f"{path}.scope", "the native timing scope")
    require(record["latency_unit"] == "ms", f"{path}.latency_unit", "ms")
    array(record["phases"], f"{path}.phases")
    if "unavailable_reason" in record:
        string(record["unavailable_reason"], f"{path}.unavailable_reason")
    if record["status"] == "unavailable":
        string(record.get("unavailable_reason"), f"{path}.unavailable_reason")
        require(not record["phases"], f"{path}.phases", "empty when unavailable")
    if record["status"] == "available":
        require(bool(record["phases"]), f"{path}.phases", "nonempty when available")
    for i, phase in enumerate(record["phases"]):
        pp = f"{path}.phases[{i}]"
        fields(phase, pp, {"name", "latency_ms", "sol", "sol_unavailable_reason", "operations"})
        string(phase["name"], f"{pp}.name")
        number(phase["latency_ms"], f"{pp}.latency_ms")
        sol(phase, pp)
        array(phase["operations"], f"{pp}.operations")
        for j, operation in enumerate(phase["operations"]):
            op = f"{pp}.operations[{j}]"
            fields(
                operation,
                op,
                {"name", "latency_ms", "source", "fallbacks", "sol", "sol_unavailable_reason", "latency_to_sol_ratio"},
            )
            string(operation["name"], f"{op}.name")
            string(operation["source"], f"{op}.source")
            number(operation["latency_ms"], f"{op}.latency_ms")
            sol(operation, op)
            if operation["latency_to_sol_ratio"] is not None:
                number(operation["latency_to_sol_ratio"], f"{op}.latency_to_sol_ratio")
                require(
                    operation["sol"] is not None and operation["sol"]["latency_ms"] > 0,
                    f"{op}.latency_to_sol_ratio",
                    "null without a positive SOL latency",
                )
            if operation["fallbacks"] is None:
                continue
            array(operation["fallbacks"], f"{op}.fallbacks")
            for k, fallback in enumerate(operation["fallbacks"]):
                fp = f"{op}.fallbacks[{k}]"
                sizes = {"requested_ep_size", "requested_node_num", "measurement_ep_size", "measurement_node_num"}
                fields(fallback, fp, {"inference_phase", "comm_backend"} | sizes)
                require(
                    fallback["inference_phase"] in ("context", "generation"),
                    f"{fp}.inference_phase",
                    "context or generation",
                )
                string(fallback["comm_backend"], f"{fp}.comm_backend")
                for key in sizes:
                    require(type(fallback[key]) is int and fallback[key] > 0, f"{fp}.{key}", "a positive integer")


def performance_diagnostics(native: dict[str, Any]) -> dict[str, Any]:
    diagnostics = native.get("performance_diagnostics")
    if diagnostics is not None:
        if not isinstance(diagnostics, dict):
            raise ValueError("performance_diagnostics must be a mapping")
        _validate_performance_diagnostics(diagnostics)
        return deepcopy(diagnostics)
    return {
        "status": "unavailable",
        "scope": "accumulated_active_forward_pass_per_gpu",
        "latency_unit": "ms",
        "phases": [],
        "unavailable_reason": "selected runner did not export operation timing, SOL, or provenance evidence",
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
            diagnostics = performance_diagnostics(native)
            result[name] = {
                "status": "available" if metrics else diagnostics["status"],
                "scope": "serving_workload",
                "latency_unit": "ms",
                "serving_metrics": metrics,
                "diagnostics": diagnostics,
            }
        elif name == "source":
            diagnostics = performance_diagnostics(native)
            result[name] = {
                "status": diagnostics["status"],
                "scope": diagnostics["scope"],
                "phases": [
                    {
                        "name": phase["name"],
                        "operations": [
                            {key: op[key] for key in ("name", "latency_ms", "source", "fallbacks")}
                            for op in phase["operations"]
                        ],
                    }
                    for phase in diagnostics["phases"]
                ],
            }
            if "unavailable_reason" in diagnostics:
                result[name]["unavailable_reason"] = diagnostics["unavailable_reason"]
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
        evidence = section.get("diagnostics", section)
        if "unavailable_reason" in evidence:
            lines.append(f"  {evidence['unavailable_reason']}")
        if name in {"time", "source"}:
            if name == "time":
                lines.append(f"  operation scope: {evidence['scope']}")
            for phase in evidence.get("phases", []):
                lines.append(f"  {phase['name']}:")
                if name == "time":
                    lines.append(f"    accumulated latency_ms: {phase['latency_ms']}")
                    lines.append(f"    SOL: {phase['sol'] or phase['sol_unavailable_reason']}")
                operations = sorted(phase["operations"], key=lambda op: (-op["latency_ms"], op["name"]))
                for op in operations[:energy_top_n]:
                    if name == "time":
                        lines.append(
                            f"    {op['name']}: {op['latency_ms']} ms; SOL: "
                            f"{op['sol'] or op['sol_unavailable_reason']}; latency/SOL: "
                            f"{op['latency_to_sol_ratio']}"
                        )
                    else:
                        fallback = op["fallbacks"]
                        lines.append(
                            f"    {op['name']}: {op['source']}; fallbacks: "
                            f"{fallback if fallback is not None else 'unavailable'}"
                        )
                if len(operations) > energy_top_n:
                    lines.append(f"    ... {len(operations) - energy_top_n} more operations in JSON")
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
