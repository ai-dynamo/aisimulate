# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo-neutral reporting helpers for single-run replay."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

TITLE = "NVIDIA AIPerf | LLM Metrics"
STAT_COLUMNS = ("avg", "min", "max", "p99", "p90", "p75", "std")
POWER_DIAGNOSTICS_TITLE = "AISimulate active forward-pass energy diagnostics (per GPU)"


def default_report_path(prefix: str = "aisimulate_replay_report") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path.cwd() / f"{prefix}_{timestamp}.json"


def write_report_json(
    report: dict[str, Any],
    output_path: str | Path | None,
    *,
    default_prefix: str = "aisimulate_replay_report",
) -> Path:
    path = Path(output_path) if output_path is not None else default_report_path(default_prefix)
    if path.exists() and path.is_dir():
        path = path / default_report_path(default_prefix).name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_per_request_jsonl(
    output_path: str | Path,
    records: list[dict[str, Any]] | None,
) -> None:
    if records is None:
        raise ValueError("replay report did not provide per_request records")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            output.write("\n")


def format_report_table(report: dict[str, Any]) -> str:
    rows: list[list[str]] = []
    for label, suffix in (
        ("Time to First Token (ms)", "ttft_ms"),
        ("Time to Second Token (ms)", "ttst_ms"),
        ("Request Latency (ms)", "e2e_latency_ms"),
        ("Inter Token Latency (ms)", "itl_ms"),
        (
            "Output Token Throughput Per User (tokens/sec/user)",
            "output_token_throughput_per_user",
        ),
    ):
        _append_stat_row(rows, report, label, suffix)
    rows.extend(
        [
            [
                "Output Token Throughput (tokens/sec)",
                _format_value(report.get("output_throughput_tok_s")),
                *["N/A"] * (len(STAT_COLUMNS) - 1),
            ],
            [
                "Request Throughput (requests/sec)",
                _format_value(report.get("request_throughput_rps")),
                *["N/A"] * (len(STAT_COLUMNS) - 1),
            ],
            [
                "Request Count (requests)",
                _format_value(report.get("completed_requests", report.get("num_requests"))),
                *["N/A"] * (len(STAT_COLUMNS) - 1),
            ],
        ]
    )
    if "power_coverage" in report:
        rows.extend(
            [
                [
                    "Active Power per GPU (W)",
                    _format_value(report.get("power_w")),
                    *["N/A"] * (len(STAT_COLUMNS) - 1),
                ],
                [
                    "Power Data Coverage (%)",
                    _format_value(float(report["power_coverage"]) * 100.0),
                    *["N/A"] * (len(STAT_COLUMNS) - 1),
                ],
            ]
        )
    lines = [TITLE, _render_table(rows)]
    wall_time_ms = report.get("wall_time_ms")
    if isinstance(wall_time_ms, int | float):
        lines.append(f"Wall Time (ms): {_format_value(wall_time_ms)}")
    prefix_ratio = report.get("prefix_cache_reused_ratio")
    if isinstance(prefix_ratio, int | float):
        lines.append(f"Prefix Cache Reused Ratio: {_format_value(prefix_ratio)}")
    first_admission_ratio = report.get("first_admission_prefix_cache_reused_ratio")
    if isinstance(first_admission_ratio, int | float):
        lines.append(f"First Admission Prefix Cache Reused Ratio: {_format_value(first_admission_ratio)}")
    return "\n".join(lines)


def format_power_diagnostics(
    diagnostics: dict[str, Any],
    *,
    top_n: int = 12,
) -> str:
    """Render a bounded view of the complete power-diagnostics JSON export."""

    if top_n < 1:
        raise ValueError("power diagnostics top_n must be at least 1")

    status = str(diagnostics.get("publication_status", "unsupported"))
    coverage = diagnostics.get("power_coverage")
    gate = diagnostics.get("coverage_gate")
    lines = [POWER_DIAGNOSTICS_TITLE]
    lines.append(
        "Aggregate: "
        f"power={_format_power(diagnostics.get('power_w'))} "
        f"coverage={_format_percent(coverage)} "
        f"gate={_format_percent(gate)} status={status}"
    )
    reason = diagnostics.get("unavailable_reason")
    if isinstance(reason, str) and reason:
        lines.append(f"Reason: {reason}")
    lines.append(
        "Energy is modeled active forward-pass evidence in W-ms per GPU; "
        "it is not wall-clock or provisioned-fleet energy."
    )

    raw_phases = diagnostics.get("phases")
    phases = raw_phases if isinstance(raw_phases, list) else []
    phase_rows: list[list[str]] = []
    operation_rows: list[list[str]] = []
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        phase_name = str(phase.get("name", "unknown"))
        phase_rows.append(
            [
                phase_name,
                _format_energy(phase.get("energy_wms")),
                _format_latency(phase.get("latency_ms")),
                _format_latency(phase.get("covered_latency_ms")),
                _format_percent(phase.get("power_coverage")),
                _format_power(phase.get("power_w")),
                str(phase.get("source_kind", "missing")),
            ]
        )
        raw_operations = phase.get("operations")
        operations = raw_operations if isinstance(raw_operations, list) else []
        ordered = sorted(
            (operation for operation in operations if isinstance(operation, dict)),
            key=_operation_sort_key,
        )
        for operation in ordered[:top_n]:
            source = str(operation.get("source", "missing"))
            source_kind = str(operation.get("source_kind", "missing"))
            status_source = f"{operation.get('status', 'missing')} {source_kind}:{source}"
            uncovered_reason = operation.get("uncovered_reason")
            if isinstance(uncovered_reason, str) and uncovered_reason:
                status_source += f"; {uncovered_reason}"
            operation_rows.append(
                [
                    phase_name,
                    str(operation.get("name", "unknown")),
                    _format_energy(operation.get("energy_wms")),
                    _format_latency(operation.get("latency_ms")),
                    _format_percent(operation.get("power_coverage")),
                    _format_percent(operation.get("energy_contribution")),
                    status_source,
                ]
            )
        if len(ordered) > top_n:
            operation_rows.append(
                [
                    phase_name,
                    f"... {len(ordered) - top_n} more in prediction.json",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )

    if phase_rows:
        lines.extend(
            [
                "",
                "Phase totals",
                _render_diagnostics_table(
                    [
                        "phase",
                        "energy",
                        "latency",
                        "covered",
                        "coverage",
                        "power",
                        "status/source",
                    ],
                    phase_rows,
                ),
            ]
        )
    if operation_rows:
        lines.extend(
            [
                "",
                f"Operations (top {top_n} per phase)",
                _render_diagnostics_table(
                    [
                        "phase",
                        "operation",
                        "energy",
                        "latency",
                        "coverage",
                        "share",
                        "source",
                    ],
                    operation_rows,
                ),
            ]
        )
    return "\n".join(lines)


def _operation_sort_key(operation: dict[str, Any]) -> tuple[bool, float, str, str]:
    energy = operation.get("energy_wms")
    has_energy = isinstance(energy, int | float)
    return (
        not has_energy,
        -float(energy) if has_energy else 0.0,
        str(operation.get("name", "")),
        str(operation.get("source", "")),
    )


def _render_diagnostics_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    rendered = ["  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))]
    rendered.append("  ".join("-" * width for width in widths))
    rendered.extend("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)) for row in rows)
    return "\n".join(rendered)


def _format_power(value: object) -> str:
    if not isinstance(value, int | float):
        return "N/A"
    return f"{float(value):,.2f} W"


def _format_energy(value: object) -> str:
    if not isinstance(value, int | float):
        return "N/A"
    return f"{float(value):,.2f} W-ms"


def _format_latency(value: object) -> str:
    if not isinstance(value, int | float):
        return "N/A"
    return f"{float(value):,.2f} ms"


def _format_percent(value: object) -> str:
    if not isinstance(value, int | float):
        return "N/A"
    return f"{float(value):.2%}"


def _append_stat_row(
    rows: list[list[str]],
    report: dict[str, Any],
    label: str,
    suffix: str,
) -> None:
    if f"mean_{suffix}" not in report:
        return
    rows.append(
        [
            label,
            *[
                _format_value(report.get(f"{prefix}_{suffix}"))
                for prefix in ("mean", "min", "max", "p99", "p90", "p75", "std")
            ],
        ]
    )


def _render_table(rows: list[list[str]]) -> str:
    headers = ["Metric", *STAT_COLUMNS]
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def render_separator(left: str, mid: str, right: str) -> str:
        return left + mid.join("━" * (width + 2) for width in widths) + right

    def render_row(row: list[str]) -> str:
        padded = []
        for index, value in enumerate(row):
            if index == 0:
                padded.append(f" {value.ljust(widths[index])} ")
            else:
                padded.append(f" {value.rjust(widths[index])} ")
        return "┃" + "┃".join(padded) + "┃"

    lines = [
        render_separator("┏", "┳", "┓"),
        render_row(headers),
        render_separator("┡", "╇", "┩"),
    ]
    lines.extend(render_row(row) for row in rows)
    lines.append(render_separator("└", "┴", "┘"))
    return "\n".join(lines)


def _format_value(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int | float):
        return f"{value:,.2f}"
    return str(value)
