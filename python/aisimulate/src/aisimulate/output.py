# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal durable-output contract for the unified CLI."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .detail import format_prediction_details
from .power import format_power_summary
from .replay.reporting import format_report_table
from .sweeper.result import SweepResult

_RECOMMENDATION_NAME = re.compile(r"^[0-9]{4}\.yaml$")


def prepare_output_directory(path: str | Path, *, overwrite: bool) -> Path:
    """Create an output directory without deleting unrelated user files."""

    root = Path(path)
    if root.exists() and not root.is_dir():
        raise ValueError(f"output path {root} exists and is not a directory")
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise ValueError(f"output directory {root} is not empty; pass --overwrite to replace known AISimulate outputs")
    root.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in (
            "prediction.json",
            "recommendation.json",
            "recommendation.csv",
            "requests.jsonl",
            "afd-replay-spec.json",
            "afd-qualification.json",
        ):
            target = root / name
            if target.is_file() or target.is_symlink():
                target.unlink()
        recommendations = root / "recommendations"
        if recommendations.is_dir():
            for target in recommendations.iterdir():
                if _RECOMMENDATION_NAME.fullmatch(target.name) and (target.is_file() or target.is_symlink()):
                    target.unlink()
    return root


def write_prediction_report(root: Path, report: dict[str, Any]) -> Path:
    path = root / "prediction.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_requests(root: Path, records: list[dict[str, Any]]) -> Path:
    path = root / "requests.jsonl"
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            output.write("\n")
    return path


def write_recommendation_result(root: Path, result: SweepResult) -> Path:
    """Write the lossless recommendation ledger and derived candidate views."""

    path = root / "recommendation.json"
    path.write_text(result.to_json() + "\n", encoding="utf-8")
    return path


def write_recommendation_csv(root: Path, result: SweepResult) -> Path:
    """Write the complete candidate ledger as an analysis-friendly CSV."""

    path = root / "recommendation.csv"
    path.write_text(result.to_csv(), encoding="utf-8")
    return path


def write_recommendations(root: Path, configs: list[Mapping[str, Any]]) -> list[Path]:
    directory = root / "recommendations"
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, config in enumerate(configs, start=1):
        path = directory / f"{index:04d}.yaml"
        path.write_text(
            yaml.safe_dump(
                dict(config),
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def format_prediction_stdout(
    summary: dict[str, Any], output_format: str, *, details: dict[str, Any] | None = None
) -> str:
    if output_format == "json":
        payload = summary if details is None else {"summary": summary, "details": details}
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if details is not None:
        return format_prediction_stdout(summary, output_format) + "\n\n" + format_prediction_details(details)
    if summary.get("metric_semantics") == "analytical_epd_overlay":
        lines = ["AISimulate analytical EPD (aggregate estimates; no encoder queue simulation)"]
        for name in (
            "mean_ttft_ms",
            "mean_tpot_ms",
            "mean_e2e_latency_ms",
            "output_throughput_tok_s",
            "completed_requests",
            "duration_ms",
            "gpu_hours",
            "encoder_gpus",
            "total_gpus",
        ):
            lines.append(f"{name}: {summary.get(name, 'N/A')}")
        lines.append(format_power_summary(summary))
        lines.append("duration_ms is a rate-derived accounting interval, not an EPD event timeline.")
        return "\n".join(lines)
    table = format_report_table(summary)
    if summary.get("agentic_qualification") == "functional_only":
        return "AgentX functional replay only; not an AgentX benchmark result.\n" + table
    return table


def format_recommendation_stdout(rows: list[dict[str, Any]], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(rows, sort_keys=True, separators=(",", ":"))
    if not rows:
        return "No feasible candidate found."
    lines = ["AISimulate recommendations"]
    for row in rows:
        objective = row.get("objectives") or {"score": row.get("score")}
        metrics = ", ".join(f"{key}={value:.4g}" for key, value in objective.items())
        power = " " + format_power_summary(row)
        lines.append(f"{row['rank']}: {metrics} used_gpus={row['used_gpus']}{power} config={row['config_path']}")
    return "\n".join(lines)
