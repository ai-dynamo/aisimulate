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


def format_prediction_stdout(summary: dict[str, Any], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(summary, sort_keys=True, separators=(",", ":"))
    return format_report_table(summary)


def format_recommendation_stdout(rows: list[dict[str, Any]], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(rows, sort_keys=True, separators=(",", ":"))
    if not rows:
        return "No feasible candidate found."
    lines = ["AISimulate recommendations"]
    for row in rows:
        objective = row.get("objectives") or {"score": row.get("score")}
        metrics = ", ".join(f"{key}={value:.4g}" for key, value in objective.items())
        power = ""
        if "power_w" in row:
            power += f" power_w={row['power_w']:.4g}W"
        if "power_coverage" in row:
            power += f" power_coverage={row['power_coverage']:.2%}"
        lines.append(
            f"{row['rank']}: {metrics} used_gpus={row['used_gpus']}"
            f"{power} config={row['config_path']}"
        )
    return "\n".join(lines)
