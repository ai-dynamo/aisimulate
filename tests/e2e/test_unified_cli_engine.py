# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only end-to-end coverage for the public engine-stack CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from aisimulate.config.cli import CorePredictionConfig

pytestmark = [
    pytest.mark.integration,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_ROOT = Path("tests/e2e/configs/unified_cli")
_EXPECTED_PREDICT_CASES = (
    "01-synthetic-default.yaml",
    "02-synthetic-poisson.yaml",
    "03-synthetic-session-constant.yaml",
    "04-trace-mooncake-speedup.yaml",
    "05-trace-mooncake-delta-concurrency.yaml",
    "06-trace-agentic-mooncake.yaml",
    "07-trace-applied-compute-agentic.yaml",
    "08-trace-dynamo-standard.yaml",
    "09-trace-dynamo-agentic.yaml",
    "10-trace-dynamo-standard-disagg.yaml",
    "11-trace-weka-agentic-lane.yaml",
    "12-trace-weka-jsonl-agentic-lane.yaml",
)
_EXPECTED_RECOMMEND_CASES = (
    "01-default-preset-throughput.yaml",
    "02-custom-preset-throughput-per-gpu.yaml",
    "03-preset-off-ttft.yaml",
    "04-mixed-disagg-pareto.yaml",
    "05-kv-fraction-goodput.yaml",
    "06-override-parallel-mappings-agg-disagg.yaml",
)
_PREDICT_CASES = tuple(
    sorted((_REPO_ROOT / _CONFIG_ROOT / "predict/engine").glob("*.yaml"))
)
_RECOMMEND_CASES = tuple(
    sorted((_REPO_ROOT / _CONFIG_ROOT / "recommend/engine").glob("*.yaml"))
)


def _run_cli(*args: str, timeout: float = 120.0, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, "-m", "aisimulate", *args],
        cwd=_REPO_ROOT,
        env=None if env is None else {**os.environ, **env},
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    assert result.returncode == 0, (
        f"aisimulate {' '.join(args)} failed with {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


def _assert_concrete(value: Any, *, path: str = "config") -> None:
    if isinstance(value, dict):
        assert "choices" not in value, f"{path} still contains a choices domain"
        assert "range" not in value, f"{path} still contains a range domain"
        assert "preset" not in value, f"{path} still contains a preset"
        for key, child in value.items():
            _assert_concrete(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_concrete(child, path=f"{path}[{index}]")


def test_engine_cli_case_matrix_is_complete() -> None:
    assert tuple(path.name for path in _PREDICT_CASES) == _EXPECTED_PREDICT_CASES
    assert tuple(path.name for path in _RECOMMEND_CASES) == _EXPECTED_RECOMMEND_CASES


@pytest.mark.parametrize("config_path", _PREDICT_CASES, ids=lambda path: path.stem)
def test_engine_predict_cli_cases(config_path: Path, tmp_path: Path) -> None:
    output = tmp_path / config_path.stem
    result = _run_cli(
        "predict",
        "--stack",
        "engine",
        "--config",
        str(config_path.relative_to(_REPO_ROOT)),
        "--output-dir",
        str(output),
        "--format",
        "json",
    )

    summary = json.loads(result.stdout)
    report = json.loads((output / "prediction.json").read_text(encoding="utf-8"))
    saved_summary = report.get("summary", report)
    assert summary["completed_requests"] > 0
    assert saved_summary["completed_requests"] == summary["completed_requests"]
    if config_path.name.startswith(("11-", "12-")):
        assert "heuristically resolved one nested timestamp basis" in result.stderr
        assert "complete Weka corpus" in result.stderr
        assert "requested='auto', resolved='absolute'" in result.stderr


@pytest.mark.parametrize("config_path", _RECOMMEND_CASES, ids=lambda path: path.stem)
def test_engine_recommend_cli_cases_round_trip(
    config_path: Path, tmp_path: Path
) -> None:
    output = tmp_path / config_path.stem
    result = _run_cli(
        "recommend",
        "--stack",
        "engine",
        "--config",
        str(config_path.relative_to(_REPO_ROOT)),
        "--output-dir",
        str(output),
        "--format",
        "json",
    )

    rows = json.loads(result.stdout)
    recommendation_paths = sorted((output / "recommendations").glob("*.yaml"))
    assert rows
    assert len(recommendation_paths) == len(rows)
    assert len({path.read_bytes() for path in recommendation_paths}) == len(
        recommendation_paths
    )

    generated_modes = set()
    for index, recommendation_path in enumerate(recommendation_paths):
        raw = yaml.safe_load(recommendation_path.read_text(encoding="utf-8"))
        _assert_concrete(raw)
        CorePredictionConfig.model_validate(raw)
        generated_modes.add(raw["engine"]["mode"])

        prediction_output = tmp_path / f"{config_path.stem}-predict-{index}"
        prediction = _run_cli(
            "predict",
            "--stack",
            "engine",
            "--config",
            str(recommendation_path),
            "--output-dir",
            str(prediction_output),
            "--format",
            "json",
        )
        assert json.loads(prediction.stdout)["completed_requests"] > 0

    if config_path.name == "06-override-parallel-mappings-agg-disagg.yaml":
        assert generated_modes == {"aggregated", "disaggregated"}
        assert [row["score"] for row in rows] == sorted((row["score"] for row in rows), reverse=True)
