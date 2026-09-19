# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate collected FPM timings through ordinary cold aggregated replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from aisimulate.config import CorePredictionConfig
from aisimulate.config.common import load_yaml
from aisimulate.output import prepare_output_directory

from .plan import check_plan, plan_lock, request_id
from .schema import SupportRequest


def add_validation_parser(actions: Any) -> None:
    parser = actions.add_parser(
        "validate-fpm",
        help="Check collected direct-FPM query coverage with a local Weka/AgentX trace.",
        description=(
            "Run ordinary predict with cold aggregated replay, one client lane, HBM-only cache and no speculation. "
            "The trace may contain one play or a selected corpus. All selected requests must finish and every "
            "native timing lookup must resolve for coverage to pass. Accuracy is not assessed. "
            "This command does not download traces or collect GPU measurements."
        ),
    )
    parser.add_argument("-c", "--config", required=True, help="Reviewed onboarding request.")
    parser.add_argument("--output-dir", required=True, help="Existing matching collection plan directory.")
    parser.add_argument("--trace", required=True, help="Local Weka JSON/JSONL file; replay its full contents.")
    parser.add_argument(
        "--validation-output-dir", required=True, help="Separate directory for config, replay and coverage evidence."
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace previous validation outputs.")


def _file_identity(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "size_bytes": path.stat().st_size}


def _fpm_artifacts(plan: Path) -> list[dict[str, Any]]:
    return [
        _file_identity(path) for path in sorted((plan / "systems/data").rglob("fpm_forward_perf.*")) if path.is_file()
    ]


def _prediction_config(request: SupportRequest, plan: Path, trace: Path) -> dict[str, Any]:
    """Retain the reviewed deployment and replace only evaluation inputs."""

    prediction = load_yaml(plan / "predict/pilot.yaml")
    prediction["traffic"] = {
        "source": {
            "type": "trace",
            "paths": [str(trace)],
            "format": "weka",
            "nested_timestamp_basis": "absolute",
        },
        "load": {"type": "trace_timestamps", "agentic_lanes": 1},
    }
    prediction["evaluation"] = {}
    engine = prediction["engine"]
    if engine["mode"] != "aggregated" or engine.get("speculation") is not None:
        raise ValueError("FPM replay validation requires aggregated serving with speculation disabled")
    worker = engine["workers"]["aggregated"]
    timing = worker["timing"]
    timing["estimation_mode"] = "fpm_interpolation"
    timing["fallback_policy"] = "deny"
    interpolation = timing.setdefault("estimator_config", {}).setdefault("fpm_interpolation", {})
    interpolation.update(method="direct", collect_coverage=True)
    if engine["model"] != request.identity.model:
        raise ValueError("saved prediction model differs from the reviewed collection identity")
    CorePredictionConfig.model_validate(prediction)
    return prediction


def _completion_issues(report: dict[str, Any]) -> list[str]:
    """Check terminal evidence, including requests that a failed play never dispatched."""

    issues = []
    graph = report.get("agentic_graph", {})
    count = graph.get("node_count", 0)
    if not isinstance(count, int) or count <= 0:
        return ["replay did not report a nonempty AgentX request graph"]
    if report.get("num_requests") != count or report.get("completed_requests") != count:
        issues.append("not every selected request completed")
    records = report.get("per_request", [])
    if len(records) != count or any(
        row.get("terminal_status") != "completed" or row.get("output_length") != row.get("requested_output_length")
        for row in records
    ):
        issues.append("request evidence is missing, failed, or has incomplete output")
    outcomes = report.get("agentic_play_outcomes", [])
    if (
        not outcomes
        or len(outcomes) != graph.get("play_count")
        or any(row.get("status") != "completed" or row.get("settled_at_ms") is None for row in outcomes)
    ):
        issues.append("not every selected play reached completed settlement")
    return issues


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_validation(args: argparse.Namespace) -> int:
    request = SupportRequest.from_yaml(args.config)
    plan = Path(args.output_dir).expanduser().resolve()
    trace = Path(args.trace).expanduser().resolve()
    output = Path(args.validation_output_dir).expanduser().resolve()
    if not trace.is_file() or trace.suffix.lower() not in {".json", ".jsonl"}:
        raise ValueError("--trace must be a local Weka JSON or JSONL file")
    if output == plan or plan in output.parents or output in plan.parents:
        raise ValueError("validation output must be separate from the collection plan directory")
    for source in (Path(args.config).expanduser().resolve(), trace):
        if source == output or output in source.parents:
            raise ValueError(f"validation output must not contain the input file {source}")
    with plan_lock(plan):
        check_plan(request, plan)
        prediction = _prediction_config(request, plan, trace)
        collection = json.loads((plan / "support-plan.json").read_text(encoding="utf-8"))
    for relative in ("predict.yaml", "validation.json", "prediction"):
        if (output / relative).is_symlink():
            raise ValueError(f"refusing symlinked validation output {output / relative}")
    prepare_output_directory(output, overwrite=args.overwrite)
    prepare_output_directory(output / "prediction", overwrite=args.overwrite)
    prediction_path = output / "predict.yaml"
    prediction_path.write_text(yaml.safe_dump(prediction, sort_keys=False), encoding="utf-8")
    report_path = output / "validation.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "incomplete",
        "accuracy": "not_assessed",
        "collection_id": request_id(request),
        "collection_plan": _file_identity(plan / "support-plan.json"),
        "trace": _file_identity(trace),
        "fpm_artifacts": _fpm_artifacts(plan),
        "scope": {
            "replay": "cold_aggregated",
            "agentic_lanes": 1,
            "nested_timestamp_basis": "absolute",
            "cache": "hbm_only",
            "speculation": "disabled",
            "model_projection": request.identity.model,
            "selected_corpus": "complete_local_file",
        },
        "prediction_config": _file_identity(prediction_path),
        "prediction_report": str(output / "prediction/prediction.json"),
        "coverage_report": str(output / "prediction/fpm-coverage.json"),
        "collection_schema_version": collection.get("schema_version"),
        "issues": ["replay has not completed"],
    }
    _write_report(report_path, report)
    # Use the existing supervised CLI: the same compiler, capacity checks,
    # timing provider, source importer and host limits apply to ordinary predict.
    from aisimulate.supervision import main as predict

    command = [
        "predict",
        "--config",
        str(prediction_path),
        "--output-dir",
        str(output / "prediction"),
        "--capture-per-request",
        "--format",
        "json",
    ]
    if args.overwrite:
        command.append("--overwrite")
    try:
        exit_code = predict(command)
    except SystemExit as exc:
        exit_code = int(exc.code) if isinstance(exc.code, int) else 1
    except KeyboardInterrupt:
        report["issues"] = ["replay interrupted; coverage remains incomplete"]
        _write_report(report_path, report)
        raise
    report["prediction_exit_code"] = exit_code
    coverage_path = Path(report["coverage_report"])
    coverage = json.loads(coverage_path.read_text(encoding="utf-8")) if coverage_path.is_file() else None
    report["fpm_query_coverage"] = coverage
    issues = []
    if exit_code != 0:
        issues.append(f"predict exited with status {exit_code}; replay coverage is incomplete")
    prediction_report = Path(report["prediction_report"])
    if prediction_report.is_file():
        replay = json.loads(prediction_report.read_text(encoding="utf-8"))
        issues.extend(_completion_issues(replay))
        report["agentic_graph"] = replay.get("agentic_graph")
        report["agentic_model_projection"] = replay.get("agentic_model_projection")
    else:
        issues.append("no completed prediction report was produced")
    if not coverage or coverage.get("status") != "covered":
        issues.append(
            "native direct-FPM query coverage did not pass; inspect the coverage report for missing coordinates"
        )
    if _file_identity(trace) != report["trace"]:
        issues.append("trace file changed during validation")
    if _fpm_artifacts(plan) != report["fpm_artifacts"]:
        issues.append("FPM library files changed during validation")
    with plan_lock(plan):
        check_plan(request, plan)
    report["issues"] = issues
    report["status"] = "covered" if not issues else "incomplete"
    _write_report(report_path, report)
    print(f"FPM replay coverage: {report['status']}; accuracy: not assessed. Saved {report_path}")
    return 0 if not issues else (exit_code or 1)
