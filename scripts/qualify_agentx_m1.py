#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify public AgentX M1 replay using one published Weka play or a local fixture.

Published rows and their materialized v2 derivative stay in a temporary directory.
Fixed timing checks execution semantics, not prediction accuracy or benchmark fidelity.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from qualify_weka_samples import DATASET, DATASET_REVISION, fetch_two_rows, qualify

from aisimulate import CorePredictionConfig, EngineReplayRunnerFactory, ReplayOutputRequirements
from aisimulate.compiler import prediction_to_replay_spec


def prediction_config(path: Path, trace_format: str, block_size: int, backend: str) -> dict[str, Any]:
    source = {"type": "trace", "paths": [str(path)], "format": trace_format, "block_size": block_size}
    return {
        "traffic": {
            "source": source,
            "load": {"type": "trace_timestamps", "speedup": 1, "agentic_lanes": 1},
        },
        "engine": {
            "mode": "aggregated",
            "backend": backend,
            "model": "example/model",
            "hardware": "h200_sxm",
            "context_length": 1_048_576,
            "workers": {
                "aggregated": {
                    "scheduler": {"max_batched_tokens": 8192, "max_sequences": 256},
                    "kv_cache": {"block_size": 64, "capacity": {"type": "fixed", "blocks": 65_536}},
                    "timing": {"type": "fixed", "prefill_ms": 1, "decode_ms": 1},
                }
            },
        },
    }


def evidence(report: dict[str, Any]) -> dict[str, Any]:
    """Compare workload and simulated-time evidence, excluding host wall-clock metrics."""
    if report.get("agentic_qualification") != "functional_only" or report.get("agentic_lanes") != 1:
        raise RuntimeError("M1 report must identify functional-only execution with one lane")
    graph = report["agentic_graph"]
    if report["completed_requests"] != graph["node_count"]:
        raise RuntimeError("published M1 play did not complete every request")
    records = report["per_request"]
    if len(records) != graph["node_count"] or any(
        row["terminal_status"] != "completed" or row["output_length"] != row["requested_output_length"]
        for row in records
    ):
        raise RuntimeError("M1 request artifact must contain every completed request and its full output")
    outcomes = report["agentic_play_outcomes"]
    if len(outcomes) != graph["play_count"] or any(
        row["status"] != "completed" or row["settled_at_ms"] is None for row in outcomes
    ):
        raise RuntimeError("M1 play did not reach completed settlement")
    # Every successful request dispatches, terminates, and quiesces; each play
    # then contributes one settlement event.
    if report["agentic_lifecycle_event_count"] != 3 * graph["node_count"] + graph["play_count"]:
        raise RuntimeError("M1 lifecycle evidence is incomplete")
    if min(row["dispatched_at_ms"] for row in records) != 0:
        raise RuntimeError("M1 replay must start at turn zero")
    return {
        key: report[key]
        for key in (
            "agentic_graph",
            "agentic_lifecycle_digest",
            "agentic_lifecycle_event_count",
            "agentic_play_outcomes",
            "agentic_model_projection",
            "per_request",
        )
    }


def run_python(config: dict[str, Any]) -> dict[str, Any]:
    runner = EngineReplayRunnerFactory().create(0)
    try:
        report = runner.run(
            prediction_to_replay_spec(CorePredictionConfig.model_validate(config)),
            output_requirements=ReplayOutputRequirements(include_raw_report=True, capture_per_request=True),
        )
        return report.metadata["native_report"]
    finally:
        runner.close()


def run_cli(root: Path, directory: Path, config: dict[str, Any]) -> dict[str, Any]:
    directory.mkdir()
    config_path = directory / "prediction-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "aisimulate",
            "predict",
            "--stack",
            "engine",
            "--config",
            str(config_path),
            "--output-dir",
            str(directory / "result"),
            "--capture-per-request",
            "--format",
            "json",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    report = json.loads((directory / "result/prediction.json").read_text(encoding="utf-8"))
    if evidence(json.loads(completed.stdout)) != evidence(report):
        raise RuntimeError("CLI stdout and saved report disagree")
    records = [json.loads(line) for line in (directory / "result/requests.jsonl").read_text().splitlines()]
    if records != report["per_request"]:
        raise RuntimeError("CLI request artifact differs from its report")
    return report


def run_gate(root: Path, source: Path, directory: Path) -> dict[str, Any]:
    materialized = directory / "agentic-v2.jsonl"
    result = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--locked",
            "-p",
            "aisimulate-core",
            "--example",
            "qualify_weka",
            "--",
            str(source),
            "--materialize",
            str(materialized),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    graph = json.loads(result.stdout)
    matrix = []
    for backend in ("vllm", "sglang"):
        baseline = None
        for trace_format, path in (("weka", source), ("agentic_mooncake", materialized)):
            config = prediction_config(path, trace_format, graph["block_size"], backend)
            reports = (
                run_python(config),
                run_python(config),
                run_cli(root, directory / f"{backend}-{trace_format}", config),
            )
            for report in reports:
                current = evidence(report)
                if report["agentic_input_format"] != trace_format:
                    raise RuntimeError(f"{backend}/{trace_format}: public runtime reported the wrong input format")
                if current["agentic_graph"]["graph_digest"] != graph["graph_digest"]:
                    raise RuntimeError(f"{backend}/{trace_format}: public runtime changed the validated graph")
                if baseline is None:
                    baseline = current
                elif current != baseline:
                    changed = [key for key in baseline if current[key] != baseline[key]]
                    raise RuntimeError(f"{backend}/{trace_format}: public replay parity failed: {changed}")
            matrix.append(
                {
                    "backend": backend,
                    "format": trace_format,
                    "python_repeat_and_cli_parity": "passed",
                    "completed_requests": reports[0]["completed_requests"],
                    "lifecycle_digest": reports[0]["agentic_lifecycle_digest"],
                }
            )
    return {
        "qualification": "functional_only",
        "topology": "aggregated",
        "agentic_lanes": 1,
        "timing": "fixed",
        "memory": "HBM-only",
        "speculative_decoding": False,
        "graph": graph,
        "matrix": matrix,
        "dynamo_compatibility": "separate qualification in Dynamo PR #14355",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, help="use a local Weka fixture instead of fetching a published sample")
    parser.add_argument("--output", type=Path, help="save the qualification summary as JSON")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="aisimulate-agentx-m1-") as temporary:
        directory = Path(temporary)
        if args.trace is None:
            rows = fetch_two_rows()
            qualify(root, rows)  # Check the pinned sample's existing source and graph digests first.
            source = directory / "published-play.jsonl"
            source.write_text(json.dumps(rows[0], separators=(",", ":")) + "\n", encoding="utf-8")
            provenance = {"dataset": DATASET, "revision": DATASET_REVISION, "play_id": rows[0]["id"]}
        else:
            source = args.trace.resolve(strict=True)
            provenance = {"local_trace": str(source)}
        report = run_gate(root, source, directory)
        report["source"] = provenance
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(serialized, encoding="utf-8")
    sys.stdout.write(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
