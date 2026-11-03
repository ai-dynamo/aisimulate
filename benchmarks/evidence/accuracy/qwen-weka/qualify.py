#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualify retained pilot evidence and native replay before spending six profile hours."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.e2e_accuracy.benchmark_features import validate_qwen_baseline
from scripts.e2e_accuracy.agentx_contract import native_completion_qualified, score
from scripts.e2e_accuracy.run_agentx_accuracy import normalize_native


def read(path):
    return json.loads(path.read_bytes())


def qualify(normalized, replay, wheel_provenance):
    features = validate_qwen_baseline(read(normalized / "feature_metadata.json"))
    receipt = read(normalized / "normalization.json")
    if receipt["benchmark_id"] != "qwen-weka-l14":
        raise ValueError("Pilot must use 14 lanes")
    duration = (int(receipt["profile_end_ns"]) - int(receipt["profile_start_ns"])) / 1e9
    if receipt["profile_requested_seconds"] != 600 or not 599 <= duration <= 641:
        raise ValueError("Pilot did not finish its 600-second admission window and bounded grace/drain")
    observations = read(normalized / "observations.json")
    measured = [row for row in observations if row["phase"] == "profile"]
    if not measured or len(measured) != receipt["measured_requests"]:
        raise ValueError("Pilot request accounting does not reconcile")
    failures = sum(row["status"] != "completed" for row in measured)
    # Use the harness's predeclared failure threshold, never an accuracy target.
    if failures / len(measured) > 0.10:
        raise ValueError("Pilot exceeds the harness's ten-percent request failure threshold")
    if not any(row["phase"] == "preparation" for row in observations):
        raise ValueError("Pilot has no retained preparation observations")
    config = read(normalized / "config.json")
    selections = config["traffic"]["load"]["agentic_snapshot"]["initial_selections"]
    if len(selections) != 14 or config["engine"]["context_length"] != 262144:
        raise ValueError("Pilot selections or native context differ")
    outcome = read(replay / "outcome.json")
    if outcome.get("outcome") != "completed":
        raise ValueError("Pilot native replay did not complete: " + str(outcome.get("reason")))
    wheel = read(wheel_provenance)
    if outcome.get("prediction_backend_version") != "0.31.0":
        raise ValueError("Pilot did not use the reviewed vLLM 0.31.0 database")
    expected = {row["file"].removeprefix("aisimulate_core/systems/data/"): row["sha256"] for row in wheel["op_data"]}
    if not outcome.get("op_data") or any(expected.get(row["file"]) != row["sha256"] for row in outcome["op_data"]):
        raise ValueError("Pilot op-data checksums differ from the evaluated wheel")
    native = read(replay / "native.json")
    telemetry = native.get("telemetry", {})
    if not native_completion_qualified(native, "successful_profile_span"):
        raise ValueError("Pilot native replay did not qualify: " + str(telemetry.get("completion_status")))
    if not telemetry.get("requests") or not telemetry.get("samples") or not telemetry.get("lifecycle"):
        raise ValueError("Pilot native request/sample/lifecycle telemetry is missing")
    scored = score(
        observations,
        normalize_native(native, observations, "self_collected", "successful_profile_span"),
        "successful_profile_span",
    )
    if not scored["coverage"]["matched"] or any(not item["count"] for item in scored["request_errors"].values()):
        raise ValueError("Pilot has no valid comparable request cohort")
    return {
        "schema": "aisimulate.qwen-weka.pilot-qualification.v1",
        "status": "qualified",
        "measured_requests": len(measured),
        "failed_requests": failures,
        "numerical_accuracy": "advisory_no_threshold",
        "feature_metadata_sha256": hashlib.sha256((normalized / "feature_metadata.json").read_bytes()).hexdigest(),
        "native_sha256": hashlib.sha256((replay / "native.json").read_bytes()).hexdigest(),
        "feature_count": len(features["features"]),
        "native_completion_status": telemetry["completion_status"],
        "native_profile": native.get("agentic_profile"),
        "evaluated_commit": wheel["evaluated_commit"],
        "wheel_sha256": wheel["wheel_sha256"],
        "matched_requests": scored["coverage"]["matched"],
        "metric_denominators": {name: item["count"] for name, item in scored["request_errors"].items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel-provenance", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(qualify(args.normalized, args.replay, args.wheel_provenance), indent=2) + "\n")
