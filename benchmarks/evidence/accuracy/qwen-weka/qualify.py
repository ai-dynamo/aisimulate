#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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


def read(path):
    return json.loads(path.read_bytes())


def qualify(normalized, replay):
    features = validate_qwen_baseline(read(normalized / "feature_metadata.json"))
    receipt = read(normalized / "normalization.json")
    if receipt["benchmark_id"] != "qwen-weka-l14":
        raise ValueError("Pilot must use 14 lanes")
    duration = (int(receipt["profile_end_ns"]) - int(receipt["profile_start_ns"])) / 1e9
    if abs(duration - 600) > 1:
        raise ValueError("Pilot did not finish its full 600-second window")
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
    native = read(replay / "native.json")
    telemetry = native.get("telemetry", {})
    if telemetry.get("completion_status") != "completed":
        raise ValueError("Pilot native replay ended early")
    if not telemetry.get("requests") or not telemetry.get("samples"):
        raise ValueError("Pilot native request/sample telemetry is missing")
    return {
        "schema": "aisimulate.qwen-weka.pilot-qualification.v1",
        "status": "qualified",
        "measured_requests": len(measured),
        "failed_requests": failures,
        "numerical_accuracy": "advisory_no_threshold",
        "feature_metadata_sha256": hashlib.sha256((normalized / "feature_metadata.json").read_bytes()).hexdigest(),
        "native_sha256": hashlib.sha256((replay / "native.json").read_bytes()).hexdigest(),
        "feature_count": len(features["features"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(qualify(args.normalized, args.replay), indent=2) + "\n")
