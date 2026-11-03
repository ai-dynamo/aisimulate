#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Archived Qwen CPU regression controls using their explicit SDK configuration."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from aisimulate import CorePredictionConfig, EngineReplayRunnerFactory, ReplayOutputRequirements
from aisimulate.compiler import prediction_to_replay_spec

CASES = {"self-p4096-l1-cold-r0": 8, "self-p4096-l4-cold-r0": 32, "self-p4096-l16-cold-r0": 128}
OMIT = object()


def numeric(value):
    if value is None or type(value) in (int, float, bool):
        return value
    if isinstance(value, dict):
        return {key: selected for key, item in value.items() if (selected := numeric(item)) is not OMIT}
    if isinstance(value, list):
        return [selected for item in value if (selected := numeric(item)) is not OMIT]
    return OMIT


def verify(reference, candidate, count):
    if len(candidate["per_request"]) != count or len(reference["per_request"]) != count:
        raise ValueError("Request count changed")
    if numeric(candidate["per_request"]) != numeric(reference["per_request"]):
        raise ValueError("Numerical request fields changed")
    if numeric(candidate["telemetry"]["requests"]) != numeric(reference["telemetry"]["requests"]):
        raise ValueError("Numerical telemetry fields changed")
    if candidate["state_cache"] != reference["state_cache"]:
        raise ValueError("Cache accounting changed")
    for key, value in reference.items():
        if ((value is None or type(value) in (int, float, bool)) and key != "wall_time_ms"
                and candidate[key] != value):
            raise ValueError(f"Numerical aggregate changed: {key}")
    for before, after in zip(reference["telemetry"]["requests"], candidate["telemetry"]["requests"], strict=True):
        if (before["request_id"] != after["request_id"] or before["phase"] != after["phase"]
                or after["terminal_status"] != "completed"):
            raise ValueError("Identity, phase or completion changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    ledger = {"scope": "Archived finite Qwen smoke cases; no new GPU collection or active coverage change",
              "planned_cases": len(CASES), "attempted_cases": 0, "passed_cases": 0, "status": "running"}
    ledger_path = args.output / "accounting.json"
    try:
        for identity, count in CASES.items():
            ledger["attempted_cases"] += 1
            ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
            directory = args.output / identity
            directory.mkdir()
            data = (args.config_root / identity / "input-config.json").read_bytes()
            (directory / "input-config.json").write_bytes(data)
            raw = json.loads(data)
            if raw["engine"]["backend"] != "vllm" or raw["engine"]["backend_version"] != "0.31.0":
                raise ValueError("Archived controls require explicit vLLM 0.31.0")
            config = CorePredictionConfig.model_validate(raw)
            spec = prediction_to_replay_spec(config)
            factory = EngineReplayRunnerFactory()
            factory.capabilities().require_compatible(spec)
            runner = factory.create(worker_id=0)
            started = time.monotonic()
            try:
                result = runner.run(spec, output_requirements=ReplayOutputRequirements(
                    include_raw_report=True, capture_per_request=True, capture_telemetry=True))
            finally:
                runner.close()
            native = result.metadata["native_report"]
            native_path = directory / "native.json"
            native_path.write_text(json.dumps(native, allow_nan=False) + "\n")
            reference_path = args.reference / f"{identity}.json"
            verify(json.loads(reference_path.read_bytes()), native, count)
            record = {"id": identity, "requests": count, "explicit_backend_version": "0.31.0",
                      "exact_numerical_request_parity": True, "exact_numerical_telemetry_parity": True,
                      "exact_cache_parity": True, "identity_and_completion_parity": True,
                      "wall_clock_execution_time_excluded": True,
                      "method": "CorePredictionConfig -> prediction_to_replay_spec -> EngineReplayRunnerFactory",
                      "config_sha256": hashlib.sha256(data).hexdigest(),
                      "native_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
                      "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                      "elapsed_seconds": time.monotonic() - started}
            results.append(record)
            (directory / "execution.json").write_text(json.dumps(record, indent=2) + "\n")
            ledger["passed_cases"] += 1
            print(json.dumps({"id": identity, "requests": count, "passed": True}), flush=True)
        ledger["status"] = "completed"
    except Exception as exc:
        ledger.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        ledger["unattempted_cases"] = len(CASES) - ledger["attempted_cases"]
        ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
        (args.output / "audit.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
