#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check a small fixed set of native prediction values against reviewed baselines.

This is numerical stability coverage, not predictive accuracy against hardware.
The broad old/new modeling report remains advisory for numerical drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def validate_cases(manifest: dict) -> list[dict]:
    cases = manifest["cases"]
    if manifest["schema_version"] != 1 or not cases:
        raise ValueError("sentinel manifest must have schema 1 and nonempty cases")
    ids = [c["id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("sentinel identities must be unique")
    for case in cases:
        if case["method"] not in {"predict_prefill_latency", "predict_decode_latency"}:
            raise ValueError(f"unsupported native query: {case['id']}")
        for field, maximum in (("expected_ms", math.inf), ("rtol", 0.1), ("atol_ms", 0.001)):
            value = case[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 < value <= maximum
            ):
                raise ValueError(f"invalid {field}: {case['id']}")
    return cases


def check_results(cases: list[dict], rows: list[dict]) -> list[str]:
    actual = {r["id"]: r for r in rows}
    expected = {c["id"] for c in cases}
    if len(actual) != len(rows) or set(actual) != expected:
        return ["missing, duplicate, or unexpected sentinel results"]
    failures = []
    for case in cases:
        row = actual[case["id"]]
        value = row.get("latency_ms")
        if (
            row.get("status") != "PASS"
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            failures.append(f"{case['id']}: native query did not produce a finite positive value: {row}")
        elif not math.isclose(value, case["expected_ms"], rel_tol=case["rtol"], abs_tol=case["atol_ms"]):
            failures.append(
                f"{case['id']}: {value:.9g} ms, expected {case['expected_ms']:.9g} ms "
                f"(rtol={case['rtol']}, atol_ms={case['atol_ms']})"
            )
    return failures


def collect(cases: list[dict]) -> list[dict]:
    from aisimulate_core.sdk import EngineHandle

    engines = {}
    rows = []
    for case in cases:
        try:
            key = json.dumps(case["compile"], sort_keys=True)
            if key not in engines:
                engines[key] = EngineHandle.compile(**case["compile"])
            value = getattr(engines[key], case["method"])(**case["arguments"])
            if not math.isfinite(value):
                raise ValueError(f"nonfinite native latency: {value}")
            rows.append({"id": case["id"], "status": "PASS", "latency_ms": value})
        except Exception as error:
            rows.append({"id": case["id"], "status": "ERROR", "error": f"{type(error).__name__}: {error}"})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".github/prediction-numerical-sentinels.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    cases = validate_cases(manifest)
    rows = collect(cases)
    failures = check_results(cases, rows)
    result = {
        "source_sha": os.environ.get("GITHUB_SHA"),
        "baseline_source_sha": manifest["baseline_source_sha"],
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "qualification": "native_prediction_stability",
        "results": rows,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for failure in failures:
        print(f"::error::{failure}")
    print(f"Native prediction sentinels: {len(cases)} cases, {len(failures)} failures")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
