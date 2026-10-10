# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare native replay timings only after validating the measured work."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from tools.simulation_perf_gate import PROTOCOL_VERSION, digest
from tools.simulation_perf_gate.contract import check_completion, check_finite, check_models, model_identity

RELATIVE_THRESHOLD = 0.10
ABSOLUTE_THRESHOLD_MS = 100.0


def validate(response: object, case: dict, revision: str, phase: str) -> dict:
    if not isinstance(response, dict):
        raise ValueError("missing worker response")
    for name, expected in (
        ("protocol_version", PROTOCOL_VERSION),
        ("case_id", case["case_id"]),
        ("case_hash", digest(case)),
        ("revision", revision),
        ("phase", phase),
    ):
        if type(response.get(name)) is not type(expected) or response[name] != expected:
            raise ValueError(f"{name} mismatch")
    if response.get("status") != "OK":
        raise ValueError(f"{response.get('status')}: {response.get('error')}")
    # Reject NaN anywhere, including fields not used by the time comparator.
    check_finite(response)
    for name in ("wall_time_ms", "worker_elapsed_ms"):
        value = response.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid {name}")
    check_models(response, case)
    report = response.get("report")
    check_completion(report, case)
    return model_identity(response["model_identity"])


def compare_case(case: dict, samples: dict, *, revisions: dict, rounds: int) -> dict:
    invalid, timings = [], []
    measured = samples.get("rounds", [])
    if len(measured) != rounds or [pair.get("round") for pair in measured] != list(range(1, rounds + 1)):
        invalid.append("missing, duplicate, or unordered measured rounds")
    reference_identity = None
    for pair in measured:
        valid = True
        for side in ("base", "head"):
            try:
                identity = validate(pair.get(side), case, revisions[side], "measure")
                if reference_identity is None:
                    reference_identity = identity
                elif identity != reference_identity:
                    raise ValueError("model identity changed")
            except (ValueError, TypeError, KeyError) as error:
                invalid.append(f"{pair.get('round')}/{side}: {error}")
                valid = False
        if not valid:
            continue
        base, head = pair["base"], pair["head"]
        b, h = base["wall_time_ms"], head["wall_time_ms"]
        timings.append(
            {
                "round": pair["round"],
                "base_ms": b,
                "head_ms": h,
                "ratio": h / b,
                "delta_ms": h - b,
                "base_worker_ms": base["worker_elapsed_ms"],
                "head_worker_ms": head["worker_elapsed_ms"],
                "exceeds": h / b > 1 + RELATIVE_THRESHOLD and h - b > ABSOLUTE_THRESHOLD_MS,
            }
        )
    exceed = sum(t["exceeds"] for t in timings)
    required = math.ceil(rounds * 0.8)
    classification = "INVALID_COMPARISON" if invalid else "PERFORMANCE_REGRESSION" if exceed >= required else "PASS"
    return {
        "case_id": case["case_id"],
        "classification": classification,
        "invalid_reasons": invalid,
        "rounds": timings,
        "exceed_count": exceed,
        "consensus_required": required,
        "base_median_ms": statistics.median(t["base_ms"] for t in timings) if timings else None,
        "head_median_ms": statistics.median(t["head_ms"] for t in timings) if timings else None,
        "base_worker_median_ms": statistics.median(t["base_worker_ms"] for t in timings) if timings else None,
        "head_worker_median_ms": statistics.median(t["head_worker_ms"] for t in timings) if timings else None,
    }


def write_report(raw: dict, output: Path) -> list[dict]:
    results = [
        compare_case(
            case, raw["samples"].get(case["case_id"], {}), revisions=raw["revisions"], rounds=raw["round_count"]
        )
        for case in raw["cases"]
    ]
    (output / "comparison.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    blockers = [result for result in results if result["classification"] != "PASS"]
    lines = [
        "## Simulation Performance (advisory)",
        "",
        f"Base `{raw['revisions']['base']}`; head `{raw['revisions']['head']}`.",
        "",
        f"{len(results)} cases; {len(blockers)} need review. Elapsed benchmark time: "
        f"{raw.get('elapsed_seconds', 0):.1f} s (builds excluded).",
        "",
        "Threshold: >10% and >100 ms in at least 80% of paired rounds.",
        "",
    ]
    for result in blockers:
        reasons = result["invalid_reasons"]
        lines += [f"- **{result['classification']}** `{result['case_id']}`" + (f": {reasons[0]}" if reasons else "")]
    for error in raw.get("qualification_errors", []):
        lines.append(f"- **QUALIFICATION_FAILED**: {error}")
    lines += [
        "",
        "| Case | Result | Base replay ms | Head replay ms | Change | Base worker ms | Head worker ms |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        b, h = result["base_median_ms"], result["head_median_ms"]
        values = f"{b:.2f} | {h:.2f} | {(h / b - 1) * 100:+.1f}%" if b is not None else "— | — | —"
        bw, hw = result["base_worker_median_ms"], result["head_worker_median_ms"]
        workers = f"{bw:.2f} | {hw:.2f}" if bw is not None else "— | —"
        lines.append(f"| {result['case_id']} | {result['classification']} | {values} | {workers} |")
    lines += ["", "Worker elapsed times, inputs, replay summaries, and individual rounds are in the artifact.", ""]
    (output / "summary.md").write_text("\n".join(lines))
    (output / "annotations.txt").write_text(
        "".join(f"{r['classification']}: {r['case_id']}\n" for r in blockers)
        + "".join(f"QUALIFICATION_FAILED: {error}\n" for error in raw.get("qualification_errors", []))
    )
    return results
