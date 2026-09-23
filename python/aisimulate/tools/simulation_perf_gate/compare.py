# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep changed behavior and invalid work separate from host speed changes."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from tools.simulation_perf_gate import PROTOCOL_VERSION, digest
from tools.simulation_perf_gate.worker import behavior

RELATIVE_THRESHOLD = 0.10
ABSOLUTE_THRESHOLD_MS = 100.0


def difference(left: object, right: object, path: str = "") -> str | None:
    """First difference, with exact integer/identity and tolerant float comparison."""
    if isinstance(left, float) and isinstance(right, float):
        if math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-6):
            return None
    elif type(left) is type(right):
        if isinstance(left, dict):
            if left.keys() != right.keys():
                return f"{path}: fields differ"
            for key in sorted(left):
                found = difference(left[key], right[key], f"{path}/{key}")
                if found:
                    return found
            return None
        if isinstance(left, list):
            if len(left) != len(right):
                return f"{path}: lengths differ ({len(left)} vs {len(right)})"
            for index, (a, b) in enumerate(zip(left, right, strict=True)):
                found = difference(a, b, f"{path}/{index}")
                if found:
                    return found
            return None
        if left == right:
            return None
    return f"{path}: {str(left)[:100]} vs {str(right)[:100]}"


def validate(response: object, case: dict, revision: str, phase: str) -> None:
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
    json.dumps(response, allow_nan=False)
    for name in ("wall_time_ms", "worker_elapsed_ms"):
        value = response.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid {name}")
    if not isinstance(response.get("model_identity"), dict) or not response["model_identity"]:
        raise ValueError("missing model identity")
    artifact = response.get("per_request_artifact")
    summary = behavior(response["behavior"], per_request=phase == "availability" and artifact is None)
    if (
        summary["completed_requests"] != case["expected_requests"]
        or summary["num_requests"] != case["expected_requests"]
    ):
        raise ValueError("incomplete request count")
    if summary["total_output_tokens"] != case["expected_output_tokens"]:
        raise ValueError("incomplete output token count")
    if phase == "availability":
        if artifact is not None:
            if (
                not isinstance(artifact, dict)
                or artifact.get("records") != case["expected_requests"]
                or artifact.get("complete") is not True
                or not isinstance(artifact.get("sha256"), str)
                or len(artifact["sha256"]) != 64
                or not artifact.get("path")
            ):
                raise ValueError("invalid per-request artifact evidence")
            return
        rows = summary["per_request"]
        if len(rows) != case["expected_requests"] or any(
            row["terminal_status"] != "completed" or row["output_length"] != row["requested_output_length"]
            for row in rows
        ):
            raise ValueError("incomplete per-request records")


def compare_case(case: dict, samples: dict, *, revisions: dict, rounds: int) -> dict:
    invalid, changed, timings = [], [], []
    valid_availability = set()
    availability = samples.get("availability", {})
    measured = samples.get("rounds", [])
    if len(measured) != rounds or [pair.get("round") for pair in measured] != list(range(1, rounds + 1)):
        invalid.append("missing, duplicate, or unordered measured rounds")
    for label, pair in [("availability", availability), *[(str(p.get("round")), p) for p in measured]]:
        phase = "availability" if label == "availability" else "measure"
        valid = True
        for side in ("base", "head"):
            try:
                validate(pair.get(side), case, revisions[side], phase)
                if phase == "availability":
                    valid_availability.add(side)
            except (ValueError, TypeError, KeyError) as error:
                invalid.append(f"{label}/{side}: {error}")
                valid = False
        if not valid:
            continue
        base, head = pair["base"], pair["head"]
        identity_diff = difference(base["model_identity"], head["model_identity"])
        if identity_diff:
            invalid.append(f"{label}: model identity changed: {identity_diff}")
        result_diff = difference(base["behavior"], head["behavior"])
        if result_diff:
            changed.append(f"{label}: {result_diff}")
        if phase == "availability" and (base.get("per_request_artifact") or head.get("per_request_artifact")):
            if "per_request_difference" not in pair:
                invalid.append("missing per-request artifact comparison")
            elif pair["per_request_difference"]:
                changed.append(str(pair["per_request_difference"]))
        if phase == "measure":
            for side in ("base", "head"):
                reference = availability.get(side, {})
                if side in valid_availability:
                    expected = {k: v for k, v in reference["behavior"].items() if k != "per_request"}
                    drift = difference(expected, pair[side]["behavior"])
                    if drift:
                        invalid.append(f"{label}/{side}: differs from its equivalence pass: {drift}")
                    if difference(reference["model_identity"], pair[side]["model_identity"]):
                        invalid.append(f"{label}/{side}: model identity differs from its equivalence pass")
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
    classification = (
        "INVALID_COMPARISON"
        if invalid
        else "BEHAVIOR_CHANGED"
        if changed
        else "PERFORMANCE_REGRESSION"
        if exceed >= required
        else "PASS"
    )
    return {
        "case_id": case["case_id"],
        "classification": classification,
        "invalid_reasons": invalid,
        "behavior_changes": changed,
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
        reasons = result["invalid_reasons"] or result["behavior_changes"]
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
    lines += ["", "Worker elapsed times, inputs, coverage evidence, and individual rounds are in the artifact.", ""]
    (output / "summary.md").write_text("\n".join(lines))
    (output / "annotations.txt").write_text(
        "".join(f"{r['classification']}: {r['case_id']}\n" for r in blockers)
        + "".join(f"QUALIFICATION_FAILED: {error}\n" for error in raw.get("qualification_errors", []))
    )
    return results
