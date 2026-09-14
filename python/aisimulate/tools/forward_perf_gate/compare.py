# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paired base-versus-head comparison and report rendering."""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from tools.forward_perf_gate import PROTOCOL_VERSION


@dataclass(frozen=True)
class Threshold:
    relative: float
    absolute_us: float


THRESHOLDS = {
    "cold": Threshold(relative=0.10, absolute_us=2.0),
    "warm": Threshold(relative=0.10, absolute_us=2.0),
}
CONSENSUS_FRACTION = 0.8


def _error_text(response: dict) -> str:
    error = response.get("error")
    if not isinstance(error, dict):
        return ""
    return f"{error.get('type', '')} {error.get('message', '')}".strip()


def _response_error(case_id: str, side: str, response: object) -> str | None:
    if not isinstance(response, dict):
        return f"{side} response is not an object"
    if response.get("protocol_version") != PROTOCOL_VERSION:
        return f"{side} protocol is {response.get('protocol_version')!r}, expected {PROTOCOL_VERSION}"
    if response.get("case_id") != case_id:
        return f"{side} returned case_id {response.get('case_id')!r}"
    if not isinstance(response.get("case_hash"), str) or not response["case_hash"]:
        return f"{side} response is missing a case hash"
    return None


def pair_disposition(case_id: str, base: object, head: object) -> tuple[str, str | None]:
    """Return COMPARE, SKIP, or INVALID for one base/head response pair."""
    for side, response in (("base", base), ("head", head)):
        reason = _response_error(case_id, side, response)
        if reason:
            return "INVALID", reason

    assert isinstance(base, dict)
    assert isinstance(head, dict)
    if base.get("case_hash") != head.get("case_hash"):
        return "INVALID", "base and head case hashes differ"

    base_status, head_status = base.get("status"), head.get("status")
    if base_status == "OK" and head_status == "OK":
        return "COMPARE", None
    if base_status == "DATA_MISS" and head_status == "DATA_MISS":
        details = list(dict.fromkeys(filter(None, (_error_text(base), _error_text(head)))))
        suffix = f": {'; '.join(details)}" if details else ""
        return "SKIP", f"DATA_MISS on base and head{suffix}"
    if base_status == "DATA_MISS" and head_status == "OK":
        detail = _error_text(base)
        suffix = f": {detail}" if detail else ""
        return "SKIP", f"base DATA_MISS{suffix}; head has no timing baseline"

    details = []
    for side, response in (("base", base), ("head", head)):
        if response.get("status") != "OK":
            suffix = _error_text(response)
            details.append(f"{side} status is {response.get('status')}" + (f": {suffix}" if suffix else ""))
    return "INVALID", "; ".join(details) or "response statuses are not comparable"


def _metric_value(response: dict, metric: str) -> tuple[float | None, str | None]:
    try:
        raw_value = response["cold_us"] if metric == "cold" else response["warm"]["call_median_us"]
    except (KeyError, TypeError):
        return None, f"missing or invalid {metric} metric"
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        return None, f"missing or invalid {metric} metric"
    value = float(raw_value)
    if not math.isfinite(value) or value <= 0:
        return None, f"{metric} metric must be finite and positive"
    return value, None


def prewarm_disposition(case_id: str, base: object, head: object) -> tuple[str, str | None]:
    """Validate a prewarm pair without comparing its one-sample timing."""
    disposition, reason = pair_disposition(case_id, base, head)
    if disposition != "COMPARE":
        return disposition, reason

    assert isinstance(base, dict)
    assert isinstance(head, dict)
    for metric in ("cold", "warm"):
        for side, response in (("base", base), ("head", head)):
            _, metric_error = _metric_value(response, metric)
            if metric_error:
                return "INVALID", f"{side} {metric_error}"
    return "COMPARE", None


def _point_identity(case: dict, metric: str) -> dict:
    threshold = THRESHOLDS[metric]
    return {
        "case_id": case["case_id"],
        "model_id": case["model_id"],
        "database_mode": case["database_mode"],
        "phase": case["phase"],
        "metric": metric,
        "relative_threshold": threshold.relative,
        "absolute_threshold_us": threshold.absolute_us,
    }


def compare_point(
    case: dict,
    rounds: list[dict],
    metric: str,
    *,
    skip_reason: str | None = None,
    availability_succeeded: bool = False,
) -> dict:
    threshold = THRESHOLDS[metric]
    result = {**_point_identity(case, metric), "rounds": []}
    if skip_reason is not None:
        result.update(
            {
                "classification": "SKIPPED",
                "skip_reason": skip_reason,
                "exceed_count": 0,
                "consensus_required": 0,
            }
        )
        return result

    round_results: list[dict] = []
    invalid_reasons: list[str] = []
    skipped_reasons: list[str] = []
    for paired in rounds:
        base, head = paired.get("base"), paired.get("head")
        disposition, reason = pair_disposition(case["case_id"], base, head)
        if disposition == "SKIP" and availability_succeeded:
            disposition = "INVALID"
            reason = "; ".join(
                f"{side} status is DATA_MISS after successful availability: {_error_text(response)}"
                for side, response in (("base", base), ("head", head))
                if response["status"] == "DATA_MISS"
            )
        if disposition == "INVALID":
            invalid_reasons.append(f"round {paired.get('round')}: {reason}")
            continue
        if disposition == "SKIP":
            skipped_reasons.append(reason or "no timing baseline")
            continue

        assert isinstance(base, dict)
        assert isinstance(head, dict)
        base_us, base_error = _metric_value(base, metric)
        head_us, head_error = _metric_value(head, metric)
        if base_error or head_error:
            errors = [
                f"{side} {error}" for side, error in (("base", base_error), ("head", head_error)) if error is not None
            ]
            invalid_reasons.append(f"round {paired.get('round')}: {'; '.join(errors)}")
            continue
        assert base_us is not None
        assert head_us is not None
        ratio = head_us / base_us
        delta_us = head_us - base_us
        exceeds = ratio > 1.0 + threshold.relative and delta_us > threshold.absolute_us
        round_results.append(
            {
                "round": paired["round"],
                "base_us": base_us,
                "head_us": head_us,
                "delta_us": delta_us,
                "ratio": ratio,
                "exceeds": exceeds,
            }
        )

    result["rounds"] = round_results
    if skipped_reasons and not invalid_reasons and not round_results and len(skipped_reasons) == len(rounds):
        unique_reasons = list(dict.fromkeys(skipped_reasons))
        if len(unique_reasons) > 1:
            invalid_reasons.append("response status changed between measured rounds")
        else:
            result.update(
                {
                    "classification": "SKIPPED",
                    "skip_reason": unique_reasons[0],
                    "exceed_count": 0,
                    "consensus_required": 0,
                }
            )
            return result

    elif skipped_reasons:
        invalid_reasons.append("response status changed between measured rounds")
    if not rounds:
        invalid_reasons.append("no paired rounds")
    if invalid_reasons or len(round_results) != len(rounds):
        result.update(
            {
                "classification": "INVALID_COMPARISON",
                "invalid_reasons": invalid_reasons or ["missing paired rounds"],
                "exceed_count": 0,
                "consensus_required": 0,
            }
        )
        return result

    consensus_required = math.ceil(CONSENSUS_FRACTION * len(round_results))
    exceed_count = sum(round_result["exceeds"] for round_result in round_results)
    base_median = statistics.median(round_result["base_us"] for round_result in round_results)
    head_median = statistics.median(round_result["head_us"] for round_result in round_results)
    median_ratio = head_median / base_median
    if exceed_count >= consensus_required:
        classification = "REGRESSION"
    elif exceed_count:
        classification = "UNSTABLE"
    elif median_ratio < 1.0:
        classification = "IMPROVEMENT"
    else:
        classification = "OK"
    result.update(
        {
            "classification": classification,
            "exceed_count": exceed_count,
            "consensus_required": consensus_required,
            "base_median_us": base_median,
            "head_median_us": head_median,
            "median_delta_us": head_median - base_median,
            "median_ratio": median_ratio,
        }
    )
    return result


def _cell_summaries(points: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], list[dict]] = {}
    for point in points:
        key = (point["model_id"], point["database_mode"], point["phase"], point["metric"])
        grouped.setdefault(key, []).append(point)

    cells = []
    for key, cell_points in sorted(grouped.items()):
        valid = [point for point in cell_points if "median_ratio" in point]
        ratios = [point["median_ratio"] for point in valid if point["median_ratio"] > 0]
        worst = max(valid, key=lambda point: point["median_ratio"], default=None)
        cells.append(
            {
                "model_id": key[0],
                "database_mode": key[1],
                "phase": key[2],
                "metric": key[3],
                "geomean_ratio": math.exp(statistics.fmean(math.log(ratio) for ratio in ratios)) if ratios else None,
                "worst_ratio": worst["median_ratio"] if worst else None,
                "worst_case_id": worst["case_id"] if worst else None,
                "regressions": sum(point["classification"] == "REGRESSION" for point in cell_points),
                "unstable": sum(point["classification"] == "UNSTABLE" for point in cell_points),
                "invalid": sum(point["classification"] == "INVALID_COMPARISON" for point in cell_points),
                "skipped": sum(point["classification"] == "SKIPPED" for point in cell_points),
            }
        )
    return sorted(
        cells,
        key=lambda cell: (
            0 if cell["regressions"] else 1 if cell["invalid"] else 2 if cell["unstable"] else 3,
            cell["model_id"],
            cell["database_mode"],
            cell["phase"],
            cell["metric"],
        ),
    )


def _validate_case_set(raw: dict) -> list[str]:
    entries = raw.get("cases")
    if not isinstance(entries, list) or not entries:
        return ["no benchmark cases were recorded"]

    case_ids = []
    errors = []
    for entry in entries:
        case = entry.get("case") if isinstance(entry, dict) else None
        case_id = case.get("case_id") if isinstance(case, dict) else None
        if not isinstance(case_id, str) or not case_id:
            errors.append("recorded case is missing a case_id")
        else:
            case_ids.append(case_id)
    if len(case_ids) != len(set(case_ids)):
        errors.append("recorded case IDs are not unique")

    expected = raw.get("configuration", {}).get("expected_case_ids")
    if isinstance(expected, list):
        missing = sorted(set(expected) - set(case_ids))
        unexpected = sorted(set(case_ids) - set(expected))
        if missing:
            errors.append(f"missing expected cases: {', '.join(missing)}")
        if unexpected:
            errors.append(f"unexpected cases: {', '.join(unexpected)}")
    return errors


def compare_raw(raw: dict) -> dict:
    run_errors = [str(error) for error in raw.get("run_errors", [])]
    run_errors.extend(_validate_case_set(raw))
    available_cases = {entry["case_id"] for entry in raw.get("prewarm", []) if entry.get("disposition") == "COMPARE"}
    points = []
    for entry in raw.get("cases", []):
        if not isinstance(entry, dict) or not isinstance(entry.get("case"), dict):
            continue
        for metric in ("cold", "warm"):
            points.append(
                compare_point(
                    entry["case"],
                    entry.get("rounds", []),
                    metric,
                    skip_reason=entry.get("skip_reason"),
                    availability_succeeded=entry["case"]["case_id"] in available_cases,
                )
            )
    blocking = bool(run_errors) or any(
        point["classification"] in {"REGRESSION", "INVALID_COMPARISON"} for point in points
    )
    configuration = raw.get("configuration", {})
    return {
        "schema_version": 1,
        "base_revision": raw.get("base_revision", ""),
        "head_revision": raw.get("head_revision", ""),
        "mode": configuration.get("mode", "full"),
        "matrix_case_count": configuration.get("matrix_case_count"),
        "selected_case_count": configuration.get("selected_case_count"),
        "run_errors": run_errors,
        "points": points,
        "cells": _cell_summaries(points),
        "blocking": blocking,
    }


def _percent(ratio: float | None) -> str:
    return "n/a" if ratio is None else f"{ratio - 1.0:+.1%}"


def _cell_status(cell: dict) -> str:
    if cell["regressions"]:
        return "❌ regression"
    if cell["invalid"]:
        return "❌ invalid"
    if cell["unstable"]:
        return "⚠️ noisy"
    if cell["skipped"]:
        return "skipped"
    return "✅ stable"


def render_markdown(comparison: dict) -> str:
    points = comparison["points"]
    regressions = [point for point in points if point["classification"] == "REGRESSION"]
    improvements = [point for point in points if point["classification"] == "IMPROVEMENT"]
    unchanged = [point for point in points if point["classification"] == "OK"]
    invalid = [point for point in points if point["classification"] == "INVALID_COMPARISON"]
    unstable = [point for point in points if point["classification"] == "UNSTABLE"]
    skipped = [point for point in points if point["classification"] == "SKIPPED"]
    stable_count = len(regressions) + len(improvements) + len(unchanged)
    result = "FAIL" if comparison["blocking"] else "PASS"
    mode_label = " — smoke" if comparison.get("mode") == "smoke" else ""

    def add_timing_table(title: str, selected: list[dict], *, collapsed: bool = False) -> None:
        if not selected:
            return
        if collapsed:
            lines.extend(["", "<details>", f"<summary>{title} ({len(selected)})</summary>", ""])
        else:
            lines.extend(["", f"### {title}", ""])
        lines.extend(
            [
                "| case | cache | base | head | change | rounds |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for point in selected:
            lines.append(
                f"| `{point['case_id']}` | {point['metric']} | {point['base_median_us']:.2f} µs "
                f"| {point['head_median_us']:.2f} µs | {_percent(point['median_ratio'])} "
                f"({point['median_delta_us']:+.2f} µs) "
                f"| {point['exceed_count']}/{len(point['rounds'])} |"
            )
        if collapsed:
            lines.extend(["", "</details>"])

    def add_reason_table(title: str, selected: list[dict], *, collapsed: bool = False) -> None:
        if not selected:
            return
        if collapsed:
            lines.extend(["", "<details>", f"<summary>{title} ({len(selected)})</summary>", ""])
        else:
            lines.extend(["", f"### {title}", ""])
        lines.extend(["| case | cache | reason |", "|---|---|---|"])
        for point in selected:
            reasons = point.get("invalid_reasons") or [point.get("skip_reason", "")]
            reason = "; ".join(reasons).replace("|", "\\|")
            lines.append(f"| `{point['case_id']}` | {point['metric']} | {reason} |")
        if collapsed:
            lines.extend(["", "</details>"])

    def add_full_matrix(cells: list[dict]) -> None:
        if not cells:
            return
        lines.extend(
            [
                "",
                "<details>",
                f"<summary>Full matrix ({len(cells)} cells)</summary>",
                "",
                "| status | model | database | phase | cache | geometric mean | worst point "
                "| regressions | noisy | invalid | skipped |",
                "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for cell in cells:
            lines.append(
                f"| {_cell_status(cell)} | {cell['model_id']} | {cell['database_mode']} | "
                f"{cell['phase']} | {cell['metric']} "
                f"| {_percent(cell['geomean_ratio'])} | {_percent(cell['worst_ratio'])} "
                f"| {cell['regressions']} | {cell['unstable']} | {cell['invalid']} | {cell['skipped']} |"
            )
        lines.extend(["", "</details>"])

    lines = [
        f"## Forward Prediction Performance (advisory{mode_label})",
        "",
        f"**{result}** — {stable_count} of {len(points)} comparisons stable: "
        f"{len(improvements)} faster, {len(unchanged)} unchanged, "
        f"{len(regressions)} regressions; {len(unstable)} noisy, "
        f"{len(invalid)} invalid, {len(skipped)} skipped.",
        "",
        f"Base `{comparison['base_revision']}` vs head `{comparison['head_revision']}`.",
    ]
    if comparison.get("mode") == "smoke":
        lines.extend(
            [
                "",
                f"Smoke mode measured {comparison.get('selected_case_count', 0)} of "
                f"{comparison.get('matrix_case_count', 0)} matrix cases.",
            ]
        )
    if comparison["run_errors"]:
        lines.extend(["", "### ❌ Run errors", ""])
        lines.extend(f"- {error}" for error in comparison["run_errors"])

    add_timing_table("❌ Confirmed regressions", regressions)
    add_reason_table("❌ Invalid comparisons", invalid)
    add_timing_table("⚠️ Noisy comparisons", unstable, collapsed=True)
    add_reason_table("Skipped comparisons", skipped, collapsed=True)
    add_full_matrix(comparison["cells"])
    lines.extend(["", "This check is advisory and is not required by branch protection."])
    return "\n".join(lines)


def write_outputs(comparison: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    (output_dir / "summary.md").write_text(render_markdown(comparison) + "\n")
    with (output_dir / "comparison.csv").open("w", newline="") as stream:
        fields = [
            "case_id",
            "model_id",
            "database_mode",
            "phase",
            "metric",
            "classification",
            "skip_reason",
            "exceed_count",
            "consensus_required",
            "base_median_us",
            "head_median_us",
            "median_delta_us",
            "median_ratio",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparison["points"])
    annotations = [f"INVALID_RUN: {error}" for error in comparison["run_errors"]]
    for point in comparison["points"]:
        if point["classification"] in {"REGRESSION", "INVALID_COMPARISON"}:
            annotations.append(f"{point['classification']}: {point['case_id']} ({point['metric']})")
    (output_dir / "annotations.txt").write_text("\n".join(annotations) + ("\n" if annotations else ""))
