# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Project retained silicon and fresh replay into safe, reproducible comparisons."""

import hashlib
import math
import re

from scripts.e2e_accuracy import agentx_curves as curves

SCHEMA = "aisimulate.agentx.comparison.v1"
SCORE_FIELDS = {
    "label",
    "metric",
    "mape_percent",
    "count",
    "coverage",
    "denominator_count",
    "zero_reference_count",
    "missing_count",
    "mae",
    "reason",
    "window_seconds",
}
SERIES_FIELDS = {"id", "label", "origin", "metric", "color", "x", "y", "kind", "step"}
REQUEST_FIELDS = {
    "uuid",
    "request_id",
    "phase",
    "arrival_time_ms",
    "terminal_time_ms",
    "first_token_ms",
    "input_length",
    "output_length",
    "ttft_ms",
    "itl_ms",
    "e2e_latency_ms",
    "terminal_status",
    "counterpart",
    "group_id",
    "origin",
}


def opaque(value):
    return hashlib.sha256(str(value).encode()).hexdigest()[:16] if value is not None else None


def build(reference, report, raw_report, scores, completed=True):
    """The reference adapter retains units and phase starts; it never fits clocks."""
    dataset = {
        "requests": raw_report["requests"],
        "samples": report["samples"],
        "reference": reference["reference"],
        "telemetry": {
            **report["telemetry"],
            "phase_boundaries": [dict(v, phase=k) for k, v in report["phases"].items()],
        },
    }
    # Qwen's preparation is identified by the measured cohort, not the native load mode.
    measured_phases = {r["request_id"]: r["phase"] for r in reference["reference"] if "key" in r}
    if measured_phases:
        dataset["requests"] = [
            dict(r, phase=measured_phases.get(r["request_id"], r["phase"])) for r in raw_report["requests"]
        ]
    native_profile = report["phases"].get("profile", {})
    if native_profile.get("admission_cutoff_ms") is not None:
        dataset["profile_duration_s"] = (native_profile["admission_cutoff_ms"] - native_profile["start_ms"]) / 1000
    comparison = curves.build_comparisons(dataset, reference["silicon"], reference["namespace_map"])
    result = {"schema": SCHEMA, "phases": {}}
    for phase, data in comparison["phases"].items():
        charts = []
        for identity, chart in data["charts"].items():
            projected = {
                k: chart[k]
                for k in ("title", "unit", "note", "foot", "series", "score_window_start_s", "score_window_end_s")
            }
            projected.update(id=identity, scores=[{k: s.get(k) for k in SCORE_FIELDS} for s in chart["scores"]])
            if phase != "profile" or not completed:
                for score in projected["scores"]:
                    score.update(
                        mape_percent=None,
                        count=0,
                        coverage=0,
                        reason="Preparation is unscored" if phase != "profile" else "Replay did not qualify",
                    )
            if identity.startswith("throughput_") and identity != "throughput_prefill":
                projected["unit"] = "output tokens/s" if identity != "throughput_both" else "tokens/s"
            charts.append(projected)
        timeline = data["timeline"]
        native_matches = timeline["native_matches"]
        native_groups = timeline["native_groups"]
        requests = []
        for row in timeline["reference_requests"]:
            requests.append(
                {
                    "uuid": opaque(row["uuid"]),
                    "request_id": opaque(row["request_id"]),
                    "phase": phase,
                    **{
                        k: row.get(k)
                        for k in (
                            "arrival_time_ms",
                            "terminal_time_ms",
                            "first_token_ms",
                            "input_length",
                            "output_length",
                            "ttft_ms",
                            "itl_ms",
                            "e2e_latency_ms",
                            "terminal_status",
                        )
                    },
                    "counterpart": opaque(row["matched_simulated_uuid"]),
                    "group_id": opaque(row["canonical_conversation"]),
                    "origin": "silicon",
                }
            )
        for row in dataset["requests"]:
            if curves.stage(row) != phase:
                continue
            requests.append(
                {
                    "uuid": opaque(row.get("uuid")),
                    "request_id": opaque(row.get("request_id")),
                    "phase": phase,
                    **{
                        k: row.get(k)
                        for k in (
                            "arrival_time_ms",
                            "terminal_time_ms",
                            "first_token_ms",
                            "input_length",
                            "output_length",
                            "ttft_ms",
                            "itl_ms",
                            "e2e_latency_ms",
                            "terminal_status",
                        )
                    },
                    "counterpart": opaque(native_matches.get(row.get("uuid"))),
                    "group_id": opaque(native_groups.get(row.get("uuid")) or curves.conversation(row)),
                    "origin": "simulated",
                }
            )
        request_scores = {}
        for metric in ("ttft_ms", "tpot_ms"):
            request_scores[metric] = (
                scores["request_errors"][metric]
                if phase == "profile" and completed
                else {"count": 0, "sum_ape": 0, "mape": None}
            )
        e2e = data["matched"]["e2e_latency_ms"]
        request_scores["e2e_ms"] = (
            {
                "count": e2e["count"],
                "sum_ape": sum(p["ape_percent"] for p in e2e["pairs"] if p["ape_percent"] is not None),
                "mape": e2e["mape_percent"],
            }
            if phase == "profile" and completed
            else {"count": 0, "sum_ape": 0, "mape": None}
        )
        result["phases"][phase] = {
            "charts": charts,
            "requests": requests,
            "request_scores": request_scores,
            "native_duration_s": data["alignment"]["native_duration_s"],
            "silicon_duration_s": data["alignment"]["silicon_duration_s"],
        }
    validate(result)
    return result


def validate(value):
    """Explicit nested allowlists: public comparisons contain numbers and opaque IDs."""

    def fields(row, expected):
        if not isinstance(row, dict) or set(row) != set(expected):
            raise ValueError("Invalid comparison fields")

    def number(value, nonnegative=True):
        return type(value) in (int, float) and math.isfinite(value) and (not nonnegative or value >= 0)

    def numeric(value, nonnegative=True):
        if value is not None and not number(value, nonnegative):
            raise ValueError("Invalid comparison number")

    def label(value):
        if value is not None and (
            not isinstance(value, str)
            or len(value) > 1500
            or re.search(r"https?://|/Users/|/home/|token_ids|Bearer |hf_[A-Za-z0-9]", value)
        ):
            raise ValueError("Unsafe comparison label")

    fields(value, {"schema", "phases"})
    if value["schema"] != SCHEMA or set(value["phases"]) != {"profile", "preparation"}:
        raise ValueError("Invalid comparison schema or phases")
    for phase, data in value["phases"].items():
        fields(data, {"charts", "requests", "request_scores", "native_duration_s", "silicon_duration_s"})
        for k in ("native_duration_s", "silicon_duration_s"):
            numeric(data[k])
        seen = set()
        for chart in data["charts"]:
            fields(
                chart,
                {
                    "id",
                    "title",
                    "unit",
                    "note",
                    "foot",
                    "series",
                    "scores",
                    "score_window_start_s",
                    "score_window_end_s",
                },
            )
            if chart["id"] in seen or not re.fullmatch(r"[a-z0-9_]+", chart["id"]):
                raise ValueError("Duplicate or invalid comparison chart")
            seen.add(chart["id"])
            for k in ("title", "unit", "note", "foot"):
                label(chart[k])
            for k in ("score_window_start_s", "score_window_end_s"):
                numeric(chart[k])
            for series in chart["series"]:
                fields(series, SERIES_FIELDS)
                if (
                    series["origin"] not in {"silicon", "simulated"}
                    or series["kind"] not in {"line", "bar"}
                    or type(series["step"]) is not bool
                    or len(series["x"]) != len(series["y"])
                ):
                    raise ValueError("Invalid comparison series")
                for k in ("id", "label", "metric", "color"):
                    label(series[k])
                for x in series["x"]:
                    if series["kind"] == "bar":
                        if x not in {
                            "completed",
                            "failed",
                            "canceled",
                            "cancelled",
                            "unknown",
                            "aborted",
                            "incomplete",
                        }:
                            raise ValueError("Invalid outcome category")
                    else:
                        numeric(x, False)
                for y in series["y"]:
                    numeric(y)
            for score in chart["scores"]:
                fields(score, SCORE_FIELDS)
                for k, item in score.items():
                    if k in {"label", "metric", "reason"}:
                        label(item)
                    elif k == "window_seconds":
                        if not isinstance(item, list) or len(item) != 2:
                            raise ValueError("Invalid comparison window")
                        for n in item:
                            numeric(n)
                    else:
                        numeric(item)
                if phase != "profile" and score["mape_percent"] is not None:
                    raise ValueError("Preparation cannot be scored")
                if score["reason"]:
                    if score["mape_percent"] is not None or score["count"] != 0:
                        raise ValueError("Unavailable curve cannot be scored")
                else:
                    selected = [s for s in chart["series"] if s["metric"] == score["metric"]]
                    actual = next((s for s in selected if s["origin"] == "silicon"), None)
                    predicted = next((s for s in selected if s["origin"] == "simulated"), None)
                    start, end = score["window_seconds"]
                    checked = curves.curve_score(
                        score["metric"], score["label"], actual, predicted, end, end, start=start
                    )
                    for key in (
                        "mape_percent",
                        "count",
                        "coverage",
                        "denominator_count",
                        "zero_reference_count",
                        "missing_count",
                    ):
                        left, right = checked[key], score[key]
                        if left != right and not (
                            number(left) and number(right) and math.isclose(left, right, rel_tol=1e-12)
                        ):
                            raise ValueError("Curve score does not reconcile")
        ids = set()
        for row in data["requests"]:
            fields(row, REQUEST_FIELDS)
            for k, item in row.items():
                if k in {"uuid", "request_id", "counterpart", "group_id"}:
                    if item is not None and not re.fullmatch(r"[0-9a-f]{16}", item):
                        raise ValueError("Comparison identities must be opaque")
                elif k in {"phase", "terminal_status", "origin"}:
                    label(item)
                else:
                    numeric(item)
            if row["uuid"] in ids or row["phase"] != phase or row["origin"] not in {"silicon", "simulated"}:
                raise ValueError("Invalid comparison request identity")
            ids.add(row["uuid"])
        by_id = {row["uuid"]: row for row in data["requests"]}
        for row in data["requests"]:
            if row["counterpart"] is not None:
                other = by_id.get(row["counterpart"])
                if other is None or other["counterpart"] != row["uuid"] or other["origin"] == row["origin"]:
                    raise ValueError("Invalid timeline counterpart")
        fields(data["request_scores"], {"ttft_ms", "tpot_ms", "e2e_ms"})
        for score in data["request_scores"].values():
            fields(score, {"count", "sum_ape", "mape"})
            for item in score.values():
                numeric(item)
            if type(score["count"]) is not int or (not score["count"] and score["sum_ape"] != 0):
                raise ValueError("Invalid request score denominator")
            if phase != "profile" and (score["count"] or score["sum_ape"] or score["mape"] is not None):
                raise ValueError("Preparation cannot be scored")
            expected = score["sum_ape"] / score["count"] if score["count"] else None
            if score["mape"] != expected and not (
                number(score["mape"]) and number(expected) and math.isclose(score["mape"], expected, rel_tol=1e-12)
            ):
                raise ValueError("Comparison request score does not reconcile")
