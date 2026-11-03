#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze private request cohorts and independently audit GLM campaign scores."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

POINTS = [(446063, 1), (446068, 4), (446069, 8), (446060, 10), (446061, 13), (446062, 14)]
METRICS = ("ttft", "tpot")


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Frozen evidence must never be silently replaced by another experiment.
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def valid(value):
    return type(value) in (int, float) and math.isfinite(value)


def metrics(rows):
    result = {}
    for metric in METRICS:
        pairs = [(r[f"silicon_{metric}_ms"], r[f"predicted_{metric}_ms"]) for r in rows]
        pairs = [(a, p) for a, p in pairs if valid(a) and a > 0 and valid(p) and p >= 0]
        signed = [100 * (p - a) / a for a, p in pairs]
        ape = [abs(v) for v in signed]
        tiny = [100 * abs(p - a) / a for a, p in pairs if a < 0.1]
        percentiles = statistics.quantiles(ape, n=100, method="inclusive") if len(ape) > 1 else None
        result[metric] = {
            "count": len(pairs),
            "sum_ape": sum(ape),
            "mape_percent": statistics.mean(ape) if ape else None,
            "median_ape_percent": statistics.median(ape) if ape else None,
            "p95_ape_percent": percentiles[94] if percentiles else ape[0] if ape else None,
            "p99_ape_percent": percentiles[98] if percentiles else ape[0] if ape else None,
            "max_ape_percent": max(ape) if ape else None,
            "signed_error_percent": statistics.mean(signed) if signed else None,
            "mae_ms": statistics.mean(abs(p - a) for a, p in pairs) if pairs else None,
            "reference_below_0_1_ms_count": len(tiny),
            "reference_below_0_1_ms_contribution_pp": sum(tiny) / len(pairs) if pairs else None,
        }
    return result


def source_key(row):
    return (row["source_trace_id"], row["source_outer_idx"], row["source_inner_idx"])


def fixed_rows(root, variant, point):
    baseline = root / "inputs" / f"point-{point}" / "historical-main-pairs.json"
    path = root / variant / f"point-{point}" / "report/paired-requests.json"
    frozen = [r for r in read(baseline) if r["strict_output_e2e_eligible"]]
    selected = [r for r in read(path) if r["strict_output_e2e_eligible"]]
    index = {source_key(r): r for r in selected}
    if len(index) != len(selected) or len({source_key(r) for r in frozen}) != len(frozen):
        raise ValueError("Duplicate fixed-cohort source identity")
    rows = []
    for actual in frozen:
        row = index.get(source_key(actual))
        if row is None:
            raise ValueError("Lost frozen request identity")
        if not actual["reference_output_tokens"] == row["simulated_output_tokens"] == row["reference_output_tokens"]:
            raise ValueError("Frozen output length mismatch")
        rows.append(
            {
                "key": list(source_key(actual)),
                **{
                    f"{side}_{metric}_ms": row[f"{field}_{metric}_ms"]
                    for metric in METRICS
                    for side, field in (("silicon", "reference"), ("predicted", "simulated"))
                },
            }
        )
    return rows, {"baseline_sha256": sha(baseline), "prediction_sha256": sha(path)}


def freeze(public, fixed, output):
    summary = read(public / "summary.json")
    groups, cohorts = [], {}
    for point, lanes in POINTS:
        identity = f"agentx-{point}"
        detail_path = public / "details" / f"{identity}.json"
        detail = read(detail_path)
        selected = [p for p in detail["pairs"] if p["status"] == "matched"]
        if len({p["request_id"] for p in selected}) != len(selected):
            raise ValueError("Duplicate dashboard pairing identity")
        actual = metrics(selected)
        for metric in METRICS:
            expected = detail["request_errors"][metric + "_ms"]
            if actual[metric]["count"] != expected["count"] or not math.isclose(
                actual[metric]["mape_percent"], expected["mape"], rel_tol=1e-10
            ):
                raise ValueError("Dashboard arithmetic does not reconcile")
        cohorts[identity] = {"dashboard": selected}
        variants = {}
        for variant in ("fresh-main", "fresh-plus-glm"):
            rows, hashes = fixed_rows(fixed, variant, point)
            cohorts[identity][variant] = rows
            variants[variant] = {"metrics": metrics(rows), **hashes}
        groups.append(
            {
                "id": identity,
                "lanes": lanes,
                "detail_sha256": sha(detail_path),
                "coverage": detail["coverage"],
                "dashboard": actual,
                "pr665": variants,
            }
        )
    pooled = {}
    for variant in ("dashboard", "fresh-main", "fresh-plus-glm"):
        pooled[variant] = metrics([row for group in cohorts.values() for row in group[variant]])
    private = {"schema": "aisimulate.glm-rsi.cohorts.v1", "snapshot": summary["snapshot"], "cohorts": cohorts}
    write(output / "frozen-cohorts.json", private)
    audit = {
        "schema": "aisimulate.glm-rsi.audit.v1",
        "status": "verified",
        "snapshot": summary["snapshot"],
        "cohorts_sha256": sha(output / "frozen-cohorts.json"),
        "points": groups,
        "pooled": pooled,
    }
    write(output / "baseline-audit.json", audit)
    return audit


def budget(jobs, limit=64 * 3600):
    """GPU-seconds: failed/startup time counts; reserve wall limits for active jobs."""
    used = 0
    identities = set()
    for row in jobs:
        if row["job_id"] in identities:
            raise ValueError("Duplicate GPU job")
        identities.add(row["job_id"])
        gpus, seconds = row["reserved_gpus"], row["charged_seconds"]
        if type(gpus) is not int or gpus <= 0 or type(seconds) is not int or seconds < 0:
            raise ValueError("Invalid GPU allocation accounting")
        used += gpus * seconds
    if used > limit:
        raise ValueError("GPU campaign budget exceeded")
    return {"limit_gpu_seconds": limit, "charged_gpu_seconds": used, "remaining_gpu_seconds": limit - used}


def compare_dashboard(frozen, candidate):
    """Missing frozen metric observations invalidate wins, even if available MAPE falls."""
    result = {}
    for metric in METRICS:
        field = metric + "_ms_ape"
        baseline = [row for row in frozen if row[field] is not None]
        index = {row["request_id"]: row for row in candidate}
        if len(index) != len(candidate):
            raise ValueError("Duplicate candidate pairing identity")
        selected, missing = [], []
        for row in baseline:
            other = index.get(row["request_id"])
            if other is None or other["status"] != "matched" or other[field] is None:
                missing.append(row["request_id"])
                continue
            if other[f"silicon_{metric}_ms"] != row[f"silicon_{metric}_ms"]:
                raise ValueError("Silicon changed inside a frozen comparison")
            expected = (
                100
                * abs(other[f"predicted_{metric}_ms"] - other[f"silicon_{metric}_ms"])
                / other[f"silicon_{metric}_ms"]
            )
            if not math.isclose(expected, other[field], rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError("Candidate APE does not reconcile")
            selected.append(other)
        result[metric] = {
            "frozen_count": len(baseline),
            "retained_count": len(selected),
            "lost_count": len(missing),
            "baseline_sum_ape": sum(row[field] for row in baseline),
            "candidate_sum_ape": sum(row[field] for row in selected),
            "baseline_mape_percent": statistics.mean(row[field] for row in baseline) if baseline else None,
            "candidate_mape_percent": statistics.mean(row[field] for row in selected) if selected else None,
            "coverage_qualified": not missing,
        }
    return result


def compare_campaign(frozen_path, baseline_root, candidate_root, lanes):
    """Audit the original metric-specific identities, including recovered matches separately."""
    frozen = read(frozen_path)["cohorts"]
    points, baseline_rows, candidate_rows = [], [], []
    for point, lane in POINTS:
        if lane not in lanes:
            continue
        identity = f"agentx-{point}"
        original = frozen[identity]["dashboard"]
        baseline_path = baseline_root / identity / "score.json"
        candidate_path = candidate_root / identity / "score.json"
        baseline = read(baseline_path)["pairs"]
        candidate = read(candidate_path)["pairs"]
        qualified = compare_dashboard(original, baseline)
        if any(not row["coverage_qualified"] for row in qualified.values()):
            raise ValueError("Research baseline lost frozen observations")
        index = {row["request_id"]: row for row in baseline}
        candidate_index = {row["request_id"]: row for row in candidate}
        selected, diagnostics = [], []
        for source in original:
            row = dict(index.get(source["request_id"], source))
            predicted = dict(candidate_index.get(source["request_id"], {}))
            for metric in METRICS:
                if source[metric + "_ms_ape"] is None:
                    row[metric + "_ms_ape"] = None
                    row["predicted_" + metric + "_ms"] = None
                    predicted["predicted_" + metric + "_ms"] = None
                if predicted.get("status") != "matched":
                    predicted["predicted_" + metric + "_ms"] = None
                if "silicon_" + metric + "_ms" not in predicted:
                    predicted["silicon_" + metric + "_ms"] = source["silicon_" + metric + "_ms"]
                    predicted["predicted_" + metric + "_ms"] = None
            selected.append(row)
            diagnostics.append(predicted)
        comparison = compare_dashboard(selected, candidate)
        recovered = {}
        for metric in METRICS:
            field = metric + "_ms_ape"
            original_ids = {row["request_id"] for row in original if row[field] is not None}
            recovered[metric] = {
                label: sum(
                    row["status"] == "matched" and row[field] is not None and row["request_id"] not in original_ids
                    for row in rows
                )
                for label, rows in (("baseline", baseline), ("candidate", candidate))
            }
        points.append(
            {
                "point": identity,
                "lanes": lane,
                "comparison": comparison,
                "recovered_outside_frozen": recovered,
                "score_checksums": {"baseline": sha(baseline_path), "candidate": sha(candidate_path)},
            }
        )
        baseline_rows.extend(selected)
        candidate_rows.extend(diagnostics)
    if len(points) != len(set(lanes)) or len(lanes) != len(set(lanes)):
        raise ValueError("Unknown or duplicate campaign lanes")
    pooled = {}
    for metric in METRICS:
        rows = [point["comparison"][metric] for point in points]
        counts = sum(row["frozen_count"] for row in rows)
        retained = sum(row["retained_count"] for row in rows)
        pooled[metric] = {
            "frozen_count": counts,
            "retained_count": retained,
            "lost_count": counts - retained,
            "baseline_mape_percent": sum(row["baseline_sum_ape"] for row in rows) / counts if counts else None,
            "candidate_mape_percent": sum(row["candidate_sum_ape"] for row in rows) / retained if retained else None,
            "coverage_qualified": counts == retained,
        }
    return {
        "schema": "aisimulate.glm-rsi.frozen-comparison.v1",
        "points": points,
        "pooled": pooled,
        "frozen_sha256": sha(frozen_path),
        "diagnostics": {"baseline": metrics(baseline_rows), "candidate": metrics(candidate_rows)},
        "qualification": "A smaller scored cohort cannot establish a win; recovered matches are separate.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-bundle", type=Path)
    parser.add_argument("--pr665-evidence", type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--lanes", type=int, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.candidate:
        if not (args.frozen and args.baseline and args.lanes):
            parser.error("--candidate requires --frozen, --baseline and --lanes")
        result = compare_campaign(args.frozen, args.baseline, args.candidate, args.lanes)
        write(args.output, result)
    else:
        if not (args.public_bundle and args.pr665_evidence):
            parser.error("freezing requires --public-bundle and --pr665-evidence")
        result = freeze(args.public_bundle, args.pr665_evidence, args.output)
    print(json.dumps(result["pooled"], indent=2))


if __name__ == "__main__":
    main()
