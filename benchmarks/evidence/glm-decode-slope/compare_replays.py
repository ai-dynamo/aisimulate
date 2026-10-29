# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare native replay variants on an explicit, retained baseline cohort.

Reuse the AgentX scorer's identity, eligibility, throughput, and native-summary
checks. Changed coverage is reported separately from common-cohort accuracy.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
from pathlib import Path

SCORER_PATH = (
    Path(__file__).resolve().parents[1] / "agentx-446063-replay" / "score_replay.py"
)
SCORER_SPEC = importlib.util.spec_from_file_location("agentx_score_replay", SCORER_PATH)
assert SCORER_SPEC is not None and SCORER_SPEC.loader is not None
SCORER = importlib.util.module_from_spec(SCORER_SPEC)
SCORER_SPEC.loader.exec_module(SCORER)

POINTS = {446063: 1, 446068: 4, 446069: 8, 446060: 10, 446061: 13, 446062: 14}
INPUT_BINS = [
    (0, 64000),
    (64000, 128000),
    (128000, 256000),
    (256000, 384000),
    (384000, None),
]


def load_json(path: Path):
    return json.loads(path.read_text())


def identity(row: dict) -> tuple:
    return row["source_trace_id"], row["source_outer_idx"], row["source_inner_idx"]


def metrics(rows: list[dict]) -> dict:
    result = {}
    for name in ("e2e", "ttft", "tpot"):
        pairs = [
            (row[f"reference_{name}_ms"], row[f"simulated_{name}_ms"])
            for row in rows
            if SCORER.ape(
                row.get(f"reference_{name}_ms"), row.get(f"simulated_{name}_ms")
            )
            is not None
        ]
        reference = SCORER.mean([left for left, _ in pairs])
        simulated = SCORER.mean([right for _, right in pairs])
        result[name] = {
            "count": len(pairs),
            "reference_mean_ms": reference,
            "simulated_mean_ms": simulated,
            "mape_percent": SCORER.mean(
                [SCORER.ape(left, right) for left, right in pairs]
            ),
            "signed_mean_percent_error": SCORER.mean(
                [100 * (right / left - 1) for left, right in pairs]
            ),
            "ratio_of_means": simulated / reference
            if reference is not None and reference > 0
            else None,
            "median_request_ratio": statistics.median(
                [right / left for left, right in pairs]
            )
            if pairs
            else None,
        }
    return result


def compare_sets(baseline: list[dict], candidate: list[dict]) -> dict:
    before = {
        identity(row): row for row in baseline if row["strict_output_e2e_eligible"]
    }
    after = {
        identity(row): row for row in candidate if row["strict_output_e2e_eligible"]
    }
    if len(before) != sum(row["strict_output_e2e_eligible"] for row in baseline):
        raise ValueError("Baseline strict cohort contains duplicate source identities")
    if len(after) != sum(row["strict_output_e2e_eligible"] for row in candidate):
        raise ValueError("Candidate strict cohort contains duplicate source identities")
    common = [key for key in before if key in after]
    for key in common:
        for field in (
            "reference_request_id",
            "reference_input_tokens",
            "reference_output_tokens",
        ):
            if before[key][field] != after[key][field]:
                raise ValueError(f"Reference changed for {key}: {field}")
        for name in ("e2e", "ttft", "tpot"):
            if (
                before[key][f"reference_{name}_ms"]
                != after[key][f"reference_{name}_ms"]
            ):
                raise ValueError(f"Reference latency changed for {key}: {name}")
    candidate_reference_rows = {
        identity(row): row
        for row in candidate
        if row["phase"] == "profile" and row["reference_request_id"]
    }
    lost = []
    for key in before:
        if key in after:
            continue
        row = candidate_reference_rows.get(key)
        reasons = (
            list(row["e2e_score_exclusions"])
            if row is not None
            else ["missing_reference_record"]
        )
        if row is not None and row["output_length_matches"] is not True:
            reasons.append("output_length_mismatch_or_unavailable")
        lost.append({"source_identity": list(key), "reasons": reasons})
    bins = []
    for low, high in INPUT_BINS:
        keys = [
            key
            for key in common
            if before[key]["reference_input_tokens"] >= low
            and (high is None or before[key]["reference_input_tokens"] < high)
        ]
        bins.append(
            {
                "reference_input_tokens_min_inclusive": low,
                "reference_input_tokens_max_exclusive": high,
                "count": len(keys),
                "baseline": metrics([before[key] for key in keys]),
                "candidate": metrics([after[key] for key in keys]),
            }
        )
    return {
        "baseline_strict_count": len(before),
        "candidate_strict_count": len(after),
        "retained_baseline_count": len(common),
        "retained_baseline_percent": SCORER.percentage(len(common), len(before)),
        "lost_baseline_requests": lost,
        "gained_strict_source_identities": [
            list(key) for key in after if key not in before
        ],
        "baseline_original_cohort": metrics(list(before.values())),
        "candidate_realized_cohort": metrics(list(after.values())),
        "common_cohort": {
            "baseline": metrics([before[key] for key in common]),
            "candidate": metrics([after[key] for key in common]),
            "changed_simulated_input_lengths": sum(
                before[key]["simulated_input_tokens"]
                != after[key]["simulated_input_tokens"]
                for key in common
            ),
        },
        "common_cohort_input_bins": bins,
    }


def compare_point(
    baseline_root: Path, candidate_root: Path, point_id: int, run_name: str
) -> dict:
    baseline_dir = baseline_root / f"point-{point_id}"
    candidate_dir = candidate_root / f"point-{point_id}"
    run_dir = candidate_dir / run_name
    baseline_pairs = load_json(
        baseline_dir / "observed-report" / "paired-requests.json"
    )
    baseline_measurement = load_json(
        baseline_dir / "observed-report" / "measurement.json"
    )
    reference = load_json(baseline_dir / "source" / "reference-requests.json")
    namespaces = load_json(baseline_dir / "source" / "source_namespace_map.json")
    native = [
        json.loads(line)
        for line in (run_dir / "telemetry-requests.jsonl").read_text().splitlines()
    ]
    execution = load_json(candidate_dir / f"{run_name}-execution.json")
    accuracy, paired, measurement = SCORER.analyze(
        load_json(run_dir / "prediction.json"),
        reference,
        native,
        namespaces,
        execution,
        "decode_slope_candidate",
        point_id,
    )
    return {
        "point_id": point_id,
        "concurrency": POINTS[point_id],
        "baseline_run_dir": str(baseline_dir / "observed-replay"),
        "candidate_run_dir": str(run_dir),
        "baseline_execution": baseline_measurement["execution"],
        "candidate_execution": execution,
        "baseline_coverage": baseline_measurement["coverage"],
        "candidate_coverage": measurement["coverage"],
        "cohort_comparison": compare_sets(baseline_pairs, paired),
        "baseline_full_profile_comparisons": baseline_measurement["comparisons"][2:],
        "candidate_full_profile_comparisons": measurement["comparisons"][2:],
        "candidate_native_summary_checks": accuracy["native_summary_checks"],
    }


def self_check() -> None:
    def row(index: int, simulated: float, eligible: bool = True) -> dict:
        return {
            "source_trace_id": "fixture",
            "source_outer_idx": index,
            "source_inner_idx": None,
            "phase": "profile",
            "reference_request_id": f"ref-{index}",
            "reference_input_tokens": 300000,
            "simulated_input_tokens": 299968,
            "reference_output_tokens": 11,
            "reference_e2e_ms": 100,
            "simulated_e2e_ms": simulated,
            "reference_ttft_ms": 20,
            "simulated_ttft_ms": 10,
            "reference_tpot_ms": 8,
            "simulated_tpot_ms": (simulated - 10) / 10,
            "strict_output_e2e_eligible": eligible,
            "output_length_matches": eligible,
            "e2e_score_exclusions": [] if eligible else ["simulated_cancelled"],
        }

    result = compare_sets(
        [row(1, 120), row(2, 200)], [row(1, 110), row(2, 0, False), row(3, 300)]
    )
    assert result["baseline_original_cohort"]["e2e"]["mape_percent"] == 60
    assert result["common_cohort"]["baseline"]["e2e"]["mape_percent"] == 20
    assert result["common_cohort"]["candidate"]["e2e"]["mape_percent"] == 10
    assert (
        result["retained_baseline_count"] == 1
        and result["retained_baseline_percent"] == 50
    )
    assert (
        len(result["lost_baseline_requests"])
        == len(result["gained_strict_source_identities"])
        == 1
    )
    assert result["common_cohort_input_bins"][3]["count"] == 1
    assert metrics([])["e2e"]["mape_percent"] is None
    print("Fixed-cohort, lost/gained coverage, bin, and empty-subset checks passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--candidate-run-name", default="observed-replay")
    parser.add_argument("--point-id", type=int, choices=list(POINTS), action="append")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not all((args.baseline_root, args.candidate_root, args.output_dir)):
        parser.error("--baseline-root, --candidate-root and --output-dir are required")
    points = args.point_id or list(POINTS)
    if len(points) != len(set(points)):
        parser.error("--point-id must not repeat")
    results = [
        compare_point(
            args.baseline_root, args.candidate_root, point, args.candidate_run_name
        )
        for point in points
    ]
    output = {
        "schema": "aisimulate.glm_decode_slope_comparison.v1",
        "methodology": [
            "The baseline strict cohort is fixed by unique source identity and equal recorded output length.",
            "Common-cohort metrics include baseline requests still strictly eligible in the candidate.",
            "Lost baseline and gained candidate requests are explicit; missing latency is never zero-filled.",
            "Full-profile throughput compares realized cohorts and is reported separately from fixed-cohort latency.",
            "Input bins use decimal token boundaries. TPOT requires positive reference and available simulated values.",
        ],
        "points": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )
    csv_rows = []
    for point in results:
        comparison = point["cohort_comparison"]
        row = {"point_id": point["point_id"], "concurrency": point["concurrency"]}
        row.update(
            {
                name: comparison[name]
                for name in (
                    "baseline_strict_count",
                    "retained_baseline_count",
                    "candidate_strict_count",
                )
            }
        )
        for variant in ("baseline", "candidate"):
            for metric in ("e2e", "ttft", "tpot"):
                value = comparison["common_cohort"][variant][metric]
                row[f"{variant}_{metric}_mape_percent"] = value["mape_percent"]
                row[f"{variant}_{metric}_mean_ms"] = value["simulated_mean_ms"]
        csv_rows.append(row)
    with (args.output_dir / "comparison.csv").open("w", newline="") as output_csv:
        writer = csv.DictWriter(output_csv, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Compared {len(results)} points; artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
