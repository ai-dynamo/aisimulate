# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export the fixed hash-closed cold E2E subset with full original denominators."""

import argparse
import csv
from pathlib import Path
import statistics

from export import read, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[2] / "gb300-silicon/report/sol-review-v2"
    pins, rows, summaries = {}, [], []
    identity = read(args.input_root / "silicon-cold-e2e-v1/started.json")
    for scope in ("off", "on", "field-off", "field-on", "service-off", "service-on"):
        fresh = scope in ("off", "on")
        path = (
            (args.input_root / "silicon-cold-e2e-v1" / f"{scope}.json.gz")
            if fresh
            else (base / scope / "e2e-silicon-results.json.gz")
        )
        source = read(path)
        if fresh and any(source[key] != value for key, value in identity.items()):
            raise ValueError("mixed current cold prediction identity")
        pins[scope] = {"source_sha256": sha(path), "fresh_predictions": fresh}
        group = []
        for ordinal, case in enumerate(source["cohorts"]):
            row = {
                "scope": "gb300-http-" + scope,
                "profile": "decoder_bounded"
                if scope == "on" or scope.endswith("-on")
                else "full",
                "input_ordinal": ordinal,
                "segment_ordinal": case.get("segment_ordinal", 0),
                "purpose": case["purpose"],
                "trial_index": case["trial_index"],
                "observed": case["observed"],
                "original_replay_spec_sha256": case["replay_spec_sha256"],
            }
            if fresh:
                row.update(
                    status=case["status"],
                    exact_spec_recovered=case["recovery_status"]
                    == "exact_original_spec_sha256",
                    previous_prediction_status=case["previous_prediction_status"],
                )
                row.update(
                    {
                        k: case[k]
                        for k in (
                            "prediction",
                            "failure",
                            "failure_type",
                            "current_cache_semantics_match",
                        )
                        if k in case
                    }
                )
                if "requests" in case:
                    row["requests"] = [
                        {k: v for k, v in r.items() if k != "request_id"}
                        for r in case["requests"]
                    ]
            else:
                row.update(
                    status="prediction_input_unavailable",
                    exact_spec_recovered=False,
                    previous_prediction_status=case["status"],
                    failure_type="RetainedInputUnavailable",
                    failure="Original field/service token-bearing plan is unavailable; no synthetic replacement",
                )
            row["previous_cache_semantics_match"] = case.get("cache_semantics_match")
            row["native_initial_cached_tokens"] = sorted(
                case.get("native_initial_cached_tokens", {}).values()
            )
            group.append(row)
        rows.extend(group)
        for metric in (
            "ttft_ms",
            "average_tpot_ms",
            "output_tokens_per_second",
            "exact_itl_ms",
            "request_latency_ms",
            "last_token_latency_ms",
        ):
            paired = [
                (r["observed"][metric], r["prediction"][metric])
                for r in group
                if r["status"] == "predicted"
            ]
            item = {
                "scope": "gb300-http-" + scope,
                "metric": metric,
                "observed_cohorts": len(group),
                "exact_specs_recovered": sum(r["exact_spec_recovered"] for r in group),
                "predicted_cohorts": len(paired),
                "missing_cohorts": len(group) - len(paired),
            }
            if paired:
                item.update(
                    mape_percent=statistics.mean(
                        100 * abs(p / o - 1) for o, p in paired
                    ),
                    wape_percent=100
                    * sum(abs(p - o) for o, p in paired)
                    / sum(o for o, _ in paired),
                )
            summaries.append(item)
    write(
        args.output / "e2e-rows.json.gz",
        {
            "scope": "Current cold replay, all original observation denominators retained",
            "rows": rows,
        },
    )
    write(args.output / "e2e-summary.json", {"summary": summaries})
    write(
        args.output / "e2e-source-bindings.json",
        {
            "input_reports": pins,
            "exporter_sha256": sha(Path(__file__)),
            "predictor_commit": identity["predictor_commit"],
            "workload_proof": read(
                args.input_root / "shared-cold-workloads-v1/completion.json"
            ),
            "current_prediction_identity": identity,
            "fresh_raw_admission": False,
            "correction_fitting": False,
        },
    )
    with (args.output / "e2e-summary.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            [
                "scope",
                "metric",
                "observed_cohorts",
                "exact_specs_recovered",
                "predicted_cohorts",
                "missing_cohorts",
                "mape_percent",
                "wape_percent",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    main()
