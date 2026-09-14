# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export this fixed current-head SILICON refresh, retaining every missing row.

Input root contains silicon-holdouts-v1, silicon-retained-native-v1 and
silicon-gb200-strict-probe-v1. It is an actual prediction artifact root, not a
calibration input. This exporter performs no prediction or fitting.
"""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    data = (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    with path.open("xb") as stream:
        stream.write(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)


def metrics(rows):
    paired = [r for r in rows if r["status"] == "predicted"]
    result = {
        "observed": len(rows),
        "predicted": len(paired),
        "missing": len(rows) - len(paired),
    }
    if paired:
        result.update(
            mape_percent=statistics.mean(
                100 * abs(r["predicted_ms"] / r["observed_ms"] - 1) for r in paired
            ),
            wape_percent=100
            * sum(abs(r["predicted_ms"] - r["observed_ms"]) for r in paired)
            / sum(r["observed_ms"] for r in paired),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, summaries, pins = [], [], {}
    model_identity = None
    for profile in ("full", "decoder_bounded"):
        rel = f"silicon-holdouts-v1/holdout-{profile}/comparison.json"
        path = args.input_root / rel
        source = read(path)
        pins[rel] = sha(path)
        group = []
        for ordinal, case in enumerate(source["cases"]):
            row = {
                k: case[k]
                for k in (
                    "observed_ms",
                    "status",
                    "prediction_input",
                    "axis_bridge",
                    "rank_max_ms",
                )
            }
            row.update(
                scope="gb300-holdout",
                profile=profile,
                input_ordinal=ordinal,
                phase=case["phase"],
            )
            row.update(
                {
                    k: case[k]
                    for k in (
                        "predicted_ms",
                        "signed_error_percent",
                        "failure_type",
                        "failure",
                    )
                    if k in case
                }
            )
            group.append(row)
        rows.extend(group)
        summaries.append(
            {
                "scope": "gb300-holdout",
                "profile": profile,
                "unit": "logical configurations",
                **metrics(group),
            }
        )
    for scope in ("off", "on", "field-off", "field-on", "service-off", "service-on"):
        rel = f"silicon-retained-native-v1/{scope}.json.gz"
        path = args.input_root / rel
        source = read(path)
        pins[rel] = sha(path)
        identity = {
            k: source[k] for k in ("predictor_commit", "native_extension_sha256")
        }
        if model_identity is not None and identity != model_identity:
            raise ValueError("mixed predictor identity")
        model_identity = identity
        profile = (
            "decoder_bounded" if scope == "on" or scope.endswith("-on") else "full"
        )
        group = []
        cohort_ordinal = 0
        for segment_ordinal, segment in enumerate(source["physical_segments"]):
            for cohort in segment["cohorts"]:
                for interval_ordinal, interval in enumerate(cohort["intervals"]):
                    row = {
                        k: interval[k]
                        for k in (
                            "observed_ms",
                            "status",
                            "scheduled_requests",
                            "native_scheduled_requests",
                            "axis_bridge",
                            "variance_bridge",
                            "previous_prediction_status",
                        )
                        if k in interval
                    }
                    row.update(
                        scope="gb300-native-" + scope,
                        profile=profile,
                        input_ordinal=len(group),
                        segment_ordinal=segment_ordinal,
                        cohort_ordinal=cohort_ordinal,
                        interval_ordinal=interval_ordinal,
                        purpose=cohort["purpose"],
                        trial_index=cohort["trial_index"],
                        phase=interval["phase"],
                    )
                    row.update(
                        {
                            k: interval[k]
                            for k in (
                                "predicted_ms",
                                "signed_error_percent",
                                "failure_type",
                                "failure",
                                "native_work_role",
                            )
                            if k in interval
                        }
                    )
                    group.append(row)
                cohort_ordinal += 1
        result = metrics(group)
        assert result["observed"] == source["summary"]["planned_points"]
        assert result["predicted"] == source["summary"]["predicted_points"]
        assert abs(result["mape_percent"] - source["summary"]["mape_percent"]) < 1e-10
        rows.extend(group)
        summaries.append(
            {
                "scope": "gb300-native-" + scope,
                "profile": profile,
                "unit": "native intervals",
                "cohorts": source["cohort_count"],
                "fully_predicted_cohorts": source["fully_predicted_cohorts"],
                **result,
            }
        )
    rel = "silicon-gb200-strict-probe-v1/result.json"
    path = args.input_root / rel
    probe = read(path)
    pins[rel] = sha(path)
    for ordinal, source_row in enumerate(probe["rows"]):
        rows.append(
            {
                "scope": "gb200-strict-op-coverage-probe",
                "profile": "full",
                "input_ordinal": ordinal,
                "phase": source_row["phase"],
                "scheduled_requests": source_row["prediction_input"][
                    "scheduled_requests"
                ],
                "status": source_row["status"],
                "failure": source_row.get("failure"),
            }
        )
    summaries.append(
        {
            "scope": "gb200-strict-op-coverage-probe",
            "profile": "full",
            "unit": "geometry queries",
            "observed": 38,
            "predicted": probe["predicted_count"],
            "missing": 38 - probe["predicted_count"],
        }
    )
    write(args.output / "rows.json.gz", {"identity": model_identity, "rows": rows})
    write(
        args.output / "summary.json", {"identity": model_identity, "summary": summaries}
    )
    with (args.output / "summary.csv").open("x", newline="") as stream:
        columns = [
            "scope",
            "profile",
            "unit",
            "observed",
            "predicted",
            "missing",
            "mape_percent",
            "wape_percent",
            "cohorts",
            "fully_predicted_cohorts",
        ]
        writer = csv.DictWriter(stream, columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)
    write(
        args.output / "source-bindings.json",
        {
            "identity": model_identity,
            "input_reports_sha256": pins,
            "exporter_sha256": sha(Path(__file__)),
            "new_measurements": False,
            "fresh_raw_serving_admission": False,
            "correction_fitting": False,
            "prediction_sources": "Current merged source and freshly built native binary; fixed profile-specific op-level SILICON overlays.",
            "observations": "Holdouts use original qualified observations. Serving uses immutable report-retained observed geometry and timing.",
            "private_identifiers": "Run, cohort, request, worker, node and dispatch identifiers omitted; ordinals preserve input ordering.",
        },
    )


if __name__ == "__main__":
    main()
