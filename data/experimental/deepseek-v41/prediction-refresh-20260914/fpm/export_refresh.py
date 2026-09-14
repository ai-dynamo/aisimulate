# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export current FPM predictions without private paths or runtime identifiers.

This is a lossless numerical projection of completed prediction outputs. Source
hashes bind the originals. It performs neither prediction nor measurement
admission and keeps unsupported rows and original denominators.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import statistics
from pathlib import Path

from refresh_reported_native import read, require, sha, write

HEAD = "430ed5e79abaf667c026d980c18ec240261fc800"
NATIVE = "50d3010ee04daaa23158fe5ef873b363a61b096ec755b1bc46a7a1ba6a8117b0"
SHAPE = (
    "num_prefill_requests",
    "sum_prefill_tokens",
    "sum_prefill_kv_tokens",
    "var_prefill_length",
    "num_decode_requests",
    "sum_decode_kv_tokens",
    "var_decode_kv_tokens",
)
METRICS = (
    "ttft_ms",
    "average_tpot_ms",
    "exact_itl_ms",
    "output_tokens_per_second",
    "request_latency_ms",
    "last_token_latency_ms",
)


def csv_write(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    raw = stream.getvalue().encode()
    with path.open("xb") as output:
        output.write(gzip.compress(raw, mtime=0) if path.suffix == ".gz" else raw)


def summary(scope, metric, rows, expected=None, role="verification"):
    paired = [r for r in rows if r["status"] == "predicted"]
    value = statistics.mean(abs(r["signed_error_percent"]) for r in paired) if paired else None
    suffix = "value" if paired and "observed_value" in paired[0] else "ms"
    wape = (
        100
        * sum(abs(r["predicted_" + suffix] - r["observed_" + suffix]) for r in paired)
        / sum(r["observed_" + suffix] for r in paired)
        if paired
        else None
    )
    if expected is not None:
        require(
            len(rows) == expected["planned_points"] and len(paired) == expected["predicted_points"], "coverage differs"
        )
        require(math.isclose(value, expected["mape_percent"], rel_tol=1e-13, abs_tol=1e-13), "MAPE projection differs")
    return {
        "scope": scope,
        "metric": metric,
        "role": role,
        "planned_points": len(rows),
        "predicted_points": len(paired),
        "mape_percent": value,
        "wape_percent": wape,
        "predictor_commit": HEAD,
        "native_extension_sha256": NATIVE,
    }


def scalar(row):
    result = {key: row[key] for key in ("status", "observed_ms", "predicted_ms", "signed_error_percent") if key in row}
    if "failure" in row:
        result["failure"] = row["failure"]
    if "historical_status" in row:
        result["historical_status"] = row["historical_status"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.input_root, args.output_dir
    require(read(root / "additional-predictions-completion.json")["valid"] is True, "additional runs incomplete")
    require(read(root / "final-cold-completion.json")["valid"] is True, "portable cold runs incomplete")
    require(read(root / "gb200-refresh-v1/completion.json")["valid"] is True, "GB200 runs incomplete")
    require(read(root / "gb300-native-v2/receipt.json")["valid"] is True, "GB300 native runs incomplete")
    out.mkdir()
    bindings, summaries, native_rows, http_rows, config_rows, diagnostic_rows = {}, [], [], [], [], []

    def load(relative):
        path = root / relative
        document = read(path)
        bindings[relative] = {"sha256": sha(path), "bytes": path.stat().st_size}
        for key in (
            "predictor_commit",
            "native_extension_sha256",
            "input_sha256",
            "inputs_sha256",
            "input_files_sha256",
            "packet_sha256",
            "script_sha256",
            "source_sha256",
            "prediction_config_sha256",
            "prediction_config",
            "prediction_sources",
            "prediction_sources_sha256",
            "systems_identity",
            "resolved_model_identity",
        ):
            if key in document:
                bindings[relative][key] = document[key]
        return document

    def native_scope(scope, document):
        selected = []
        physical = {}
        for ordinal, cohort in enumerate(document["cohorts"]):
            run = cohort.get("physical_run_id", "single")
            segment = physical.setdefault(run, len(physical))
            for index, row in enumerate(cohort["intervals"]):
                value = {
                    "scope": scope,
                    "segment_ordinal": segment,
                    "cohort_ordinal": ordinal,
                    "interval_ordinal": index,
                    "purpose": cohort["purpose"],
                    "trial_index": cohort["trial_index"],
                    "phase": row["phase"],
                    **scalar(row),
                }
                value.update({key: row["scheduled_requests"][key] for key in SHAPE})
                selected.append(value)
        native_rows.extend(selected)
        summaries.append(summary(scope, "native_forward_ms", selected, document["summary"]))

    for profile in ("off", "on"):
        for population in ("core", "field", "service"):
            native_scope(
                f"gb300/{profile}/{population}", load(f"gb300-native-v2/gb300-{profile}-{population}-native.json.gz")
            )
    native_scope("gb200/off/ordinary", load("gb200-refresh-v1/ordinary-trace.json"))

    http_documents = [
        ("gb200/off/ordinary", load("gb200-refresh-v1/ordinary-e2e.json")),
        ("gb300/off/core/cold-subset", load("gb300-off-cold-v3.json.gz")),
        ("gb300/on/core/cold-subset", load("gb300-on-cold-v3.json.gz")),
    ]
    repo = Path(__file__).resolve().parents[5]
    for profile in ("off", "on"):
        for population in ("field", "service"):
            relative = (
                f"data/experimental/deepseek-v41/gb300-fpm/{profile}-union-v3-tracewait2"
                f"/reports/{population}/e2e.json.gz"
            )
            path = repo / relative
            original = read(path)
            bindings[relative] = {"sha256": sha(path), "original_input_sha256": original["input_sha256"]}
            cohorts = [
                {
                    **{key: row[key] for key in ("purpose", "trial_index", "observed")},
                    "status": "prediction_unavailable",
                    "historical_status": row["status"],
                    "failure": "original complete token plan and replay specification unavailable",
                }
                for row in original["cohorts"]
            ]
            http_documents.append((f"gb300/{profile}/{population}", {"cohorts": cohorts}))
    for scope, document in http_documents:
        for metric in METRICS:
            selected = []
            for ordinal, row in enumerate(document["cohorts"]):
                value = {
                    "scope": scope,
                    "cohort_ordinal": ordinal,
                    "purpose": row["purpose"],
                    "trial_index": row["trial_index"],
                    "metric": metric,
                    "status": row["status"],
                    "observed_value": row.get("observed", {}).get(metric),
                    "cache_semantics_match": row.get("cache_semantics_match"),
                    "metric_unit": "tokens/second" if metric == "output_tokens_per_second" else "ms",
                }
                if row["status"] == "predicted":
                    value.update(
                        predicted_value=row["prediction"][metric],
                        signed_error_percent=row["signed_error_percent"][metric],
                        request_count=len(row["requests"]),
                    )
                else:
                    value["failure"] = row.get("failure", "unavailable")
                selected.append(value)
            http_rows.extend(selected)
            result = summary(scope, metric, selected)
            if metric in document.get("metrics", {}):
                require(
                    math.isclose(result["mape_percent"], document["metrics"][metric]["mape_percent"], rel_tol=1e-13),
                    "HTTP MAPE differs",
                )
            summaries.append(result)
    for scope, relative in (
        ("gb200/off/holdout38", "gb200-refresh-v1/independent-holdout.json"),
        ("gb300/off/forward-validation46", "gb300-off-forward46-v1.json"),
        ("gb300/on/forward-validation46", "gb300-on-forward46-v1.json"),
    ):
        document = load(relative)
        selected = []
        for ordinal, row in enumerate(document["cases"]):
            value = {"scope": scope, "configuration_ordinal": ordinal, "phase": row["phase"], **scalar(row)}
            value.update({key: row["prediction_input"]["scheduled_requests"][key] for key in SHAPE})
            selected.append(value)
        config_rows.extend(selected)
        role = "independent_holdout" if scope.startswith("gb200/") else "forward_validation_corpus"
        summaries.append(summary(scope, "native_forward_ms", selected, document["summary"], role=role))
    diagnostic = load("dl-diagnostic-v2.json")
    for arm in ("native", "ordinary"):
        scope = f"gb200/off/dl-diagnostic/{arm}"
        selected = []
        for row in diagnostic["rows"]:
            if row["arm"] != arm:
                continue
            value = {
                "scope": scope,
                "geometry": row["geometry"],
                "repeat_index": row["repeat_index"],
                "role": row["role"],
                **scalar(row),
                **row["scheduled_requests"],
            }
            diagnostic_rows.append(value)
            if row["role"] == "diagnostic_measurement":
                selected.append(value)
        summaries.append(
            summary(scope, "native_forward_ms", selected, diagnostic["summaries"][arm], role="diagnostic_only")
        )
    csv_write(out / "summary.csv", summaries)
    csv_write(out / "native-predictions.csv.gz", native_rows)
    csv_write(out / "http-predictions.csv.gz", http_rows)
    csv_write(out / "configuration-predictions.csv", config_rows)
    csv_write(out / "diagnostic-predictions.csv", diagnostic_rows)
    trace = read(root / "gb200-refresh-v1/ordinary-trace.json")
    e2e = http_documents[0][1]
    write(
        out / "gb200-statistics.json",
        {
            "native_independent_trial_summary": trace["independent_trial_summary"],
            "http_purpose_statistics": e2e["points"],
            "note": "Current prediction statistics; whole-trial bootstrap, conditional on frozen calibration.",
        },
    )
    write(
        out / "source-bindings.json",
        {
            "predictor_commit": HEAD,
            "native_extension_sha256": NATIVE,
            "inputs": bindings,
            "exporter_sha256": sha(__file__),
            "no_new_observations": True,
            "no_refit": True,
            "projection": "Exact numeric pairs and denominators; private runtime identifiers omitted.",
        },
    )
    for path in out.iterdir():
        raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
        require(
            not any(token in raw for token in (b"/home/", b"/tmp/", b"/campaign/", b"/raid/", b"/Users/")),
            "private filesystem path in public projection",
        )
    write(out / "artifact-hashes.json", {path.name: sha(path) for path in sorted(out.iterdir()) if path.is_file()})
    print(
        json.dumps(
            {
                "summary_rows": len(summaries),
                "native_rows": len(native_rows),
                "http_metric_rows": len(http_rows),
                "configuration_rows": len(config_rows),
                "diagnostic_rows_including_warmup": len(diagnostic_rows),
            }
        )
    )


if __name__ == "__main__":
    main()
