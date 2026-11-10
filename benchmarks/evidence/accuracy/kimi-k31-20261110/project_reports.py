# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Project verified accuracy reports onto explicit public fields; never fit or select."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

METRICS = (
    "selected",
    "paired",
    "mape_percent",
    "p95_ape_percent",
    "max_ape_percent",
    "mean_signed_error_percent",
)
COORDS = ("batch_size", "total_prefill_tokens", "total_kv_read_tokens")
CSV_FIELDS = (
    "method",
    "phase",
    *COORDS,
    "predicted_ms",
    "measured_ms",
    "signed_error_percent",
    "ape_percent",
    "status",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pick(value, fields):
    return {key: value[key] for key in fields if key in value}


def summarize(value):
    return {phase: pick(metrics, METRICS) for phase, metrics in value.items()}


def emit(path, value):
    data = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if path.exists():
        assert path.read_text() == data, f"Existing public report differs: {path.name}"
    else:
        path.write_text(data)


def emit_points(path, method, points):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for point in points:
        writer.writerow(
            {
                "method": method,
                **{k: point.get(k, "") for k in CSV_FIELDS if k != "method"},
            }
        )
    data = out.getvalue()
    if path.exists():
        assert path.read_text() == data
    else:
        path.write_text(data)


def point_fields(point, method):
    coordinates = point if method == "op_based" else point["coordinates"]
    out = {
        "phase": point["phase"],
        **pick(coordinates, COORDS),
        **pick(point, ("predicted_ms", "measured_ms", "status")),
    }
    if "predicted_ms" in point and point["predicted_ms"] is not None:
        signed = 100 * (point["predicted_ms"] - point["measured_ms"]) / point["measured_ms"]
        reference = point["signed_error_percent"] if method == "op_based" else 100 * point["relative_error"]
        assert math.isclose(signed, reference, rel_tol=1e-12, abs_tol=1e-12)
        out.update(signed_error_percent=reference, ape_percent=abs(reference))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accuracy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    original_hashes = {}

    def read(relative):
        path = args.accuracy_root / relative
        original_hashes[relative] = sha(path)
        return json.loads(path.read_text())

    op = read("full-truth-v4/full-op-accuracy.json")
    freeze = read("full-truth-v4/prediction-freeze-v4-final.json")
    plan = read("full-truth-v4/full-op-plan.json")
    op_notes = read("full-truth-v4/full-op-runtime-notes.json")
    producer = read("fpm-predictor-producer-binding-v3.json")
    assert op_notes["accuracy_report_sha256"] == original_hashes["full-truth-v4/full-op-accuracy.json"]
    assert op["input_sha256"]["frozen_predictions"] == original_hashes["full-truth-v4/prediction-freeze-v4-final.json"]
    assert op["input_sha256"]["plan"] == freeze["plan_sha256"] == original_hashes["full-truth-v4/full-op-plan.json"]
    assert not op["calibration_or_predictor_changes_after_truth"] and not freeze["truth_consumed"]
    op_points = [point_fields(point, "op_based") for point in op["results"]]
    assert len(op_points) == 8
    op_public = {
        "schema": "kimi-k31-public-op-accuracy/v1",
        "method": "op_based",
        "model": "moonshotai/Kimi-K3.1-Pro",
        "system": "gb300",
        "tp": 8,
        "scope": (
            "Full trained-model native SGLang eager ModelRunner.forward CUDA-event timing, eight "
            "preselected shapes. Not serving TTFT/TPOT, throughput, text quality, or a "
            "SemiAnalysis match."
        ),
        "calibration_scope": (
            "Predictions froze against a private 20-point GB300 DSA calibration snapshot plus the "
            "existing Op profiles. This is not an accuracy score for the later 96-point ARM "
            "coverage table."
        ),
        "truth_protocol": (
            "Three warmup and ten retained target forwards per point and each of eight TP ranks; "
            "measured_ms is the maximum rank median. Fresh native request/state preparation and "
            "matching observed input-token hashes across all TP ranks were checked before target "
            "timing."
        ),
        "runtime": pick(plan["runtime"], ("image_index", "source_commit", "package", "arm_manifest")),
        "predictor": pick(
            freeze,
            (
                "source_commit",
                "runtime_extension_sha256",
                "wheel_sha256",
                "build_receipt_sha256",
            ),
        ),
        "source_report_sha256": original_hashes["full-truth-v4/full-op-accuracy.json"],
        "runtime_notes_sha256": original_hashes["full-truth-v4/full-op-runtime-notes.json"],
        "native_log_sha256": op_notes["native_log_sha256"],
        "frozen_evidence_sha256": dict(op["input_sha256"]),
        "native_source_files_sha256": dict(op["native_source"]["files"]),
        "source_profile_manifest_digest": hashlib.sha256(
            json.dumps(freeze["source_files"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "source_profile_manifest_digest_encoding": (
            "SHA256 of canonical JSON: sorted keys, separators comma and colon, UTF-8; full "
            "manifest is sealed by frozen_predictions hash."
        ),
        "summary": summarize(op["summary"]),
        "points": op_points,
        "raw_truth_samples": [
            {
                "phase": p["phase"],
                **pick(p, COORDS),
                "ranks": [{"rank": r["rank"], "samples_ms": list(r["samples_ms"])} for r in p["raw_samples"]],
            }
            for p in op["results"]
        ],
        "calibration_or_predictor_changes_after_truth": False,
        "limitations": [
            (
                "Native MoE logs retained 32 untuned-shape tactic=-1 fallback warnings. These differ "
                "from predictor fallback and do not establish the cause of prediction error."
            ),
            "The eight-point test does not establish full-domain accuracy or production prompt-distribution accuracy.",
        ],
    }
    emit(args.output / "op-full-model.json", op_public)
    emit_points(args.output / "op-full-model.csv", "op_based", op_points)

    phases = []
    for phase in ("prefill", "decode"):
        report_name = f"fpm-holdout-v6-{phase}/accuracy-summary.json"
        if not (args.accuracy_root / report_name).exists():
            continue
        report = read(report_name)
        canonical_name = f"fpm-holdout-v6-{phase}/interpolation-validation.json"
        canonical = read(canonical_name)
        selection = read(f"fpm-holdout-v6-{phase}/holdout-plan.json")
        sampling_name = f"fpm-native-sampling-v6-{phase}.json"
        sampling = read(sampling_name)
        assert report["canonical_report_sha256"] == original_hashes[canonical_name]
        assert report["results"] == canonical["predictions"]
        assert (
            canonical["source_artifacts_unchanged"] and canonical["native_query_coverage"]["queries"]["measured"] == 0
        )
        assert report["assessed_phases"] == [phase]
        points = [point_fields(point, "fpm") for point in report["results"]]
        assert len(points) == report["summary"][phase]["selected"]
        public = {
            "schema": "kimi-k31-public-fpm-holdout-accuracy/v1",
            "method": "fpm_direct_interpolation",
            "phase": phase,
            "model": "moonshotai/Kimi-K3.1-Pro",
            "system": "gb300",
            "tp": 8,
            "runtime": "Dynamo/vLLM 0.1.dev24216+g266e88c05",
            "timing_boundary": "Native scheduler-wall forward-pass estimates; not CUDA-event latency.",
            "scope": (
                "Same GB300, TP8, runtime and original worker role. Seed 42, 20% capped at 16 "
                "coordinates, retained envelope/native expected mode boundaries; simultaneously "
                "remove all selected coordinates before direct native interpolation. No calibration "
                "self-lookups or predictor fallback."
            ),
            "not_assessed": [
                "Cross-TP or cross-GPU SOL transfer/extrapolation",
                "Independent repeatability qualification",
                "Serving TTFT/TPOT or SemiAnalysis accuracy",
            ],
            "source_report_sha256": original_hashes[report_name],
            "producer_binding_sha256": original_hashes["fpm-predictor-producer-binding-v3.json"],
            "producer": pick(
                producer,
                (
                    "producer_source",
                    "producer_archive_sha256",
                    "producer_runtime_validator_sha256",
                    "producer_measurement_evidence_sha256",
                    "producer_execution_evidence_sha256",
                ),
            ),
            "native_artifact_sha256": sorted(
                {
                    graph["source"]["sha256"]
                    for cell in canonical["capture_boundaries"]["cells"]
                    for graph in cell["native_graph_config"]
                }
            ),
            "canonical_report_sha256": original_hashes[canonical_name],
            "holdout_plan_sha256": report["source_sha256"]["plan"],
            "source_hashes": pick(
                report["source_sha256"],
                (
                    "request",
                    "execution_evidence",
                    "implementation",
                    "runtime_extension",
                    "build_receipt",
                    "scorer",
                ),
            ),
            "original_table_sha256": selection["inputs"]["parquet"]["sha256"],
            "original_sidecar_sha256": selection["inputs"]["metadata"]["sha256"],
            "predictor": pick(report, ("predictor_build_source_commit", "predictor_wheel_sha256")),
            "formal_rows": selection["matching_row_count"],
            "retained_calibration_rows": selection["matching_row_count"] - selection["removed_row_count"],
            "withheld_rows": selection["removed_row_count"],
            "native_query_coverage": pick(
                canonical["native_query_coverage"]["queries"],
                ("interpolated", "measured", "unsupported"),
            ),
            "canonical_status": report["canonical_status"],
            "capture_boundary_status": report["capture_boundary_status"],
            "capture_boundary_limitation": (
                "Native expected modes and resolved graph configuration are available; per-call "
                "observed dispatch is not reported."
            ),
            "sampling_audit_sha256": original_hashes[sampling_name],
            "sampling": sampling["distributions"][phase]["formal_selected"],
            "sampling_note": (
                "One formal collection, one retained native estimate per point/DP stream. Internal "
                "sample counts and reducers are shown above. No separate repeatability qualification "
                "was performed; the generic canonical limitation must not be read as claiming one "
                "internal forward or existing repeatability evidence."
            ),
            "summary": pick(report["summary"][phase], METRICS),
            "points": points,
            "all_selected_coordinates_retained": True,
            "reselection_or_tuning_after_truth": False,
        }
        emit(args.output / f"fpm-{phase}.json", public)
        emit_points(args.output / f"fpm-{phase}.csv", "fpm_direct_interpolation", points)
        phases.append(phase)

    aggregate_name = "fpm-holdout-v6-both.json"
    aggregate_summary = None
    if (args.accuracy_root / aggregate_name).exists():
        aggregate = read(aggregate_name)
        assert phases == ["prefill", "decode"] and aggregate["both_phases_have_accuracy_pairs"]
        combined = {
            "schema": "kimi-k31-public-fpm-both-role-accuracy/v1",
            "source_report_sha256": original_hashes[aggregate_name],
            "selection_protocol_sha256": aggregate["selection_protocol_sha256"],
            "phase_summary_sha256": {phase: aggregate["inputs"][phase]["summary_sha256"] for phase in phases},
            "summary": summarize(aggregate["summary"]),
            "phase_canonical_status": aggregate["phase_canonical_status"],
            "all_phase_accuracy_gates_passed": all(
                value == "passed" for value in aggregate["phase_canonical_status"].values()
            ),
            "acceptance_note": (
                "Each phase keeps its original gate result. The aggregate error does not override a "
                "failed phase gate. Complete formal collection and calibration publication do not "
                "imply accuracy acceptance."
            ),
            "aggregation": (
                "Equal weight across all paired selected coordinates, with selected/paired and "
                "unsupported coverage retained per phase."
            ),
            "comparison_with_op": (
                "Different backend/runtime/timing boundaries; these scores do not rank Op versus FPM "
                "on the same runtime."
            ),
        }
        emit(args.output / "fpm-both-phases.json", combined)
        aggregate_summary = aggregate["summary"]["all"]

    rows = [
        "| Method / phase | Paired / selected | MAPE | P95 APE | Max APE |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in [
        ("Op / all", op["summary"]["all"]),
        ("Op / prefill", op["summary"]["prefill"]),
        ("Op / decode", op["summary"]["decode"]),
    ]:
        rows.append(
            f"| {name} | {metrics['paired']} / {metrics['selected']} | "
            f"{metrics['mape_percent']:.4f}% | {metrics['p95_ape_percent']:.4f}% | "
            f"{metrics['max_ape_percent']:.4f}% |"
        )
    for phase in phases:
        value = json.loads((args.output / f"fpm-{phase}.json").read_text())["summary"]
        rows.append(
            f"| FPM / {phase} | {value['paired']} / {value['selected']} | "
            f"{value['mape_percent']:.4f}% | {value['p95_ape_percent']:.4f}% | {value['max_ape_percent']:.4f}% |"
        )
    if aggregate_summary is not None:
        value = aggregate_summary
        rows.append(
            f"| FPM / all selected points | {value['paired']} / {value['selected']} | "
            f"{value['mape_percent']:.4f}% | {value['p95_ape_percent']:.4f}% | {value['max_ape_percent']:.4f}% |"
        )
    readme = (
        (
            "<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All "
            "rights reserved. -->\n<!-- SPDX-License-Identifier: Apache-2.0 -->\n# Kimi K3.1 Pro "
            "measured accuracy\n\nThese are two separate full-model timing experiments on GB300 "
            "TP8. Op uses native SGLang eager CUDA-event timing; FPM uses Dynamo/vLLM native "
            "scheduler-wall estimates. They do not establish an apples-to-apples method ranking "
            "or a SemiAnalysis serving comparison.\n\n"
        )
        + "\n".join(rows)
        + (
            "\n\nOp predictions were frozen before the eight full-model measurements, using a "
            "private 20-point GB300 DSA calibration snapshot and existing Op profiles. The later "
            "96-point ARM coverage table is not the calibration snapshot scored here. All eight "
            "shapes and all ten retained observations from each TP rank remain in the JSON; the "
            "truth is the maximum rank median after three warmup forwards. Native untuned MoE "
            "tactic fallback warnings are retained as runtime context, without assigning them as "
            "the cause of error.\n\nFPM withholds coordinates from the formal table simultaneously, "
            "predicts them from the remaining rows, and retains every selected point and outlier. "
            "This assesses direct interpolation at the same GPU/TP/runtime; it does not assess "
            "SOL transfer across TP sizes or GPUs. Each phase has one formal collection and one "
            "retained native estimate per point. The JSON records the actual internal sample "
            "counts and native reducers; no independent repeatability certification has been "
            "completed.\n\n`project_reports.py` is an explicit field projection. It does not select "
            "points, recompute predictions, fit, or tune. CSV copies coordinates, prediction, "
            "truth and signed/absolute percent error from the sealed reports; JSON adds "
            "source/frozen/native hashes and scopes. APE is `100 * abs(predicted_ms - "
            "measured_ms) / measured_ms`; MAPE gives equal weight to paired points, and P95 "
            "linearly interpolates sorted APE at `(N-1)*0.95`. Hashes identify the unchanged "
            "private original reports and source artifacts. Hostnames, PIDs, environments, logs "
            "and absolute paths are excluded.\n\nTo reproduce the projection from the private audit "
            "archive, run `python project_reports.py --accuracy-root AUDIT_ARCHIVE --output "
            "OUTPUT_DIRECTORY`. The archive must contain the relative report names used in the "
            "script. No GPU or package upload is performed.\n"
        )
    )
    if "prefill" not in phases:
        readme += "\nPrefill FPM remains pending; this staging snapshot currently contains decode only.\n"
    else:
        failures = []
        for phase in phases:
            public_phase = json.loads((args.output / f"fpm-{phase}.json").read_text())
            if public_phase["canonical_status"] != "passed":
                phase_plan = read(f"fpm-holdout-v6-{phase}/holdout-plan.json")
                threshold = 100 * phase_plan["policy"]["max_p95_relative_error"]
                failures.append(
                    f"{phase} P95 APE is {public_phase['summary']['p95_ape_percent']:.4f}% "
                    f"against the original {threshold:g}% gate"
                )
        if failures:
            marker = (
                "\n**FPM accuracy acceptance: FAILED.** "
                + "; ".join(failures)
                + (
                    ". The aggregate MAPE does not override this phase failure. Both formal collections "
                    "completed successfully; the native measurements remain development calibration data, "
                    "with every held-out point and outlier retained. Neither the table nor the threshold "
                    "was adjusted.\n"
                )
            )
            readme = readme.replace("\nOp predictions were frozen", marker + "\nOp predictions were frozen")
    (args.output / "README.md").write_text(readme)
    assert all(sha(args.accuracy_root / name) == value for name, value in original_hashes.items())
    # Generated files are data, not original native artifacts. LF is intentional.
    for path in args.output.iterdir():
        if path.suffix not in {".json", ".csv", ".md"}:
            continue
        text = path.read_text()
        assert not any(
            token in text
            for token in (
                "/home/",
                "/scratch/",
                "/lustre/",
                "OMPI_MCA_",
                "worker_pid",
                "mountinfo",
            )
        )
    print(
        json.dumps(
            {
                "phases": phases,
                "originals_unchanged": True,
                "files": {p.name: sha(p) for p in sorted(args.output.iterdir()) if p.is_file()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
