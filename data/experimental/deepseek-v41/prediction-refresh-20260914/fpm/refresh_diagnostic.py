# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Predict source-qualified DL diagnostic shapes without changing calibration.

The caller must first qualify original archive/source closure. This script checks
the retained point joins and geometry; it does not perform raw-data admission.
The six shapes coincide with calibration coordinates, so this is a distinct
cross-run diagnostic, not an independent holdout or formal serving study.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

from refresh_reported_native import read, require, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-native-sha256", required=True)
    args = parser.parse_args()
    repo, root = args.repo.resolve(), args.diagnostic_root.resolve()
    require(Path.cwd() == repo, "run from current repository")
    require(
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == args.expected_head, "head differs"
    )
    sys.path[:0] = [
        str(repo / "data/experimental/deepseek-v41/verification-plan"),
        str(repo / "python/aisimulate"),
        str(repo / "python/aisimulate/src"),
    ]
    from compare_trace import systems_identity, trace_summary
    from normalize_fpm import prediction_input

    import aisimulate._runtime as native
    from aiconfigurator_core.sdk.rust_engine_step import RustForwardPassPerfModel

    require(Path(native.__file__).resolve().is_relative_to(repo), "foreign native binary")
    require(sha(native.__file__) == args.expected_native_sha256, "native differs")
    paths = {
        "point_index": root / "runtime-plan/v1/point-index.json",
        "paired_report": root / "results-through-v4/results/attempt-v4/ordinary/artifacts/paired-diagnostic.json",
        "completion": root / "results-through-v4/results/attempt-v4/ordinary/artifacts/completion.json",
        "independent_closure": root / "analysis/final-historical-v4-independent/source-bindings.json",
        "prefill": root / "results-through-v4/results/attempt-v2/native-prefill/artifacts/benchmark.json",
        "decode": root / "results-through-v4/results/attempt-v2/native-decode/artifacts/benchmark.json",
    }
    pins = {key: sha(path) for key, path in paths.items()}
    require(pins["point_index"] == "865727cb66aae227c10c6c85014a7fd5b8b2f7a40ad174e43774d527f1a767dd", "plan differs")
    require(
        pins["paired_report"] == "10a1ec96087fd8c983310d232e6340048d9d1a28af3c58df7990d84e1bc14e3c", "report differs"
    )
    require(
        pins["completion"] == "bbcde88ca1de407fa2d4d9ba210b2d58bca48db67878f8842d76d27e2ad67a44", "completion differs"
    )
    paired, plan = read(paths["paired_report"]), read(paths["point_index"])
    require(paired["capture_audit_valid"] is True and paired["formal_study_admission"] is False, "wrong capture scope")
    planned = {(p["mode"], p["benchmark_id"]): p for p in plan["points"]}
    original = {}
    for mode in ("prefill", "decode"):
        document = read(paths[mode])
        require(document["valid"] is True and len(document["results"]) == 18, "native incomplete")
        original.update({(mode, row["point"]["benchmark_id"]): row for row in document["results"]})
    config_path = repo / "data/experimental/deepseek-v41/gb200-fpm/holdout-sol-review-v2/fpm-prediction-config.json"
    config = read(config_path)
    identity = systems_identity(config)
    model = RustForwardPassPerfModel.from_native(config)
    seen, rows = set(), []
    for point in paired["points"]:
        spec = point["specification"]
        key = spec["mode"], spec["benchmark_id"]
        require(key not in seen and spec == planned[key], "point identity differs")
        seen.add(key)
        require(point["native_all_fpms"] == original[key]["fpms"], "original native FPM differs")
        require(point["native_selected_fpm_index"] == 0 and len(point["native_all_fpms"]) == 1, "selection differs")
        native_fpm = point["native_all_fpms"][0]
        require(native_fpm["scheduled_requests"] == point["target_scheduled_requests"], "target differs")
        require(native_fpm["wall_time"] * 1000 == point["native_wall_ms"], "native observation differs")
        selected = point["selected_ordinary_dispatches"]
        require(len(selected) == point["matched_dispatch_count"], "match count differs")
        require(point["unique_geometry_match"] is (len(selected) == 1), "unique match differs")
        if len(selected) == 1:
            require(
                selected[0]["fpm"]["scheduled_requests"] == point["target_scheduled_requests"], "ordinary shape differs"
            )
            require(selected[0]["fpm"]["wall_time"] * 1000 == point["ordinary_wall_ms"], "ordinary observation differs")
        for arm in ("native", "ordinary"):
            row = {
                "arm": arm,
                "geometry": spec["geometry_id"],
                "mode": spec["mode"],
                "benchmark_id": spec["benchmark_id"],
                "role": spec["role"],
                "repeat_index": spec["measured_repeat_index"],
                "scheduled_requests": point["target_scheduled_requests"],
                "matched_dispatch_count": len(selected),
            }
            if arm == "ordinary" and len(selected) != 1:
                row.update(
                    status="observation_unavailable",
                    observed_ms=None,
                    failure="no unique ordinary dispatch with the planned exact geometry",
                )
            else:
                fpm = native_fpm if arm == "native" else selected[0]["fpm"]
                query, bridge = prediction_input(
                    {"scheduled_requests": fpm["scheduled_requests"]},
                    producer_semantics="vllm_past_kv",
                    target_axis="whole_forward_past_kv",
                )
                row.update(observed_ms=fpm["wall_time"] * 1000, axis_bridge=bridge)
                try:
                    value = model.estimate_forward_pass_time_ms(query)
                    require(type(value) in (int, float) and math.isfinite(value) and value > 0, "no finite prediction")
                except Exception as error:
                    row.update(status="prediction_unavailable", failure_type=type(error).__name__, failure=str(error))
                else:
                    row.update(
                        status="predicted",
                        predicted_ms=value,
                        signed_error_percent=100 * (value / row["observed_ms"] - 1),
                    )
            rows.append(row)
    require(seen == set(planned), "missing planned point")
    require(all(sha(path) == pins[key] for key, path in paths.items()), "input changed")
    require(identity == systems_identity(config), "calibration changed")
    measured = [r for r in rows if r["role"] == "diagnostic_measurement"]
    require(len(measured) == 60, "measured denominator differs")
    write(
        args.output,
        {
            "schema": "dsv41.current-source.dl-diagnostic-refresh.v1",
            "diagnostic_only": True,
            "formal_study_admission": False,
            "independent_holdout": False,
            "correction_fitting": False,
            "predictor_commit": args.expected_head,
            "native_extension_sha256": args.expected_native_sha256,
            "inputs_sha256": pins,
            "source_sha256": sha(__file__),
            "prediction_config_sha256": sha(config_path),
            "prediction_config": config,
            "systems_identity": identity,
            "rows": rows,
            "summaries": {
                arm: trace_summary([r for r in measured if r["arm"] == arm]) for arm in ("native", "ordinary")
            },
            "notes": [
                "Five correlated repetitions at each of six existing calibration coordinates; no new fit.",
                "Six diagnostic warmup points and all prefix setup excluded from MAPE by prospective role.",
                "Ten unmatched ordinary points remain unavailable; native30 and ordinary20 are different populations.",
                "MAPE compares fresh predictions with observations; "
                "it is distinct from benchmark/serving latency ratios.",
            ],
        },
    )


if __name__ == "__main__":
    main()
