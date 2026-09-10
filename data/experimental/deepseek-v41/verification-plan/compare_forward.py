# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare admitted native forward holdouts through the shared prediction API.

Run with the SILICON checkout's source and rebuilt native extension available.
This does not fit correction factors or turn missing predictions into zeroes.
The observed target includes native preparation and sampling, not HTTP E2E.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from analyze_e2e import quantile
from normalize_fpm import prediction_input


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def case_metrics(case):
    batch = case["batch_size"]
    context = case["phase"] == "context"
    if case["phase"] not in {"context", "generation"}:
        raise ValueError("unknown native workload phase")
    return {
        "version": 1,
        "scheduled_requests": {
            "num_prefill_requests": batch if context else 0,
            "sum_prefill_tokens": batch * case["query"] if context else 0,
            "sum_prefill_kv_tokens": batch * case["prefix"] if context else 0,
            "num_decode_requests": 0 if context else batch,
            "sum_decode_kv_tokens": 0 if context else batch * case["native_inclusive_kv"],
            "var_prefill_length": 0.0,
            "var_decode_kv_tokens": 0.0,
        },
    }


def error_summary(rows):
    paired = [row for row in rows if row["status"] == "predicted"]
    if not paired:
        return {"planned_points": len(rows), "predicted_points": 0}
    signed = [row["signed_error_percent"] for row in paired]
    absolute = [abs(value) for value in signed]
    return {
        "planned_points": len(rows),
        "predicted_points": len(paired),
        "weighting": "equal logical configurations; observed median of repeated rank maxima",
        "mean_signed_error_percent": statistics.mean(signed),
        "median_absolute_error_percent": statistics.median(absolute),
        "p90_absolute_error_percent_across_configurations": quantile(absolute, 0.9),
        "wape_percent": 100
        * sum(abs(r["predicted_ms"] - r["observed_ms"]) for r in paired)
        / sum(r["observed_ms"] for r in paired),
    }


def compare_cases(cases, predictor, *, forward_model):
    if forward_model not in {"op_level", "fpm"}:
        raise ValueError("unknown forward model")
    target = "whole_forward_past_kv" if forward_model == "fpm" else "op_level_inclusive_query"
    rows = []
    for case in cases:
        repeats = case["rank_max_ms"]
        if len(repeats) < 3 or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in repeats):
            raise ValueError("invalid or incomplete measured repetitions")
        observed = statistics.median(repeats)
        if observed != case["median_ms"]:
            raise ValueError("reported observation differs from measured median")
        metrics, bridge = prediction_input(
            case_metrics(case), producer_semantics="sglang_inclusive_query", target_axis=target
        )
        row = dict(case, observed_ms=observed, prediction_input=metrics, axis_bridge=bridge)
        try:
            predicted = predictor(metrics)
            if type(predicted) not in (int, float) or not math.isfinite(predicted) or predicted <= 0:
                raise ValueError("native estimator returned no finite positive prediction")
        except Exception as error:
            row.update(status="prediction_unavailable", failure_type=type(error).__name__, failure=str(error))
        else:
            row.update(
                status="predicted", predicted_ms=predicted, signed_error_percent=100 * (predicted / observed - 1)
            )
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--heldout-plan", type=Path, required=True)
    parser.add_argument("--prediction-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from collector.sglang.dsv41_forward_results import BOUNDARY
    from collector.sglang.dsv41_workloads import freeze_workloads

    import aisimulate._runtime as native
    from aiconfigurator_core.sdk import engine
    from aiconfigurator_core.sdk.models import deepseek_v41
    from aiconfigurator_core.sdk.rust_engine_step import RustForwardPassPerfModel

    observations = json.loads(args.observations.read_bytes())
    plan = freeze_workloads(json.loads(args.heldout_plan.read_bytes()))
    config = json.loads(args.prediction_config.read_bytes())
    if (
        observations.get("status") != "accepted"
        or observations["timing_boundary"] != BOUNDARY
        or observations["plan_sha256"] != plan["source_sha256"]
        or len(observations["cases"]) != len(plan["cases"])
    ):
        raise ValueError("only complete admitted observations for this frozen holdout plan are accepted")
    for observed, planned in zip(observations["cases"], plan["cases"], strict=True):
        if any(observed.get(key) != value for key, value in planned.items()):
            raise ValueError("observation geometry/order differs from frozen holdout plan")
    if (
        config["model_name"] != "deepseek-ai/DeepSeek-V4.1-Flash"
        or config["backend"] != "sglang"
        or config["system_name"] != "gb300"
        or config.get("tp_size") != 4
        or config.get("moe_tp_size") != 4
        or config.get("moe_ep_size") != 1
        or bool(config.get("decoder_replay", False)) != (observations["execution_profile"] == "decoder_bounded")
    ):
        raise ValueError("prediction model/topology/profile differs from native GB300 observations")
    model = RustForwardPassPerfModel.from_native(config)
    rows = compare_cases(
        observations["cases"],
        model.estimate_forward_pass_time_ms,
        forward_model=config.get("forward_model", "op_level"),
    )
    report = {
        "schema": "dsv41.forward.comparison.v1",
        "observed_target": BOUNDARY,
        "prediction_input_origin": "frozen logical workload manifest, not observed FPM timing",
        "execution_profile": observations["execution_profile"],
        "database_mode": config["database_mode"],
        "forward_model": config.get("forward_model", "op_level"),
        "correction_fitting": False,
        "observation_sha256": file_hash(args.observations),
        "heldout_plan_sha256": file_hash(args.heldout_plan),
        "prediction_config_sha256": file_hash(args.prediction_config),
        "prediction_sources": {
            "model_python_sha256": file_hash(deepseek_v41.__file__),
            "engine_python_sha256": file_hash(engine.__file__),
            "native_extension_sha256": file_hash(native.__file__),
        },
        "summary": error_summary(rows),
        "by_phase": {
            phase: error_summary([r for r in rows if r["phase"] == phase]) for phase in ("context", "generation")
        },
        "cases": rows,
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
