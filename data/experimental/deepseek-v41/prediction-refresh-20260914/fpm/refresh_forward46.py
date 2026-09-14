# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Query the existing GB300 46-configuration observations with current FPM.

The original observation/plan gate is preserved. The FP8 FPM table selector is
qualified separately because the older shared op-level config allowlist predates
that selector. The actual predictor receives the original complete FPM config.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from refresh_reported_native import read, require, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--input-task", type=Path, required=True)
    parser.add_argument("--qualifiers", type=Path, required=True)
    parser.add_argument("--profile", choices=("off", "on"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-native-sha256", required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    require(Path.cwd() == repo, "run from current repository")
    require(
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == args.expected_head, "head differs"
    )
    tools = repo / "data/experimental/deepseek-v41/verification-plan"
    sys.path[:0] = [str(args.qualifiers), str(tools), str(repo / "python/aisimulate/src")]
    from collector.sglang import dsv41_forward_results, dsv41_workloads
    from compare_forward import (
        compare_cases,
        error_summary,
        resolved_model_identity,
        system_data_identity,
        validate_prediction_contract,
    )

    import aisimulate._runtime as native
    from aiconfigurator_core.sdk.rust_engine_step import RustForwardPassPerfModel

    require(Path(native.__file__).resolve().is_relative_to(repo), "foreign native binary")
    require(sha(native.__file__) == args.expected_native_sha256, "native differs")
    task = read(args.input_task)
    paths = {key: Path(task["inputs"][key]) for key in ("observations", "heldout-plan")}
    for key, path in paths.items():
        require(sha(path) == task["input_pins"][key]["sha256"], "original input differs")
    observations = read(paths["observations"])
    plan = dsv41_workloads.freeze_workloads(read(paths["heldout-plan"]))
    require(
        observations["status"] == "accepted" and observations["timing_boundary"] == dsv41_forward_results.BOUNDARY,
        "original observation not qualified",
    )
    require(
        observations["plan_sha256"] == plan["source_sha256"] and len(observations["cases"]) == len(plan["cases"]) == 46,
        "original plan differs",
    )
    for observed, planned in zip(observations["cases"], plan["cases"], strict=True):
        require(all(observed.get(key) == value for key, value in planned.items()), "original case differs")
    config_path = (
        repo
        / "data/experimental/deepseek-v41/gb300-fpm"
        / f"{args.profile}-union-v3-tracewait2/reports/core/prediction-config.json"
    )
    config = read(config_path)
    require(
        config["fpm_fmha_dtype"] == "fp8" and config["forward_model"] == "fpm" and config["database_mode"] == "SILICON",
        "FPM selector/config differs",
    )
    common = dict(config)
    common.pop("fpm_fmha_dtype")
    validate_prediction_contract(common, observations)
    model_identity = resolved_model_identity(config, observations)
    identity = system_data_identity(config["systems_path"])
    model = RustForwardPassPerfModel.from_native(config)
    rows = compare_cases(observations["cases"], model.estimate_forward_pass_time_ms, forward_model="fpm")
    require(identity == system_data_identity(config["systems_path"]), "calibration differs")
    require(model_identity == resolved_model_identity(config, observations), "checkpoint differs")
    for key, path in paths.items():
        require(sha(path) == task["input_pins"][key]["sha256"], "input changed")
    write(
        args.output,
        {
            "schema": "dsv41.current-source.fpm-forward46-refresh.v1",
            "scope": args.profile,
            "predictor_commit": args.expected_head,
            "native_extension_sha256": args.expected_native_sha256,
            "input_sha256": {key: sha(path) for key, path in paths.items()},
            "source_sha256": sha(__file__),
            "qualifier_sources": {
                key: sha(module.__file__)
                for key, module in (
                    ("dsv41_forward_results", dsv41_forward_results),
                    ("dsv41_workloads", dsv41_workloads),
                )
            },
            "prediction_config": config,
            "prediction_config_sha256": sha(config_path),
            "systems_identity": identity,
            "resolved_model_identity": model_identity,
            "summary": error_summary(rows),
            "cases": rows,
            "correction_fitting": False,
            "new_observations": False,
            "observed_target": dsv41_forward_results.BOUNDARY,
            "notes": [
                "All 46 configurations retained; observed median of rank maxima across original repetitions.",
                "No FPM refit or calibration self-query. Current Decoder ON API guard failures remain unavailable.",
            ],
        },
    )


if __name__ == "__main__":
    main()
