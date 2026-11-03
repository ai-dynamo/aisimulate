# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise actual candidate-native and prior-main Rust configuration readers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixtures():
    base = {
        "model": "compatibility/model",
        "system": "compatibility_system",
        "backend": "vllm",
        "worker_type": "decode",
    }
    cases = []
    for mode in ("auto", "op_level", "fpm_interpolation", "fpm_regression"):
        for explicit in (False, True):
            config = {**base, "estimation_mode": mode}
            if explicit:
                config["estimator_config"] = {"fpm_learned": {}}
            cases.append(
                {
                    "name": f"{mode}-{'explicit-default' if explicit else 'omitted'}",
                    "config": config,
                    "construct": mode == "fpm_regression",
                }
            )
    for key, value in {
        "update_every": 7,
        "recent_window": 123,
        "history_weight": 0.2,
        "mutable_tree_tail": 2,
        "derivative_refresh": "lazy",
        "split_sampling_rate": 0.5,
        "recent_max_age_observations": 5,
    }.items():
        cases.append(
            {
                "name": f"nondefault-{key}",
                "config": {
                    **base,
                    "estimation_mode": "fpm_regression",
                    "estimator_config": {"fpm_learned": {key: value}},
                },
            }
        )
    return cases


def native_child(binary):
    loader = importlib.machinery.ExtensionFileLoader("_runtime", binary)
    spec = importlib.util.spec_from_file_location("_runtime", binary, loader=loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.RustForwardPassPerfModel
    results = []
    for case in json.load(sys.stdin):
        normalized = json.loads(model.normalize_config(json.dumps(case["config"])))
        result = {"name": case["name"], "accepted": True, "normalized": normalized}
        if case.get("construct"):
            result["diagnostics"] = json.loads(model.best_available(json.dumps(normalized)).diagnostics())
        results.append(result)
    print(json.dumps({"cases": results}))


def run(command, cases):
    output = subprocess.check_output(command, input=json.dumps(cases), text=True)
    return json.loads(output)["cases"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-child")
    parser.add_argument("--candidate-runtime")
    parser.add_argument("--prior-reader")
    parser.add_argument("--output", default="compatibility-result.json")
    args = parser.parse_args()
    if args.native_child:
        native_child(args.native_child)
        return
    if not args.candidate_runtime or not args.prior_reader:
        parser.error("--candidate-runtime and --prior-reader are required")
    command = [sys.executable, __file__, "--native-child", args.candidate_runtime]
    inputs = fixtures()
    candidate = run(command, inputs)
    saved = [{**case, "config": output["normalized"]} for case, output in zip(inputs, candidate, strict=True)]
    prior = run([args.prior_reader], saved)
    restored = run(command, saved)
    results = []
    for case, current, old, reloaded in zip(inputs, candidate, prior, restored, strict=True):
        assert case["name"] == current["name"] == old["name"] == reloaded["name"]
        assert current["accepted"] and reloaded["accepted"]
        assert current["normalized"] == reloaded["normalized"]
        controls = current["normalized"]["estimator_config"]
        nondefault = case["name"].startswith("nondefault-")
        row = {"name": case["name"], "prior_accepted": old["accepted"], "candidate_roundtrip_equal": True}
        if nondefault:
            expected = case["config"]["estimator_config"]["fpm_learned"]
            assert all(controls["fpm_learned"][key] == value for key, value in expected.items())
            assert not old["accepted"] and "unknown field `fpm_learned`" in old["error"]
            row.update(preserved_override=expected, prior_error=old["error"])
        else:
            assert "fpm_learned" not in controls and old["accepted"]
            row["default_learned_block_omitted"] = True
        if case.get("construct"):
            for output in (current, old, reloaded):
                assert output["diagnostics"]["source"] == "fallback_regression"
                assert output["diagnostics"]["readiness"] == "insufficient_data"
            row["regression_constructed_in_both_readers"] = True
        row["normalized_config_sha256"] = hashlib.sha256(
            json.dumps(current["normalized"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        results.append(row)
    report = {
        "schema": "pr533-configuration-compatibility-v1",
        "summary": {
            "default_configurations_accepted_by_prior": sum(row["prior_accepted"] for row in results),
            "nondefault_controls_preserved_and_rejected_by_prior": sum("preserved_override" in row for row in results),
            "exact_candidate_roundtrips": len(results),
        },
        "binary_sha256": {"candidate": digest(args.candidate_runtime), "prior_main_reader": digest(args.prior_reader)},
        "script_sha256": digest(__file__),
        "fixture_inputs_sha256": hashlib.sha256(
            json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "cases": results,
    }
    assert tuple(report["summary"].values()) == (8, 7, 15)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
