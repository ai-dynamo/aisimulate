# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Re-query preserved GB300 workload descriptors with a source-bound FPM build.

This is prediction refresh from published, previously qualified reports. It does
not reconstruct missing raw traces or perform a new measurement admission.
Only scheduled workload fields enter the estimator; observed and old predicted
latencies are never estimator inputs. No tuning or fallback model is used.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(gzip.compress(raw, mtime=0) if path.suffix == ".gz" else raw)


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-native-sha256", required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    require(Path.cwd() == repo, "run from the current source repository root")
    require(
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == args.expected_head,
        "predictor head differs",
    )
    require(
        not subprocess.check_output(["git", "diff", "--name-only", "HEAD", "--", "crates", "python"], text=True),
        "predictor sources are dirty",
    )
    tools = repo / "data/experimental/deepseek-v41/verification-plan"
    sys.path[:0] = [str(tools), str(repo / "python/aisimulate"), str(repo / "python/aisimulate/src")]
    from compare_trace import systems_identity, trace_summary
    from normalize_fpm import prediction_input

    import aisimulate._runtime as native
    from aiconfigurator_core.sdk.rust_engine_step import RustForwardPassPerfModel

    require(Path(native.__file__).resolve().is_relative_to(repo), "native binary is outside current repository")
    require(sha(native.__file__) == args.expected_native_sha256, "native binary differs")
    source = repo / "data/experimental/deepseek-v41/fpm-aggregate-admission-v1/audit.py"
    spec = importlib.util.spec_from_file_location("preserved_inputs", source)
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    frozen_audit = frozen.audit(repo)
    pins = {str(Path(__file__).resolve().relative_to(repo)): sha(__file__), str(source.relative_to(repo)): sha(source)}
    tracked = subprocess.check_output(
        ["git", "ls-files", "crates", "python/aisimulate/src", str(tools.relative_to(repo))], text=True
    ).splitlines()
    for name in tracked:
        if name.endswith((".rs", ".py", ".toml", ".json", ".yaml")):
            pins[name] = sha(repo / name)
    outputs = []
    args.output_dir.mkdir()
    for scope, expected_count in frozen.EXPECTED_COUNTS.items():
        profile, population = scope.split("/")
        base = repo / f"data/experimental/deepseek-v41/gb300-fpm/{profile}-union-v3-tracewait2"
        input_path = base / f"reports/{population}/{'segments' if scope == 'on/core' else 'trace'}.json.gz"
        config_path = base / f"reports/{population}/prediction-config.json"
        config = read(config_path)
        require(config["system_name"] == "gb300" and config["forward_model"] == "fpm", "wrong predictor")
        require(config["decoder_replay"] is (profile == "on"), "wrong execution profile")
        require(config["fpm_fmha_dtype"] == "fp8" and "activation_dtype" not in config, "precision changed")
        for path in [input_path, config_path, *(base / "systems").rglob("*")]:
            if path.is_file():
                pins[str(path.relative_to(repo))] = sha(path)
        old = read(input_path)
        original_cohorts = (
            [(segment["physical_run_id"], c) for segment in old["segments"] for c in segment["trace_cases"]]
            if scope == "on/core"
            else [(old["run_id"], c) for c in old["cohorts"]]
        )
        identity = systems_identity(config)
        model = RustForwardPassPerfModel.from_native(config)
        started = time.time()
        cohorts, all_rows, seen = [], [], set()
        for run_id, cohort in original_cohorts:
            result = {k: cohort[k] for k in ("cohort_id", "purpose", "trial_index", "trial_seed")}
            result.update(physical_run_id=run_id, intervals=[])
            for row in cohort["intervals"]:
                key = run_id, row["counter_id"]
                require(key not in seen, "duplicate physical native counter")
                seen.add(key)
                query, bridge = prediction_input(
                    {"scheduled_requests": row["native_scheduled_requests"]},
                    producer_semantics="sglang_inclusive_query",
                    target_axis="whole_forward_past_kv",
                )
                require(bridge == row["axis_bridge"], "published native axis bridge differs")
                require(query["scheduled_requests"] == row["scheduled_requests"], "published query differs")
                variance = row["variance_bridge"]
                require(
                    variance["native_prefill_axis"] == variance["prediction_prefill_axis"] == "current_query_tokens"
                    and variance["native_prefill_variance"]
                    == variance["prediction_prefill_variance"]
                    == query["scheduled_requests"]["var_prefill_length"],
                    "published variance bridge differs",
                )
                require(
                    math.isclose(
                        variance["native_prefill_variance"],
                        variance["current_query_variance"],
                        rel_tol=1e-6,
                        abs_tol=1e-6,
                    ),
                    "source-qualified query variance differs beyond original audit tolerance",
                )
                require(math.isfinite(row["observed_ms"]) and row["observed_ms"] > 0, "invalid observation")
                item = {
                    k: row[k]
                    for k in (
                        "counter_id",
                        "dispatch_id",
                        "phase",
                        "observed_ms",
                        "axis_bridge",
                        "variance_bridge",
                        "native_scheduled_requests",
                        "scheduled_requests",
                    )
                }
                item["historical_status"] = row["status"]
                if "failure" in row:
                    item["historical_failure"] = row["failure"]
                try:
                    predicted = model.estimate_forward_pass_time_ms(query)
                    require(
                        type(predicted) in (int, float) and math.isfinite(predicted) and predicted > 0,
                        "no positive finite current prediction",
                    )
                except Exception as error:
                    item.update(
                        status="prediction_unavailable",
                        failure_type=type(error).__name__,
                        failure=re.sub(r"(?<![\w:])/(?:[^\s'\"()]+)", "<path>", str(error)),
                    )
                else:
                    item.update(
                        status="predicted",
                        predicted_ms=predicted,
                        signed_error_percent=100 * (predicted / row["observed_ms"] - 1),
                    )
                result["intervals"].append(item)
                all_rows.append(item)
            result["summary"] = trace_summary(result["intervals"])
            cohorts.append(result)
        require(len(all_rows) == expected_count, "denominator changed")
        require(identity == systems_identity(config), "calibration changed")
        output = args.output_dir / f"gb300-{profile}-{population}-native.json.gz"
        report = {
            "schema": "dsv41.current-source.reported-native-refresh.v1",
            "scope": scope,
            "predictor_commit": args.expected_head,
            "native_extension_sha256": args.expected_native_sha256,
            "input_file": str(input_path.relative_to(repo)),
            "input_sha256": sha(input_path),
            "prediction_config": config,
            "systems_identity": identity,
            "summary": trace_summary(all_rows),
            "cohorts": cohorts,
            "failure_counts": dict(Counter(r.get("failure", "") for r in all_rows if r["status"] != "predicted")),
            "started_unix": started,
            "finished_unix": time.time(),
            "new_measurement_admission": False,
            "new_observations": False,
            "correction_fitting": False,
            "confidence_intervals": None,
            "limits": [
                "Native intervals are correlated; no independent-interval CI is claimed.",
                "All rows retained, including current and historical unsupported predictions.",
                "Previously qualified published descriptors are re-queried; unavailable raw files are not re-admitted.",
                "Historical prediction values are not used as predictor inputs or fallback outputs.",
            ],
        }
        write(output, report)
        outputs.append({"scope": scope, "file": output.name, "sha256": sha(output), "summary": report["summary"]})
        print(json.dumps(outputs[-1]), flush=True)
    require(all(sha(repo / path) == value for path, value in pins.items()), "source/input changed")
    require(sha(native.__file__) == args.expected_native_sha256, "native changed")
    write(
        args.output_dir / "receipt.json",
        {
            "schema": "dsv41.current-source.reported-native-refresh.receipt.v1",
            "valid": True,
            "predictor_commit": args.expected_head,
            "native_sha256": args.expected_native_sha256,
            "files_sha256": pins,
            "outputs": outputs,
            "original_input_predicate_audit": frozen_audit,
            "source_and_inputs_unchanged": True,
        },
    )


if __name__ == "__main__":
    main()
