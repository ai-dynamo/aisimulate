#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay published SOL geometries with an explicitly identified native build.

HTTP metric definitions follow AISimulate compare_e2e.py at df04d9edec7efc1a97dbf67782c026a94f4b0761;
see README.md for provenance and sampling limits. This does not repeat raw admission.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import gzip
import hashlib
import io
import importlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
NATIVE_SHA256 = "eaa67bc03d410356b767f9fd285452613ae3a4c0e9f83734c751590d667b686b"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_bytes(path):
    manifest = json.loads((ROOT / "artifact-hashes.json").read_text())
    relative = str(path.relative_to(ROOT))
    data = path.read_bytes()
    require(
        hashlib.sha256(data).hexdigest() == manifest[relative]["sha256"],
        "published input hash mismatch",
    )
    return data


def config_for(name):
    config = json.loads(checked_bytes(ROOT / (name + ".json")))
    require(
        config["database_mode"] == "SOL" and config["forward_model"] == "op_level",
        "requires SOL/op_level",
    )
    require(
        "activation_dtype" not in config and "fpm_fmha_dtype" not in config,
        "unexpected precision override",
    )
    config["systems_path"] = str((ROOT / config["systems_path"]).resolve(strict=True))
    for relative, expected in json.loads(checked_bytes(ROOT / "systems-pins.json"))[
        name
    ].items():
        require(
            digest(Path(config["systems_path"]) / relative) == expected["sha256"],
            "system overlay changed",
        )
    return config


def provider(config):
    return {
        "model": config["model_name"],
        "backend": config["backend"],
        "system": config["system_name"],
        "backend_version": config["backend_version"],
        "tp": 4,
        "pp": 1,
        "attention_dp": 1,
        "moe_tp_size": 4,
        "moe_ep_size": 1,
        "nextn": 0,
        "decoder_replay": config["decoder_replay"],
        "database_mode": "SOL",
        "enable_shared_layer": False,
        "strict_provenance": True,
        "systems_path": config["systems_path"],
        "forward_model": "op_level",
    }


def current_spec(row, config):
    require(
        row["recovery_status"] == "exact_original_spec_sha256",
        "original replay input unavailable",
    )
    spec = deepcopy(row["replay_spec"])
    require(
        hashlib.sha256(canonical(spec).encode()).hexdigest()
        == row["replay_spec_sha256"],
        "original ReplaySpec hash mismatch",
    )
    spec["engine"]["rank"]["timing_model"]["config"] = provider(config)
    return spec


def http_metrics(report, request_ids, cohort_start_ms):
    records = report["per_request"]
    by_id = {r["request_id"]: r for r in records}
    require(
        len(by_id) == len(records) and set(request_ids).issubset(by_id),
        "missing/duplicate request",
    )
    selected = [by_id[rid] for rid in request_ids]
    require(
        all(r["terminal_status"] == "completed" for r in selected), "incomplete request"
    )
    duration = max(r["terminal_time_ms"] for r in selected) - cohort_start_ms
    require(math.isfinite(duration) and duration > 0, "invalid completion time")
    metrics = {
        "ttft_ms": statistics.mean(r["ttft_ms"] for r in selected),
        "request_latency_ms": statistics.mean(
            r["terminal_time_ms"] - r["arrival_time_ms"] for r in selected
        ),
        "last_token_latency_ms": statistics.mean(
            r["last_token_ms"] - r["arrival_time_ms"] for r in selected
        ),
        "output_tokens_per_second": 1000
        * sum(r["output_length"] for r in selected)
        / duration,
    }
    if all(r["output_length"] > 1 for r in selected):
        metrics["average_tpot_ms"] = statistics.mean(
            (r["last_token_ms"] - r["first_token_ms"]) / (r["output_length"] - 1)
            for r in selected
        )
        metrics["exact_itl_ms"] = statistics.mean(r["itl_ms"] for r in selected)
    require(
        all(
            type(v) in (int, float) and math.isfinite(v) and v > 0
            for v in metrics.values()
        ),
        "invalid metric",
    )
    return metrics


def qualify_python_sources():
    identity = json.loads(checked_bytes(ROOT / "predictor-identity.json"))
    actual = {}
    for name, expected in identity["modules"].items():
        if name == "aisimulate._runtime":
            continue
        module = importlib.import_module(name)
        actual[name] = digest(module.__file__)
        require(
            actual[name] == expected["sha256"],
            "Python predictor source mismatch: " + name,
        )
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--native-sha256",
        default=NATIVE_SHA256,
        help="SHA of an explicitly rebuilt current native extension; defaults to the recorded build",
    )
    args = parser.parse_args()
    require(not args.output.exists(), "output must be new")
    from aisimulate import _runtime as native
    from aiconfigurator_core.sdk.rust_engine_step import RustForwardPassPerfModel

    require(
        digest(native.__file__) == args.native_sha256,
        "native build does not match explicit identity",
    )
    before = digest(native.__file__)
    python_sources = qualify_python_sources()
    results = []
    if args.dataset in ("gb300-retained-e2e-core-off", "gb300-retained-e2e-core-on"):
        scope = args.dataset.rsplit("-", 1)[1]
        packet = json.loads(
            gzip.decompress(
                checked_bytes(ROOT / f"core-{scope}-replay-workloads.json.gz")
            )
        )
        config = config_for(f"gb300-{scope}-sol")
        for ordinal, row in enumerate(packet["cohorts"]):
            result = {"row": ordinal, "observed": row["observed"]}
            if row["recovery_status"] != "exact_original_spec_sha256":
                result.update(
                    status="prediction_unavailable",
                    failure_type="UnavailableOriginalReplayInput",
                )
            else:
                spec = current_spec(row, config)
                try:
                    report = json.loads(native.run_replay_json(canonical(spec)))
                    result.update(
                        status="predicted",
                        prediction=http_metrics(
                            report, row["request_ids"], row["cohort_start_ms"]
                        ),
                    )
                except Exception as error:
                    result.update(
                        status="prediction_unavailable",
                        failure_type=type(error).__name__,
                        failure=str(error),
                    )
            results.append(result)
    else:
        rows = list(
            csv.DictReader(
                io.StringIO(
                    gzip.decompress(
                        checked_bytes(ROOT / "observations-and-predictions.csv.gz")
                    ).decode()
                )
            )
        )
        selected = [
            row
            for row in rows
            if row["dataset"] == args.dataset and row["metric"] == "forward_ms"
        ]
        require(bool(selected), "dataset has no published native input")
        require(len({r["config"] for r in selected}) == 1, "mixed config")
        model = RustForwardPassPerfModel.from_native(config_for(selected[0]["config"]))
        for row in selected:
            result = {
                "row": int(row["row"]),
                "population": row["population"],
                "observed_ms": row["observed"],
            }
            if (
                row["failure_type"] == "RetainedSourceQualificationError"
                or row["status"] == "unmatched_observation"
            ):
                result.update(status=row["status"], failure_type=row["failure_type"])
            else:
                metrics = json.loads(row["prediction_input"])
                require(
                    "wall_time" not in metrics
                    and set(metrics) == {"version", "scheduled_requests"},
                    "unexpected predictor input",
                )
                try:
                    prediction = model.estimate_forward_pass_time_ms(metrics)
                    require(
                        math.isfinite(prediction) and prediction > 0,
                        "invalid prediction",
                    )
                    result.update(status="predicted", predicted_ms=prediction)
                except Exception as error:
                    result.update(
                        status="prediction_unavailable",
                        failure_type=type(error).__name__,
                        failure=str(error),
                    )
            results.append(result)
    require(before == digest(native.__file__), "native changed during replay")
    require(
        python_sources == qualify_python_sources(),
        "Python predictor changed during replay",
    )
    with args.output.open("x") as stream:
        json.dump(
            {
                "dataset": args.dataset,
                "native_extension_sha256": before,
                "new_raw_admission": False,
                "new_gpu_observations": False,
                "correction_fitting": False,
                "results": results,
            },
            stream,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")


if __name__ == "__main__":
    main()
