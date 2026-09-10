# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Verify real calibration-table consumption; this is not an accuracy holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from aiconfigurator_core.sdk.rust_engine_step import RustForwardPassPerfModel

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / "prediction-config.json").read_text())
    config["systems_path"] = str(ROOT / "systems")
    model = RustForwardPassPerfModel.from_native(config)
    files = list((ROOT / "systems").rglob("fpm_forward_perf.parquet"))
    if len(files) != 1:
        raise ValueError("one explicit calibration table is required")
    table = files[0]
    before = digest(table)
    points = pq.read_table(table).to_pylist()
    if (
        len(points) != 126
        or len(
            {
                (r["workload_kind"], r["batch_size"], r["total_prefill_tokens"], r["total_kv_read_tokens"])
                for r in points
            }
        )
        != 126
    ):
        raise ValueError("calibration geometry count is incomplete or duplicated")
    results = []
    for row in points:
        prefill = row["workload_kind"] == "prefill"
        metrics = {
            "version": 1,
            "scheduled_requests": {
                "num_prefill_requests": row["batch_size"] if prefill else 0,
                "sum_prefill_tokens": row["total_prefill_tokens"] if prefill else 0,
                "sum_prefill_kv_tokens": row["total_kv_read_tokens"] if prefill else 0,
                "num_decode_requests": 0 if prefill else row["batch_size"],
                "sum_decode_kv_tokens": 0 if prefill else row["total_kv_read_tokens"],
                "var_prefill_length": 0.0,
                "var_decode_kv_tokens": 0.0,
            },
        }
        measured = row["latency_ms"]
        loaded = model.estimate_forward_pass_time_ms(metrics)
        if loaded != measured:
            raise ValueError("native whole-forward consumer differs from the exact table cell")
        results.append(
            {
                "phase": row["workload_kind"],
                "metrics": metrics,
                "measured_table_ms": measured,
                "native_consumer_ms": loaded,
            }
        )
    incompatible = dict(config, activation_dtype="bfloat16")
    wrong_model = RustForwardPassPerfModel.from_native(incompatible)
    try:
        wrong_model.estimate_forward_pass_time_ms(results[0]["metrics"])
    except Exception as error:
        if "No FPM cell" not in str(error):
            raise
        rejected = type(error).__name__
    else:
        raise ValueError("incompatible FMHA precision reused the measured table")
    if digest(table) != before:
        raise ValueError("calibration input changed during verification")
    import aisimulate._runtime as native

    result = {
        "schema": "dsv41.fpm.calibration.consumer.v1",
        "valid": True,
        "role": "calibration_consumer_roundtrip",
        "formal_accuracy_claim": False,
        "points": results,
        "table_sha256": before,
        "native_extension_sha256": digest(Path(native.__file__)),
        "verification_source_sha256": digest(Path(__file__)),
        "wrong_fmha_rejected": rejected,
        "decode_axis": "whole-forward past KV; current query is not added",
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
