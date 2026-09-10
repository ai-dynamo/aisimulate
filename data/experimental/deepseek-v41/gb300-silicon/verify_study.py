# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rebuild calibration tables and independently validate native forward evidence."""

import argparse
import gzip
import json
import shutil
import statistics
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
from collector.sglang.collect_dsv41_module import (
    aggregate_baseline_records,
    aggregate_rank_records,
)
from collector.sglang.dsv41_contract import build_manifest, write_parquet
from collector.sglang.dsv41_forward_results import forward_admission_report
from collector.sglang.dsv41_workloads import projected_keys

import aiconfigurator_core._aiconfigurator_core as native
from aiconfigurator_core.sdk.engine import _evaluate_single_op
from aiconfigurator_core.sdk.perf_database import PerfDatabase


def unpack_evidence(source, destination):
    destination.mkdir()
    for path in source.iterdir():
        if path.suffix == ".gz":
            (destination / path.name.removesuffix(".gz")).write_bytes(gzip.decompress(path.read_bytes()))
        else:
            shutil.copyfile(path, destination / path.name)


def verify(profile):
    root = Path(__file__).resolve().parent / "study" / profile
    systems = root / "systems"
    with tempfile.TemporaryDirectory(prefix="dsv41-study-") as temporary:
        scratch = Path(temporary)
        calibration, heldout = scratch / "calibration", scratch / "heldout"
        unpack_evidence(root / "calibration/evidence", calibration)
        unpack_evidence(root / "heldout/evidence", heldout)
        assert (calibration / "COMPLETE").is_file()
        plan = json.loads((calibration / "workload-plan.json").read_text())
        assert len(plan["cases"]) == 126
        for rank in range(4):
            progress = [
                json.loads(line) for line in (calibration / f"workloads-rank-{rank}.jsonl").read_text().splitlines()
            ]
            assert len(progress) == 504
            assert {(r["case_index"], r["sample"]) for r in progress} == {(i, s) for i in range(126) for s in range(4)}
            assert all(r["status"] == "passed" and r["tp_rank"] == rank for r in progress)
        rows = aggregate_rank_records(sorted(calibration.glob("rank-*.jsonl")), 4)
        expected = set().union(
            *(projected_keys(build_manifest(4, profile == "decoder_bounded"), c) for c in plan["cases"])
        )
        assert {tuple(r[k] for k in ("component", "geometry", "batch_size", "prefix", "x")) for r in rows} == expected
        rebuilt = scratch / "module.parquet"
        write_parquet(rows, rebuilt)
        table = systems / "data/gb300/dsv41/sglang/0.0.0.dev0/dsv41_module_perf.parquet"
        assert pq.read_table(rebuilt).equals(pq.read_table(table))
        baselines = aggregate_baseline_records(sorted(calibration.glob("baseline-rank-*.jsonl")), 4)
        for kind, values in baselines.items():
            relative = (
                f"{kind}/sglang/0.0.0.dev0/{kind}_perf.parquet"
                if kind != "nccl"
                else "comm/nccl/2.29.7/nccl_perf.parquet"
            )
            assert pq.read_table(systems / "data/gb300" / relative).to_pylist() == values
        forward = forward_admission_report(heldout)
        assert forward["status"] == "accepted" and forward["case_count"] == 38
        assert forward == json.loads((root / "heldout/forward-results.json").read_text())
        precision_root = root / "precision-v2"
        if precision_root.exists():
            precision_raw = scratch / "precision-v2"
            unpack_evidence(precision_root / "evidence", precision_raw)
            precision = forward_admission_report(precision_raw)
            assert precision["status"] == "accepted" and precision["case_count"] == 38
            assert precision["iterations"] == 10 and precision["warmup"] == 1
            assert precision == json.loads((precision_root / "forward-results.json").read_text())
            for key in (
                "source_sha256",
                "plan_sha256",
                "input_provenance",
                "timing_boundary",
                "component_recorder",
                "used_cuda_graph",
                "execution_profile",
            ):
                assert precision[key] == forward[key]
            comparison = json.loads((precision_root / "comparison-to-v1.json").read_text())
            assert comparison["case_count"] == 38 and comparison["profile"] == profile
            assert comparison["original_repeats"] == 3 and comparison["precision_repeats"] == 10
            assert comparison["calibration_unchanged"] and comparison["original_observations_retained"]
            assert comparison["pooling"] is False and comparison["plan_sha256"] == forward["plan_sha256"]
            assert len(comparison["cases"]) == 38
            for old, new, row in zip(forward["cases"], precision["cases"], comparison["cases"], strict=True):
                geometry = {k: v for k, v in old.items() if k not in ("rank_max_ms", "median_ms")}
                assert geometry == {k: v for k, v in new.items() if k not in ("rank_max_ms", "median_ms")}
                assert all(row[k] == value for k, value in geometry.items())
                for label, result in (("original", old), ("precision", new)):
                    assert row[f"{label}_rank_max_ms"] == result["rank_max_ms"]
                    assert row[f"{label}_median_ms"] == statistics.median(result["rank_max_ms"])
                    assert row[f"{label}_cv_percent"] == (
                        100 * statistics.stdev(result["rank_max_ms"]) / statistics.mean(result["rank_max_ms"])
                    )
                assert row["median_change_percent"] == 100 * (new["median_ms"] / old["median_ms"] - 1)
    database = PerfDatabase(
        "gb300",
        "sglang",
        "0.0.0.dev0",
        str(systems.resolve()),
        database_mode="SILICON",
        shared_layer=False,
        strict_provenance=True,
    )
    for row in rows:
        variant = (
            "Dsv41"
            + {
                "attention": "Attention",
                "mhc": "Mhc",
                "linear": "Linear",
                "engram": "Engram",
            }[row["component"]]
        )
        geometry = json.loads(row["geometry"])
        op = native.op_from_spec_json(json.dumps({variant: geometry | {"name": "study_point_check"}}))
        result = _evaluate_single_op(
            database,
            op,
            is_context=geometry.get("is_context", True),
            batch_size=row["batch_size"],
            s=row["x"],
            prefix=row["prefix"],
            x=row["x"],
        )
        assert result.source == "silicon" and abs(float(result) - row["latency"]) < 1e-6
    print(
        json.dumps(
            {
                "profile": profile,
                "strict_module_points": len(rows),
                "independent_forward_cases": 38,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", nargs="*")
    args = parser.parse_args()
    for profile in args.profiles or ["full", "decoder_bounded"]:
        if profile not in ("full", "decoder_bounded"):
            parser.error(f"unknown profile: {profile}")
        verify(profile)


if __name__ == "__main__":
    main()
