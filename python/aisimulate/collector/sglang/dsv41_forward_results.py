# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Admit independent native benchmark-forward observations without module rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from collector.sglang.dsv41_workloads import freeze_workloads

BOUNDARY = "sglang_one_batch_synchronized_wall_including_prepare_forward_sample"
IDENTITY = ("source_sha256", "config_sha256", "runtime_digest", "execution_profile")


def aggregate_forward_results(attempt: Path, *, tp_size: int = 4) -> dict:
    """Require the complete frozen plan, then retain rank maxima per repetition."""
    if not (attempt / "COMPLETE").is_file():
        raise ValueError("native forward attempt has no completion receipt")
    if list(attempt.glob("rank-*.jsonl")) or list(attempt.glob("baseline-rank-*.jsonl")):
        raise ValueError("heldout forward attempt contains module or baseline measurements")
    plan = json.loads((attempt / "workload-plan.json").read_text())
    if plan != freeze_workloads(plan["source_payload"]):
        raise ValueError("forward plan differs from frozen source")
    contract = json.loads((attempt / "execution-contract.json").read_text())
    if (
        contract["mode"] != "native_benchmark_forward"
        or contract["component_recorder"] is not False
        or contract["timing_boundary"] != BOUNDARY
        or contract["warmup"] < 1
        or contract["iterations"] < 3
    ):
        raise ValueError("independent forward timing contract is not qualified")
    argv = contract["native_cli_args"]
    for flag in (
        "--disable-custom-all-reduce",
        "--enforce-disable-flashinfer-allreduce-fusion",
        "--disable-shared-experts-fusion",
    ):
        if flag not in argv:
            raise ValueError("forward collective/shared-expert contract is not qualified")
    for flag in ("--cuda-graph-backend-decode", "--cuda-graph-backend-prefill"):
        if flag not in argv or argv[argv.index(flag) + 1] != "disabled":
            raise ValueError("forward eager execution is not qualified")
    sources = json.loads((attempt / "source_hashes.json").read_text())
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    inputs = json.loads((attempt / "input_provenance.json").read_text())
    if inputs["source_sha256"] != digest or inputs["source"] != "tokenizer_text":
        raise ValueError("native source or text provenance mismatch")
    provenance = {key: inputs[key] for key in IDENTITY}
    bounded = provenance["execution_profile"] == "decoder_bounded"
    if provenance["execution_profile"] not in ("full", "decoder_bounded") or bounded != (
        "--enable-decoder-swa-bounded-replay" in argv
    ):
        raise ValueError("native decoder profile mismatch")
    paths = sorted(attempt.glob("forward-rank-*.jsonl"))
    if {p.name for p in paths} != {f"forward-rank-{rank}.jsonl" for rank in range(tp_size)}:
        raise ValueError("missing or unexpected forward rank files")
    case_keys = {tuple(c[k] for k in ("phase", "batch_size", "query", "prefix")): c for c in plan["cases"]}
    groups = defaultdict(dict)
    for path in paths:
        file_rank = int(path.stem.rsplit("-", 1)[1])
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = tuple(row[k] for k in ("phase", "batch_size", "query", "prefix"))
            if key not in case_keys or {k: row[k] for k in IDENTITY} != provenance:
                raise ValueError("unknown forward case or mixed provenance")
            if (
                row["component_recorder"] is not False
                or row["used_cuda_graph"] is not False
                or row["finite_logits"] is not True
                or row["timing_boundary"] != BOUNDARY
                or row["real_kv"] != (row["phase"] == "generation" or row["prefix"] > 0)
            ):
                raise ValueError("forward timing/state witness is not qualified")
            if row["phase"] == "generation" and (
                row["canonical_past_kv"] != row["prefix"] or row["native_inclusive_kv"] != row["prefix"] + 1
            ):
                raise ValueError("forward decode KV axis mismatch")
            latency = row["native_benchmark_forward_ms"]
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError("invalid native benchmark latency")
            sample, rank = row["sample"], row["tp_rank"]
            if rank != file_rank or not contract["warmup"] <= sample < contract["warmup"] + contract["iterations"]:
                raise ValueError("forward rank or repetition mismatch")
            ranks = groups[(key, sample)]
            if rank in ranks:
                raise ValueError("duplicate forward rank within repetition")
            ranks[rank] = row
    expected = {
        (key, sample)
        for key in case_keys
        for sample in range(contract["warmup"], contract["warmup"] + contract["iterations"])
    }
    if set(groups) != expected:
        raise ValueError("incomplete forward plan or repetition set")
    cases = []
    for key, case in case_keys.items():
        repetitions = []
        for sample in range(contract["warmup"], contract["warmup"] + contract["iterations"]):
            ranks = groups[(key, sample)]
            if set(ranks) != set(range(tp_size)) or len({r["invocation"] for r in ranks.values()}) != 1:
                raise ValueError("incomplete or inconsistent forward invocation ranks")
            repetitions.append(max(r["native_benchmark_forward_ms"] for r in ranks.values()))
        cases.append(case | {"rank_max_ms": repetitions, "median_ms": statistics.median(repetitions)})
    return {
        "schema_version": 1,
        "timing_boundary": BOUNDARY,
        "source_sha256": digest,
        "plan_sha256": plan["source_sha256"],
        "input_provenance": inputs,
        "execution_profile": provenance["execution_profile"],
        "component_recorder": False,
        "used_cuda_graph": False,
        "rank_count": tp_size,
        "warmup": contract["warmup"],
        "iterations": contract["iterations"],
        "case_count": len(cases),
        "cases": cases,
    }


def forward_admission_report(attempt: Path) -> dict:
    """Preserve rejected-attempt evidence; never turn partial rows into completion."""
    try:
        return {"status": "accepted", **aggregate_forward_results(attempt)}
    except (ValueError, KeyError, IndexError, OSError, json.JSONDecodeError) as error:
        failures = []
        for path in sorted(attempt.glob("workloads-rank-*.jsonl")):
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    failures.append({"file": path.name, "status": "malformed_progress_record"})
                    continue
                if row.get("status") != "passed":
                    failures.append(row)
        missing_observations = []
        try:
            plan = json.loads((attempt / "workload-plan.json").read_text())
            contract = json.loads((attempt / "execution-contract.json").read_text())
            observed = defaultdict(set)
            for path in sorted(attempt.glob("forward-rank-*.jsonl")):
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    key = tuple(row[k] for k in ("phase", "batch_size", "query", "prefix", "sample"))
                    observed[key].add(row["tp_rank"])
            for case in plan["cases"]:
                geometry = tuple(case[k] for k in ("phase", "batch_size", "query", "prefix"))
                for sample in range(contract["warmup"], contract["warmup"] + contract["iterations"]):
                    missing = sorted(set(range(4)) - observed[(*geometry, sample)])
                    if missing:
                        missing_observations.append(
                            {"case_id": case["case_id"], "sample": sample, "missing_ranks": missing}
                        )
        except (ValueError, KeyError, OSError, json.JSONDecodeError):
            # Keep the primary admission error when even the inventory is unreadable.
            missing_observations = None
        return {
            "status": "rejected",
            "complete": False,
            "admission_error": str(error),
            "missing_rank_files": [
                f"forward-rank-{rank}.jsonl"
                for rank in range(4)
                if not (attempt / f"forward-rank-{rank}.jsonl").is_file()
            ],
            "failed_workloads": failures,
            "missing_observations": missing_observations,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = forward_admission_report(args.attempt)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "accepted":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
