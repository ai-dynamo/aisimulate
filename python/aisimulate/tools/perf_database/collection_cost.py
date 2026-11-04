#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare collection inventories and account for collection-only pilot GPU time.

Inventory inputs are the CSVs written by collect.py --plan-only --plan-table.
No framework imports or device-memory predictions are needed for comparison.
"""

from __future__ import annotations

import argparse
import ast
import csv
import fcntl
import hashlib
import itertools
import json
import math
import os
import re
import signal
import subprocess
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path


def case_key(row: dict) -> str:
    """The planner's deduplicated case identity, including dispatch and model semantics.

    Plan tables contain case fields, not arbitrary run labels. In particular a
    module model_path selects checkpoint/runtime semantics and cannot be dropped.
    """
    return json.dumps(
        {key: value for key, value in row.items() if not key.startswith("_")}, sort_keys=True, separators=(",", ":")
    )


def inner_cells(row: dict) -> list[dict]:
    """Expand grouped MoE/KDA task axes before comparing physical work.

    The original task remains available for grouping/native invocation, but is
    not part of a physical cell's identity. An appended token anchor must not
    make all old cells appear removed and re-added.
    """
    axes = [key for key in ("num_tokens", "batch_sizes", "seq_lens") if key in row and row[key].startswith(("(", "["))]
    if not axes:
        return [row]
    values = [ast.literal_eval(row[key]) for key in axes]
    if any(not isinstance(axis, (tuple, list)) or not axis for axis in values):
        raise ValueError("grouped axes must be non-empty integer lists")
    return [
        {**row, **{key: str(value) for key, value in zip(axes, point, strict=True)}, "_task_case": row}
        for point in itertools.product(*values)
    ]


def load_tables(directory: Path) -> dict[str, dict[str, dict]]:
    tables = {}
    for path in sorted(directory.glob("*.csv")):
        with path.open(newline="") as handle:
            rows = [cell for row in csv.DictReader(handle) for cell in inner_cells(row)]
        keyed = {case_key(row): row for row in rows}
        if len(keyed) != len(rows):
            raise ValueError(f"duplicate physical plan rows in {path}")
        tables[path.stem] = keyed
    if not tables:
        raise ValueError(f"no plan-table CSV files in {directory}")
    return tables


def compare(baseline: Path, candidate: Path) -> dict:
    before, after = load_tables(baseline), load_tables(candidate)
    if before.keys() != after.keys():
        raise ValueError("baseline and candidate must contain the same system/backend/op tables")
    result = {}
    for name in before:
        old, new = before[name], after[name]
        result[name] = {
            "baseline": len(old),
            "candidate": len(new),
            "added": [new[key] for key in sorted(new.keys() - old.keys())],
            "removed": [old[key] for key in sorted(old.keys() - new.keys())],
        }
    return result


def stratum(row: dict) -> tuple:
    """Keep different native kernels and precisions in separate timing buckets."""
    return tuple(
        (key, value)
        for key, value in sorted(row.items())
        if "dtype" in key or key in {"dispatch", "gemm_type", "moe_type", "phase"}
    )


def work_proxy(row: dict, *, dense_context: bool = False) -> float:
    """Ordering only; never convert this proxy directly into seconds."""

    def number(*names, default=1):
        return next((float(row[name]) for name in names if row.get(name)), default)

    batch = number("batch_size", "batch", "batch_sizes")
    length = number(
        "input_len", "seq_len", "seq_length", "sequence_length", "kv_cache_length", "sequence", "num_tokens", "seq_lens"
    )
    prefix = number("prefix_len", "prefix", default=0)
    heads = number("num_heads", "head_num", "local_heads")
    width = number("head_dim")
    return batch * (length + prefix) ** (2 if dense_context else 1) * heads * width


def select_anchors(difference: dict, per_stratum: int = 10) -> dict:
    """Deterministic work-quantile anchors, preserving every dtype/dispatch bucket."""
    if per_stratum < 3:
        raise ValueError("at least three anchors per stratum are required")
    result = {}
    for table, delta in difference.items():
        buckets = defaultdict(list)
        for row in delta["added"]:
            buckets[stratum(row)].append(row)
        selected = []
        for bucket in sorted(buckets):
            rows = buckets[bucket]
            ordered = sorted(
                rows,
                key=lambda row: (work_proxy(row, dense_context=table.endswith("attention_context")), case_key(row)),
            )
            indices = {round(i * (len(ordered) - 1) / (per_stratum - 1)) for i in range(per_stratum)}
            interior = sorted(
                indices - {0, len(ordered) - 1}, key=lambda i: hashlib.sha256(case_key(ordered[i]).encode()).hexdigest()
            )
            holdouts = set(interior[: min(len(interior), round(len(indices) * 0.2))])
            for i in sorted(indices):
                row = ordered[i]
                selected.append(
                    {
                        "case": row,
                        "key": hashlib.sha256(case_key(row).encode()).hexdigest(),
                        "role": "holdout" if i in holdouts else "fit",
                        "population": len(rows),
                    }
                )
        result[table] = selected
    return result


def read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@contextmanager
def ledger_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def reserve(path: Path, run_id: str, gpu_seconds: float, limit: float) -> None:
    """Reserve a hard collection timeout before concurrent pilots can launch."""
    if not math.isfinite(gpu_seconds) or gpu_seconds <= 0 or not math.isfinite(limit) or limit <= 0:
        raise ValueError("reservation and budget must be finite and positive")
    with ledger_lock(path):
        rows = read_ledger(path)
        if any(row["run_id"] == run_id for row in rows):
            raise ValueError(f"run already reserved: {run_id}")
        finished = {row["run_id"]: row["gpu_seconds"] for row in rows if row["event"] == "finished"}
        used = sum(finished.get(row["run_id"], row["gpu_seconds"]) for row in rows if row["event"] == "reserved")
        if used + gpu_seconds > limit:
            raise ValueError(f"collection budget exhausted: {used:g}+{gpu_seconds:g}>{limit:g} GPU-seconds")
        with path.open("a") as handle:
            handle.write(json.dumps({"event": "reserved", "run_id": run_id, "gpu_seconds": gpu_seconds}) + "\n")


def finish(path: Path, run_id: str, gpu_seconds: float) -> None:
    with ledger_lock(path):
        rows = read_ledger(path)
        if not any(row["run_id"] == run_id and row["event"] == "reserved" for row in rows):
            raise ValueError(f"no reservation for {run_id}")
        if any(row["run_id"] == run_id and row["event"] == "finished" for row in rows):
            raise ValueError(f"run already finished: {run_id}")
        if not math.isfinite(gpu_seconds) or gpu_seconds < 0:
            raise ValueError("actual time must be finite and nonnegative")
        with path.open("a") as handle:
            handle.write(json.dumps({"event": "finished", "run_id": run_id, "gpu_seconds": gpu_seconds}) + "\n")


def phase_timings(log: Path) -> list[dict]:
    """Read native point clocks; bench_* fields are nested, not additive phases."""
    points = []
    for line in log.read_text(errors="replace").splitlines():
        if "point timing]" not in line and " module timing]" not in line:
            continue
        phases = {key: float(value) for key, value in re.findall(r"([A-Za-z_]+)=([0-9]+(?:\.[0-9]+)?)s", line)}
        if "total" in phases:
            points.append(
                {
                    "native_log": line.strip(),
                    "phase_seconds": phases,
                    "inclusive_seconds": phases["total"],
                    "nested_phase_warning": "bench_* phases already belong to bench; do not sum twice",
                }
            )
    return points


def run(spec: dict, output: Path) -> dict:
    """Launch inside an existing GPU container; scheduler/container time stays outside."""
    output.mkdir(parents=True, exist_ok=False)
    gpus, timeout = int(spec["gpus"]), float(spec["timeout_seconds"])
    if gpus < 1 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("positive GPU count and collection timeout required")
    started = time.perf_counter()
    with (output / "collection.log").open("w") as log:
        process = subprocess.Popen(
            spec["argv"], cwd=spec["cwd"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            code = process.wait(timeout=timeout)
            status = "passed" if code == 0 else "failed"
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
            status = "timeout"
    seconds = time.perf_counter() - started
    result = {
        "spec": spec,
        "status": status,
        "exit_code": code,
        "collection_seconds": seconds,
        "collection_gpu_hours": seconds * gpus / 3600,
        "allocation_seconds_included": False,
        "native_point_timings": phase_timings(output / "collection.log"),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    diff = sub.add_parser("compare")
    diff.add_argument("--baseline", type=Path, required=True)
    diff.add_argument("--candidate", type=Path, required=True)
    diff.add_argument("--output", type=Path, required=True)
    sample = sub.add_parser("sample")
    sample.add_argument("--difference", type=Path, required=True)
    sample.add_argument("--per-stratum", type=int, default=10)
    sample.add_argument("--output", type=Path, required=True)
    execution = sub.add_parser("run")
    execution.add_argument("--spec", type=Path, required=True)
    execution.add_argument("--output", type=Path, required=True)
    execution.add_argument("--ledger", type=Path, required=True)
    execution.add_argument("--budget-gpu-hours", type=float, required=True)
    args = parser.parse_args()
    if args.command == "compare":
        payload = compare(args.baseline, args.candidate)
    elif args.command == "sample":
        payload = select_anchors(json.loads(args.difference.read_text()), args.per_stratum)
    else:
        spec = json.loads(args.spec.read_text())
        reserve(args.ledger, spec["run_id"], spec["gpus"] * (spec["timeout_seconds"] + 5), args.budget_gpu_hours * 3600)
        payload = run(spec, args.output)
        finish(args.ledger, spec["run_id"], payload["collection_gpu_hours"] * 3600)
        print(json.dumps(payload))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
