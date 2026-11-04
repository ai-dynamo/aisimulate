#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run selected native planner cases in audit-only output, preserving batch rows.

Run inside the manifest-pinned framework container, through collection_cost.py's
bounded wrapper. Dynamic imports select the registry's native implementation;
there are no replacement kernels or fabricated shapes.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import importlib
import importlib.metadata
import json
import os
import sys
import time
import traceback
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG_ROOT))


def select_plan(planned, requested: list[dict]):
    """Select exact generated cases/dispatches; refuse missing or duplicate requests."""
    wanted = {json.dumps(row, sort_keys=True) for row in requested}
    if len(wanted) != len(requested):
        raise ValueError("duplicate requested case keys")
    cases, dispatches, found = [], [], set()
    runs = planned.dispatches or [("",)] * len(planned.cases)
    for case, runners in zip(planned.cases, runs, strict=True):
        kept = []
        for runner in runners:
            key = json.dumps(
                {
                    **{name: "" if value is None else str(value) for name, value in case._asdict().items()},
                    "dispatch": runner,
                },
                sort_keys=True,
            )
            if key in wanted:
                kept.append(runner)
                found.add(key)
        if kept:
            cases.append(case)
            dispatches.append(tuple(kept))
    if found != wanted:
        raise ValueError(f"{len(wanted - found)} requested cases absent from the actual plan")
    return dataclasses.replace(planned, cases=cases, dispatches=dispatches if planned.dispatches else [])


def run(spec: dict, output: Path) -> dict:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    result = {"spec": spec, "publish_eligible": False, "allocation_seconds_included": False, "calls": []}
    try:
        actual_ref = os.environ.get("AISIM_COLLECTOR_REF")
        if actual_ref != spec["collector_commit"]:
            raise ValueError(f"collector revision mismatch: {actual_ref!r}")
        if os.environ.get("AIS_SM") or os.environ.get("AIC_SM"):
            raise ValueError("CPU SM overrides must not be set for native sampling")
        torch = importlib.import_module("torch")
        torch.cuda.set_device(0)
        result["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "sm": int("".join(map(str, torch.cuda.get_device_capability(0)))),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        }
        if result["gpu"]["sm"] != spec["sm"]:
            raise ValueError(f"GPU SM does not match requested system: {result['gpu']}")
        if spec["gpu_name_contains"] not in result["gpu"]["name"]:
            raise ValueError(f"GPU SKU does not match requested system: {result['gpu']}")
        package = "tensorrt_llm" if spec["backend"] == "trtllm" else spec["backend"]
        result["package_version"] = importlib.metadata.version(package)
        if result["package_version"] != spec["package_version"]:
            raise ValueError("native framework version differs from manifest")
        plans = importlib.import_module("collector.op_plans")
        inventory = importlib.import_module("tools.perf_database.collection_cost_inventory")
        registry = importlib.import_module(f"collector.{spec['backend']}.registry").REGISTRY
        entry = next(entry for entry in registry if entry.op == spec["op"])
        if entry.unverified or spec["sm"] in entry.unverified_sms:
            raise ValueError("normal collector marks this operation unverified; exclude it from eligible cost")
        if entry.module is None:
            raise ValueError("version-routed registries require the normal resolver; no fallback is permitted")
        module = importlib.import_module(entry.module)
        function = getattr(module, entry.run_func)
        policy = inventory.shared_policy() if spec["scenario"] == "shared" else contextlib.nullcontext()
        with policy:
            options = {"include_long_history": True} if spec["scenario"] == "model" else {}
            selected = select_plan(plans.plan(spec["op"], spec["backend"], spec["sm"], **options), spec["cases"])
            rows = selected.rows
            result["planned_cases"] = len(selected.cases)
            result["native_groups"] = len(rows)
            result["initialization_and_planning_seconds"] = time.perf_counter() - started
            perf = output / str(entry.perf_filename)
            for index, row in enumerate(rows):
                params = row["params"] if isinstance(row, dict) and "params" in row else row
                call_started = time.perf_counter()
                call = {"index": index, "row": str(row)}
                try:
                    kwargs = {"perf_filename": str(perf), "device": "cuda:0", **entry.run_kwargs_dict}
                    if entry.extra_perf_filenames:
                        kwargs["extra_perf_filenames"] = tuple(
                            str(output / str(name)) for name in entry.extra_perf_filenames
                        )
                    function(*params, **kwargs)
                    torch.cuda.synchronize()
                    call["status"] = "passed"
                except Exception as error:
                    call.update(status="failed", error=str(error), traceback=traceback.format_exc())
                    traceback.print_exc()
                call["native_call_seconds"] = time.perf_counter() - call_started
                result["calls"].append(call)
                (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            result["forward_rows"] = list(csv.DictReader(perf.open())) if perf.exists() else []
            if not result["forward_rows"]:
                raise ValueError("native calls produced no forward measurements")
            result["status"] = "passed" if all(call["status"] == "passed" for call in result["calls"]) else "failed"
    except Exception as error:
        result.update(status="failed", error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        result["worker_seconds"] = time.perf_counter() - started
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(json.loads(args.spec.read_text()), args.output)
    print(json.dumps({"status": result["status"], "worker_seconds": result["worker_seconds"]}))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
