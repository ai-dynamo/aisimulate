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
import ast
import contextlib
import csv
import dataclasses
import hashlib
import importlib
import importlib.metadata
import itertools
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG_ROOT))


def save_result(output: Path, result: dict) -> None:
    temporary = output / "result.json.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output / "result.json")


def select_plan(planned, requested: list[dict]):
    """Select exact generated cases/dispatches; refuse missing or duplicate requests."""
    wanted = {json.dumps({k: v for k, v in row.items() if not k.startswith("_")}, sort_keys=True) for row in requested}
    if len(wanted) != len(requested):
        raise ValueError("duplicate requested case keys")
    cases, dispatches, found = [], [], set()
    runs = planned.dispatches or [("",)] * len(planned.cases)
    for case, runners in zip(planned.cases, runs, strict=True):
        for runner in runners:
            actual = {
                **{name: "" if value is None else str(value) for name, value in case._asdict().items()},
                "dispatch": runner,
            }
            axes = [
                name
                for name in ("num_tokens", "batch_sizes", "seq_lens")
                if isinstance(getattr(case, name, None), tuple)
            ]
            points = set()
            for request in requested:
                row = {k: v for k, v in request.items() if not k.startswith("_")}
                key = json.dumps(row, sort_keys=True)
                if set(row) != set(actual):
                    continue
                if any(row.get(name) != value for name, value in actual.items() if name not in axes):
                    continue
                if not axes:
                    if row == actual:
                        points.add(())
                        found.add(key)
                    continue
                values = []
                for name in axes:
                    raw = row.get(name, "")
                    parsed = ast.literal_eval(raw) if raw.startswith(("(", "[")) else (int(raw),)
                    if (
                        not isinstance(parsed, (tuple, list))
                        or not parsed
                        or not set(parsed) <= set(getattr(case, name))
                    ):
                        break
                    values.append(parsed)
                if len(values) == len(axes):
                    points.update(itertools.product(*values))
                    found.add(key)
            if not points:
                continue
            if len(axes) > 2:
                raise ValueError("unsupported grouped audit axes")
            if not axes:
                selected = [case]
            elif len(axes) == 1:
                values = tuple(value for value in getattr(case, axes[0]) if (value,) in points)
                selected = [case._replace(**{axes[0]: values})]
            else:
                # Equal second-axis sets can share initialization. Irregular
                # samples split into rectangles without inventing extra cells.
                rectangles = {}
                for first in getattr(case, axes[0]):
                    seconds = tuple(second for second in getattr(case, axes[1]) if (first, second) in points)
                    if seconds:
                        rectangles.setdefault(seconds, []).append(first)
                selected = [
                    case._replace(**{axes[0]: tuple(firsts), axes[1]: seconds})
                    for seconds, firsts in rectangles.items()
                ]
            cases.extend(selected)
            dispatches.extend([(runner,)] * len(selected))
    if found != wanted:
        raise ValueError(f"{len(wanted - found)} requested cases absent from the actual plan")
    return dataclasses.replace(planned, cases=cases, dispatches=dispatches if planned.dispatches else [])


def run(spec: dict, output: Path) -> dict:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    result = {"spec": spec, "publish_eligible": False, "allocation_seconds_included": False, "calls": []}
    try:
        os.environ["COLLECTOR_MEASURE_POWER"] = "0"
        if spec.get("model_path"):
            os.environ["COLLECTOR_MODEL_PATH"] = spec["model_path"]
        else:
            os.environ.pop("COLLECTOR_MODEL_PATH", None)
        result["effective_protocol"] = {
            "measure_power": False,
            "model_path": os.environ.get("COLLECTOR_MODEL_PATH"),
            "coverage_scenario": spec["scenario"],
        }
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
        executor = importlib.import_module("collector.collect")
        helper = importlib.import_module("collector.helper")
        function = getattr(module, entry.run_func)
        observations = importlib.import_module("tools.perf_database.collection_cost_observations")
        result["audit_driver_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        observer_file = getattr(observations, "__file__", None)
        result["audit_observer_sha256"] = (
            hashlib.sha256(Path(observer_file).read_bytes()).hexdigest() if observer_file else None
        )
        result["audit_driver_commit_declared"] = spec.get("audit_driver_commit")
        properties = torch.cuda.get_device_properties(0)
        uuid = getattr(properties, "uuid", None)
        result["gpu"]["uuid"] = str(uuid) if uuid is not None else None
        image_receipt = output.parent / (output.name.removeprefix("native-") + "-image.sha256")
        result["runtime_protocol"] = {
            "installed_framework_version_observed": result["package_version"],
            "torch_version_observed": getattr(torch, "__version__", None),
            "torch_cuda_build_version_observed": getattr(getattr(torch, "version", None), "cuda", None),
            "loaded_cuda_runtimes_observed": observations.loaded_cuda_runtimes(),
            "python_version_observed": platform.python_version(),
            "measure_power_observed": False,
            "omp_num_threads_observed": os.environ.get("OMP_NUM_THREADS"),
            "torch_num_threads_observed": getattr(torch, "get_num_threads", lambda: None)(),
            "image_squashfs_sha256_observed": image_receipt.read_text().split()[0] if image_receipt.exists() else None,
            "host_concurrent_workers_observed": None,
            "allocation_observation_enabled": bool(spec.get("observe_allocations")),
            "native_passes_requested": int(spec.get("native_passes", 1)),
        }
        result["allocation_observations"] = []
        current_call = {"index": None, "pass_index": None}

        def record_allocation(record):
            event = {
                "native_call_index": current_call["index"], "native_pass_index": current_call["pass_index"], **record
            }
            result["allocation_observations"].append(event)
            print(json.dumps({"event": "allocation_observation", **event}), flush=True)

        # Direct collectors expose their benchmark helper in this module. Native
        # module subprocesses keep their existing point clocks in the raw log.
        benchmark = getattr(module, "benchmark_with_power", None)
        measurements = []

        @contextlib.contextmanager
        def record_benchmark(*args, **kwargs):
            if spec.get("observe_allocations"):
                kernel = kwargs.get("kernel_func")
                if kernel is not None:
                    try:
                        record_allocation(
                            {"kind": "forward_input_arguments", **observations.forward_inputs(kernel, spec["backend"])}
                        )
                    except Exception as error:
                        record_allocation({"kind": "forward_input_arguments", "observation_error": str(error)})
            with benchmark(*args, **kwargs) as measured:
                yield measured
            measurements.append(
                {
                    "native_call_index": current_call["index"],
                    "native_pass_index": current_call["pass_index"],
                    "forward_latency_ms": measured.get("latency_ms"),
                    "num_runs_executed": measured.get("num_runs_executed"),
                    "requested_warmups": kwargs.get("num_warmups", 3),
                    "repeat_n": kwargs.get("repeat_n", 1),
                    "phase_seconds": dict(measured.get("phase_seconds") or {}),
                }
            )

        if benchmark is not None:
            module.benchmark_with_power = record_benchmark
        policy = inventory.shared_policy() if spec["scenario"] == "shared" else contextlib.nullcontext()
        cache_observer = (
            observations.observe_native_caches(module, spec["backend"], record_allocation)
            if spec.get("observe_allocations")
            else contextlib.nullcontext()
        )
        with policy, cache_observer:
            options = {"include_long_history": True} if spec["scenario"] == "model" else {}
            selected = select_plan(plans.plan(spec["op"], spec["backend"], spec["sm"], **options), spec["cases"])
            rows = selected.rows
            passes = int(spec.get("native_passes", 1))
            if passes not in (1, 2):
                raise ValueError("native_passes must be one or two for bounded cold/warm observations")
            execution_rows = rows * passes
            result["native_calls_requested"] = len(execution_rows)
            result["planned_cases"] = len(selected.cases)
            result["native_groups"] = len(rows)
            result["initialization_and_planning_seconds"] = time.perf_counter() - started
            perf = output / str(entry.perf_filename)
            for index, row in enumerate(execution_rows):
                params = row["params"] if isinstance(row, dict) and "params" in row else row
                call_started = time.perf_counter()
                pass_index = index // len(rows) + 1
                call = {
                    "index": index,
                    "row": str(row),
                    "native_pass_index": pass_index,
                    "pass_scope": "first_in_worker" if pass_index == 1 else "subsequent_same_worker_pass",
                }
                current_call.update(index=index, pass_index=call["native_pass_index"])
                rows_before = len(list(csv.DictReader(perf.open()))) if perf.exists() else 0
                result["pending_call"] = {
                    **call,
                    "status": "running",
                    "started_elapsed_seconds": call_started - started,
                }
                save_result(output, result)
                print(json.dumps({"event": "native_call_start", **result["pending_call"]}), flush=True)
                try:
                    kwargs = {"perf_filename": str(perf), "device": "cuda:0", **entry.run_kwargs_dict}
                    if entry.extra_perf_filenames:
                        kwargs["extra_perf_filenames"] = tuple(
                            str(output / str(name)) for name in entry.extra_perf_filenames
                        )
                    outcome = function(*params, **kwargs)
                    if isinstance(outcome, helper.WorkerRestartSignal):
                        raise SystemExit(helper.EXIT_CODE_RESTART)
                    torch.cuda.synchronize()
                    call["status"] = "passed"
                except SystemExit as error:
                    rows_after = len(list(csv.DictReader(perf.open()))) if perf.exists() else 0
                    expected = error.code == helper.EXIT_CODE_RESTART and rows_after > rows_before
                    call.update(
                        status="passed" if expected else "failed",
                        worker_exit_code=error.code,
                        persisted_forward_rows=rows_after - rows_before,
                    )
                    result["worker_recycle_required"] = True
                    result["worker_recycle_reason"] = (
                        "normal_executor_restart" if expected else "unexpected_worker_exit"
                    )
                    result["unattempted_rows"] = [str(pending) for pending in execution_rows[index + 1 :]]
                except Exception as error:
                    call.update(status="failed", error=str(error), traceback=traceback.format_exc())
                    traceback.print_exc()
                    fatal = executor._is_cuda_fatal_exception(error, torch)
                    context_error = None if fatal else helper.cuda_context_error("cuda:0")
                    if fatal or context_error is not None:
                        call["cuda_context_error"] = context_error
                        result["worker_recycle_required"] = True
                        result["unattempted_rows"] = [str(pending) for pending in execution_rows[index + 1 :]]
                call["native_call_seconds"] = time.perf_counter() - call_started
                result["calls"].append(call)
                result.pop("pending_call", None)
                save_result(output, result)
                if result.get("worker_recycle_required"):
                    break
            result["forward_rows"] = list(csv.DictReader(perf.open())) if perf.exists() else []
            result["direct_benchmark_measurements"] = measurements
            result["subprocess_repetition_policy"] = spec.get("source_checked_subprocess_repetition_policy")
            if not result["forward_rows"]:
                raise ValueError("native calls produced no forward measurements")
            result["status"] = (
                ("partial" if result.get("unattempted_rows") else "passed")
                if all(call["status"] == "passed" for call in result["calls"])
                else "failed"
            )
    except Exception as error:
        result.update(status="failed", error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        result["worker_seconds"] = time.perf_counter() - started
        save_result(output, result)
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
