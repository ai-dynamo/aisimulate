#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Matched synthetic host-cost qualification, using one fresh process per trial.

Run this identical file with each checkout's interpreter, --checkout, and
--native-sha256. Five trials are required for measurements. --dry-run exercises
the complete sequence once and records no timing samples. Both builds use the same caller-managed Gym
initial fitting policy and unchanged public predictor interface. No native
source-free initialization is implemented or measured. Scope extraction
is benchmark-only and is used to time the same public SDK prediction/save/load
APIs after the wrapper has created its model. Synthetic targets establish no
accuracy result. Construction excludes interpreter/module import; initial fitting
includes the wrapper's complete admitted twentieth-observation call. Checkpoint
and snapshot timers include their complete public calls, including filesystem I/O.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path

THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    pending = path.with_suffix(".partial")
    pending.write_bytes(encoded(value) + b"\n")
    pending.replace(path)


def command(root, *args):
    return subprocess.check_output(args, cwd=root, text=True).strip()


def identity(root, expected_native):
    native_path = Path(importlib.import_module("aisimulate._runtime").__file__).resolve()
    if not native_path.is_relative_to(root) or sha(native_path) != expected_native:
        raise ValueError(f"Unexpected native binary: {native_path}")
    expected_environment = (root / "python/aisimulate/.venv").resolve()
    if Path(sys.prefix).resolve() != expected_environment:
        raise ValueError(f"Run with the selected checkout's environment: {expected_environment}")
    for name in ("aisimulate", "aisimulate_core"):
        path = Path(importlib.import_module(name).__file__).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Python package loaded from another checkout: {path}")
    paths = command(root, "git", "ls-files", "-co", "--exclude-standard").splitlines()
    sources = {
        path: sha(root / path)
        for path in sorted(set(paths))
        if Path(path).suffix in {".py", ".rs", ".toml", ".lock"}
        and (root / path).is_file()
        and not path.startswith(("benchmarks/evidence/", "work/"))
    }
    difference = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)
    dependencies = sorted(
        (distribution.metadata["Name"].lower(), distribution.version)
        for distribution in importlib.metadata.distributions()
    )
    return {
        "checkout": str(root),
        "revision": command(root, "git", "rev-parse", "HEAD"),
        "tracked_diff_sha256": hashlib.sha256(difference).hexdigest(),
        "code_files": sources,
        "code_sha256": hashlib.sha256(encoded(sources)).hexdigest(),
        "harness_path": str(Path(__file__).resolve()),
        "harness_sha256": sha(__file__),
        "native_path": str(native_path),
        "native_sha256": sha(native_path),
        "native_bytes": native_path.stat().st_size,
        "python": sys.version,
        "python_executable": sys.executable,
        "python_environment": sys.prefix,
        "dependencies": dependencies,
        "dependency_freeze_sha256": hashlib.sha256(encoded(dependencies)).hexdigest(),
        "lock_sha256": sha(root / "python/aisimulate/uv.lock"),
        "machine": platform.uname()._asdict(),
        "cpu_count": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "thread_environment": {name: os.environ.get(name) for name in THREAD_ENV},
    }


def memory():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {"process_peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024}
    try:
        import psutil

        result["process_rss_bytes"] = psutil.Process().memory_info().rss
    except (ImportError, OSError) as error:
        result["process_rss_bytes"] = None
        result["current_rss_error"] = str(error)
    return result


def disk_bytes(directory):
    files = [path for path in directory.rglob("*") if path.is_file()]
    return {
        "file_count": len(files),
        "bytes": sum(path.stat().st_size for path in files),
    }


def observations():
    from scripts.fpm_accuracy.types.forward_pass import (
        ForwardPassIteration,
        ForwardPassMetric,
        RequestMetrics,
    )

    # The first twenty inputs cover five deterministic decode load levels.
    # Later targets introduce a modest deterministic shift; they are synthetic
    # measured-label stand-ins, never predictions fed back as truth.
    return tuple(
        ForwardPassIteration.single_rank(
            ForwardPassMetric(
                fpm_id=index,
                configuration_id="synthetic-host-cost-v1",
                version=1,
                worker_id="host-cost-worker",
                dp_rank=0,
                counter_id=index,
                wall_time_s=(10 + 0.4 * (index % 5) + (1.0 if index >= 20 else 0.0)) / 1000,
                scheduled=RequestMetrics(
                    num_decode_requests=1 + index % 5,
                    sum_decode_kv_tokens=(1 + index % 5) * 128,
                ),
                queued=RequestMetrics(),
                capture_id="synthetic-host-cost-v1",
            )
        )
        for index in range(71)
    )


def valid_prediction(value):
    if value is None or not math.isfinite(value) or value <= 0:
        raise AssertionError(f"Expected a finite positive prediction, got {value}")
    return value


def forest_shape(model):
    return {
        kind: {
            "trees": len(store["trees"]),
            "nodes": sum(len(tree["value"]) for tree in store["trees"]),
            "leaves": sum(sum(child < 0 for child in tree["left"]) for tree in store["trees"]),
        }
        for kind, store in model.export_learned_artifact()["stores"].items()
    }


def runtime_checkpoint_config(config, path):
    result = deepcopy(config)
    estimator = result["estimator_config"]
    estimator.pop("state_snapshot", None)
    learned = estimator["fpm_learned"]
    for field in (
        "artifact_path",
        "bundle_path",
        "source_sha256",
        "resolved_source",
        "resolved_fallback",
    ):
        learned.pop(field, None)
    learned["checkpoint_path"] = str(path)
    return result


def worker(args):
    root = args.checkout.resolve()
    sys.path.insert(0, str(root))
    from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel, fpm_learned
    from scripts.fpm_accuracy.models.learned_scenario import LearnedScenarioPredictor

    run_identity = identity(root, args.native_sha256)
    data = observations()
    payloads = [observation.tuning_payload() for observation in data]
    query = data[-1].prediction_input()
    query_payload = query.aic_payload()
    request = ForwardPassPerfModelConfig(
        model="Qwen/Qwen3-8B",
        system="h200_sxm",
        worker_type="decode",
        backend="vllm",
        backend_version="0.25.1",
    )
    stages = {}
    rss = {"before_construction": memory()}

    def measure(name, callback, operations=1):
        started = None if args.dry_run else time.perf_counter_ns()
        result = callback()
        elapsed = None if started is None else time.perf_counter_ns() - started
        stages[name] = {"operations": operations, "elapsed_ns": elapsed}
        return result

    predictor = measure(
        "construction",
        lambda: LearnedScenarioPredictor(
            "gbdt",
            RustForwardPassPerfModel,
            request,
            fpm_learned,
            online=True,
            cold_start=True,
        ),
    )
    restored_models = []
    try:
        if measure("first_cold_prediction", lambda: predictor.predict(query)).value_ms is not None:
            raise AssertionError("Strict C unexpectedly used a fitted model before its first admission")
        measure(
            "collect_first_19",
            lambda: [predictor.tune([row]) for row in data[:19]],
            operations=19,
        )
        if predictor.predict(query).value_ms is not None:
            raise AssertionError("The initial fit ran before observation twenty")
        rss["after_collect_19"] = memory()
        measure("initial_fit_20th_admission", lambda: predictor.tune([data[19]]))
        first = valid_prediction(measure("first_ready_prediction", lambda: predictor.predict(query)).value_ms)
        rss["after_initial_fit"] = memory()
        warm = measure(
            "warm_wrapper_predictions",
            lambda: [predictor.predict(query).value_ms for _ in range(args.warm_queries)],
            operations=args.warm_queries,
        )
        if any(value != first for value in warm):
            raise AssertionError("Prediction changed without admitted feedback")
        # Deliberately outside product APIs: identify the wrapper-created model
        # for timing public SDK calls without adding a production escape hatch.
        scopes = list(predictor._scopes.values())
        if len(scopes) != 1 or scopes[0].model is None:
            raise AssertionError("Expected exactly one ready worker/store scope")
        native, config = scopes[0].model, scopes[0].config
        online_after_fit = native.learned_online_diagnostics()
        shape_after_fit = forest_shape(native)
        forest_after_fit = hashlib.sha256(encoded(native.export_learned_artifact()["stores"])).hexdigest()
        warm_sdk = measure(
            "warm_sdk_predictions",
            lambda: [native.estimate_forward_pass_time_ms(query_payload) for _ in range(args.warm_queries)],
            operations=args.warm_queries,
        )
        if any(value != first for value in warm_sdk):
            raise AssertionError("Wrapper and public SDK predictions differ")
        measure(
            "ordinary_updates_50",
            lambda: [predictor.tune([row]) for row in data[20:70]],
            operations=50,
        )
        expected = valid_prediction(native.estimate_forward_pass_time_ms(query_payload))
        online_after_updates = native.learned_online_diagnostics()
        shape_after_updates = forest_shape(native)
        forest_after_updates = hashlib.sha256(encoded(native.export_learned_artifact()["stores"])).hexdigest()
        rss["after_updates_50"] = memory()
        with tempfile.TemporaryDirectory(prefix="gbdt-host-cost-") as temporary:
            temporary = Path(temporary)
            checkpoint = temporary / "checkpoint"
            measure("checkpoint_save", lambda: native.save_online_checkpoint(checkpoint))
            checkpoint_sizes = disk_bytes(checkpoint)
            checkpoint_config = runtime_checkpoint_config(config, checkpoint)
            restored = measure(
                "checkpoint_restore",
                lambda: RustForwardPassPerfModel.best_available(checkpoint_config),
            )
            restored_models.append(restored)
            if restored.estimate_forward_pass_time_ms(query_payload) != expected:
                raise AssertionError("Checkpoint changed the prediction")
            snapshot = temporary / "snapshot"
            measure("snapshot_save", lambda: native.save_state_snapshot(snapshot))
            snapshot_sizes = disk_bytes(snapshot)
            restored_snapshot = measure(
                "snapshot_restore",
                lambda: RustForwardPassPerfModel.best_available(state_snapshot=snapshot),
            )
            restored_models.append(restored_snapshot)
            if restored_snapshot.estimate_forward_pass_time_ms(query_payload) != expected:
                raise AssertionError("Full state snapshot changed the prediction")
            rss["with_restored_models"] = memory()
        # Metadata collection and diagnostics are excluded from timed stages.
        result = {
            "schema_version": 1,
            "complete": True,
            "dry_run": args.dry_run,
            "trial": args.trial,
            "identity": run_identity,
            "synthetic_input_sha256": hashlib.sha256(encoded(payloads)).hexdigest(),
            "synthetic_input_count": len(payloads),
            "accuracy_qualified": False,
            "scope": "one worker, decode store, strict C: 20 bootstrap observations then 50 ordinary updates",
            "stages": stages,
            "rss": rss,
            "first_ready_prediction_ms": first,
            "after_updates_prediction_ms": expected,
            "native_state_bytes_after_fit": online_after_fit["training_state_bytes"],
            "native_state_bytes_after_updates": online_after_updates["training_state_bytes"],
            "forest_shape_after_fit": shape_after_fit,
            "forest_sha256_after_fit": forest_after_fit,
            "forest_sha256_after_updates": forest_after_updates,
            "forest_shape_after_updates": shape_after_updates,
            "online_after_fit": online_after_fit,
            "online_after_updates": online_after_updates,
            "checkpoint": checkpoint_sizes,
            "snapshot": snapshot_sizes,
        }
    finally:
        for model in restored_models:
            model.close()
        predictor.close()
        gc.collect()
    final_identity = identity(root, args.native_sha256)
    for key in ("revision", "code_sha256", "native_sha256", "dependency_freeze_sha256"):
        if final_identity[key] != run_identity[key]:
            raise ValueError(f"Build or environment changed during trial: {key}")
    write(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--native-sha256", required=True)
    parser.add_argument(
        "--comparison-checkout",
        type=Path,
        help="Optional candidate checkout; alternates AB/BA order across paired trials",
    )
    parser.add_argument("--comparison-native-sha256")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New result directory; worker uses a JSON path",
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warm-queries", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--trial", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.warm_queries < 1 or (not args.dry_run and args.trials < 5):
        parser.error("Use positive warm-queries and at least five measurement trials")
    if args.worker:
        if any(os.environ.get(name) != "1" for name in THREAD_ENV):
            parser.error("Worker thread environment must be fixed to one")
        worker(args)
        return
    if (args.comparison_checkout is None) != (args.comparison_native_sha256 is None):
        parser.error("comparison-checkout and comparison-native-sha256 are required together")
    args.output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, **dict.fromkeys(THREAD_ENV, "1"), PYTHONHASHSEED="0")
    environment.pop("PYTHONPATH", None)
    builds = (
        {"selected": (args.checkout, args.native_sha256)}
        if args.comparison_checkout is None
        else {
            "baseline": (args.checkout, args.native_sha256),
            "candidate": (args.comparison_checkout, args.comparison_native_sha256),
        }
    )
    results = {label: [] for label in builds}
    trial_files = []
    for index in range(1 if args.dry_run else args.trials):
        order = list(builds)
        if index % 2:
            order.reverse()
        for label in order:
            checkout, native_hash = builds[label]
            output = args.output / f"{label}-trial-{index}.json"
            invocation = [
                str(checkout.resolve() / "python/aisimulate/.venv/bin/python"),
                str(Path(__file__).resolve()),
                "--checkout",
                str(checkout.resolve()),
                "--native-sha256",
                native_hash,
                "--output",
                str(output),
                "--worker",
                "--trial",
                str(index),
                "--warm-queries",
                str(args.warm_queries),
            ]
            if args.dry_run:
                invocation.append("--dry-run")
            with (args.output / f"{label}-trial-{index}.log").open("w") as log:
                subprocess.run(
                    invocation,
                    cwd=checkout,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            result = json.loads(output.read_text())
            if results[label]:
                for key in (
                    "revision",
                    "code_sha256",
                    "native_sha256",
                    "dependency_freeze_sha256",
                ):
                    if result["identity"][key] != results[label][0]["identity"][key]:
                        raise ValueError(f"Build or environment changed between trials: {key}")
            results[label].append(result)
            trial_files.append(
                {
                    "label": label,
                    "trial": index,
                    "path": output.name,
                    "sha256": sha(output),
                }
            )
            print(
                json.dumps(
                    {
                        "build": label,
                        "trial": index,
                        "complete": True,
                        "dry_run": args.dry_run,
                    }
                ),
                flush=True,
            )
        if len(builds) > 1:
            baseline, candidate = results["baseline"][-1], results["candidate"][-1]
            if (
                baseline["synthetic_input_sha256"] != candidate["synthetic_input_sha256"]
                or baseline["identity"]["dependency_freeze_sha256"] != candidate["identity"]["dependency_freeze_sha256"]
            ):
                raise ValueError("Paired inputs/dependencies do not match")
            for key in (
                "first_ready_prediction_ms",
                "after_updates_prediction_ms",
                "forest_sha256_after_fit",
                "forest_sha256_after_updates",
            ):
                if baseline[key] != candidate[key]:
                    raise ValueError(f"Paired numerical behavior differs: {key}")
    summary = {}
    if not args.dry_run:
        for label, trials in results.items():
            summary[label] = {}
            for stage in trials[0]["stages"]:
                samples = [row["stages"][stage]["elapsed_ns"] for row in trials]
                operations = trials[0]["stages"][stage]["operations"]
                summary[label][stage] = {
                    "samples_ns": samples,
                    "operations": operations,
                    "median_ns": statistics.median(samples),
                    "min_ns": min(samples),
                    "max_ns": max(samples),
                    "median_ns_per_operation": statistics.median(samples) / operations,
                    "sample_stdev_ns": statistics.stdev(samples),
                }
        if len(builds) > 1:
            summary["paired_candidate_over_baseline"] = {
                stage: {
                    "ratios": [
                        c["stages"][stage]["elapsed_ns"] / b["stages"][stage]["elapsed_ns"]
                        for b, c in zip(results["baseline"], results["candidate"], strict=True)
                    ],
                    "median_ratio": statistics.median(
                        c["stages"][stage]["elapsed_ns"] / b["stages"][stage]["elapsed_ns"]
                        for b, c in zip(results["baseline"], results["candidate"], strict=True)
                    ),
                }
                for stage in results["baseline"][0]["stages"]
            }
    write(
        args.output / "summary.json",
        {
            "schema_version": 1,
            "complete": True,
            "dry_run": args.dry_run,
            "trials_per_build": len(next(iter(results.values()))),
            "execution_order": trial_files,
            "timings": summary,
            "accuracy_qualified": False,
            "interpretation": (
                "Synthetic host costs only. Fresh processes, paired AB/BA ordering when comparison checkout supplied; "
                "inspect trial variation. Interpreter startup and metadata are excluded. "
                "RSS covers the whole fresh process; native state bytes are separate."
            ),
        },
    )


if __name__ == "__main__":
    main()
