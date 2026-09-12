#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare installed AIC/AISimulate runtimes on one explicitly shared data tree.

See docs/cli/migrate-from-aiconfigurator.md for boundaries and reproduction.
The stdlib-only controller alternates variants and runs one child at a time.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MODEL = "meta-llama/Meta-Llama-3.1-8B"
SYSTEM = "b200_sxm"
BACKEND_VERSION = "0.24.0"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def prediction_digest(report):
    # These three fields measure host execution rather than simulated serving.
    excluded = {"wall_time_ms", "processed_tokens_per_s", "processed_output_tokens_per_s"}
    return digest({key: value for key, value in report.items() if key not in excluded})


def reports_match(left, right):
    excluded = {"wall_time_ms", "processed_tokens_per_s", "processed_output_tokens_per_s"}
    if left.keys() != right.keys():
        return False
    # Floating-point aggregation order may differ in the unmodified runtime.
    # Counts and schema remain exact; only real-valued metrics get a tolerance.
    return all(
        key in excluded
        or (
            math.isclose(value, right[key], rel_tol=1e-12, abs_tol=1e-9)
            if isinstance(value, float) and isinstance(right[key], float)
            else value == right[key]
        )
        for key, value in left.items()
    )


def prediction(requests):
    return {
        "traffic": {
            "source": {"type": "synthetic", "input_tokens": 1024, "output_tokens": 128},
            "load": {"type": "concurrency", "concurrency": 16},
            "stop": {"requests": requests},
        },
        "engine": {
            "mode": "aggregated",
            "model": MODEL,
            "hardware": SYSTEM,
            "backend": "vllm",
            "backend_version": BACKEND_VERSION,
            "workers": {
                "aggregated": {
                    "parallelism": {
                        "replicas": 1,
                        "tensor": 4,
                        "pipeline": 1,
                        "attention_data": 1,
                        "moe_tensor": 4,
                        "moe_expert": 1,
                    }
                }
            },
        },
    }


def worker(args):
    # Both distributions must use exactly the same system, model, and profile
    # inputs. This bootstrap is included in the process wall-time measurements.
    from aiconfigurator.sdk import perf_database

    perf_database.set_systems_paths([args.systems_path])
    os.environ["AICONFIGURATOR_SYSTEMS_PATH"] = args.systems_path
    result = {"python": sys.version, "sdk_path": perf_database.__file__}
    start = time.perf_counter()
    if args.worker == "estimate":
        from aiconfigurator.main import main

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            main(
                [
                    "cli",
                    "estimate",
                    "--model-path",
                    MODEL,
                    "--system",
                    SYSTEM,
                    "--backend",
                    "vllm",
                    "--backend-version",
                    BACKEND_VERSION,
                    "--estimate-mode",
                    "agg",
                    "--batch-size",
                    "16",
                    "--tp-size",
                    "4",
                    "--isl",
                    "1024",
                    "--osl",
                    "128",
                    "--systems-paths",
                    args.systems_path,
                ]
            )
        result["stdout"] = stdout.getvalue()
        result["result_sha256"] = digest(
            result["stdout"].split("\n============================================================\n", 1)[-1]
        )
    elif args.worker == "sdk":
        from aiconfigurator.sdk.backends.factory import get_backend
        from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig
        from aiconfigurator.sdk.inference_session import InferenceSession
        from aiconfigurator.sdk.models import get_model

        database = perf_database.get_database(SYSTEM, "vllm", BACKEND_VERSION)
        model = get_model(
            MODEL, ModelConfig(tp_size=4, pp_size=1, attention_dp_size=1, moe_tp_size=4, moe_ep_size=1), "vllm"
        )
        session = InferenceSession(model, database, get_backend("vllm"))
        runtime = RuntimeConfig(batch_size=16, beam_width=1, isl=1024, osl=2, prefix=0, engine_step_backend="rust")
        result["session_setup_s"] = time.perf_counter() - start
        result["phases"] = {}
        for phase in ("static_ctx", "static_gen"):
            first = time.perf_counter()
            value = session.run_static_latency_only(runtime, mode=phase, stride=1)
            first_s = time.perf_counter() - first
            for _ in range(10):
                session.run_static_latency_only(runtime, mode=phase, stride=1)
            samples = []
            for _ in range(100):
                begin = time.perf_counter_ns()
                actual = session.run_static_latency_only(runtime, mode=phase, stride=1)
                samples.append((time.perf_counter_ns() - begin) / 1000)
                if actual != value:
                    raise RuntimeError("repeated SDK calls changed the predicted value")
            result["phases"][phase] = {"predicted_ms": value, "first_query_s": first_s, "warm_us": samples}
        result["result_sha256"] = digest({p: r["predicted_ms"] for p, r in result["phases"].items()})
    elif args.worker == "predict":
        import yaml

        from aisimulate.main import main

        with tempfile.TemporaryDirectory(prefix="aisim-runtime-") as temp:
            root = Path(temp)
            config_path = root / "prediction.yaml"
            config_path.write_text(yaml.safe_dump(prediction(args.requests)))
            code = main(
                ["predict", "--config", str(config_path), "--output-dir", str(root / "out"), "--format", "json"]
            )
            if code:
                raise RuntimeError(f"predict returned {code}")
            result["report"] = json.loads((root / "out/prediction.json").read_text())
            result["result_sha256"] = prediction_digest(result["report"])
    else:
        from aisimulate.compiler import prediction_to_replay_spec
        from aisimulate.config import CorePredictionConfig
        from aisimulate.runner import EngineReplayRunnerFactory
        from aisimulate.sweeper.replay import ReplayOutputRequirements

        spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(prediction(args.requests)))
        runner = EngineReplayRunnerFactory().create(0)
        samples = []
        native_samples = []
        try:
            # Each run builds a fresh provider/cache. Warm means process/data
            # access is primed; completed replay results are never memoized.
            for repeat in range(4):
                begin = time.perf_counter()
                output = runner.run(spec, output_requirements=ReplayOutputRequirements(include_raw_report=True))
                elapsed = time.perf_counter() - begin
                report = output.metadata["native_report"]
                if repeat == 0:
                    result["report"] = report
                    result["result_sha256"] = prediction_digest(report)
                elif not reports_match(report, result["report"]):
                    raise RuntimeError("repeated replay changed the report")
                else:
                    samples.append(elapsed)
                    native_samples.append(report["wall_time_ms"] / 1000)
        finally:
            runner.close()
        result["warm_replay_s"] = samples
        result["warm_native_replay_s"] = native_samples
    result["worker_operation_s"] = time.perf_counter() - start
    result["process_cpu_s"] = time.process_time()
    result["native_modules"] = [
        module.__file__
        for module in list(sys.modules.values())
        if getattr(module, "__file__", "")
        and module.__file__.endswith((".so", ".pyd"))
        and Path(module.__file__).name.startswith(("_runtime.", "_aiconfigurator_core."))
    ]
    result["dependencies"] = {
        name: importlib.metadata.version(name) for name in ("numpy", "pandas", "pyarrow", "pyyaml")
    }

    Path(args.result).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def attest(variant, env):
    root = Path(variant["root"]).resolve()
    for path in variant["sources"]:
        Path(path).resolve().relative_to(root)
    # Untracked Python modules can shadow imports; native build products are
    # normally ignored by Git and are attested separately below.
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True
    ).strip()
    if dirty:
        raise RuntimeError(f"{variant['label']} has uncommitted source changes: {dirty}")
    module = "aisimulate._runtime" if variant["kind"] == "aisimulate" else "aiconfigurator_core._aiconfigurator_core"
    dependencies = ["numpy", "pandas", "pyarrow", "pyyaml"]
    if variant["kind"] == "aisimulate":
        dependencies += ["jax", "jaxlib", "google-vizier", "equinox"]
    code = (
        "import importlib,importlib.metadata,json,sys; "
        f"native=importlib.import_module({module!r}); "
        "print(json.dumps({'python':sys.version,'native':native.__file__,"
        f"'dependencies':{{n:importlib.metadata.version(n) for n in {dependencies!r}}}}}))"
    )
    result = json.loads(subprocess.check_output([variant["python"], "-c", code], env=env, text=True))
    native = Path(result["native"]).resolve()
    native.relative_to(root)  # Reject a stale extension loaded from another checkout.
    if native.suffix not in {".so", ".pyd"}:
        raise RuntimeError(f"{variant['label']} did not load a native extension")
    result["native"] = str(native)
    result["native_sha256"] = hashlib.sha256(native.read_bytes()).hexdigest()
    result["revision"] = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    return result


def controller(args):
    variants = json.loads(Path(args.variants).read_text())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    systems = Path(args.systems_path).resolve()
    files = sorted(p for p in systems.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    manifest = {str(p.relative_to(systems)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    evidence = {
        "protocol": 1,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "rounds": args.rounds,
        "systems_sha256": digest(manifest),
        "systems_manifest": manifest,
        "model": MODEL,
        "system": SYSTEM,
        "backend_version": BACKEND_VERSION,
        "variants": variants,
        "prediction_tolerance": {"relative": 1e-12, "absolute": 1e-9},
        "prediction_exclusions": ["wall_time_ms", "processed_tokens_per_s", "processed_output_tokens_per_s"],
        "samples": [],
    }
    for variant in variants:
        root = Path(variant["root"]).resolve()
        variant["revision"] = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(str(Path(p).resolve()) for p in variant["sources"])
        variant["attestation"] = attest(variant, env)
    # A full unrecorded pass pays first-install/import/filesystem warmup costs.
    # Recorded samples still start a fresh interpreter; OS caches are not purged.
    for round_number in range(-1, args.rounds):
        order = variants if round_number % 2 == 0 else list(reversed(variants))
        for variant in order:
            cases = [("estimate", 0), ("sdk", 0)]
            if variant["kind"] == "aisimulate":
                cases += [("predict", 100), ("predict", 1000), ("replay", 100), ("replay", 1000)]
            for case, requests in cases:
                env = os.environ.copy()
                env["PYTHONPATH"] = os.pathsep.join(str(Path(p).resolve()) for p in variant["sources"])
                # Keep unrelated native thread pools from oversubscribing the host.
                for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                    env[name] = "1"
                with tempfile.TemporaryDirectory(prefix="migration-runtime-") as temp:
                    result_path = Path(temp) / "result.json"
                    command = [
                        variant["python"],
                        str(Path(__file__).resolve()),
                        "--worker",
                        case,
                        "--requests",
                        str(requests),
                        "--systems-path",
                        str(systems),
                        "--result",
                        str(result_path),
                    ]
                    begin = time.perf_counter()
                    child = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
                    wall = time.perf_counter() - begin
                    if child.returncode:
                        raise RuntimeError(f"{variant['label']}/{case}: {child.stdout}\n{child.stderr}")
                    row = json.loads(result_path.read_text())
                    expected = variant["attestation"]
                    loaded = {str(Path(p).resolve()) for p in row["native_modules"]}
                    if expected["native"] not in loaded:
                        raise RuntimeError(f"{variant['label']} loaded an unexpected native runtime")
                    if hashlib.sha256(Path(expected["native"]).read_bytes()).hexdigest() != expected["native_sha256"]:
                        raise RuntimeError(f"{variant['label']} native runtime changed during measurement")
                    row.update(
                        variant=variant["label"], case=case, requests=requests, round=round_number, process_wall_s=wall
                    )
                    if round_number >= 0:
                        evidence["samples"].append(row)
                        output.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
                    print(f"{round_number}: {variant['label']} {case}/{requests}: {wall:.3f}s", flush=True)
    comparisons = []
    for case, count in (
        ("estimate", 0),
        ("sdk", 0),
        ("predict", 100),
        ("predict", 1000),
        ("replay", 100),
        ("replay", 1000),
    ):
        rows = [r for r in evidence["samples"] if r["case"] == case and r["requests"] == count]
        if not rows:
            continue
        if case in {"predict", "replay"}:
            match = all(reports_match(rows[0]["report"], r["report"]) for r in rows)
        else:
            match = len({r["result_sha256"] for r in rows}) == 1
        comparisons.append({"case": case, "requests": count, "predictions_match": match})
    evidence["comparisons"] = comparisons
    output.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    if not all(row["predictions_match"] for row in comparisons):
        raise RuntimeError(f"prediction mismatch; inspect {output} before interpreting speed ratios")
    for variant in variants:
        for case, count in (("estimate", 0), ("predict", 100), ("predict", 1000)):
            rows = [
                r
                for r in evidence["samples"]
                if r["variant"] == variant["label"] and r["case"] == case and r["requests"] == count
            ]
            if rows:
                field = "process_wall_s"
                print(variant["label"], case, count, field, statistics.median(r[field] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", help="JSON list of label, kind, root, python, sources objects")
    parser.add_argument("--systems-path", required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", default="migration-runtime.json")
    parser.add_argument("--worker", choices=["estimate", "sdk", "predict", "replay"], help=argparse.SUPPRESS)
    parser.add_argument("--requests", type=int, default=100, help=argparse.SUPPRESS)
    parser.add_argument("--result", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
    elif args.variants and args.rounds > 0:
        controller(args)
    else:
        parser.error("--variants and a positive --rounds are required")


if __name__ == "__main__":
    main()
