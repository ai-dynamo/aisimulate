#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run paired revision-local workers and produce the advisory report."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PYTHON_ROOT))

from tools.forward_perf_gate import PROTOCOL_VERSION, compare
from tools.forward_perf_gate import cases as case_matrix

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}
WORKER_ENV = {**THREAD_ENV, "AIC_ALLOW_UNLISTED_VERSIONS": "1"}


def _command_version(command: list[str]) -> str:
    try:
        return subprocess.run(command, text=True, capture_output=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _worker_error(case: dict, revision: str, status: str, message: str) -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "revision": revision,
        "case_id": case["case_id"],
        "case_hash": "",
        "status": status,
        "error": {"type": status, "message": message[:2_000]},
    }


def _invoke_worker(
    *,
    python: Path,
    worker: Path,
    request: dict,
    cpu: int,
    timeout: float,
) -> tuple[object | None, str | None, str | None, str]:
    command = ["taskset", "--cpu-list", str(cpu), str(python), str(worker)]
    try:
        completed = subprocess.run(
            command,
            input=json.dumps(request),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
            env={**os.environ, **WORKER_ENV},
        )
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT", f"worker exceeded {timeout:.0f}s", ""
    if completed.returncode:
        return (
            None,
            "WORKER_ERROR",
            f"exit {completed.returncode}: {completed.stderr.strip()}",
            completed.stderr[-2_000:],
        )
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return (
            None,
            "WORKER_ERROR",
            f"invalid JSON: {exc}: {completed.stdout[:1000]}",
            completed.stderr[-2_000:],
        )
    return response, None, None, completed.stderr[-2_000:]


def run_worker(
    *,
    python: Path,
    worker: Path,
    revision: str,
    case: dict,
    warmup: int,
    iterations: int,
    cpu: int,
    timeout: float,
) -> dict:
    response, error_status, error, worker_stderr = _invoke_worker(
        python=python,
        worker=worker,
        request={
            "protocol_version": PROTOCOL_VERSION,
            "revision": revision,
            "case": case,
            "warmup": warmup,
            "iterations": iterations,
        },
        cpu=cpu,
        timeout=timeout,
    )
    if error_status:
        return _worker_error(case, revision, error_status, error or "worker failed")
    if not isinstance(response, dict):
        return _worker_error(
            case,
            revision,
            "WORKER_ERROR",
            f"worker JSON must be an object, got {type(response).__name__}",
        )
    if worker_stderr.strip():
        response["worker_stderr"] = worker_stderr
    return response


def run_worker_batch(
    *,
    python: Path,
    worker: Path,
    revision: str,
    cases: list[dict],
    warmup: int,
    iterations: int,
    cpu: int,
    timeout: float,
) -> tuple[list[dict], str | None]:
    response, error_status, error, worker_stderr = _invoke_worker(
        python=python,
        worker=worker,
        request={
            "protocol_version": PROTOCOL_VERSION,
            "revision": revision,
            "cases": cases,
            "warmup": warmup,
            "iterations": iterations,
        },
        cpu=cpu,
        timeout=timeout,
    )
    if error_status:
        return [], f"{error_status}: {error}"
    if not isinstance(response, dict):
        return [], f"worker JSON must be an object, got {type(response).__name__}"
    if response.get("protocol_version") != PROTOCOL_VERSION:
        return [], f"worker protocol is {response.get('protocol_version')!r}, expected {PROTOCOL_VERSION}"
    if response.get("revision") != revision:
        return [], f"worker revision is {response.get('revision')!r}, expected {revision!r}"
    results = response.get("results")
    if not isinstance(results, list):
        return [], f"worker results must be an array, got {type(results).__name__}"
    if not all(isinstance(result, dict) for result in results):
        return [], "worker results must contain only objects"

    expected_ids = [case["case_id"] for case in cases]
    actual_ids = [result.get("case_id") for result in results]
    if not all(isinstance(case_id, str) and case_id for case_id in actual_ids):
        return [], "worker result case_id fields must be non-empty strings"
    if actual_ids != expected_ids:
        duplicates = sorted({case_id for case_id in actual_ids if actual_ids.count(case_id) > 1})
        missing = [case_id for case_id in expected_ids if case_id not in actual_ids]
        unexpected = [case_id for case_id in actual_ids if case_id not in expected_ids]
        return (
            [],
            "worker result case IDs do not match the request: "
            f"missing={missing}, duplicate={duplicates}, unexpected={unexpected}, ordered={actual_ids}",
        )
    if worker_stderr.strip():
        for result in results:
            result["worker_stderr"] = worker_stderr
    return results, None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-python", type=Path, required=True)
    parser.add_argument("--base-worker", type=Path, required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--head-python", type=Path, required=True)
    parser.add_argument("--head-worker", type=Path, required=True)
    parser.add_argument("--head-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--worker-timeout", type=float, default=120.0)
    parser.add_argument("--skip-prewarm", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run one short case.")
    return parser.parse_args()


def _checkpoint(raw: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "raw_results.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(output_path)


def _finish(raw: dict, output_dir: Path) -> int:
    _checkpoint(raw, output_dir)
    comparison = compare.compare_raw(raw)
    compare.write_outputs(comparison, output_dir)
    print(compare.render_markdown(comparison))
    return 1 if comparison["blocking"] else 0


def main() -> int:
    args = _parse_args()
    if shutil.which("taskset") is None:
        print("error: taskset is required to pin benchmark workers", file=sys.stderr)
        return 2
    for path in (args.base_python, args.base_worker, args.head_python, args.head_worker):
        if not path.exists():
            print(f"error: required path does not exist: {path}", file=sys.stderr)
            return 2
    if args.rounds <= 0 or args.warmup < 0 or args.iterations <= 0:
        print("error: invalid round, warmup, or iteration count", file=sys.stderr)
        return 2

    cpu = min(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 0
    matrix_cases = case_matrix.expand_cases()
    selected_cases = matrix_cases[:1] if args.smoke else matrix_cases
    rounds, warmup, iterations = args.rounds, args.warmup, args.iterations
    if args.smoke:
        rounds, warmup, iterations = 1, 1, 3

    raw = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "base_revision": args.base_revision,
        "head_revision": args.head_revision,
        "configuration": {
            "mode": "smoke" if args.smoke else "full",
            "rounds": rounds,
            "warmup": warmup,
            "iterations": iterations,
            "cpu": cpu,
            "thread_env": THREAD_ENV,
            "matrix_case_count": len(matrix_cases),
            "selected_case_count": len(selected_cases),
            "expected_case_ids": [case["case_id"] for case in selected_cases],
        },
        "host": {"platform": platform.platform(), "processor": _cpu_model()},
        "toolchain": {
            "controller_python": platform.python_version(),
            "rustc": _command_version(["rustc", "--version"]),
            "cargo": _command_version(["cargo", "--version"]),
            "maturin": _command_version(["maturin", "--version"]),
        },
        "run_errors": [],
        "worker_processes": 0,
        "prewarm": [],
        "cases": [],
    }
    _checkpoint(raw, args.output_dir)
    if not selected_cases:
        raw["run_errors"].append("case matrix is empty")
        return _finish(raw, args.output_dir)

    sides = {
        "base": (args.base_python, args.base_worker, args.base_revision),
        "head": (args.head_python, args.head_worker, args.head_revision),
    }
    skipped_cases: dict[str, str] = {}
    if not args.skip_prewarm and not args.smoke:
        prewarm_responses = {}
        for side in ("base", "head"):
            python, worker, revision = sides[side]
            print(f"prewarm {side} batch ({len(selected_cases)} cases)", file=sys.stderr, flush=True)
            results, error = run_worker_batch(
                python=python,
                worker=worker,
                revision=revision,
                cases=selected_cases,
                warmup=0,
                iterations=1,
                cpu=cpu,
                timeout=args.worker_timeout,
            )
            raw["worker_processes"] += 1
            if error:
                raw["run_errors"].append(f"prewarm {side} batch failed: {error}")
                return _finish(raw, args.output_dir)
            prewarm_responses[side] = {result["case_id"]: result for result in results}

        for case in selected_cases:
            responses = {
                "base": prewarm_responses["base"][case["case_id"]],
                "head": prewarm_responses["head"][case["case_id"]],
            }
            disposition, reason = compare.prewarm_disposition(
                case["case_id"],
                responses["base"],
                responses["head"],
            )
            raw["prewarm"].append(
                {
                    "case_id": case["case_id"],
                    "base": responses["base"],
                    "head": responses["head"],
                    "disposition": disposition,
                    "reason": reason,
                }
            )
            if disposition == "SKIP":
                skipped_cases[case["case_id"]] = reason or "no timing baseline"
            elif disposition == "INVALID":
                raw["run_errors"].append(f"prewarm failed for {case['case_id']}: {reason}")
                _checkpoint(raw, args.output_dir)
                return _finish(raw, args.output_dir)
        _checkpoint(raw, args.output_dir)

    entries = {}
    measured_cases = []
    for case in selected_cases:
        if case["case_id"] in skipped_cases:
            raw["cases"].append(
                {
                    "case": case,
                    "rounds": [],
                    "skip_reason": skipped_cases[case["case_id"]],
                }
            )
            continue
        entry = {"case": case, "rounds": []}
        raw["cases"].append(entry)
        entries[case["case_id"]] = entry
        measured_cases.append(case)

    for round_index in range(rounds):
        if not measured_cases:
            break
        ordered_cases = measured_cases if round_index % 2 == 0 else list(reversed(measured_cases))
        case_order = "forward" if round_index % 2 == 0 else "reverse"
        side_order = ("base", "head") if round_index % 2 == 0 else ("head", "base")
        round_responses = {}
        for side in side_order:
            python, worker, revision = sides[side]
            print(
                f"measure round={round_index + 1}/{rounds} {side} {case_order} batch ({len(ordered_cases)} cases)",
                file=sys.stderr,
                flush=True,
            )
            results, error = run_worker_batch(
                python=python,
                worker=worker,
                revision=revision,
                cases=ordered_cases,
                warmup=warmup,
                iterations=iterations,
                cpu=cpu,
                timeout=args.worker_timeout,
            )
            raw["worker_processes"] += 1
            if error:
                raw["run_errors"].append(f"measurement round {round_index + 1} {side} batch failed: {error}")
                return _finish(raw, args.output_dir)
            round_responses[side] = {result["case_id"]: result for result in results}

        for case in measured_cases:
            case_id = case["case_id"]
            entries[case_id]["rounds"].append(
                {
                    "round": round_index + 1,
                    "order": list(side_order),
                    "case_order": case_order,
                    "base": round_responses["base"][case_id],
                    "head": round_responses["head"][case_id],
                }
            )
        _checkpoint(raw, args.output_dir)

    return _finish(raw, args.output_dir)


if __name__ == "__main__":
    sys.exit(main())
