#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One bounded order-control experiment for PR 585; not a replacement CI gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python/aisimulate"))

from tools.forward_perf_gate import PROTOCOL_VERSION, cases, compare
from tools.forward_perf_gate.run import THREAD_ENV, _command_version, _cpu_model, run_worker_batch

BASE_SHA = "912e91755a23a79724426a0301f3ab157086be2f"
HEAD_SHA = "a75c4a62cba4e734395d26885afbcf9a5fa8cb83"
GATE = Path("python/aisimulate/tools/forward_perf_gate")
CONDITIONS = (
    ("forward", ("left", "right")),
    ("reverse", ("right", "left")),
    ("forward", ("right", "left")),
    ("reverse", ("left", "right")),
)
ARMS = {"base_vs_head": ("base", "head"), "base_vs_base": ("base", "control")}


def schedule() -> list[dict]:
    """Three balanced blocks; alternate arm order and rotate condition order."""
    result = []
    for block in range(3):
        conditions = CONDITIONS[block:] + CONDITIONS[:block]
        for index, (traversal, order) in enumerate(conditions):
            arms = list(ARMS) if (block + index) % 2 == 0 else list(reversed(ARMS))
            for arm in arms:
                result.append(
                    {
                        "sequence": len(result) + 1,
                        "block": block + 1,
                        "arm": arm,
                        "arm_order": arms,
                        "traversal": traversal,
                        "order": list(order),
                    }
                )
    return result


def selected_cases() -> list[dict]:
    selected = [case for case in cases.expand_cases() if case["model_id"] == "qwen3-32b"]
    if len(selected) != 18 or any(
        sum(case["database_mode"] == mode for case in selected) != 9 for mode in ("SILICON", "EMPIRICAL")
    ):
        raise ValueError("expected the complete two nine-case Qwen3-32B groups")
    return selected


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def describe_installation(python: Path) -> dict:
    code = """
import hashlib, importlib.metadata, json, pathlib, platform, sys
import aisimulate._runtime as runtime
path = pathlib.Path(runtime.__file__).resolve()
print(json.dumps({
    'python': platform.python_version(), 'prefix': sys.prefix,
    'native_path': str(path), 'native_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'packages': sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions()),
}))
"""
    completed = subprocess.run([str(python), "-I", "-c", code], capture_output=True, text=True, check=True)
    result = json.loads(completed.stdout)
    prefix = python.parent.parent.resolve()
    if Path(result["prefix"]).resolve() != prefix or not Path(result["native_path"]).is_relative_to(prefix):
        raise ValueError(f"runtime did not load from the intended installation: {prefix}")
    return result


def provenance(args: argparse.Namespace) -> dict:
    sources = {"base": args.base_source, "head": args.head_source}
    expected = {"base": BASE_SHA, "head": HEAD_SHA}
    files = [path.relative_to(REPO) for path in sorted((REPO / GATE).glob("*.py"))]
    files.append(Path("python/aisimulate/tools/prediction_regression_gate/grid.py"))
    hashes = {}
    for side, source in sources.items():
        if git(source, "rev-parse", "HEAD") != expected[side] or git(source, "status", "--porcelain"):
            raise ValueError(f"{side} source is not the clean pinned revision")
        hashes[side] = {str(path): sha256(source / path) for path in files}
    controller_hashes = {str(path): sha256(REPO / path) for path in files}
    if hashes["base"] != hashes["head"] or hashes["base"] != controller_hashes:
        raise ValueError("worker, matrix or comparison helpers differ between the pinned revisions/controller")
    installs = {side: describe_installation(getattr(args, f"{side}_python")) for side in ("base", "head", "control")}
    if installs["base"]["native_sha256"] != installs["control"]["native_sha256"]:
        raise ValueError("A/A native library hashes differ")
    if len({str(Path(installs[side]["prefix"]).resolve()) for side in installs}) != 3:
        raise ValueError("three distinct installations are required")
    for side in ("base", "head", "control"):
        wheel = args.head_wheel if side == "head" else args.base_wheel
        with zipfile.ZipFile(wheel) as archive:
            native = archive.read("aisimulate/" + Path(installs[side]["native_path"]).name)
        if hashlib.sha256(native).hexdigest() != installs[side]["native_sha256"]:
            raise ValueError(f"{side} native library differs from its recorded wheel")
    for side in ("head", "control"):
        if installs[side]["packages"] != installs["base"]["packages"]:
            raise ValueError(f"{side} package versions differ from baseline")
        if installs[side]["python"] != installs["base"]["python"]:
            raise ValueError(f"{side} Python version differs from baseline")
    return {
        "source_revisions": expected,
        "controller_revision": git(REPO, "rev-parse", "HEAD"),
        "controller_sha256": sha256(Path(__file__)),
        "harness_sha256": controller_hashes,
        "wheel_sha256": {"base": sha256(args.base_wheel), "head": sha256(args.head_wheel)},
        "control_wheel": "same base wheel installed into a second venv",
        "installations": installs,
        "build_flags": ["--release", "--locked"],
        "toolchain": {
            "python": platform.python_version(),
            "rustc": _command_version(["rustc", "--version"]),
            "cargo": _command_version(["cargo", "--version"]),
            "maturin": _command_version(["maturin", "--version"]),
        },
    }


def validate_result(case: dict, result: dict, revision: str, iterations: int) -> None:
    expected_hash = hashlib.sha256(json.dumps(case, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    disposition, reason = compare.prewarm_disposition(case["case_id"], result, result)
    if disposition != "COMPARE":
        raise ValueError(reason)
    if result.get("revision") != revision or result.get("case_hash") != expected_hash:
        raise ValueError("worker returned the wrong revision or canonical case hash")
    if (
        result.get("group_case_count") != 9
        or result.get("steady_state_setup_queries") != 2
        or result.get("cold_definition") != "steady_state_unseen_query"
    ):
        raise ValueError("worker did not preserve complete group/cold-query semantics")
    samples = result.get("warm_samples_us", [])
    if not isinstance(samples, list):
        raise ValueError("warm samples must be an array")
    values = [result.get("predicted_value_ms"), *samples]
    if len(samples) != iterations or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
        for value in values
    ):
        raise ValueError("invalid prediction or incomplete/non-finite warm samples")
    if statistics.median(samples) != result["warm"]["call_median_us"]:
        raise ValueError("warm median does not match raw samples")


def checkpoint(raw: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "raw.json.tmp"
    temporary.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    temporary.replace(output / "raw.json")


def analyze(raw: dict) -> dict:
    points = []
    for arm in ARMS:
        for traversal, order in CONDITIONS:
            observations = [
                observation
                for observation in raw["observations"]
                if observation["arm"] == arm
                and observation["traversal"] == traversal
                and observation["order"] == list(order)
            ]
            if len(observations) != 3:
                raise ValueError("incomplete balanced stratum")
            for index, case in enumerate(raw["cases"]):
                paired = [
                    {
                        "round": observation["block"],
                        "base": observation["lanes"]["left"][index],
                        "head": observation["lanes"]["right"][index],
                    }
                    for observation in observations
                ]
                for metric in ("cold", "warm"):
                    point = compare.compare_point(case, paired, metric)
                    if point["classification"] == "INVALID_COMPARISON":
                        raise ValueError(point["invalid_reasons"])
                    point["left_median_us"] = point.pop("base_median_us")
                    point["right_median_us"] = point.pop("head_median_us")
                    for measurement in point["rounds"]:
                        measurement["left_us"] = measurement.pop("base_us")
                        measurement["right_us"] = measurement.pop("head_us")
                    point.update({"arm": arm, "traversal": traversal, "lane_order": list(order)})
                    point["physical_revisions"] = {
                        label: raw["lanes"][installation]["revision"]
                        for label, installation in zip(("left", "right"), ARMS[arm], strict=True)
                    }
                    points.append(point)
    return {"points": points, "thresholds_unchanged": True, "purpose": "diagnostic strata, not PR approval"}


def collect(lanes: dict, output: Path, evidence: dict, cpu: int) -> dict:
    selected = selected_cases()
    raw = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": "isolate revision, traversal and temporal order; not a replacement PR merge check",
        "scope_limit": (
            "Complete Qwen3-32B groups and off-matrix worker priming are preserved. Other model groups are omitted; "
            "cross-model process/cache history differs from the full 64-case matrix, so this cannot clear that matrix."
        ),
        "provenance": evidence,
        "host": {"platform": platform.platform(), "processor": _cpu_model()},
        "configuration": {"blocks": 3, "warmup": 10, "iterations": 100, "cpu": cpu, "thread_env": THREAD_ENV},
        "lanes": {name: {key: str(value) for key, value in lane.items()} for name, lane in lanes.items()},
        "arms": ARMS,
        "cases": selected,
        "schedule": schedule(),
        "prewarm": [],
        "observations": [],
        "errors": [],
        "invalid_batches": [],
        "complete": False,
    }
    checkpoint(raw, output)

    def invoke(name: str, ordered: list[dict], *, warmup: int, iterations: int) -> list[dict]:
        results, error = run_worker_batch(
            **lanes[name], cases=ordered, warmup=warmup, iterations=iterations, cpu=cpu, timeout=120.0
        )
        try:
            if error:
                raise ValueError(f"{name}: {error}")
            if len(results) != len(ordered):
                raise ValueError(f"{name}: incomplete worker batch")
            for case, result in zip(ordered, results, strict=True):
                validate_result(case, result, lanes[name]["revision"], iterations)
        except ValueError:
            raw["invalid_batches"].append({"installation": name, "results": results, "worker_error": error})
            raise
        return results

    try:
        for name in lanes:
            print(f"prewarm {name}", flush=True)
            raw["prewarm"].append({"installation": name, "results": invoke(name, selected, warmup=0, iterations=1)})
            checkpoint(raw, output)
        for condition in raw["schedule"]:
            observation = {**condition, "lanes": {}}
            raw["observations"].append(observation)
            ordered = selected if condition["traversal"] == "forward" else list(reversed(selected))
            installations = dict(zip(("left", "right"), ARMS[condition["arm"]], strict=True))
            for label in condition["order"]:
                name = installations[label]
                print(
                    f"block={condition['block']} {condition['arm']} {condition['traversal']} {label}={name}", flush=True
                )
                results = invoke(name, ordered, warmup=10, iterations=100)
                by_id = {result["case_id"]: result for result in results}
                observation["lanes"][label] = [by_id[case["case_id"]] for case in selected]
                checkpoint(raw, output)
            for left, right in zip(observation["lanes"]["left"], observation["lanes"]["right"], strict=True):
                if left["predicted_value_ms"] != right["predicted_value_ms"]:
                    raise ValueError(f"predictions differ for {left['case_id']}")
        analysis = analyze(raw)
        (output / "comparison.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
        raw["complete"] = True
    except (ValueError, OSError) as error:
        raw["errors"].append(str(error))
        raise
    finally:
        checkpoint(raw, output)
    return analysis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "base-source",
        "head-source",
        "base-wheel",
        "head-wheel",
        "base-python",
        "head-python",
        "control-python",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if shutil.which("taskset") is None or not hasattr(os, "sched_getaffinity"):
        parser.error("this diagnostic requires Linux CPU affinity/taskset")
    evidence = provenance(args)
    lanes = {
        "base": {"python": args.base_python, "worker": args.base_source / GATE / "worker.py", "revision": BASE_SHA},
        "head": {"python": args.head_python, "worker": args.head_source / GATE / "worker.py", "revision": HEAD_SHA},
        "control": {
            "python": args.control_python,
            "worker": args.base_source / GATE / "worker.py",
            "revision": BASE_SHA,
        },
    }
    analysis = collect(lanes, args.output_dir, evidence, min(os.sched_getaffinity(0)))
    flagged = [point for point in analysis["points"] if point["classification"] == "REGRESSION"]
    summary = [
        "## PR 585 cold-query order diagnostic",
        "",
        f"Base `{BASE_SHA}`; head `{HEAD_SHA}`. Complete: 24 paired rounds, 18 cases, two arms.",
        "",
        "Three repeats per traversal/lane-order stratum; unchanged >10% and >2 us thresholds with 80% consensus.",
        (
            "A/A lanes use the same baseline wheel/revision. "
            "Stratum threshold crossings require interpretation with the A/A controls."
        ),
        (
            "A green collection job means valid complete data, not a clean PR or absence of a slowdown. "
            "No per-call CPU-time instrumentation was added."
        ),
        (
            "The two full nine-case groups retain worker priming; other model groups are omitted, "
            "so cross-model process/cache history differs from the full matrix."
        ),
        "",
        (
            f"{len(flagged)} strata crossed the threshold in all three repeats. "
            "All strata and raw samples are retained in the artifact."
        ),
        "",
        "| Arm | Traversal | First lane | Case | Metric | Median change | Crossings |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for point in flagged:
        summary.append(
            f"| {point['arm']} | {point['traversal']} | {point['lane_order'][0]} | {point['case_id']} | "
            f"{point['metric']} | {point['median_ratio'] - 1:+.2%} | {point['exceed_count']}/3 |"
        )
    (args.output_dir / "summary.md").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
