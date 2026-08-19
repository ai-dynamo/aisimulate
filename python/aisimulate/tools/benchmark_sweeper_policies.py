# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reproducible rapid-versus-thorough Sweeper comparison fixture."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

from aisimulate.sweeper import (
    ReplayReport,
    RunnerCapabilities,
    SearchPolicy,
    SmartSearchConfig,
    Sweeper,
)


class _DeterministicRunner:
    def run(self, spec, *, output_requirements=None):
        del output_requirements
        deployment = spec.backend_deployment
        engine_args = deployment.agg_engine_args or deployment.decode_engine_args or {}
        max_num_seqs = float(engine_args.get("max_num_seqs", 1))
        return ReplayReport(
            metrics={
                "output_throughput_tok_s": max_num_seqs * deployment.num_workers,
                "gpu_hours": 1.0,
            }
        )

    def close(self):
        pass


class _DeterministicRunnerFactory:
    def capabilities(self):
        return RunnerCapabilities(supported_backend_topologies=(("*", "*"),))

    def create(self, worker_id):
        del worker_id
        return _DeterministicRunner()


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    diff_bytes = diff.stdout if diff.returncode == 0 else b""
    return {
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": bool(diff_bytes),
        "diff_sha256": hashlib.sha256(diff_bytes).hexdigest() if diff_bytes else None,
    }


def _run_one(config_path: Path, policy: SearchPolicy) -> dict[str, Any]:
    config = SmartSearchConfig.from_yaml(config_path)
    config = config.model_copy(update={"sweep": config.sweep.model_copy(update={"policy": policy})})
    sweeper = Sweeper(
        runner_factory=_DeterministicRunnerFactory(),
        show_progress=False,
    )

    tracemalloc.start()
    cpu_started = time.process_time()
    wall_started = time.perf_counter()
    candidates = sweeper.run(config)
    wall_seconds = time.perf_counter() - wall_started
    cpu_seconds = time.process_time() - cpu_started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    report = sweeper.last_report
    if report is None:
        raise RuntimeError("Sweeper completed without SearchExecutionReport")
    best = candidates[0] if candidates else None
    return {
        "policy": policy.value,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "peak_python_bytes": peak_bytes,
        "result_count": len(candidates),
        "best_score": best.score if best is not None else None,
        "best_config": best.config if best is not None else None,
        "execution": report.as_dict(),
    }


def _child_command(config_path: Path, policy: SearchPolicy) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(config_path),
        "--_policy",
        policy.value,
    ]


def _run_isolated(config_path: Path, policy: SearchPolicy) -> dict[str, Any]:
    result = subprocess.run(
        _child_command(config_path, policy),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{policy.value} benchmark failed with exit {result.returncode}:\n{result.stderr}")
    return json.loads(result.stdout)


def _score_ratio(rapid: float | None, thorough: float | None) -> float | None:
    if rapid is None or thorough is None:
        return None
    if thorough <= 0:
        raise ValueError("--min-rapid-score-ratio requires a positive thorough score; use a throughput-like fixture")
    return rapid / thorough


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-rapid-score-ratio", type=float)
    parser.add_argument("--max-rapid-time-ratio", type=float)
    parser.add_argument("--_policy", choices=[policy.value for policy in SearchPolicy])
    args = parser.parse_args()
    config_path = args.config.resolve()

    if args._policy is not None:
        print(json.dumps(_run_one(config_path, SearchPolicy(args._policy)), sort_keys=True))
        return

    repo_root = Path(__file__).resolve().parents[3]
    raw_config = config_path.read_bytes()
    thorough = _run_isolated(config_path, SearchPolicy.THOROUGH)
    rapid = _run_isolated(config_path, SearchPolicy.RAPID)
    score_ratio = _score_ratio(rapid["best_score"], thorough["best_score"])
    time_ratio = rapid["wall_seconds"] / max(thorough["wall_seconds"], 1e-12)
    artifact = {
        "schema_version": 1,
        "source": _git_provenance(repo_root),
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "aisimulate": _version("aisimulate"),
            "google_vizier": _version("google-vizier"),
            "jax": _version("jax"),
        },
        "rapid": rapid,
        "thorough": thorough,
        "comparison": {
            "rapid_score_ratio": score_ratio,
            "rapid_time_ratio": time_ratio,
        },
    }

    failures = []
    if args.min_rapid_score_ratio is not None and (score_ratio is None or score_ratio < args.min_rapid_score_ratio):
        failures.append(f"rapid score ratio {score_ratio!r} < {args.min_rapid_score_ratio}")
    if args.max_rapid_time_ratio is not None and time_ratio > args.max_rapid_time_ratio:
        failures.append(f"rapid time ratio {time_ratio:.6g} > {args.max_rapid_time_ratio}")

    rendered = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered)
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
