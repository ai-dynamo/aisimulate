#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the strict-native AISimulate FPE coverage matrix."""

from __future__ import annotations

import argparse
import importlib.metadata
import logging
import subprocess
import sys
from pathlib import Path

_APPLICATION_ROOT = Path(__file__).resolve().parents[2]
# Import the installed package and its compiled runtime. Prepending src here
# shadows a correctly installed wheel with a source-only aisimulate package.
sys.path.insert(0, str(_APPLICATION_ROOT))

from tools.support_matrix.fpe_support_matrix import (
    ProbeWorkload,
    build_probe_plans,
    run_probe_plans,
    write_outputs,
)


def _source_sha() -> str:
    # A manually requested SHA may differ from the workflow event's GITHUB_SHA.
    # Record the checkout that supplied the generator and discovery code.
    repository_root = _APPLICATION_ROOT.parents[1]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_version() -> str:
    try:
        return importlib.metadata.version("aisimulate")
    except importlib.metadata.PackageNotFoundError:
        import tomllib

        with (_APPLICATION_ROOT / "pyproject.toml").open("rb") as handle:
            return str(tomllib.load(handle)["project"]["version"])


def _optional_set(values: list[str] | None) -> set[str] | None:
    return set(values) if values else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a strict-native FPE coverage matrix. This does not run or certify the CLI, "
            "Sweeper, scheduler, Replay, disaggregated rate matching, deployment validity, or accuracy."
        )
    )
    parser.add_argument(
        "--output-dir", default="fpe-support-matrix", help="Destination for JSON, CSV, Markdown, and metrics"
    )
    parser.add_argument("--model", action="append", help="Only probe this model; repeat for multiple models")
    parser.add_argument("--system", action="append", help="Only probe this system; repeat for multiple systems")
    parser.add_argument("--backend", action="append", help="Only probe this backend; repeat for multiple backends")
    parser.add_argument(
        "--backend-version",
        action="append",
        help="Only probe this backend version; repeat for multiple versions",
    )
    parser.add_argument(
        "--forward-model",
        action="append",
        choices=("op_level",),
        help="Estimator mode to probe; the FPE support matrix currently requires op_level",
    )
    parser.add_argument(
        "--max-topologies-per-role",
        type=int,
        default=None,
        help="Optional deterministic cap for focused/smoke runs; full runs enumerate all resolved role topologies",
    )
    parser.add_argument("--max-workers", type=int, default=None, help="Maximum threads inside each database group")
    parser.add_argument(
        "--sdk-log-level",
        choices=("ERROR", "WARNING", "INFO"),
        default="ERROR",
        help="SDK console log level; ERROR avoids repeated per-topology warnings in full CI runs",
    )
    parser.add_argument("--isl", type=int, default=256)
    parser.add_argument("--osl", type=int, default=256)
    parser.add_argument("--prefix", type=int, default=128)
    parser.add_argument("--prefill-batch", type=int, default=1)
    parser.add_argument("--decode-batch", type=int, default=8)
    parser.add_argument("--mixed-context-tokens", type=int, default=256)
    parser.add_argument("--mixed-decode-tokens", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.getLogger("aiconfigurator_core").setLevel(args.sdk_log_level)
    workload = ProbeWorkload(
        isl=args.isl,
        osl=args.osl,
        prefix=args.prefix,
        prefill_batch=args.prefill_batch,
        decode_batch=args.decode_batch,
        mixed_context_tokens=args.mixed_context_tokens,
        mixed_decode_tokens=args.mixed_decode_tokens,
    )
    workload.validate()
    source_sha = _source_sha()
    source_version = _source_version()
    plans = build_probe_plans(
        models=_optional_set(args.model),
        systems=_optional_set(args.system),
        backends=_optional_set(args.backend),
        backend_versions=_optional_set(args.backend_version),
        forward_models=tuple(args.forward_model or ("op_level",)),
        max_topologies_per_role=args.max_topologies_per_role,
    )
    if not plans:
        build_parser().error("No FPE support-matrix plans matched the requested filters.")

    results, metrics = run_probe_plans(
        plans,
        workload=workload,
        source_version=source_version,
        source_sha=source_sha,
        max_workers=args.max_workers,
    )
    paths = write_outputs(
        results=results,
        metrics=metrics,
        output_dir=args.output_dir,
        source_version=source_version,
        source_sha=source_sha,
        workload=workload,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    print(
        f"plans={metrics.plan_count} results={metrics.result_count} "
        f"wall={metrics.wall_time_seconds:.3f}s cpu={metrics.cpu_time_seconds:.3f}s "
        f"peak_rss_kib={metrics.peak_rss_kib} workers={metrics.max_workers}"
    )


if __name__ == "__main__":
    main()
