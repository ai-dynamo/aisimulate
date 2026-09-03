# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from tools.cuda_graph_profiles.infx import download_locked_sources, resolve_manifest
from tools.cuda_graph_profiles.publish import publish_database, validate_database
from tools.cuda_graph_profiles.train import train_model

TOOL_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TOOL_DIR.parents[1]
DEFAULT_MANIFEST = TOOL_DIR / "infx_sources.yaml"
DEFAULT_LOCK = TOOL_DIR / "infx_sources.lock.json"
DEFAULT_DATABASE = PACKAGE_DIR / "src/aiconfigurator_core/systems/cuda_graph_profiles/v1"
DEFAULT_REPORTS = TOOL_DIR / "reports/v1"


def _write_validation_report(reports_dir: Path, result: dict[str, object]) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "validation.report.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _concise_result(result: dict[str, object]) -> dict[str, object]:
    if "sources" in result:
        sources = result["sources"]
        artifacts = [artifact for source in sources for artifact in source["artifacts"]]
        return {
            "artifact_count": len(artifacts),
            "locked_file_count": sum(len(artifact.get("files", [])) for artifact in artifacts),
            "source_count": len(sources),
        }
    if "artifact_version" in result:
        return {
            "enabled": result["enabled"],
            "gate_failures": result["gate_failures"],
            "holdout_metrics": result["holdout_metrics"],
            "training_profile_count": result["training_profile_count"],
        }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce the reviewed InfX CUDA graph profile database")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--database-dir", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("resolve", "resolve reviewed run/artifact names into immutable GitHub IDs"),
        ("download", "download and checksum every locked extracted file"),
        ("parse", "parse, reconcile, and publish the measurement database"),
        ("validate", "validate the packaged database and checksums"),
        ("train", "fit and safety-gate the deterministic reservation model"),
        ("reproduce", "run resolve, download, parse, validate, and train"),
    ):
        subparsers.add_parser(command, help=help_text)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "resolve":
        result = resolve_manifest(args.manifest, args.lock)
    elif args.command == "download":
        result = download_locked_sources(args.lock, args.cache_dir)
    elif args.command == "parse":
        result = publish_database(args.lock, args.cache_dir, args.database_dir, args.reports_dir)
    elif args.command == "validate":
        result = validate_database(args.database_dir)
        _write_validation_report(args.reports_dir, result)
    elif args.command == "train":
        validate_database(args.database_dir, validate_model=False)
        result = train_model(
            args.database_dir / "cuda_graph_profiles.parquet",
            args.database_dir / "cuda_graph_reservation_model.json",
        )
        validation = validate_database(args.database_dir)
        _write_validation_report(args.reports_dir, validation)
    else:
        resolve_manifest(args.manifest, args.lock)
        download_locked_sources(args.lock, args.cache_dir)
        publish_database(args.lock, args.cache_dir, args.database_dir, args.reports_dir)
        result = train_model(
            args.database_dir / "cuda_graph_profiles.parquet",
            args.database_dir / "cuda_graph_reservation_model.json",
        )
        validation = validate_database(args.database_dir)
        _write_validation_report(args.reports_dir, validation)
    print(json.dumps(_concise_result(result), indent=2, sort_keys=True))
