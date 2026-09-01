#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the explicitly allowlisted public AISimulate GitHub Pages artifact."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = Path("python/aisimulate/docs")
SYSTEMS_ROOT = Path("python/aisimulate/src/aiconfigurator_core/systems")

# Directories are opt-in so adding internal documentation under docs/ never
# publishes it accidentally. Every listed page is now part of the required
# public surface and its absence must fail the build.
PUBLIC_PAGE_DIRECTORIES = {
    "support-matrix": True,
    "e2e-accuracy": True,
    "fpe-support-matrix": True,
}
PUBLIC_ASSET_SUFFIXES = {".css", ".html", ".js", ".json", ".png", ".svg", ".webp"}
PUBLIC_DATASETS = {
    "support-matrix": "support_matrix",
    "fpe-support-matrix": "fpe_support_matrix",
}


class PagesBuildError(RuntimeError):
    """Raised when the public Pages artifact contract is invalid."""


def _copy_file(source: Path, destination: Path) -> None:
    if source.is_symlink():
        raise PagesBuildError(f"public artifact source cannot be a symlink: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_page_directory(source: Path, destination: Path) -> None:
    if not (source / "index.html").is_file():
        raise PagesBuildError(f"public page has no index.html: {source}")
    for asset in sorted(source.rglob("*")):
        if not asset.is_file() or asset.suffix.lower() not in PUBLIC_ASSET_SUFFIXES:
            continue
        _copy_file(asset, destination / asset.relative_to(source))


def _copy_dataset(source: Path, destination: Path) -> None:
    index_path = source / "index.json"
    if not index_path.is_file():
        raise PagesBuildError(f"public dataset index is missing: {index_path}")

    try:
        index = json.loads(index_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise PagesBuildError(f"cannot read public dataset index: {index_path}") from exc

    files = index.get("files") if isinstance(index, dict) else None
    if not isinstance(files, list) or not files:
        raise PagesBuildError(f"public dataset index has no files: {index_path}")

    _copy_file(index_path, destination / "index.json")
    for filename in files:
        if not isinstance(filename, str) or Path(filename).name != filename or Path(filename).suffix.lower() != ".csv":
            raise PagesBuildError(f"unsafe public dataset entry in {index_path}: {filename!r}")
        csv_path = source / filename
        if not csv_path.is_file():
            raise PagesBuildError(f"public dataset file is missing: {csv_path}")
        _copy_file(csv_path, destination / filename)


def build_site(repo_root: Path, output_dir: Path) -> set[Path]:
    """Build the public site and return its files relative to ``output_dir``."""
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir == repo_root:
        raise PagesBuildError("the Pages output directory cannot be the repository root")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise PagesBuildError(f"the Pages output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    docs_root = repo_root / DOCS_ROOT
    index_path = docs_root / "index.html"
    if not index_path.is_file():
        raise PagesBuildError(f"public landing page is missing: {index_path}")
    _copy_file(index_path, output_dir / "index.html")
    (output_dir / ".nojekyll").touch()

    for public_name, required in PUBLIC_PAGE_DIRECTORIES.items():
        page_source = docs_root / public_name
        if not page_source.is_dir():
            if required:
                raise PagesBuildError(f"required public page is missing: {page_source}")
            continue
        _copy_page_directory(page_source, output_dir / public_name)

        dataset_name = PUBLIC_DATASETS.get(public_name)
        if dataset_name:
            _copy_dataset(
                repo_root / SYSTEMS_ROOT / dataset_name,
                output_dir / "data" / public_name,
            )

    return {path.relative_to(output_dir) for path in output_dir.rglob("*") if path.is_file()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    files = build_site(args.repo_root, args.output_dir)
    print(f"Built {len(files)} public files in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
