#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the explicitly allowlisted public AISimulate GitHub Pages artifact."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from datetime import UTC, datetime
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


def _copy_fpe_branches(source: Path, destination: Path, *, require_catalog: bool = False) -> None:
    """Copy only cataloged branch datasets, retaining the legacy main data path."""
    catalog_path = source / "branches.json"
    if require_catalog and not catalog_path.is_file():
        raise PagesBuildError("prepared FPE data requires branches.json")
    catalog = (
        json.loads(catalog_path.read_text())
        if catalog_path.is_file()
        else {
            "schema_version": 1,
            "default": "main",
            "preview": True,
            "branches": [{"name": "main", "path": ".", "status": "available"}],
        }
    )
    if catalog.get("schema_version") != 1 or catalog.get("default") != "main":
        raise PagesBuildError("invalid FPE branch catalog")
    if require_catalog and catalog.get("preview"):
        raise PagesBuildError("prepared FPE data cannot be a repository preview")
    branches = catalog.get("branches")
    if not isinstance(branches, list) or not branches:
        raise PagesBuildError("empty FPE branch catalog")
    names = set()
    for branch in branches:
        name = branch.get("name", "")
        if (
            not isinstance(name, str)
            or name in names
            or not (name == "main" or re.fullmatch(r"release/[A-Za-z0-9][A-Za-z0-9._-]*", name))
        ):
            raise PagesBuildError("invalid or duplicate FPE branch name")
        names.add(name)
        if branch.get("status") == "unavailable" and name != "main" and "path" not in branch:
            continue
        path = "." if name == "main" else f"branches/{name}"
        if branch.get("status") != "available" or branch.get("path") != path:
            raise PagesBuildError("invalid FPE branch dataset path or status")
        if name != "main":
            _copy_dataset(source / path, destination / path)
    if "main" not in names:
        raise PagesBuildError("FPE branch catalog must include main")
    (destination / "branches.json").write_text(json.dumps(catalog, indent=2) + "\n")


def _record_legacy_snapshot(repo_root: Path, destination: Path) -> None:
    """Describe the copied legacy data without treating a commit date as a test run."""
    index_path = destination / "index.json"
    index = json.loads(index_path.read_text())
    snapshot = {"kind": "historical", "qualification": "not_recorded"}
    dataset = SYSTEMS_ROOT / "support_matrix"
    paths = [str(dataset / name) for name in ["index.json", *index["files"]]]

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=repo_root, text=True, stderr=subprocess.DEVNULL, timeout=10
        ).strip()

    try:
        # Archives, shallow clones, and locally changed data cannot establish
        # the last committed update of the exact dataset being displayed.
        if Path(git("rev-parse", "--show-toplevel")).resolve() != repo_root.resolve():
            raise ValueError("dataset is outside the repository root")
        if git("rev-parse", "--is-shallow-repository") != "false":
            raise ValueError("dataset history is incomplete")
        git("ls-files", "--error-unmatch", "--", *paths)
        git("diff", "--quiet", "HEAD", "--", *paths)
        commit, updated_at = git("log", "-1", "--format=%H%n%cI", "HEAD", "--", *paths).splitlines()
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("invalid data commit")
        updated = datetime.fromisoformat(updated_at)
        if updated.tzinfo is None:
            raise ValueError("data commit timestamp requires a time zone")
        snapshot.update(
            data_commit=commit,
            data_updated_at=updated.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        # Keep the historical matrix useful without inventing a date or
        # inheriting unverified metadata from its source index.
        pass
    index["snapshot"] = snapshot
    index_path.write_text(json.dumps(index, indent=2) + "\n")


def build_site(repo_root: Path, output_dir: Path, *, fpe_data_dir: Path | None = None) -> set[Path]:
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
                fpe_data_dir
                if public_name == "fpe-support-matrix" and fpe_data_dir is not None
                else repo_root / SYSTEMS_ROOT / dataset_name,
                output_dir / "data" / public_name,
            )
            if public_name == "fpe-support-matrix":
                _copy_fpe_branches(
                    fpe_data_dir if fpe_data_dir is not None else repo_root / SYSTEMS_ROOT / dataset_name,
                    output_dir / "data" / public_name,
                    require_catalog=fpe_data_dir is not None,
                )
            elif public_name == "support-matrix":
                _record_legacy_snapshot(repo_root, output_dir / "data" / public_name)

    return {path.relative_to(output_dir) for path in output_dir.rglob("*") if path.is_file()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fpe-data-dir", type=Path, help="Qualified FPE data prepared for this deployment")
    args = parser.parse_args()

    files = build_site(args.repo_root, args.output_dir, fpe_data_dir=args.fpe_data_dir)
    print(f"Built {len(files)} public files in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
