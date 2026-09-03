# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from tools.cuda_graph_profiles.common import SCHEMA_VERSION, sha256_file, stable_hash


class SourceResolutionError(RuntimeError):
    """Raised when a reviewed InfX source cannot be pinned safely."""


def _gh_json(endpoint: str) -> dict[str, Any]:
    result = subprocess.run(
        ["gh", "api", endpoint],
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise SourceResolutionError(f"GitHub endpoint returned a non-object: {endpoint}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise SourceResolutionError("source manifest has an unsupported schema")
    return value


def _run_artifacts(repository: str, run_id: int) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    page = 1
    while True:
        response = _gh_json(f"repos/{repository}/actions/runs/{run_id}/artifacts?per_page=100&page={page}")
        batch = response.get("artifacts", [])
        if not isinstance(batch, list):
            raise SourceResolutionError(f"run {run_id} returned an invalid artifact list")
        artifacts.extend(batch)
        if len(batch) < 100:
            return artifacts
        page += 1


def resolve_manifest(manifest_path: Path, lock_path: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    repository = str(manifest["repository"])
    locked_sources: list[dict[str, Any]] = []
    for source in manifest.get("sources", []):
        run_id = int(source["run_id"])
        run = _gh_json(f"repos/{repository}/actions/runs/{run_id}")
        if run.get("status") != "completed" or run.get("conclusion") != "success":
            raise SourceResolutionError(f"run {run_id} is not completed successfully")
        expected_attempt = int(source["run_attempt"])
        if int(run["run_attempt"]) != expected_attempt:
            raise SourceResolutionError(
                f"run {run_id} attempt changed: expected {expected_attempt}, got {run['run_attempt']}"
            )

        artifacts_by_name: dict[str, list[dict[str, Any]]] = {}
        for artifact in _run_artifacts(repository, run_id):
            artifacts_by_name.setdefault(artifact["name"], []).append(artifact)
        locked_artifacts: list[dict[str, Any]] = []
        for requested in source.get("artifacts", []):
            name = str(requested["name"])
            matches = artifacts_by_name.get(name, [])
            if not matches:
                raise SourceResolutionError(f"run {run_id} does not contain reviewed artifact {name!r}")
            if len(matches) != 1:
                raise SourceResolutionError(f"run {run_id} contains multiple artifacts named {name!r}")
            artifact = matches[0]
            if artifact.get("expired"):
                raise SourceResolutionError(f"reviewed artifact {artifact['id']} has expired")
            locked_artifacts.append(
                {
                    "artifact_id": int(artifact["id"]),
                    "artifact_name": name,
                    "artifact_size_bytes": int(artifact["size_in_bytes"]),
                    "created_at": artifact["created_at"],
                    "files": [],
                    "profile_key": requested["profile_key"],
                    "role": requested["role"],
                }
            )
        locked_sources.append(
            {
                "artifacts": locked_artifacts,
                "head_sha": run["head_sha"],
                "html_url": run["html_url"],
                "run_attempt": int(run["run_attempt"]),
                "run_id": run_id,
            }
        )

    lock = {
        "manifest_sha256": sha256_file(manifest_path),
        "repository": repository,
        "schema_version": SCHEMA_VERSION,
        "sources": locked_sources,
    }
    _write_json(lock_path, lock)
    return lock


def _validated_member_path(root: Path, name: str) -> Path:
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts:
        raise SourceResolutionError(f"archive contains unsafe member {name!r}")
    destination = root.joinpath(*pure.parts)
    if not destination.resolve().is_relative_to(root.resolve()):
        raise SourceResolutionError(f"archive member escapes extraction root: {name!r}")
    return destination


def _extract_zip(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = _validated_member_path(destination, member.filename)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _extract_nested_tar(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:*") as bundle:
        for member in bundle.getmembers():
            if not member.isfile():
                if member.issym() or member.islnk():
                    raise SourceResolutionError(f"nested archive contains a link: {member.name!r}")
                continue
            target = _validated_member_path(destination, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise SourceResolutionError(f"cannot read nested archive member {member.name!r}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _download_artifact(repository: str, artifact_id: int, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip") as stream:
        subprocess.run(
            ["gh", "api", f"repos/{repository}/actions/artifacts/{artifact_id}/zip"],
            check=True,
            stdout=stream,
        )
        stream.flush()
        _extract_zip(Path(stream.name), destination)
    for archive in sorted(destination.rglob("*.tar.gz")):
        nested = archive.parent / f"{archive.name.removesuffix('.tar.gz')}.extracted"
        _extract_nested_tar(archive, nested)


def download_locked_sources(lock_path: Path, cache_dir: Path) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for source in lock["sources"]:
        for artifact in source["artifacts"]:
            destination = cache_dir / str(source["run_id"]) / str(artifact["artifact_id"])
            if destination.exists():
                shutil.rmtree(destination)
            _download_artifact(lock["repository"], int(artifact["artifact_id"]), destination)
            files = []
            for path in sorted(candidate for candidate in destination.rglob("*") if candidate.is_file()):
                files.append(
                    {
                        "relative_path": path.relative_to(destination).as_posix(),
                        "sha256": sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                )
            if not files:
                raise SourceResolutionError(f"artifact {artifact['artifact_id']} extracted no files")
            artifact["files"] = files
            artifact["extracted_content_sha256"] = stable_hash(files)
    _write_json(lock_path, lock)
    return lock


def verify_cache(lock: dict[str, Any], cache_dir: Path) -> None:
    for source in lock["sources"]:
        for artifact in source["artifacts"]:
            root = cache_dir / str(source["run_id"]) / str(artifact["artifact_id"])
            if not artifact.get("files"):
                raise SourceResolutionError(f"artifact {artifact['artifact_id']} has no locked file hashes")
            for expected in artifact["files"]:
                path = root / expected["relative_path"]
                if not path.is_file() or sha256_file(path) != expected["sha256"]:
                    raise SourceResolutionError(
                        f"artifact {artifact['artifact_id']} file failed checksum: {expected['relative_path']}"
                    )
