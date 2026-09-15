#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select qualified main and release FPE data for Pages without executing artifact code."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
import zipfile
from pathlib import Path

ARTIFACT_NAME = "fpe-support-matrix-web"
DATA_PREFIX = "python/aisimulate/src/aiconfigurator_core/systems/fpe_support_matrix/"
WORKFLOWS = {".github/workflows/fpe-support-matrix.yml", ".github/workflows/nightly-ci.yml"}
QUALIFICATION = "complete_native_fpe_reports_and_required_probes"
# Producer contract: tools/support_matrix/qualify_fpe_support_matrix.py::STATUSES.
PROBE_STATUSES = {
    "PASS",
    "SDK_UNREPRESENTABLE",
    "PERF_DATA_MISSING",
    "MODEL_UNSUPPORTED",
    "HW_INCOMPATIBLE",
    "FRAMEWORK_INCOMPATIBLE",
    "BUILD_FAILED",
    "QUERY_FAILED",
}
CAPABILITY_FIELDS = ("HuggingFaceID", "Architecture", "System", "Backend", "Version")


class SnapshotUnavailable(ValueError):
    """No retained qualification is available for this branch."""


def branch_path(branch: str) -> str:
    if branch == "main":
        return "."
    if not branch.startswith("release/") or not all(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", part) for part in branch.split("/")
    ):
        raise ValueError(f"unsupported FPE branch: {branch}")
    return f"branches/{branch}"


def github(repository: str, endpoint: str) -> bytes:
    return subprocess.check_output(["gh", "api", f"repos/{repository}/{endpoint}"])


def _strict_json(data: bytes):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON member: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"invalid JSON constant: {value}")

    return json.loads(data, object_pairs_hook=unique_object, parse_constant=reject_constant)


def qualified_files(archive: bytes, source_sha: str) -> dict[str, bytes] | None:
    """Read only qualified CSVs; old artifacts without a qualification are ineligible."""
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive entries")
        if "fpe-qualification.json" not in names:
            return None
        report = _strict_json(bundle.read("fpe-qualification.json"))
        if (
            type(report.get("schema_version")) is not int
            or report["schema_version"] != 1
            or report.get("qualification") != QUALIFICATION
            or report.get("source_sha") != source_sha
            or not re.fullmatch(r"[0-9a-f]{64}", str(report.get("wheel_sha256", "")))
            or type(report.get("shard_count")) is not int
            or report["shard_count"] <= 0
            or type(report.get("required_probe_count")) is not int
            or report["required_probe_count"] <= 0
        ):
            raise ValueError("invalid FPE qualification or mismatched source commit")
        statuses = report.get("status_counts", {})
        if not isinstance(statuses, dict) or not statuses.keys() <= PROBE_STATUSES:
            raise ValueError("FPE qualification has unknown status-count keys")
        if any(type(count) is not int or count < 0 for count in statuses.values()):
            raise ValueError("FPE qualification status counts must be nonnegative integers")
        if not statuses.get("PASS", 0) or statuses.get("BUILD_FAILED", 0) or statuses.get("QUERY_FAILED", 0):
            raise ValueError("FPE qualification has no passes or unexpected native failures")
        index = _strict_json(bundle.read(DATA_PREFIX + "index.json"))
        files = index.get("files") if isinstance(index, dict) else None
        if not isinstance(files, list) or not files or len(set(files)) != len(files):
            raise ValueError("empty or duplicate FPE dataset index")
        selected = {"index.json": json.dumps(index).encode()}
        shards = set()
        capabilities = set()
        for filename in files:
            if not isinstance(filename, str) or not re.fullmatch(r"[a-z0-9_]+\.csv", filename):
                raise ValueError("unsafe FPE dataset filename")
            data = bundle.read(DATA_PREFIX + filename)
            reader = csv.DictReader(io.StringIO(data.decode("utf-8")), strict=True)
            headers = reader.fieldnames or []
            if not headers or len(headers) != len(set(headers)) or any(not field.strip() for field in headers):
                raise ValueError(f"empty or duplicate CSV headers: {filename}")
            rows = list(reader)
            if not rows:
                raise ValueError(f"empty FPE dataset: {filename}")
            for row in rows:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"inconsistent CSV record width: {filename}")
                if row.get("SourceSHA") != source_sha or row.get("System") != filename[:-4]:
                    raise ValueError(f"mixed source or system in FPE dataset: {filename}")
                if row.get("Status") not in {"PASS", "FAIL", "HW_INCOMPATIBLE", "FRAMEWORK_INCOMPATIBLE"}:
                    raise ValueError(f"invalid FPE status: {filename}")
                if not all(row.get(key) for key in CAPABILITY_FIELDS):
                    raise ValueError(f"incomplete FPE identity: {filename}")
                identity = tuple(row[key] for key in CAPABILITY_FIELDS)
                if identity in capabilities:
                    raise ValueError(f"duplicate FPE capability: {identity}")
                capabilities.add(identity)
                shards.add((row["System"], row["Backend"]))
            selected[filename] = data
        if len(shards) != report["shard_count"]:
            raise ValueError("FPE dataset does not contain all qualified shards")
        return selected


def list_artifacts(repository: str, *, api=github) -> list[dict]:
    artifacts = []
    page = 1
    while True:
        batch = json.loads(api(repository, f"actions/artifacts?name={ARTIFACT_NAME}&per_page=100&page={page}"))[
            "artifacts"
        ]
        artifacts.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return artifacts


def prepare(
    repository: str,
    repo_root: Path,
    destination: Path,
    *,
    branch="main",
    history_ref="HEAD",
    artifacts=None,
    api=github,
) -> dict:
    """Prefer the newest tested commit in this branch's history over a later old rerun."""
    branch_path(branch)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("invalid repository")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("FPE output directory must be empty")
    ancestors = subprocess.check_output(
        ["git", "rev-list", "--topo-order", history_ref, "--"], cwd=repo_root, text=True
    ).splitlines()
    ranks = {sha: index for index, sha in enumerate(ancestors)}
    if artifacts is None:
        artifacts = list_artifacts(repository, api=api)
    candidates = []
    for artifact in artifacts:
        identity = artifact.get("workflow_run") or {}
        if (
            artifact.get("name") != ARTIFACT_NAME
            or identity.get("head_branch") != branch
            or identity.get("head_sha") not in ranks
        ):
            continue
        candidates.append(artifact)
    candidates.sort(key=lambda a: (ranks[a["workflow_run"]["head_sha"]], -a["id"]))
    for artifact in candidates:
        run_id = artifact["workflow_run"]["id"]
        run = json.loads(api(repository, f"actions/runs/{run_id}"))
        source_sha = artifact["workflow_run"]["head_sha"]
        if (
            run.get("path") not in WORKFLOWS
            or run.get("head_repository", {}).get("full_name") != repository
            or run.get("head_branch") != branch
            or run.get("head_sha") != source_sha
            or run.get("event") not in {"schedule", "workflow_dispatch"}
            or run.get("status") != "completed"
            or run.get("conclusion") != "success"
        ):
            continue
        if artifact.get("expired"):
            raise SnapshotUnavailable(
                f"newest successful {branch} FPE artifact has expired; refresh FPE before publishing"
            )
        files = qualified_files(api(repository, f"actions/artifacts/{artifact['id']}/zip"), source_sha)
        if files is None:
            continue
        snapshot = {
            "branch": branch,
            "source_sha": source_sha,
            "generated_at": artifact["created_at"],
            "run_url": f"https://github.com/{repository}/actions/runs/{run_id}",
            "artifact_id": artifact["id"],
        }
        index = json.loads(files["index.json"])
        index["snapshot"] = snapshot
        files["index.json"] = (json.dumps(index, indent=2) + "\n").encode()
        destination.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            (destination / name).write_bytes(data)
        return snapshot
    raise SnapshotUnavailable(f"no retained qualified {branch} FPE artifact; run FPE Support Matrix on that branch")


def prepare_branches(repository: str, repo_root: Path, destination: Path, *, api=github) -> dict:
    """Publish a catalog from fetched origin refs; never borrow another branch's data."""
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("FPE output directory must be empty")
    refs = subprocess.check_output(
        ["git", "for-each-ref", "--format=%(refname)", "refs/remotes/origin/release/"],
        cwd=repo_root,
        text=True,
    ).splitlines()
    branches = {"main": "HEAD"}
    for ref in sorted(refs):
        name = ref.removeprefix("refs/remotes/origin/")
        branch_path(name)
        branches[name] = ref
    artifacts = list_artifacts(repository, api=api)
    catalog = {"schema_version": 1, "default": "main", "branches": []}
    for branch, ref in branches.items():
        path = branch_path(branch)
        entry = {"name": branch, "path": path, "status": "available"}
        try:
            prepare(
                repository, repo_root, destination / path, branch=branch, history_ref=ref, artifacts=artifacts, api=api
            )
        except SnapshotUnavailable as exc:
            if branch == "main":
                raise
            entry = {"name": branch, "status": "unavailable", "reason": str(exc)}
        catalog["branches"].append(entry)
    (destination / "branches.json").write_text(json.dumps(catalog, indent=2) + "\n")
    return catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_branches(args.repository, args.repo_root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
