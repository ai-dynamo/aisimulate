#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select qualified main-branch FPE data for Pages without executing artifact code."""

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


def github(repository: str, endpoint: str) -> bytes:
    return subprocess.check_output(["gh", "api", f"repos/{repository}/{endpoint}"])


def qualified_files(archive: bytes, source_sha: str) -> dict[str, bytes] | None:
    """Read only qualified CSVs; old artifacts without a qualification are ineligible."""
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive entries")
        if "fpe-qualification.json" not in names:
            return None
        report = json.loads(bundle.read("fpe-qualification.json"))
        if (
            report.get("schema_version") != 1
            or report.get("qualification") != QUALIFICATION
            or report.get("source_sha") != source_sha
            or not re.fullmatch(r"[0-9a-f]{64}", str(report.get("wheel_sha256", "")))
            or not isinstance(report.get("shard_count"), int)
            or report["shard_count"] <= 0
            or not isinstance(report.get("required_probe_count"), int)
            or report["required_probe_count"] <= 0
        ):
            raise ValueError("invalid FPE qualification or mismatched source commit")
        statuses = report.get("status_counts", {})
        if not statuses.get("PASS", 0) or statuses.get("BUILD_FAILED", 0) or statuses.get("QUERY_FAILED", 0):
            raise ValueError("FPE qualification has no passes or unexpected native failures")
        index = json.loads(bundle.read(DATA_PREFIX + "index.json"))
        files = index.get("files") if isinstance(index, dict) else None
        if not isinstance(files, list) or not files or len(set(files)) != len(files):
            raise ValueError("empty or duplicate FPE dataset index")
        selected = {"index.json": json.dumps(index).encode()}
        shards = set()
        for filename in files:
            if not isinstance(filename, str) or not re.fullmatch(r"[a-z0-9_]+\.csv", filename):
                raise ValueError("unsafe FPE dataset filename")
            data = bundle.read(DATA_PREFIX + filename)
            rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
            if not rows:
                raise ValueError(f"empty FPE dataset: {filename}")
            for row in rows:
                if row.get("SourceSHA") != source_sha or row.get("System") != filename[:-4]:
                    raise ValueError(f"mixed source or system in FPE dataset: {filename}")
                if row.get("Status") not in {"PASS", "FAIL", "HW_INCOMPATIBLE", "FRAMEWORK_INCOMPATIBLE"}:
                    raise ValueError(f"invalid FPE status: {filename}")
                if not all(row.get(key) for key in ("HuggingFaceID", "Architecture", "Backend", "Version")):
                    raise ValueError(f"incomplete FPE identity: {filename}")
                shards.add((row["System"], row["Backend"]))
            selected[filename] = data
        if len(shards) != report["shard_count"]:
            raise ValueError("FPE dataset does not contain all qualified shards")
        return selected


def prepare(repository: str, repo_root: Path, destination: Path, *, api=github) -> dict:
    """Prefer the newest tested main commit, not the most recently finished old run."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("invalid repository")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("FPE output directory must be empty")
    ancestors = subprocess.check_output(
        ["git", "rev-list", "--topo-order", "HEAD"], cwd=repo_root, text=True
    ).splitlines()
    ranks = {sha: index for index, sha in enumerate(ancestors)}
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
    candidates = []
    for artifact in artifacts:
        identity = artifact.get("workflow_run") or {}
        if (
            artifact.get("name") != ARTIFACT_NAME
            or identity.get("head_branch") != "main"
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
            or run.get("head_branch") != "main"
            or run.get("head_sha") != source_sha
            or run.get("event") not in {"schedule", "workflow_dispatch"}
            or run.get("status") != "completed"
            or run.get("conclusion") != "success"
        ):
            continue
        if artifact.get("expired"):
            raise ValueError("newest successful main FPE artifact has expired; refresh FPE before publishing")
        files = qualified_files(api(repository, f"actions/artifacts/{artifact['id']}/zip"), source_sha)
        if files is None:
            continue
        snapshot = {
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
    raise ValueError("no retained qualified main FPE artifact; run FPE Support Matrix against the current main SHA")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.repository, args.repo_root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
