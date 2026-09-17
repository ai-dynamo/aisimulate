#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select qualified FPM aggregates from trusted GitHub Actions campaigns."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import urllib.error
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fpm_accuracy.contract import REPO, artifact_key, eligible_branch, keys, require, strict_json, validate_summary
from prepare_e2e_accuracy_pages import ancestor, api, api_items

WORKFLOW = ".github/workflows/fpm-accuracy.yml"


def completed_runs():
    # Include all retained campaigns, even after many failed/manual runs.
    since = (datetime.now(UTC) - timedelta(days=90)).date().isoformat()
    page = 1
    while True:
        batch = api(
            "actions/workflows/fpm-accuracy.yml/runs?status=completed&branch=main"
            f"&created=%3E%3D{since}&per_page=100&page={page}"
        )["workflow_runs"]
        yield from batch
        if len(batch) < 100:
            return
        page += 1


def unpack(archive: bytes) -> dict:
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        require(sorted(bundle.namelist()) == ["qualification.json", "summary.json"], "unexpected FPM artifact files")
        require(all(info.file_size <= 16 * 1024 * 1024 for info in bundle.infolist()), "oversized FPM artifact")
        raw = bundle.read("summary.json")
        summary = validate_summary(strict_json(raw))
        qualification = strict_json(bundle.read("qualification.json"))
    keys(qualification, ("schema_version", "snapshot", "summary_sha256"))
    require(
        type(qualification["schema_version"]) is int and qualification["schema_version"] == 1,
        "invalid qualification schema",
    )
    require(qualification["snapshot"] == summary["snapshot"], "qualification identity mismatch")
    require(qualification["summary_sha256"] == hashlib.sha256(raw).hexdigest(), "summary checksum mismatch")
    return summary


def trusted_run(run):
    return (
        run.get("event") in {"schedule", "workflow_dispatch"}
        and run.get("head_branch") == "main"
        and run.get("path") == WORKFLOW
        and run.get("status") == "completed"
        and run.get("conclusion") in {"success", "failure"}
        and run.get("repository", {}).get("full_name") == REPO
        and run.get("head_repository", {}).get("full_name") == REPO
    )


def validate_artifact(archive, run, name, jobs):
    require(trusted_run(run), "untrusted FPM campaign")
    summary = unpack(archive)
    snapshot = summary["snapshot"]
    require(
        snapshot["run_id"] == str(run["id"]) and snapshot["run_attempt"] == str(run["run_attempt"]), "wrong run attempt"
    )
    require(snapshot["evaluator_sha"] == run["head_sha"], "wrong evaluator revision")
    key = artifact_key(snapshot["branch"])
    require(name == "fpm-accuracy-web-" + key, "wrong branch artifact")
    expected = f"Qualify FPM accuracy ({key})"
    matches = [job for job in jobs if job["name"] == expected or job["name"].endswith(" / " + expected)]
    require(len(matches) == 1, "missing or ambiguous branch job")
    job = matches[0]
    require(
        job.get("status") == "completed"
        and job.get("conclusion") == "success"
        and job.get("run_id") == run["id"]
        and job.get("head_sha") == run["head_sha"],
        "branch job did not qualify",
    )
    return summary


def prepare(repo: Path, output: Path):
    refs = subprocess.check_output(
        ["git", "-C", str(repo), "for-each-ref", "--format=%(refname:strip=3)", "refs/remotes/origin/release/"],
        text=True,
    ).splitlines()
    branches = {"main", *(name for name in refs if eligible_branch(name))}
    selected = {}
    try:
        runs = list(completed_runs())
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        runs = []  # Pages may deploy before the first scheduled evaluation.
    for listed in runs:
        run = api(f"actions/runs/{listed['id']}")
        if not trusted_run(run) or not ancestor(repo, run["head_sha"], "origin/main"):
            continue
        attempts = {}
        for artifact in api_items(f"actions/runs/{run['id']}/artifacts", "artifacts"):
            if artifact["expired"] or not re.fullmatch(r"fpm-accuracy-web-[0-9a-f]{16}", artifact["name"]):
                continue
            try:
                archive = api(f"actions/artifacts/{artifact['id']}/zip", binary=True)
                snapshot = unpack(archive)["snapshot"]
                number = int(snapshot["run_attempt"])
                require(snapshot["run_id"] == str(run["id"]) and number <= run["run_attempt"], "wrong campaign attempt")
                if number not in attempts:
                    endpoint = f"actions/runs/{run['id']}/attempts/{number}"
                    attempt = run if number == run["run_attempt"] else api(endpoint)
                    require(attempt["id"] == run["id"] and attempt["head_sha"] == run["head_sha"], "wrong campaign")
                    attempts[number] = attempt, api_items(endpoint + "/jobs", "jobs")
                attempt, jobs = attempts[number]
                summary = validate_artifact(archive, attempt, artifact["name"], jobs)
            except (ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
                print(f"Ignoring invalid FPM artifact {artifact['id']}: {exc}")
                continue
            branch, commit = snapshot["branch"], snapshot["commit_sha"]
            if branch not in branches or not ancestor(repo, commit, "origin/" + branch):
                continue
            prior = selected.get(branch)
            if prior is not None:
                previous = prior["snapshot"]
                if commit != previous["commit_sha"]:
                    if ancestor(repo, commit, previous["commit_sha"]):
                        continue
                    require(ancestor(repo, previous["commit_sha"], commit), "incomparable FPM revisions")
                elif datetime.fromisoformat(snapshot["completed_at"]) <= datetime.fromisoformat(
                    previous["completed_at"]
                ):
                    continue
            selected[branch] = summary
    output.mkdir(parents=True, exist_ok=True)
    require(not any(output.iterdir()), "FPM output must be empty")
    for branch, summary in selected.items():
        (output / (artifact_key(branch) + ".json")).write_text(
            json.dumps(summary, allow_nan=False, sort_keys=True) + "\n"
        )
    print(f"Prepared {len(selected)} qualified FPM branch snapshots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.repo_root, args.output)
