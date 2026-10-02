#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select qualified FPM aggregates from trusted GitHub Actions campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import urllib.error
import zipfile
from datetime import UTC, datetime, timedelta
from functools import cmp_to_key
from pathlib import Path

from fpm_accuracy.contract import REPO, artifact_key, eligible_branch, keys, require, strict_json, validate_summary
from fpm_accuracy.dashboard_contract import (
    BASELINE,
    MAX_BUNDLE,
    archive_files,
    population,
    validate_details,
    validate_visualization,
)
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
    files = archive_files(archive)
    require({"qualification.json", "summary.json"} <= files.keys(), "missing FPM artifact files")
    raw = files["summary.json"]
    summary = validate_summary(strict_json(raw))
    qualification = strict_json(files["qualification.json"])
    version = qualification.get("schema_version")
    require(type(version) is int and version in (1, 2), "invalid qualification schema")
    expected = ("schema_version", "snapshot", "summary_sha256")
    keys(qualification, (*expected, "details_sha256") if version == 2 else expected)
    require(
        set(files)
        == (
            {"qualification.json", "summary.json", "details.json"}
            if version == 2
            else {"qualification.json", "summary.json"}
        ),
        "unexpected FPM artifact files",
    )
    if version == 2:
        require(
            hashlib.sha256(files["details.json"]).hexdigest() == qualification["details_sha256"],
            "detail checksum mismatch",
        )
        validate_details(strict_json(files["details.json"]), summary)

    require(qualification["snapshot"] == summary["snapshot"], "qualification identity mismatch")
    require(qualification["summary_sha256"] == hashlib.sha256(raw).hexdigest(), "summary checksum mismatch")
    return summary


def trusted_run(run, *, allow_manual_branch=False):
    return (
        run.get("event") in {"schedule", "workflow_dispatch"}
        and (run.get("head_branch") == "main" or (allow_manual_branch and run.get("event") == "workflow_dispatch"))
        and run.get("path") == WORKFLOW
        and run.get("status") == "completed"
        and run.get("conclusion") in {"success", "failure"}
        and run.get("repository", {}).get("full_name") == REPO
        and run.get("head_repository", {}).get("full_name") == REPO
    )


def validate_artifact(archive, run, name, jobs, *, allow_manual_branch=False):
    require(trusted_run(run, allow_manual_branch=allow_manual_branch), "untrusted FPM campaign")
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
    history = {}
    measurements = []
    run_revisions = {}
    run_summaries = {}
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
            is_measurement = artifact["name"] == "fpm-accuracy-measurements"
            if artifact["expired"] or not (
                is_measurement or re.fullmatch(r"fpm-accuracy-web-[0-9a-f]{16}", artifact["name"])
            ):
                continue
            if is_measurement:
                measurements.append((run, artifact))
                continue
            try:
                archive = api(f"actions/artifacts/{artifact['id']}/zip", binary=True)
                files = archive_files(archive)
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
            except urllib.error.HTTPError as exc:
                if exc.code not in {404, 410}:
                    raise
                print(f"Ignoring unavailable FPM artifact {artifact['id']}: HTTP {exc.code}")
                continue
            except (ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
                print(f"Ignoring invalid FPM artifact {artifact['id']}: {exc}")
                continue
            branch, commit = snapshot["branch"], snapshot["commit_sha"]
            if branch not in branches or not ancestor(repo, commit, "origin/" + branch):
                continue
            run_revisions[(run["id"], snapshot["run_attempt"])] = snapshot["hf_revision"]
            run_summaries[(run["id"], snapshot["run_attempt"])] = summary
            identity = (branch, commit, population(summary))
            previous_run = history.get(identity)
            if previous_run is None or snapshot["completed_at"] > previous_run[0]["snapshot"]["completed_at"]:
                history[identity] = (summary, files.get("details.json"))
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
    dashboard = output / "dashboard"
    dashboard.mkdir()
    entries = []
    for summary, detail in sorted(history.values(), key=lambda item: item[0]["snapshot"]["completed_at"]):
        snapshot = summary["snapshot"]
        key = f"{snapshot['run_id']}-{snapshot['run_attempt']}-{artifact_key(snapshot['branch'])}"
        summary_path = key + ".json"
        (dashboard / summary_path).write_text(json.dumps(summary, allow_nan=False, sort_keys=True) + "\n")
        detail_path = key + "-details.json" if detail else None
        if detail_path:
            (dashboard / detail_path).write_bytes(detail)
        entries.append(
            dict(
                snapshot=snapshot,
                population=population(summary),
                summary_path=summary_path,
                details_path=detail_path,
                trend=snapshot["branch"] == "main" and ancestor(repo, BASELINE, snapshot["commit_sha"]),
            )
        )

    def compare(left, right):
        a, b = left["snapshot"], right["snapshot"]
        if a["branch"] == b["branch"] and a["commit_sha"] != b["commit_sha"]:
            return -1 if ancestor(repo, a["commit_sha"], b["commit_sha"]) else 1
        return (a["completed_at"] > b["completed_at"]) - (a["completed_at"] < b["completed_at"])

    for branch in branches:
        ordered = sorted((e for e in entries if e["snapshot"]["branch"] == branch), key=cmp_to_key(compare))
        for index, entry in enumerate(ordered):
            entry["revision_order"] = index
    (dashboard / "history.json").write_text(
        json.dumps(dict(schema_version=1, baseline=BASELINE, entries=entries)) + "\n"
    )
    revision = selected.get("main", {}).get("snapshot", {}).get("hf_revision")

    # Download only candidate point bundles for the selected dataset, newest first.
    # Do not keep 90 days of full measurement points resident in memory.
    def measurement_order(pair):
        run, artifact = pair
        revisions = {value for (run_id, _), value in run_revisions.items() if run_id == run["id"]}
        return revision in revisions, artifact["id"]

    for run, artifact in sorted(measurements, key=measurement_order, reverse=True):
        try:
            files = archive_files(api(f"actions/artifacts/{artifact['id']}/zip", binary=True, max_bytes=MAX_BUNDLE))
            snapshot = strict_json(files["qualification.json"])
            number = int(snapshot["run_attempt"])
            require(
                snapshot["schema_version"] == 1
                and snapshot["run_id"] == str(run["id"])
                and 0 < number <= run["run_attempt"]
                and snapshot["evaluator_sha"] == run["head_sha"],
                "wrong measurement campaign",
            )
            measurement_revision = snapshot["hf_revision"]
            require(
                run_revisions.get((run["id"], str(number))) == measurement_revision, "measurement/evaluation mismatch"
            )
            endpoint = f"actions/runs/{run['id']}/attempts/{number}"
            attempt = run if number == run["run_attempt"] else api(endpoint)
            require(trusted_run(attempt) and attempt["head_sha"] == run["head_sha"], "untrusted measurement attempt")
            jobs = api_items(endpoint + "/jobs", "jobs")
            matches = [job for job in jobs if job["name"] == "Qualify FPM measurements"]
            require(
                len(matches) == 1
                and matches[0].get("conclusion") == "success"
                and matches[0].get("status") == "completed"
                and matches[0].get("run_id") == run["id"]
                and matches[0].get("head_sha") == run["head_sha"],
                "unqualified measurements",
            )
            require(
                hashlib.sha256(files["manifest.json"]).hexdigest() == snapshot["manifest_sha256"],
                "measurement manifest checksum mismatch",
            )
            validate_visualization(files, measurement_revision, run_summaries[(run["id"], str(number))])
        except urllib.error.HTTPError as exc:
            if exc.code not in {404, 410}:
                raise
            continue
        except (ValueError, KeyError, TypeError, OSError, EOFError, zipfile.BadZipFile) as exc:
            print(f"Ignoring invalid measurement artifact {artifact['id']}: {exc}")
            continue
        target = dashboard / "visualization"
        target.mkdir()
        for name, content in files.items():
            (target / name).write_bytes(content)
        (target / "publication.json").write_text(
            json.dumps(
                {
                    "completed_at": snapshot["completed_at"],
                    "hf_revision": measurement_revision,
                    "current_hf_revision": revision,
                    "run_id": snapshot["run_id"],
                    "run_attempt": snapshot["run_attempt"],
                }
            )
            + "\n"
        )
        break
    print(f"Prepared {len(selected)} qualified FPM branch snapshots and {len(entries)} history entries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.repo_root, args.output)
