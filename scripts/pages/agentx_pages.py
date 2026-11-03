# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate AgentX campaign data and package it with the trusted replay UI."""

import argparse
import hashlib
import io
import re
import subprocess
import sys
import urllib.error
import zipfile
from datetime import datetime
from pathlib import Path

from plotly.offline import get_plotlyjs

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.e2e_accuracy.agentx_contract import (
    SUMMARY_SCHEMA,
    canonical,
    driver_digest,
    validate_public,
    validate_result,
)
from scripts.e2e_accuracy.benchmark_features import validate as validate_features
from scripts.pages.prepare_e2e_accuracy_pages import ancestor, api, api_items, artifact_key, strict_json, trusted_run


def validate_snapshot(snapshot, *, preview=False):
    fields = {
        "branch",
        "commit_sha",
        "dataset_revision",
        "wheel_sha256",
        "driver_sha256",
        "run_id",
        "run_attempt",
        "cohort_sha256",
        "preview",
        "completed_at",
    }
    partial = snapshot.get("measurement_campaign_status")
    if partial is not None:
        if not preview or partial != "partial":
            raise ValueError("Partial measurement campaigns require a preview")
        fields.add("measurement_campaign_status")
    if set(snapshot) != fields or snapshot["preview"] is not preview:
        raise ValueError("Invalid AgentX qualification identity")
    if not re.fullmatch(r"main|release/[A-Za-z0-9][A-Za-z0-9._/-]*", snapshot["branch"]) and not preview:
        raise ValueError("Invalid AgentX evaluated branch")
    for key, length in (
        ("commit_sha", 40),
        ("dataset_revision", 40),
        ("wheel_sha256", 64),
        ("cohort_sha256", 64),
        ("driver_sha256", 64),
    ):
        if not re.fullmatch("[0-9a-f]{" + str(length) + "}", snapshot[key]):
            raise ValueError("Invalid AgentX revision or digest")
    if (
        not re.fullmatch(r"[0-9]+", snapshot["run_id"])
        or type(snapshot["run_attempt"]) is not int
        or snapshot["run_attempt"] < 1
    ):
        raise ValueError("Invalid AgentX workflow identity")
    if datetime.fromisoformat(snapshot["completed_at"]).tzinfo is None:
        raise ValueError("AgentX completion time requires a timezone")


def validate_bundle(files, *, preview=False):
    if "qualification.json" not in files or "summary.json" not in files:
        raise ValueError("Incomplete AgentX publication")
    qualification = strict_json(files["qualification.json"])
    if (
        set(qualification) != {"schema", "status", "snapshot", "files"}
        or qualification["schema"] != "aisimulate.agentx.qualification.v1"
        or qualification["status"] != "complete"
    ):
        raise ValueError("Invalid AgentX qualification")
    validate_snapshot(qualification["snapshot"], preview=preview)
    if set(files) != {*qualification["files"], "qualification.json"}:
        raise ValueError("Uncataloged or missing AgentX files")
    for name, expected in qualification["files"].items():
        if not re.fullmatch(r"summary\.json|details/[a-z0-9-]+\.json", name):
            raise ValueError("Unsafe AgentX publication path")
        if hashlib.sha256(files[name]).hexdigest() != expected:
            raise ValueError("AgentX publication checksum mismatch")
        value = validate_public(strict_json(files[name]))
        if value.get("snapshot") != qualification["snapshot"] or value.get("schema") != SUMMARY_SCHEMA:
            raise ValueError("Mixed AgentX provenance")
    summary = strict_json(files["summary.json"])
    if set(summary) != {"schema", "snapshot", "runs"} or not isinstance(summary["runs"], list) or not summary["runs"]:
        raise ValueError("Invalid AgentX summary")
    ids = [run["id"] for run in summary["runs"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate AgentX result")
    expected = {"summary.json", *(f"details/{identity}.json" for identity in ids)}
    if set(qualification["files"]) != expected:
        raise ValueError("Incomplete AgentX details")
    for run in summary["runs"]:
        if run["detail_path"] != f"details/{run['id']}.json" or run["outcome"] not in {
            "completed",
            "unsupported",
            "failed",
        }:
            raise ValueError("Invalid AgentX outcome or detail link")
        detail = strict_json(files[run["detail_path"]])
        if {key: value for key, value in run.items() if key != "detail_path"} != {
            key: value for key, value in detail.items() if key not in {"schema", "snapshot", "report", "pairs"}
        }:
            raise ValueError("AgentX overview and detail disagree")
        validate_result({k: v for k, v in detail.items() if k not in {"schema", "snapshot"}})
        if (
            len(detail["pairs"]) != run["coverage"]["measured"]
            or run["coverage"]["measured"] != run["measured_requests"]
        ):
            raise ValueError("AgentX request accounting mismatch")
    return summary


def unpack(archive, *, preview=False):
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        entries = zipped.infolist()
        if len({e.filename for e in entries}) != len(entries) or sum(e.file_size for e in entries) > 512 * 1024 * 1024:
            raise ValueError("Invalid AgentX archive size or duplicate paths")
        if any(e.file_size > 32 * 1024 * 1024 or (e.external_attr >> 16) & 0o170000 == 0o120000 for e in entries):
            raise ValueError("Invalid AgentX archive member")
        files = {e.filename: zipped.read(e) for e in entries}
    validate_bundle(files, preview=preview)
    return files


def prepare(repo, output):
    branches = {
        "main",
        *subprocess.check_output(
            ["git", "-C", str(repo), "for-each-ref", "--format=%(refname:strip=3)", "refs/remotes/origin/release/"],
            text=True,
        ).splitlines(),
    }
    selected, updates, finite_archive = {}, {}, {}
    try:
        runs = api("actions/workflows/e2e-accuracy.yml/runs?status=completed&branch=main&per_page=100")["workflow_runs"]
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        runs = []
    for listed in runs:
        run = api(f"actions/runs/{listed['id']}")
        if not trusted_run(run) or not ancestor(repo, run["head_sha"], "origin/main"):
            continue
        latest_path = f"actions/runs/{run['id']}/attempts/{run['run_attempt']}"
        latest_jobs = api_items(latest_path + "/jobs", "jobs")
        for branch in branches:
            suffix = f"Qualify AgentX accuracy ({artifact_key(branch)})"
            matches = [j for j in latest_jobs if j.get("name", "").endswith(suffix)]
            shard_failed = any(
                "Evaluate AgentX shard " in j.get("name", "")
                and f"({artifact_key(branch)})" in j["name"]
                and j.get("conclusion") in {"failure", "timed_out"}
                for j in latest_jobs
            )
            if len(matches) == 1 and (
                matches[0].get("conclusion") in {"success", "failure", "timed_out"} or shard_failed
            ):
                updates.setdefault(
                    branch,
                    {
                        "status": "success" if matches[0]["conclusion"] == "success" else "failed",
                        "run_id": str(run["id"]),
                    },
                )
        artifacts = api_items(f"actions/runs/{run['id']}/artifacts", "artifacts")
        seen = set()
        for artifact in artifacts:
            if artifact["expired"] or not re.fullmatch(r"e2e-agentx-web-[0-9a-f]{16}", artifact["name"]):
                continue
            files = unpack(api(f"actions/artifacts/{artifact['id']}/zip", binary=True, max_bytes=512 * 1024 * 1024))
            summary = strict_json(files["summary.json"])
            snapshot = summary["snapshot"]
            branch, number = snapshot["branch"], snapshot["run_attempt"]
            expected_driver = driver_digest(
                lambda name: subprocess.check_output(["git", "-C", str(repo), "show", f"{run['head_sha']}:{name}"])
            )
            if snapshot["driver_sha256"] != expected_driver:
                raise ValueError("AgentX driver differs from the trusted producer")
            if branch not in branches or artifact["name"] != "e2e-agentx-web-" + artifact_key(branch):
                raise ValueError("AgentX artifact branch mismatch")
            if snapshot["run_id"] != str(run["id"]) or number > run["run_attempt"]:
                raise ValueError("AgentX artifact attempt mismatch")
            if branch in seen:
                raise ValueError("Duplicate AgentX branch artifact")
            seen.add(branch)
            path = f"actions/runs/{run['id']}/attempts/{number}"
            attempt = run if number == run["run_attempt"] else api(path)
            jobs = latest_jobs if number == run["run_attempt"] else api_items(path + "/jobs", "jobs")
            expected = f"Qualify AgentX accuracy ({artifact_key(branch)})"
            matches = [job for job in jobs if job["name"] == expected or job["name"].endswith(" / " + expected)]
            if (
                not trusted_run(attempt)
                or attempt["head_sha"] != run["head_sha"]
                or len(matches) != 1
                or attempt["id"] != run["id"]
                or attempt["run_attempt"] != number
            ):
                raise ValueError("Untrusted AgentX producer")
            if matches[0].get("run_id") != run["id"] or matches[0].get("head_sha") != run["head_sha"]:
                raise ValueError("Untrusted AgentX qualification job")
            updates.setdefault(
                branch,
                {"status": "success" if matches[0]["conclusion"] == "success" else "failed", "run_id": str(run["id"])},
            )
            if matches[0].get("conclusion") != "success" or matches[0].get("status") != "completed":
                continue
            if not ancestor(repo, snapshot["commit_sha"], "origin/" + branch):
                continue
            qwen = [row for row in summary["runs"] if row["category"] == "self_collected"]
            if qwen and all(row["boundary"] == "complete_finite_cohort" for row in qwen):
                validate_bundle(files)
                finite_archive.setdefault(branch, files)
            prior = selected.get(branch)
            if prior:
                previous = strict_json(prior["summary.json"])["snapshot"]
                commit, old = snapshot["commit_sha"], previous["commit_sha"]
                if commit == old:
                    if datetime.fromisoformat(snapshot["completed_at"]) <= datetime.fromisoformat(
                        previous["completed_at"]
                    ):
                        continue
                elif ancestor(repo, commit, old):
                    continue
                elif not ancestor(repo, old, commit):
                    raise ValueError("Incomparable AgentX evaluated revisions")
            selected[branch] = files
    for branch, files in selected.items():
        for name, content in files.items():
            path = output / artifact_key(branch) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    for branch, files in finite_archive.items():
        for name, content in files.items():
            path = output / "archive/finite-v1" / artifact_key(branch) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    output.mkdir(parents=True, exist_ok=True)
    (output / "updates.json").write_bytes(canonical(updates))


def write_assets(repo, destination):
    """One shared Plotly asset and the existing replay renderer, from trusted main."""
    source = repo / "python/aisimulate/src/aisimulate/replay"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("dashboard.js", "dashboard.css", "dashboard-body.html"):
        (destination / name).write_bytes((source / name).read_bytes())
    (destination / "plotly.min.js").write_text(get_plotlyjs())
    (destination / "PLOTLY-LICENSE.txt").write_text((source / "plotly-js-LICENSE.txt").read_text())


def validate_feature_evidence(value, summary):
    """Backfilled silicon metadata has its own revision, never a replacement prediction snapshot."""
    fields = {
        "schema",
        "measurement_dataset_revision",
        "feature_dataset_revision",
        "measurement_catalog_sha256",
        "feature_catalog_sha256",
        "records",
    }
    if not isinstance(value, dict) or set(value) != fields or value["schema"] != "aisimulate.agentx.features.v1":
        raise ValueError("Invalid AgentX feature evidence fields")
    for key in fields - {"schema", "records"}:
        length = 40 if key.endswith("revision") else 64
        if not isinstance(value[key], str) or not re.fullmatch(f"[0-9a-f]{{{length}}}", value[key]):
            raise ValueError("Feature evidence requires pinned revisions and checksums")
    if value["measurement_dataset_revision"] != summary["snapshot"]["dataset_revision"]:
        raise ValueError("Feature evidence belongs to another measurement dataset")
    if not isinstance(value["records"], list):
        raise ValueError("Invalid feature evidence records")
    expected, seen = {run["id"] for run in summary["runs"]}, set()
    for record in value["records"]:
        if not isinstance(record, dict) or set(record) != {
            "benchmark_id",
            "feature_artifact_sha256",
            "feature_metadata",
        }:
            raise ValueError("Invalid feature evidence record fields")
        identity = record["benchmark_id"]
        if not isinstance(identity, str) or identity not in expected or identity in seen:
            raise ValueError("Unknown or duplicate feature evidence identity")
        seen.add(identity)
        metadata = validate_features(record["feature_metadata"], identity)
        if hashlib.sha256(canonical(metadata)).hexdigest() != record["feature_artifact_sha256"]:
            raise ValueError("Feature artifact checksum mismatch")
    if seen != expected:
        raise ValueError("Incomplete feature evidence accounting")
    return value


def attach_feature_evidence(destination, branch, evidence_path, *, preview=False):
    """Attach a reviewed feature sidecar after packaging; numerical artifacts remain byte-identical."""
    catalog_path = destination / "branches.json"
    catalog = strict_json(catalog_path.read_bytes())
    matches = [entry for entry in catalog["branches"] if entry["branch"] == branch]
    if len(matches) != 1 or matches[0]["status"] != "evaluated":
        raise ValueError("Feature evidence requires an evaluated branch")
    entry = matches[0]
    key = artifact_key(branch)
    if entry["summary_path"] != f"branches/{key}/summary.json":
        raise ValueError("Invalid feature evidence destination")
    base = destination / "branches" / key
    paths = list(base.rglob("*"))
    if any(path.is_symlink() for path in (destination, destination / "branches", base, *paths)):
        raise ValueError("Feature evidence destination cannot contain symlinks")
    files = {
        path.relative_to(base).as_posix(): path.read_bytes()
        for path in paths
        if path.is_file() and path.relative_to(base).as_posix() != "feature-evidence.json"
    }
    summary = validate_bundle(files, preview=preview)
    if summary["snapshot"]["branch"] != branch:
        raise ValueError("Feature evidence branch mismatch")
    value = validate_feature_evidence(strict_json(evidence_path.read_bytes()), summary)
    content = canonical(value)
    (base / "feature-evidence.json").write_bytes(content)
    entry["feature_evidence"] = {
        "path": f"branches/{key}/feature-evidence.json",
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    catalog_path.write_bytes(canonical(catalog))


def package(repo, destination, branches, artifacts=None, *, preview=False, _archive=False):
    if not _archive:
        write_assets(repo, destination)
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    updates = (
        strict_json((artifacts / "updates.json").read_bytes())
        if artifacts and (artifacts / "updates.json").exists()
        else {}
    )
    for branch in branches:
        key = artifact_key(branch)
        source = artifacts / key if artifacts else None
        entry = {"branch": branch, "status": "unavailable"}
        if branch in updates:
            entry["last_update"] = updates[branch]
        if source is not None and source.exists():
            paths = list(source.rglob("*"))
            if any(p.is_symlink() for p in paths):
                raise ValueError("AgentX artifacts cannot contain symlinks")
            files = {p.relative_to(source).as_posix(): p.read_bytes() for p in paths if p.is_file()}
            summary = validate_bundle(files, preview=preview)
            if summary["snapshot"]["branch"] != branch:
                raise ValueError("AgentX publication belongs to another branch")
            for name, content in files.items():
                target = destination / "branches" / key / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            entry.update(status="evaluated", summary_path=f"branches/{key}/summary.json")
        entries.append(entry)
    (destination / "branches.json").write_bytes(canonical({"schema_version": 1, "branches": entries}))
    archive = artifacts / "archive/finite-v1" if artifacts and not _archive else None
    if archive is not None and archive.exists():
        package(repo, destination / "archive/finite-v1", branches, archive, preview=preview, _archive=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.repo, args.output)
