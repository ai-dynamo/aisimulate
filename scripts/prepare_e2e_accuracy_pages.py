#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fetch and validate complete accuracy artifacts using the trusted main publisher."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

if __package__:
    from .build_pages_site import _accuracy_summary
else:
    from build_pages_site import _accuracy_summary

REPO = "ai-dynamo/aisimulate"
WORKFLOW = ".github/workflows/e2e-accuracy.yml"
CAMPAIGN_KEYS = {
    "schema_version",
    "branch",
    "commit_sha",
    "wheel_sha256",
    "dataset_sha256",
    "measurement_sha256",
    "cohort_sha256",
    "driver_sha256",
    "run_id",
    "run_attempt",
    "selection_policy",
    "measurement_filter_counts",
    "selected",
    "published",
    "outcomes",
    "exclusion_reasons",
    "backend_versions",
    "release_tag",
    "started_at",
    "completed_at",
    "status",
    "advisory",
}
METRICS = {
    "points",
    "ttft_mape_pct",
    "tpot_mape_pct",
    "ttft_shape_error_pct",
    "tpot_shape_error_pct",
}


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("non-finite JSON value")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def keys(value, allowed):
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError("unexpected public artifact fields")


def completion_time(value):
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid accuracy completion timestamp") from exc
    if parsed.utcoffset() is None:
        raise ValueError("accuracy completion timestamp requires a timezone")
    return parsed


def public_contract(summary):
    """Apply a recursive allowlist before the existing semantic validator."""
    keys(summary, {"schema_version", "title", "snapshot", "scope", "totals", "models"})
    snapshot = summary["snapshot"]
    keys(
        snapshot,
        {
            "release_tag",
            "measurement_source",
            "measurement_source_url",
            "measurement_date_through",
            "prediction_generated_at",
            "predictions_sha256",
            "aic_commit_sha",
            "aisimulate_completed_at",
            "aisimulate_method",
            "aisimulate_packages",
            "aisimulate_sot_sha256",
            "corrections",
            "evaluated_revision",
            "aic_source",
            "campaign",
        },
    )
    keys(snapshot["evaluated_revision"], {"branch", "commit_sha"})
    keys(snapshot["aic_source"], {"repository", "branch", "commit_sha"})
    keys(snapshot["aisimulate_packages"], {"aisimulate"})
    if snapshot["corrections"] != []:
        raise ValueError("nightly campaign cannot contain unreviewed corrections")
    keys(snapshot["campaign"], CAMPAIGN_KEYS)
    keys(
        summary["scope"],
        {
            "measurement_scope",
            "latency_scope",
            "metrics",
            "multinode",
            "raw_rows",
            "published_rows",
            "excluded_multinode_rows",
            "claim",
        },
    )

    def aggregate(item, extra):
        keys(item, {"rows", "aic", "aisimulate"} | extra)
        keys(item["aic"], METRICS)
        keys(item["aisimulate"], METRICS | {"status_counts", "coverage_pct"})
        keys(
            item["aisimulate"]["status_counts"],
            {"success", "failed", "unsupported", "unknown"},
        )

    dimensions = {"gpu_skus", "frameworks", "precisions", "workloads"}
    aggregate(summary["totals"], dimensions | {"models"})
    for model in summary["models"]:
        aggregate(model, dimensions | {"model", "hf_model_paths"})
        for workload in model["workloads"]:
            aggregate(workload, {"identity", "label", "gpu_skus", "precisions", "gpus"})
            for gpu in workload["gpus"]:
                aggregate(gpu, {"gpu", "precisions", "topologies"})
                for topology in gpu["topologies"]:
                    aggregate(
                        topology,
                        {
                            "id",
                            "framework",
                            "precision",
                            "serving",
                            "spec_method",
                            "parallelism",
                            "points",
                        },
                    )
                    keys(
                        topology["parallelism"],
                        {
                            "tp_size",
                            "pp_size",
                            "attention_dp_size",
                            "moe_tp_size",
                            "moe_ep_size",
                        },
                    )
                    for point in topology["points"]:
                        keys(
                            point,
                            {"concurrency", "status", "measured", "aic", "aisimulate"},
                        )
                        for name in ("measured", "aic", "aisimulate"):
                            keys(
                                point[name],
                                {"ttft_relative", "tpot_relative"}
                                | (set() if name == "measured" else {"ttft_error_pct", "tpot_error_pct"}),
                            )
    return _accuracy_summary(json.dumps(summary, allow_nan=False))


def unpack_artifact(archive: bytes) -> dict:
    """Validate the public payload before using any of its provenance."""
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        infos = z.infolist()
        if (
            len(infos) != 2
            or {info.filename for info in infos} != {"summary.json", "qualification.json"}
            or any(
                info.file_size > 32 * 1024 * 1024 or (info.external_attr >> 16) & 0o170000 == 0o120000 for info in infos
            )
        ):
            raise ValueError("invalid accuracy artifact ZIP")
        data = z.read("summary.json")
        summary = public_contract(strict_json(data))
        qualification = strict_json(z.read("qualification.json"))
    keys(qualification, CAMPAIGN_KEYS | {"summary_sha256"})
    if set(qualification) != CAMPAIGN_KEYS | {"summary_sha256"}:
        raise ValueError("incomplete qualification")
    if qualification.pop("summary_sha256") != hashlib.sha256(data).hexdigest():
        raise ValueError("summary checksum mismatch")
    if qualification != summary["snapshot"]["campaign"]:
        raise ValueError("summary and qualification disagree")
    q = qualification
    completion_time(q["completed_at"])
    if (
        q["schema_version"] != 1
        or q["status"] != "complete"
        or q["advisory"] is not True
        or q["selection_policy"] != "latest-complete-config-run-v1"
    ):
        raise ValueError("campaign is not complete")
    for field in (
        "wheel_sha256",
        "dataset_sha256",
        "measurement_sha256",
        "cohort_sha256",
        "driver_sha256",
    ):
        if not isinstance(q[field], str) or not re.fullmatch(r"[0-9a-f]{64}", q[field]):
            raise ValueError("invalid campaign digest")
    if q["branch"] != "main" and not re.fullmatch(r"release/[A-Za-z0-9][A-Za-z0-9._/-]*", q["branch"]):
        raise ValueError("invalid evaluated branch")
    revision = {key: q[key] for key in ("branch", "commit_sha")}
    if (
        summary["snapshot"]["evaluated_revision"] != revision
        or not re.fullmatch(r"[0-9a-f]{40}", q["commit_sha"])
        or q["release_tag"] != summary["snapshot"]["release_tag"]
    ):
        raise ValueError("invalid evaluated revision or dataset")
    outcomes = q["outcomes"]
    keys(outcomes, {"evaluated", "unsupported", "baseline_failed"})
    keys(
        q["exclusion_reasons"],
        {"recipe_required", "adapter_unsupported", "baseline_failed"},
    )
    for count in q["exclusion_reasons"].values():
        if type(count) is not int or count < 0:
            raise ValueError("invalid exclusion reason count")
    keys(
        q["measurement_filter_counts"],
        {
            "incomplete_measurement_run",
            "nonstandard_or_error",
            "multinode",
            "stale",
            "missing_mean_latency",
            "mixed_image_curve",
        },
    )
    if any(type(count) is not int or count < 0 for count in q["measurement_filter_counts"].values()):
        raise ValueError("invalid measurement filter counts")
    if (
        type(q["selected"]) is not int
        or q["selected"] <= 0
        or any(type(count) is not int or count < 0 for count in outcomes.values())
        or sum(outcomes.values()) != q["selected"]
        or outcomes.get("evaluated") != summary["scope"]["raw_rows"]
        or type(q["published"]) is not int
        or q["published"] != summary["totals"]["rows"]
        or summary["totals"]["aisimulate"]["points"] <= 0
    ):
        raise ValueError("incomplete campaign coverage")
    return summary


def artifact_key(branch: str) -> str:
    return hashlib.sha256(branch.encode()).hexdigest()[:16]


class UnqualifiedBranch(ValueError):
    """A branch without a successful qualification job cannot publish."""


def trusted_run(run: dict) -> bool:
    return (
        run.get("event") in {"schedule", "workflow_dispatch"}
        and run.get("head_branch") == "main"
        and run.get("path") == WORKFLOW
        and run.get("conclusion") in {"success", "failure"}
        and run.get("repository", {}).get("full_name") == REPO
        and run.get("head_repository", {}).get("full_name") == REPO
    )


def validate_artifact(archive: bytes, run: dict, *, artifact_name="e2e-accuracy-web", jobs=None) -> dict:
    if not trusted_run(run):
        raise ValueError("untrusted accuracy workflow run")
    summary = unpack_artifact(archive)
    q = summary["snapshot"]["campaign"]
    if q["run_id"] != str(run["id"]) or q["run_attempt"] != str(run["run_attempt"]):
        raise ValueError("campaign does not belong to this completed run attempt")
    if artifact_name == "e2e-accuracy-web":
        # Legacy single-branch producers qualified the entire run.
        if run["conclusion"] != "success":
            raise UnqualifiedBranch("legacy accuracy workflow did not succeed")
    else:
        key = artifact_key(q["branch"])
        if artifact_name != "e2e-accuracy-web-" + key:
            raise ValueError("accuracy artifact name does not match its branch")
        expected = f"Qualify E2E accuracy ({key})"
        matches = [job for job in (jobs or []) if job["name"] == expected or job["name"].endswith(" / " + expected)]
        if (
            run.get("status") != "completed"
            or len(matches) != 1
            or matches[0].get("status") != "completed"
            or matches[0].get("conclusion") != "success"
            or matches[0].get("run_id") != run["id"]
            or matches[0].get("head_sha") != run["head_sha"]
        ):
            raise UnqualifiedBranch("branch qualification job did not succeed in this attempt")
    return summary


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def api(path: str, *, binary=False):
    request = urllib.request.Request(
        "https://api.github.com/repos/" + REPO + "/" + path,
        headers={
            "Authorization": "Bearer " + os.environ["GH_TOKEN"],
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        response = urllib.request.build_opener(NoRedirect).open(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code != 302 or not binary:
            raise
        # Never forward the GitHub credential to signed artifact blob storage.
        url = exc.headers["Location"]
        if not url.startswith("https://"):
            raise ValueError("insecure artifact redirect") from exc
        response = urllib.request.urlopen(url, timeout=60)
    with response:
        body = response.read(64 * 1024 * 1024 + 1)
    if len(body) > 64 * 1024 * 1024:
        raise ValueError("oversized Actions response")
    return body if binary else strict_json(body)


def api_items(path: str, key: str) -> list:
    items = []
    page = 1
    while True:
        batch = api(f"{path}?per_page=100&page={page}")[key]
        items.extend(batch)
        if len(batch) < 100:
            return items
        page += 1


def ancestor(repo: Path, commit: str, ref: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", commit, ref],
        capture_output=True,
    )
    if result.returncode not in (0, 1):
        raise ValueError("cannot establish accuracy source ancestry")
    return result.returncode == 0


def prepare(repo: Path, output: Path) -> None:
    branches = {"main"}
    refs = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo),
            "for-each-ref",
            "--format=%(refname:strip=3)",
            "refs/remotes/origin/release/",
        ],
        text=True,
    ).splitlines()
    branches.update(refs)
    selected = {}
    # Ninety-day artifacts outlive ordinary docs pushes. Inspect up to 100
    # completed campaigns; absent/expired artifacts retain committed evidence.
    try:
        runs = api("actions/workflows/e2e-accuracy.yml/runs?status=completed&branch=main&per_page=100")["workflow_runs"]
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        # First Pages build can precede registration of this new workflow.
        runs = []
    for listed in runs:
        run = api(f"actions/runs/{listed['id']}")
        if not trusted_run(run) or not ancestor(repo, run["head_sha"], "origin/main"):
            continue
        artifacts = api_items(f"actions/runs/{run['id']}/artifacts", "artifacts")
        matches = [
            a
            for a in artifacts
            if (a["name"] == "e2e-accuracy-web" or re.fullmatch(r"e2e-accuracy-web-[0-9a-f]{16}", a["name"]))
            and not a["expired"]
        ]
        seen = set()
        attempts = {}
        for artifact in matches:
            data = api(f"actions/artifacts/{artifact['id']}/zip", binary=True)
            q = unpack_artifact(data)["snapshot"]["campaign"]
            branch, commit = q["branch"], q["commit_sha"]
            if branch in seen:
                raise ValueError("ambiguous accuracy branch artifact")
            seen.add(branch)
            if artifact["name"] == "e2e-accuracy-web":
                attempt, jobs = run, None
            else:
                number = q["run_attempt"]
                if (
                    q["run_id"] != str(run["id"])
                    or not isinstance(number, str)
                    or not re.fullmatch(r"[1-9][0-9]*", number)
                    or int(number) > run["run_attempt"]
                ):
                    raise ValueError("invalid campaign run attempt")
                if number not in attempts:
                    path = f"actions/runs/{run['id']}/attempts/{number}"
                    attempt = run if int(number) == run["run_attempt"] else api(path)
                    if (
                        attempt["id"] != run["id"]
                        or attempt["run_attempt"] != int(number)
                        or attempt["head_sha"] != run["head_sha"]
                        or not trusted_run(attempt)
                    ):
                        raise ValueError("untrusted campaign attempt")
                    attempts[number] = attempt, api_items(path + "/jobs", "jobs")
                attempt, jobs = attempts[number]
            try:
                summary = validate_artifact(data, attempt, artifact_name=artifact["name"], jobs=jobs)
            except UnqualifiedBranch as exc:
                print(f"Skipping {artifact['name']}: {exc}")
                continue
            if branch not in branches or not ancestor(repo, commit, "origin/" + branch):
                continue
            prior = selected.get(branch)
            if prior:
                old = prior["snapshot"]["campaign"]
                if commit == old["commit_sha"]:
                    if completion_time(q["completed_at"]) <= completion_time(old["completed_at"]):
                        continue
                elif ancestor(repo, commit, old["commit_sha"]):
                    continue
                elif not ancestor(repo, old["commit_sha"], commit):
                    raise ValueError("incomparable accuracy revisions")
            selected[branch] = summary
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("accuracy output must be empty")
    written = 0
    for branch, summary in selected.items():
        committed = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "show",
                "origin/" + branch + ":python/aisimulate/docs/e2e-accuracy/summary.json",
            ],
            capture_output=True,
            text=True,
        )
        if committed.returncode == 0:
            previous = _accuracy_summary(committed.stdout)["snapshot"]
            revision = previous.get("evaluated_revision")
            current = summary["snapshot"]["campaign"]
            if revision and revision["branch"] == branch:
                if current["commit_sha"] != revision["commit_sha"] and ancestor(
                    repo, current["commit_sha"], revision["commit_sha"]
                ):
                    continue
                if (
                    current["commit_sha"] == revision["commit_sha"]
                    and previous.get("aisimulate_completed_at") is not None
                    and completion_time(previous["aisimulate_completed_at"]) >= completion_time(current["completed_at"])
                ):
                    continue
        (output / (artifact_key(branch) + ".json")).write_text(
            json.dumps(summary, sort_keys=True, allow_nan=False) + "\n"
        )
        written += 1
    print(f"Prepared {written} qualified branch accuracy snapshots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.repo_root, args.output)
