#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preview or deliver one daily accuracy message plus its detail thread."""

from __future__ import annotations

import argparse
import functools
import hashlib
import io
import json
import math
import os
import re
import subprocess
import urllib.error
import urllib.request
import zipfile
import zlib
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from accuracy_digest import compare, decode_points, e2e_snapshot, fpm_snapshot, messages
from prepare_e2e_accuracy_pages import (
    api,
    api_items,
    strict_json,
    unpack_artifact,
    validate_artifact,
)
from prepare_fpm_accuracy_pages import unpack as unpack_fpm
from prepare_fpm_accuracy_pages import validate_artifact as validate_fpm

REPO = "ai-dynamo/aisimulate"
LA = ZoneInfo("America/Los_Angeles")
WORKFLOWS = {"e2e": "e2e-accuracy.yml", "fpm": "fpm-accuracy.yml"}
REPORT_WORKFLOW = ".github/workflows/accuracy-digest.yml"


def parse_time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def report_day(run):
    return parse_time(run["created_at"]).astimezone(LA).date()


def scheduled_runs(kind, since, until):
    page = 1
    while True:
        path = (
            f"actions/workflows/{WORKFLOWS[kind]}/runs?event=schedule&branch=main"
            f"&created={since}..{until}&per_page=100&page={page}"
        )
        batch = api(path)["workflow_runs"]
        yield from batch
        if len(batch) < 100:
            return
        page += 1


def select_run(kind, day):
    candidates = [r for r in scheduled_runs(kind, day, day + timedelta(days=1)) if report_day(r) == day]
    return max(candidates, key=lambda r: (r["created_at"], r["id"]), default=None)


def ready(runs, day, now):
    return all(r and r["status"] == "completed" for r in runs.values()) or (
        now.astimezone(LA) >= datetime.combine(day, time(9), LA)
    )


def validate_run(run, kind, *, allow_manual_branch=False):
    if not (
        run["path"] == ".github/workflows/" + WORKFLOWS[kind]
        and (run["head_branch"] == "main" or (allow_manual_branch and run["event"] == "workflow_dispatch"))
        and run["event"] in {"schedule", "workflow_dispatch"}
        and run["repository"]["full_name"] == REPO
        and run["head_repository"]["full_name"] == REPO
    ):
        raise ValueError("unexpected accuracy producer")


@functools.lru_cache(maxsize=32)
def rules_digest(kind, sha):
    paths = (
        [
            "scripts/run_e2e_accuracy.py",
            "scripts/build_e2e_accuracy_overview.py",
            "scripts/fetch_accuracy_measurements.py",
        ]
        if kind == "e2e"
        else ["scripts/run_fpm_accuracy.py", "scripts/fpm_accuracy"]
    )
    paths += [f".github/workflows/{kind}-accuracy-branch.yml"]
    exists = subprocess.run(["git", "cat-file", "-e", sha], capture_output=True)
    if exists.returncode:
        subprocess.run(
            ["git", "fetch", "--no-tags", "origin", sha],
            check=True,
            capture_output=True,
        )
    tree = subprocess.check_output(["git", "ls-tree", "-r", sha, "--", *paths])
    if not tree:
        raise ValueError("missing evaluator source")
    return hashlib.sha256(tree).hexdigest()


def json_zip(archive, name, limit=256 * 1024 * 1024):
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        if bundle.namelist() != [name] or bundle.getinfo(name).file_size > limit:
            raise ValueError("unexpected notification artifact")
        return strict_json(bundle.read(name))


def fpm_points(archive, summary, public_archive):
    data = json_zip(archive, "comparison.json")
    with zipfile.ZipFile(io.BytesIO(public_archive)) as bundle:
        digest = hashlib.sha256(bundle.read("summary.json")).hexdigest()
    if data["schema_version"] != 1 or data["snapshot"] != summary["snapshot"] or data["summary_sha256"] != digest:
        raise ValueError("FPM point evidence identity/checksum mismatch")
    expected = {row["configuration_id"] + "/" + row["snapshot_id"]: row for row in summary["rows"]}
    if set(data["points"]) != set(expected):
        raise ValueError("FPM point evidence configuration mismatch")
    for key, row in expected.items():
        evidence = data["points"][key]
        if not re.fullmatch(r"[0-9a-f]{64}", evidence["order_sha256"]):
            raise ValueError("invalid FPM observation order hash")
        if set(evidence["methods"]) != set(row["results"]):
            raise ValueError("FPM point evidence predictor mismatch")
        for method, result in row["results"].items():
            points = decode_points(evidence["methods"][method])
            metric = result["metrics"]["all"]
            if any(not math.isfinite(v) or (v < 0 and v != -1) for v in points):
                raise ValueError("invalid FPM point error")
            values = [v for v in points if v >= 0]
            if len(points) != metric["measured_count"] or len(values) != metric["predicted_count"]:
                raise ValueError("FPM point evidence coverage mismatch")
            if values and not math.isclose(sum(values) / len(values), metric["mape_pct"], rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("FPM point evidence aggregate mismatch")
    return data["points"]


def load_snapshots(kind, run, *, allow_manual_branch=False):
    """Validate each artifact against its original attempt, including preserved retry successes."""
    if not run or run["status"] != "completed":
        return {}, []
    validate_run(run, kind, allow_manual_branch=allow_manual_branch)
    artifacts = api_items(f"actions/runs/{run['id']}/artifacts", "artifacts")
    companions = {a["name"]: a for a in artifacts if not a["expired"]}
    snapshots, warnings, attempts = {}, [], {}
    for artifact in artifacts:
        if not re.fullmatch(kind + r"-accuracy-web-[0-9a-f]{16}", artifact["name"]):
            continue
        try:
            if artifact["expired"]:
                raise ValueError("artifact expired")
            archive = api(f"actions/artifacts/{artifact['id']}/zip", binary=True)
            summary = unpack_artifact(archive) if kind == "e2e" else unpack_fpm(archive)
            snapshot = summary["snapshot"]["campaign"] if kind == "e2e" else summary["snapshot"]
            number = int(snapshot["run_attempt"])
            if snapshot["run_id"] != str(run["id"]) or not 1 <= number <= run["run_attempt"]:
                raise ValueError("artifact run attempt mismatch")
            if number not in attempts:
                endpoint = f"actions/runs/{run['id']}/attempts/{number}"
                attempt = run if number == run["run_attempt"] else api(endpoint)
                attempts[number] = attempt, api_items(endpoint + "/jobs", "jobs")
            attempt, jobs = attempts[number]
            if attempt["head_sha"] != run["head_sha"] or attempt["id"] != run["id"]:
                raise ValueError("attempt source mismatch")
            if kind == "e2e":
                validate_artifact(
                    archive, attempt, artifact_name=artifact["name"], jobs=jobs, allow_manual_branch=allow_manual_branch
                )
                normalized = e2e_snapshot(summary, rules_digest(kind, run["head_sha"]), run["html_url"])
            else:
                validate_fpm(
                    archive,
                    attempt,
                    artifact["name"],
                    jobs,
                    allow_manual_branch=allow_manual_branch,
                )
                evidence = None
                name = artifact["name"].replace("-web-", "-comparison-")
                if name in companions:
                    evidence = fpm_points(
                        api(
                            f"actions/artifacts/{companions[name]['id']}/zip",
                            binary=True,
                        ),
                        summary,
                        archive,
                    )
                normalized = fpm_snapshot(
                    summary,
                    rules_digest(kind, run["head_sha"]),
                    run["html_url"],
                    evidence,
                )
            branch = normalized["branch"]
            if branch in snapshots:
                raise ValueError("duplicate qualified branch")
            snapshots[branch] = normalized
        except (ValueError, KeyError, TypeError, zipfile.BadZipFile, zlib.error) as exc:
            warnings.append(f"{kind.upper()} {artifact['name']}: invalid or unavailable evidence ({exc}).")
        except urllib.error.HTTPError as exc:
            if exc.code not in {404, 410}:
                raise
            warnings.append(f"{kind.upper()} {artifact['name']}: evidence unavailable (HTTP {exc.code}).")
    return snapshots, warnings


def pipeline_status(kind, run, now, *, allow_manual_branch=False):
    url = f"https://github.com/{REPO}/actions/workflows/{WORKFLOWS[kind]}"
    if not run:
        return {"url": url, "status": "NOT STARTED"}, [f"{kind.upper()}: scheduled run missing."]
    validate_run(run, kind, allow_manual_branch=allow_manual_branch)
    url = run["html_url"]
    status = run.get("conclusion") or "still running at deadline"
    problems = [] if status == "success" else [f"{kind.upper()}: {status} ({url})."]
    if run["status"] == "completed":
        jobs = api_items(f"actions/runs/{run['id']}/attempts/{run['run_attempt']}/jobs", "jobs")
        for job in jobs:
            if job["conclusion"] in {
                "failure",
                "cancelled",
                "timed_out",
                "action_required",
            }:
                steps = ", ".join(s["name"] for s in job.get("steps", []) if s["conclusion"] == "failure")
                problems.append(f"{kind.upper()} {job['name']}: {steps or job['conclusion']} ({job['html_url']}).")
        end = max(
            (parse_time(j["completed_at"]) for j in jobs if j["completed_at"]),
            default=now,
        )
        minutes = (end - parse_time(run["run_started_at"])).total_seconds() / 60
        status += f" ({minutes:.0f} min, attempt {run['run_attempt']})"
    return {"url": url, "status": status}, problems


def prior_state(day):
    # Successful scheduled deliveries only. Dry runs and test sends cannot change production baselines.
    page = 1
    while page <= 10:
        try:
            runs = api(
                f"actions/workflows/accuracy-digest.yml/runs?status=success&branch=main&per_page=100&page={page}"
            )["workflow_runs"]
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return {"baselines": {}, "active": []}
            raise
        for run in runs:
            if run["event"] not in {"schedule", "workflow_run"} or run["path"] != REPORT_WORKFLOW:
                continue
            for item in api_items(f"actions/runs/{run['id']}/artifacts", "artifacts"):
                if item["name"] != "accuracy-delivery-state" or item["expired"]:
                    continue
                state = json_zip(
                    api(f"actions/artifacts/{item['id']}/zip", binary=True),
                    "state.json",
                )
                if (
                    state.get("schema_version") == 1
                    and state.get("trigger_accepted") is True
                    and state["day"] < str(day)
                ):
                    return state
        if len(runs) < 100:
            break
        page += 1
    return {"baselines": {}, "active": []}


def initial_baseline(kind, branch, day):
    # Bounded to the producers' 90-day artifact retention. No manual baseline substitution.
    for listed in scheduled_runs(kind, day - timedelta(days=90), day):
        if report_day(listed) >= day or listed["status"] != "completed":
            continue
        run = api(f"actions/runs/{listed['id']}")
        found, _ = load_snapshots(kind, run)
        if branch in found:
            return found[branch]
    return None


def build_report(day, runs, state, now, historical=True, allow_manual_branch=False):
    pipelines, snapshots, alerts, notes, active = {}, {}, [], [], []
    baselines = dict(state.get("baselines", {}))
    comparable = set()
    for kind, run in runs.items():
        pipelines[kind], failures = pipeline_status(kind, run, now, allow_manual_branch=allow_manual_branch)
        alerts.extend(failures)
        if failures:
            active.append(kind + ":pipeline")
        snapshots[kind], warnings = load_snapshots(kind, run, allow_manual_branch=allow_manual_branch)
        alerts.extend(warnings)
        if warnings:
            active.append(kind + ":evidence")
        # Discover evaluated branches from jobs, rather than reporting a newly created branch as missing.
        if run and run["status"] == "completed":
            jobs = api_items(f"actions/runs/{run['id']}/jobs", "jobs")
            branches = {m.group(1) for j in jobs if (m := re.match(r"(?:E2E|FPM) accuracy \((.+)\) /", j["name"]))}
            if not branches and run["conclusion"] == "success":
                alerts.append(f"{kind.upper()}: completed run has no evaluated branch jobs.")
                active.append(kind + ":evidence")
            for branch in sorted(branches - snapshots[kind].keys()):
                alerts.append(f"{kind.upper()} {branch}: no qualified result in today's run.")
                active.append(kind + ":" + branch)
        for branch, current in snapshots[kind].items():
            key = kind + ":" + branch
            previous = baselines.get(key)
            if previous is None and historical:
                previous = initial_baseline(kind, branch, day)
            regressions, limitations = compare(current, previous)
            if previous is not None and not limitations:
                comparable.add(key)
            if kind == "fpm" and any(g["points"] is None for g in current["groups"].values()):
                limitations.append("Point evidence unavailable in this run; point regression checks unavailable.")
            alerts.extend(f"{kind.upper()} {branch}: {a}" for a in regressions)
            notes.extend(f"{kind.upper()} {branch}: {n}" for n in limitations)
            if regressions:
                active.append(key)
                baselines[key] = previous  # Keep the pre-alert baseline until recovery.
            else:
                baselines[key] = current
            if kind == "e2e":
                lost = sum(v is None for g in current["groups"].values() for v in g["points"].values())
                if lost:
                    notes.append(f"E2E {branch}: {lost} unsuccessful eligible predictions (not necessarily new).")
    recovered = []
    for key in state.get("active", []):
        if key in active:
            continue
        kind, branch = key.split(":", 1)
        if (branch == "pipeline" and runs[kind] and runs[kind]["conclusion"] == "success") or (key in comparable):
            recovered.append(key)
    root, replies = messages(str(day), pipelines, snapshots, alerts, notes, recovered)
    return {
        "schema_version": 1,
        "day": str(day),
        "root": root,
        "replies": replies,
        "alerts": alerts,
        "notes": notes,
        "pipelines": pipelines,
        "state": {
            "schema_version": 1,
            "day": str(day),
            "trigger_accepted": False,
            "baselines": baselines,
            "active": active,
        },
    }


def webhook_payload(report, *, test=False):
    payload = {
        "message": ("[TEST] " if test else "") + report["root"],
        "accuracy_details": "\n\n".join(report["replies"]) or "No additional details.",
    }
    if any(len(text) > 35000 for text in payload.values()):
        raise ValueError("Workflow message exceeds 35000 characters; inspect the report artifact")
    return payload


def webhook_url():
    url = os.environ.get("SLACK_ACCURACY_WEBHOOK_URL", "")
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc != "hooks.slack.com" or not parsed.path.startswith("/triggers/"):
        raise ValueError("Set SLACK_ACCURACY_WEBHOOK_URL to the Slack Workflow Builder web request URL")
    return url


def send_webhook(url, payload):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    # An uncertain POST must not be retried: Slack may already have accepted it.
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Slack workflow returned HTTP {exc.code}; inspect workflow activity before retrying"
        ) from None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        raise RuntimeError("Slack trigger outcome unknown; inspect workflow activity before retrying") from None
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise RuntimeError("Slack did not acknowledge the workflow trigger; inspect workflow activity")


def delivery_attempt(day):
    """A pre-POST Actions artifact reserves the day, including failed/uncertain sends."""
    name = f"accuracy-attempt-{day}"
    artifacts = api_items(f"actions/artifacts?name={name}", "artifacts")
    for item in artifacts:
        if item["name"] != name:
            continue
        run = api(f"actions/runs/{item['workflow_run']['id']}")
        if (
            run["path"] != REPORT_WORKFLOW
            or run["head_branch"] != "main"
            or run["event"] not in {"schedule", "workflow_run"}
            or run["head_repository"]["full_name"] != REPO
        ):
            continue
        if item["expired"]:
            raise ValueError("Daily delivery reservation expired; refusing an uncertain resend")
        claim = json_zip(api(f"actions/artifacts/{item['id']}/zip", binary=True), "attempt.json")
        if claim["day"] != str(day) or claim["run_id"] != str(run["id"]):
            raise ValueError("Daily delivery reservation identity mismatch")
        return claim
    return None


def report_hash(report):
    return hashlib.sha256(json.dumps(report, sort_keys=True, allow_nan=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--deliver", type=Path, help="Deliver a previously persisted report")
    parser.add_argument("--mode", choices=("dry-run", "test", "daily"), default="dry-run")
    parser.add_argument("--e2e-run-id", type=int)
    parser.add_argument("--fpm-run-id", type=int)
    parser.add_argument("--output", type=Path, default=Path("accuracy-preview"))
    args = parser.parse_args()
    now = datetime.now(UTC)
    day = now.astimezone(LA).date()
    if args.mode == "daily" and (args.e2e_run_id or args.fpm_run_id):
        parser.error("daily mode accepts scheduled runs only")
    if args.mode == "daily" and not (args.prepare_only or args.deliver):
        parser.error("daily delivery requires prepare, upload reservation, then deliver")
    if not args.deliver and args.mode != "daily" and not (args.e2e_run_id and args.fpm_run_id):
        parser.error("dry-run/test requires both explicit run IDs")
    url = webhook_url() if args.mode != "dry-run" else None
    claim = delivery_attempt(day) if args.mode == "daily" else None
    if claim and not args.deliver:
        print(
            "Daily trigger already reserved/attempted; no resend. Check the original run and Slack workflow activity."
        )
        return
    if args.deliver:
        report = strict_json(args.deliver.read_text())
        if report["day"] != str(day):
            raise ValueError("do not send a report for a previous local day")
        if args.mode == "daily" and (
            not claim
            or claim
            != {
                "day": str(day),
                "run_id": os.environ.get("GITHUB_RUN_ID"),
                "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
                "report_sha256": report_hash(report),
            }
        ):
            raise ValueError("daily delivery requires this run attempt's persisted reservation and exact report")
    else:
        runs = {
            kind: api(f"actions/runs/{run_id}") if run_id else select_run(kind, day)
            for kind, run_id in (("e2e", args.e2e_run_id), ("fpm", args.fpm_run_id))
        }
        if args.mode == "daily" and not ready(runs, day, now):
            print("Waiting for both scheduled pipelines; no message sent.")
            return
        report = build_report(day, runs, prior_state(day), now, allow_manual_branch=args.mode != "daily")
    payload = webhook_payload(report, test=args.mode == "test")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, allow_nan=False, indent=2) + "\n")
    (args.output / "payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    preview = payload["message"] + "\n\n--- Thread reply ---\n\n" + payload["accuracy_details"]
    (args.output / "preview.md").write_text(preview + "\n")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as stream:
            stream.write(preview[:900000] + "\n")
    if args.mode == "daily" and args.prepare_only:
        run_id, attempt = os.environ.get("GITHUB_RUN_ID"), os.environ.get("GITHUB_RUN_ATTEMPT")
        if not run_id or not attempt:
            raise ValueError("prepare production delivery inside GitHub Actions")
        (args.output / "attempt.json").write_text(
            json.dumps(
                {
                    "day": str(day),
                    "run_id": run_id,
                    "run_attempt": attempt,
                    "report_sha256": report_hash(report),
                }
            )
            + "\n"
        )
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            stream.write(f"day={day}\n")
    if url and not args.prepare_only:
        send_webhook(url, payload)
        if args.mode == "daily":
            report["state"]["trigger_accepted"] = True
            state_dir = args.output / "delivery"
            state_dir.mkdir(exist_ok=True)
            (state_dir / "state.json").write_text(json.dumps(report["state"], allow_nan=False) + "\n")
        print("Slack accepted the workflow trigger; check Slack workflow activity for message delivery.")
    print(f"{args.mode}: {len(report['alerts'])} alert(s); preview: {args.output / 'preview.md'}")


if __name__ == "__main__":
    main()
