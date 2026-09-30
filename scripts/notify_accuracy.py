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
import uuid
import zipfile
import zlib
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from urllib.parse import urlencode
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
    # Successful scheduled deliveries only. Dry runs and test-channel sends cannot change production baselines.
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
                    and state.get("production_sent") is True
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
            "production_sent": False,
            "baselines": baselines,
            "active": active,
        },
    }


class Slack:
    def __init__(self, token, channel):
        if not token or not re.fullmatch(r"[CG][A-Z0-9]+", channel):
            raise ValueError("configure a Slack bot token and channel ID")
        self.token, self.channel = token, channel

    def call(self, method, **payload):
        url = "https://slack.com/api/" + method
        data = json.dumps(payload).encode()
        if method in {"conversations.history", "conversations.replies"}:
            query = {
                key: str(value).lower() if isinstance(value, bool) else value
                for key, value in payload.items()
                if value != ""
            }
            url += "?" + urlencode(query)
            data = None
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": "Bearer " + self.token,
                "Content-Type": "application/json; charset=utf-8",
            },
        )
        # Do not blindly retry POST: a timeout can occur after Slack accepted the message.
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
        if not result.get("ok"):
            raise RuntimeError(f"Slack {method}: {result.get('error', 'unknown error')}")
        return result

    def history(self, day):
        cursor = ""
        oldest = str(datetime.combine(day, time(), LA).timestamp())
        latest = str(datetime.combine(day + timedelta(days=1), time(), LA).timestamp())
        while True:
            result = self.call(
                "conversations.history",
                channel=self.channel,
                oldest=oldest,
                latest=latest,
                limit=100,
                cursor=cursor,
                include_all_metadata=True,
            )
            yield from result["messages"]
            cursor = result.get("response_metadata", {}).get("next_cursor", "")
            if not cursor:
                return

    def existing(self, day):
        for message in self.history(day):
            meta = message.get("metadata", {})
            if (
                meta.get("event_type") == "aisim_accuracy_daily"
                and meta.get("event_payload", {}).get("day") == str(day)
                and meta.get("event_payload", {}).get("repo") == REPO
            ):
                return message
        return None

    def send(self, report, test=False):
        day = report["day"]
        existing = None if test else self.existing(date.fromisoformat(day))
        key = f"{REPO}:{self.channel}:{day}" + (
            ":" + os.environ.get("GITHUB_RUN_ID", str(uuid.uuid4())) if test else ""
        )
        payload = {
            "channel": self.channel,
            "text": ("[TEST] " if test else "") + report["root"],
            "unfurl_links": False,
            "unfurl_media": False,
            "parse": "none",
            "mrkdwn": True,
            "link_names": False,
            "client_msg_id": str(uuid.uuid5(uuid.NAMESPACE_URL, key)),
            "metadata": {
                "event_type": "aisim_accuracy_test" if test else "aisim_accuracy_daily",
                "event_payload": {
                    "day": day,
                    "repo": REPO,
                    "report_run": os.environ.get("GITHUB_RUN_ID", "local"),
                },
            },
        }
        root = existing or self.call("chat.postMessage", **payload)
        sent, cursor = set(), ""
        if existing:
            while True:
                result = self.call(
                    "conversations.replies",
                    channel=self.channel,
                    ts=root["ts"],
                    cursor=cursor,
                    limit=100,
                    include_all_metadata=True,
                )
                for message in result["messages"]:
                    meta = message.get("metadata", {})
                    if meta.get("event_type") == "aisim_accuracy_detail":
                        sent.add(meta.get("event_payload", {}).get("part"))
                cursor = result.get("response_metadata", {}).get("next_cursor", "")
                if not cursor:
                    break
        for index, reply in enumerate(report["replies"]):
            if str(index) in sent:
                continue
            self.call(
                "chat.postMessage",
                channel=self.channel,
                thread_ts=root["ts"],
                text=reply,
                metadata={
                    "event_type": "aisim_accuracy_detail",
                    "event_payload": {"part": str(index)},
                },
                unfurl_links=False,
                unfurl_media=False,
                parse="none",
                mrkdwn=True,
                link_names=False,
                client_msg_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{key}:{index}")),
            )
        return True


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
    if not args.deliver and args.mode != "daily" and not (args.e2e_run_id and args.fpm_run_id):
        parser.error("dry-run/test requires both explicit run IDs")
    slack = None
    if args.mode != "dry-run":
        channel = os.environ.get("SLACK_ACCURACY_CHANNEL_ID", "")
        slack = Slack(os.environ.get("SLACK_ACCURACY_BOT_TOKEN", ""), channel)
    existing = slack.existing(day) if slack and args.mode == "daily" else None
    if args.deliver:
        report = strict_json(args.deliver.read_text())
        if report["day"] != str(day):
            raise ValueError("do not send a report for a previous local day")
    elif existing:
        # Resume the original frozen thread, never recompute after late pipeline results.
        producer = existing["metadata"]["event_payload"]["report_run"]
        if not str(producer).isdigit():
            raise ValueError("cannot resume report without Actions provenance")
        run = api(f"actions/runs/{producer}")
        if (
            run["path"] != REPORT_WORKFLOW
            or run["head_branch"] != "main"
            or run["event"] not in {"schedule", "workflow_run"}
            or run["head_repository"]["full_name"] != REPO
        ):
            raise ValueError("untrusted notification producer")
        artifacts = api_items(f"actions/runs/{producer}/artifacts", "artifacts")
        artifact = next(a for a in artifacts if a["name"] == "accuracy-report" and not a["expired"])
        report = json_zip(api(f"actions/artifacts/{artifact['id']}/zip", binary=True), "report.json")
        if report["day"] != str(day):
            raise ValueError("restored report date mismatch")
    else:
        runs = {
            kind: api(f"actions/runs/{run_id}") if run_id else select_run(kind, day)
            for kind, run_id in (("e2e", args.e2e_run_id), ("fpm", args.fpm_run_id))
        }
        if args.mode == "daily" and not ready(runs, day, now):
            print("Waiting for both scheduled pipelines; no message sent.")
            return
        report = build_report(day, runs, prior_state(day), now, allow_manual_branch=args.mode != "daily")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, allow_nan=False, indent=2) + "\n")
    preview = report["root"] + "\n\n" + "\n\n--- Thread reply ---\n\n".join(report["replies"])
    (args.output / "preview.md").write_text(preview + "\n")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as stream:
            stream.write(preview[:900000] + "\n")
    if slack and not args.prepare_only and slack.send(report, test=args.mode == "test") and args.mode == "daily":
        report["state"]["production_sent"] = True
        state_dir = args.output / "delivery"
        state_dir.mkdir(exist_ok=True)
        (state_dir / "state.json").write_text(json.dumps(report["state"], allow_nan=False) + "\n")
    print(f"{args.mode}: {len(report['alerts'])} alert(s); preview: {args.output / 'preview.md'}")


if __name__ == "__main__":
    main()
