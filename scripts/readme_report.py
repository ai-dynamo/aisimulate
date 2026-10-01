#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Send stateful daily README events to a Slack workflow webhook."""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import urllib.request
import zipfile
from pathlib import Path

PROFILES = {"development", "release", "dynamo", "macos"}


def gh(path: str, *, binary: bool = False):
    data = subprocess.check_output(["gh", "api", path])
    return data if binary else json.loads(data)


def artifact_json(repository: str, artifact: dict) -> list[dict]:
    archive = gh(f"repos/{repository}/actions/artifacts/{artifact['id']}/zip", binary=True)
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        records = []
        for member in zipped.infolist():
            if Path(member.filename).name not in {"report.json", "state.json"}:
                continue
            if member.file_size > 1_000_000:
                raise ValueError("Report artifact exceeds size limit")
            records.append(json.loads(zipped.read(member)))
        return records


def summarize(reports: list[dict], conclusion: str, sha: str) -> tuple[list[str], bool]:
    """Missing evidence never qualifies a recovery."""
    failures = []
    seen = set()
    for report in reports:
        profile = report["profile"]
        if profile not in PROFILES or profile in seen or report["sha"] != sha:
            raise ValueError("Unexpected, duplicate, or wrong-SHA report")
        seen.add(profile)
        checks = {check["id"]: check["status"] for check in report["checks"]}
        if len(checks) != len(report["checks"]) or not report["expected"]:
            raise ValueError("Invalid command evidence")
        for name in sorted(set(checks) | set(report["expected"])):
            if checks.get(name) != "passed":
                failures.append(f"{profile}/{name}: {checks.get(name, 'missing')}")
        if not report["complete"] or set(checks) != set(report["expected"]):
            failures.append(f"{profile}: incomplete")
    failures += [f"{profile}: missing report" for profile in sorted(PROFILES - seen)]
    if conclusion != "success":
        failures.append(f"workflow: {conclusion}")
    failures = sorted(set(failures))
    return failures, not failures


def transition(previous: dict, failures: list[str], healthy: bool, run: dict) -> tuple[dict | None, dict]:
    url = run["html_url"]
    state = dict(previous)
    incident = state.get("incident")
    if healthy:
        if not incident:
            return None, state
        payload = {
            "event": "recovery",
            "incident_id": incident["id"],
            "message": "✅ AISimulate README validation recovered",
            "details": (
                "All README profiles and full Rust/Python suites passed.\n"
                f"Commit: {run['head_sha']}\nRun: {url}\nOriginal failure: {incident['url']}"
            ),
        }
        return payload, {}
    if not failures:
        raise ValueError("Non-healthy result needs a failure reason")
    if incident and incident["failures"] == failures:
        return None, state
    if incident:
        event = "update"
    else:
        event = "failure"
        incident = {"id": f"readme-{run['id']}", "url": url}
    required = set(incident.get("required_checks", []))
    required.update(failure.split(":", 1)[0] for failure in failures if "/" in failure.split(":", 1)[0])
    incident = {**incident, "failures": failures, "required_checks": sorted(required)}
    state["incident"] = incident
    payload = {
        "event": event,
        "incident_id": incident["id"],
        "message": f"❌ AISimulate README validation: {len(failures)} failing checks",
        "details": f"Commit: {run['head_sha']}\nRun: {url}\n" + "\n".join(f"• {failure}" for failure in failures),
    }
    return payload, state


def previous_state(repository: str, current_run: int) -> dict:
    # Only this trusted consumer on main owns the delivery ledger.
    endpoint = f"repos/{repository}/actions/workflows/readme-report.yml/runs?branch=main&status=success&per_page=100"
    for run in gh(endpoint)["workflow_runs"]:
        if run["id"] >= current_run:
            continue
        artifacts = gh(f"repos/{repository}/actions/runs/{run['id']}/artifacts")["artifacts"]
        for artifact in artifacts:
            if artifact["name"] == "readme-report-state" and not artifact["expired"]:
                records = artifact_json(repository, artifact)
                if len(records) != 1:
                    raise ValueError("Invalid prior delivery ledger")
                return records[0]
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("readme-report"))
    args = parser.parse_args()
    repository = os.environ["GITHUB_REPOSITORY"]
    run = gh(f"repos/{repository}/actions/runs/{args.run_id}")
    if (
        run["path"] != ".github/workflows/readme-daily.yml"
        or run["head_branch"] != "main"
        or run["event"] not in {"schedule", "workflow_dispatch"}
        or run["status"] != "completed"
    ):
        raise ValueError("Only completed full daily/manual runs on main may notify")
    artifacts = gh(f"repos/{repository}/actions/runs/{args.run_id}/artifacts")["artifacts"]
    reports = []
    for artifact in artifacts:
        if artifact["name"] in {f"readme-{profile}" for profile in PROFILES} and not artifact["expired"]:
            reports.extend(artifact_json(repository, artifact))
    failures, healthy = summarize(reports, run["conclusion"], run["head_sha"])
    previous = previous_state(repository, int(os.environ["GITHUB_RUN_ID"]))
    if healthy and previous.get("incident"):
        passed = {
            f"{report['profile']}/{check['id']}"
            for report in reports
            for check in report["checks"]
            if check["status"] == "passed"
        }
        removed = sorted(set(previous["incident"].get("required_checks", [])) - passed)
        if removed:
            failures = [f"{name}: previous failing check was not rerun" for name in removed]
            healthy = False
    # workflow_run can be delivered out of order. Never recover from older evidence.
    if previous.get("last_run", 0) >= run["id"]:
        payload, state = None, previous
    else:
        payload, state = transition(previous, failures, healthy, run)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "payload.json").write_text(json.dumps(payload, indent=2) + "\n")
    webhook = os.environ.get("SLACK_README_WEBHOOK_URL")
    enabled = os.environ.get("SLACK_README_ENABLED") == "true"
    if payload and enabled and webhook:
        request = urllib.request.Request(
            webhook,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # Do not blindly retry an ambiguous POST: the workflow must deduplicate incident/event.
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if json.load(response).get("ok") is not True:
                    raise ValueError("Slack workflow did not acknowledge the event")
        except Exception:
            # Do not expose the webhook URL through an HTTP exception traceback.
            raise SystemExit("Slack workflow delivery failed; inspect the workflow before retrying") from None
    elif payload:
        print("Delivery disabled; payload saved for preview")
        state = previous
    state["last_run"] = max(previous.get("last_run", 0), run["id"])
    (args.output / "state.json").write_text(json.dumps(state, indent=2) + "\n")
    print("No notification needed" if payload is None else payload["message"])


if __name__ == "__main__":
    main()
