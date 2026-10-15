#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Require a standalone Fast CI run for this exact Full CI branch and commit."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlencode

WORKFLOW = ".github/workflows/fast-ci.yml"
REQUIRED_JOBS = {"Repository Policy", "Python Static Checks", "Rust Format", "Fast CI Success"}


class GateError(RuntimeError):
    """The prerequisite has failed or cannot be verified."""


def github_api(endpoint: str) -> list[dict]:
    """Read every result page, preserving API errors as failures."""
    try:
        result = subprocess.run(
            ["gh", "api", "--paginate", "--slurp", endpoint],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        pages = json.loads(result.stdout)
        if not isinstance(pages, list) or not pages or not all(isinstance(page, dict) for page in pages):
            raise ValueError("expected object pages")
        return pages
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        raise GateError(f"Cannot verify Fast CI: GitHub API request failed for {endpoint}") from error


def latest_run(pages: list[dict], sha: str, branch: str, event: str, repository: str) -> dict | None:
    # A lifecycle push must use its own push validation. Manual Full CI may reuse
    # a push on the same branch, or an explicitly dispatched standalone Fast run.
    events = {"push"} if event == "push" else {"push", "workflow_dispatch"}
    candidates = []
    for page in pages:
        for run in page["workflow_runs"]:
            if (
                run.get("path") == WORKFLOW
                and run.get("head_sha") == sha
                and run.get("head_branch") == branch
                and run.get("event") in events
                and (run.get("head_repository") or {}).get("full_name") == repository
            ):
                candidates.append(run)
    return max(candidates, key=lambda run: run["id"], default=None)


def verify_jobs(pages: list[dict], sha: str) -> None:
    jobs = [job for page in pages for job in page["jobs"]]
    counts = Counter(job["name"] for job in jobs)
    if any(counts[name] != 1 for name in REQUIRED_JOBS):
        raise GateError("Fast CI did not execute exactly one of every required job")
    if any(
        job.get("head_sha") != sha or job.get("status") != "completed" or job.get("conclusion") != "success"
        for job in jobs
    ):
        raise GateError("Fast CI contains a failed, cancelled, skipped, incomplete, or wrong-commit job")


def require_fast_ci(
    repository: str,
    sha: str,
    ref: str,
    event: str,
    *,
    api: Callable = github_api,
    timeout: float = 600,
    interval: float = 15,
    clock: Callable = time.monotonic,
    sleep: Callable = time.sleep,
) -> dict:
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repository) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise GateError("Expected a repository owner/name and full lowercase commit SHA")
    if event not in {"push", "workflow_dispatch"} or not ref.startswith("refs/heads/"):
        raise GateError("Fast CI prerequisites are supported only for branch pushes and manual runs")
    if not math.isfinite(timeout) or not math.isfinite(interval) or timeout < 0 or interval <= 0:
        raise GateError("Invalid prerequisite wait settings")
    branch = ref.removeprefix("refs/heads/")
    query = urlencode({"head_sha": sha, "per_page": 100})
    runs_endpoint = f"repos/{repository}/actions/workflows/fast-ci.yml/runs?{query}"
    deadline = clock() + timeout
    previous = None
    while True:
        run = latest_run(api(runs_endpoint), sha, branch, event, repository)
        state = (run["id"], run["status"], run.get("conclusion")) if run else None
        if state != previous or run is None:
            print(f"Waiting for standalone Fast CI on {branch}@{sha}: {state or 'not found'}", flush=True)
            previous = state
        if run and run["status"] == "completed":
            if run.get("conclusion") != "success":
                raise GateError(f"Latest Fast CI run {run['id']} finished with {run.get('conclusion')}")
            attempt = run["run_attempt"]
            jobs_endpoint = f"repos/{repository}/actions/runs/{run['id']}/attempts/{attempt}/jobs?per_page=100"
            verify_jobs(api(jobs_endpoint), sha)
            # Do not accept an older success if a new run or rerun appeared
            # while its jobs were being checked.
            current = latest_run(api(runs_endpoint), sha, branch, event, repository)
            if current and (current["id"], current["run_attempt"], current["status"], current.get("conclusion")) == (
                run["id"],
                attempt,
                "completed",
                "success",
            ):
                return run
        if clock() >= deadline:
            raise GateError(
                f"No complete Fast CI evidence for {branch}@{sha} within {timeout:g}s. "
                "Run standalone Fast CI on this branch and commit, then rerun Full CI."
            )
        sleep(min(interval, max(0, deadline - clock())))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--sha", default=os.environ.get("GITHUB_SHA", ""))
    parser.add_argument("--ref", default=os.environ.get("GITHUB_REF", ""))
    parser.add_argument("--event", default=os.environ.get("GITHUB_EVENT_NAME", ""))
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    try:
        run = require_fast_ci(args.repository, args.sha, args.ref, args.event, timeout=args.timeout)
    except (GateError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"::error::{error}") from error
    summary = (
        f"### Standalone Fast CI verified\n\nCommit: `{args.sha}`\n\n"
        f"Run: {run['html_url']} (attempt {run['run_attempt']})\n\n"
        "All Fast CI jobs completed successfully. No Fast CI tests were rerun in Full CI.\n"
    )
    print(summary)
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(path).open("a", encoding="utf-8") as handle:
            handle.write(summary)


if __name__ == "__main__":
    main()
