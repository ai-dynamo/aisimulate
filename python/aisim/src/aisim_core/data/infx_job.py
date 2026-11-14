# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Weekly/manual InferenceX ingestion job: state, promotion, staleness and alerts.

``python -m aisim_core.data.infx_job {run,promote,staleness}``. The internal GitLab
job (``scripts/ci/gitlab/aisim-infx-ingest.gitlab-ci.yml``) only calls these entry points.

Rules enforced here (AISIM-2334 / DEP #435):

* An **unchanged** refresh (snapshot ID equals the qualified or pending candidate) writes
  nothing, so the job has no commit to make.
* A **failed** or **empty** run writes nothing under the data root: the last qualified
  snapshot stays in ``ingestion-state.json`` untouched.
* A successful run with new content writes a **candidate** only. The job has no code path
  that sets ``qualified``; :func:`promote` needs a named reviewer and is a separate manual step.
* Staleness is measured from the last qualified snapshot: warn at 8 days, stale at 14
  (alert only; nothing is deleted or demoted).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aisim_core.data.e2e import read_snapshot
from aisim_core.data.infx_ingest import (
    Acquirer,
    AcquisitionCache,
    InfxClient,
    InfxError,
    RetryPolicy,
    Transport,
    diff_members,
    plan_snapshot,
    snapshot_path,
    urllib_transport,
    write_candidate,
)

STATE_FILE = "ingestion-state.json"
STATE_SCHEMA = "aisim-infx-ingestion-state/v1"
OWNER = "Simone Chen"
SLACK_CHANNEL = "#swdl-dynamo-aisim-daily"
WARN_DAYS = 8
STALE_DAYS = 14


class PromotionError(Exception):
    pass


def _stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_state(root: Path) -> dict[str, Any]:
    path = Path(root) / STATE_FILE
    if not path.exists():
        return {"schema_version": STATE_SCHEMA, "qualified": None, "candidate": None}
    state = json.loads(path.read_text())
    if state.get("schema_version") != STATE_SCHEMA:
        raise ValueError(f"unsupported ingestion state schema {state.get('schema_version')!r}")
    return state


def save_state(root: Path, state: Mapping[str, Any]) -> None:
    path = Path(root) / STATE_FILE
    with tempfile.NamedTemporaryFile("w", dir=root, delete=False) as handle:
        handle.write(json.dumps(state, indent=2, sort_keys=True) + "\n")
    os.replace(handle.name, path)


def _members_of(root: Path, entry: Mapping[str, Any] | None) -> dict[str, str]:
    if not entry:
        return {}
    _, members = read_snapshot(root, entry["snapshot_path"])
    return {m.observation_id: m.record_sha256 for m in members}


@dataclass
class RunResult:
    status: str  # candidate | unchanged | empty | failed
    snapshot_id: str | None = None
    changes: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    requests_made: int = 0

    def to_json(self) -> dict[str, Any]:
        counts = {k: len(v) for k, v in self.changes.items()}
        return {
            "status": self.status,
            "snapshot_id": self.snapshot_id,
            "change_counts": counts,
            "warnings": self.warnings,
            "error": self.error,
            "requests_made": self.requests_made,
        }


def run_ingestion(
    root: Path,
    cache_dir: Path,
    *,
    models: list[str] | None = None,
    transport: Transport = urllib_transport,
    policy: RetryPolicy = RetryPolicy(),  # noqa: B008
    sleep: Callable[[float], None] | None = None,
    now: Callable[[], datetime] | None = None,
) -> RunResult:
    """Acquire, plan and (only if new) write a candidate. Never changes ``qualified``."""
    root = Path(root)
    clock = now or (lambda: datetime.now(UTC))
    kwargs = {"sleep": sleep} if sleep else {}
    client = InfxClient(transport, policy, **kwargs)
    state = load_state(root)
    qualified, candidate = state.get("qualified"), state.get("candidate")
    try:
        prior = _members_of(root, qualified)
        acq = Acquirer(client, AcquisitionCache(cache_dir), clock).acquire(models, list(prior))
    except InfxError as exc:
        return RunResult("failed", error=f"{type(exc).__name__}: {exc}", requests_made=client.requests_made)
    if acq.empty_scope:
        return RunResult(
            "empty", warnings=["no InferenceX data declared for the scope"], requests_made=acq.requests_made
        )
    plan = plan_snapshot(acq)
    known = {e["snapshot_id"] for e in (qualified, candidate) if e}
    changes = diff_members(prior, plan.members)
    if plan.snapshot_id in known:
        return RunResult("unchanged", plan.snapshot_id, requests_made=acq.requests_made)
    path = write_candidate(root, acq, plan)
    state["candidate"] = {
        "snapshot_id": plan.snapshot_id,
        "snapshot_path": path,
        "created_at": _stamp(clock()),
        "change_counts": {k: len(v) for k, v in changes.items()},
    }
    save_state(root, state)
    warnings = [f"availability lags benchmarks: {u}" for u in acq.undeclared]
    return RunResult("candidate", plan.snapshot_id, changes, warnings, requests_made=acq.requests_made)


def promote(root: Path, snapshot_id: str, reviewer: str, now: datetime | None = None) -> dict[str, Any]:
    """Manual review step: the pending candidate becomes the qualified snapshot."""
    root = Path(root)
    if not reviewer.strip():
        raise PromotionError("promotion requires a named reviewer")
    state = load_state(root)
    candidate = state.get("candidate")
    if not candidate or candidate["snapshot_id"] != snapshot_id:
        raise PromotionError(f"{snapshot_id} is not the pending candidate")
    read_snapshot(root, snapshot_path(snapshot_id))  # re-verify membership, tables and ID before promoting
    state["qualified"] = {**candidate, "qualified_at": _stamp(now or datetime.now(UTC)), "reviewer": reviewer}
    state["candidate"] = None
    save_state(root, state)
    return state["qualified"]


def staleness(state: Mapping[str, Any], now: datetime) -> dict[str, Any]:
    """``level`` is ok < 8 days <= warn < 14 days <= stale; ``never`` when nothing was qualified."""
    qualified = state.get("qualified")
    if not qualified:
        return {"level": "never", "age_days": None}
    age = (now - datetime.fromisoformat(qualified["qualified_at"].replace("Z", "+00:00"))).total_seconds() / 86400
    level = "stale" if age >= STALE_DAYS else "warn" if age >= WARN_DAYS else "ok"
    return {"level": level, "age_days": round(age, 2), "snapshot_id": qualified["snapshot_id"]}


# --------------------------------------------------------------------------- alerts


def post_slack(webhook: str, text: str) -> None:
    request = urllib.request.Request(
        webhook, json.dumps({"channel": SLACK_CHANNEL, "text": text}).encode(), {"Content-Type": "application/json"}
    )
    urllib.request.urlopen(request, timeout=30).close()


def send_mail(host: str, sender: str, to: str, subject: str, text: str) -> None:
    import smtplib
    from email.message import EmailMessage

    message = EmailMessage()
    message["From"], message["To"], message["Subject"] = sender, to, subject
    message.set_content(text)
    with smtplib.SMTP(host, timeout=30) as smtp:
        smtp.send_message(message)


def alert(
    subject: str,
    text: str,
    *,
    slack: Callable[[str], None] | None,
    mail: Callable[[str, str], None] | None,
) -> list[str]:
    """Send to Slack and to the owner's email independently; returns delivery failures (never raises)."""
    failures = []
    for name, send in (
        ("slack", (lambda: slack(f"{subject}\n{text}")) if slack else None),
        ("email", (lambda: mail(subject, text)) if mail else None),
    ):
        if send is None:
            failures.append(f"{name}: not configured")
            continue
        try:
            send()
        except Exception as exc:
            failures.append(f"{name}: {exc}")
    return failures


def _env_alert(subject: str, text: str) -> list[str]:
    hook, host = os.environ.get("SLACK_WEBHOOK_URL"), os.environ.get("SMTP_HOST")
    to = os.environ.get("AISIM_JOB_OWNER_EMAIL")
    return alert(
        f"[aisim-infx-ingest] {subject}",
        f"owner: {OWNER}\n{text}",
        slack=(lambda t: post_slack(hook, t)) if hook else None,
        mail=(lambda s, t: send_mail(host, os.environ.get("SMTP_FROM", to or ""), to, s, t)) if host and to else None,
    )


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="infx_job")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--root", type=Path, required=True, help="working tree of the private cache repository")
    run.add_argument("--cache-dir", type=Path, required=True, help="mutable acquisition cache (not committed)")
    run.add_argument("--model", action="append", help="display name or DB key; default all")
    run.add_argument("--report", type=Path, help="write the run report JSON here")
    pro = sub.add_parser("promote")
    pro.add_argument("--root", type=Path, required=True)
    pro.add_argument("--snapshot-id", required=True)
    pro.add_argument("--reviewer", required=True)
    notify = sub.add_parser("notify", help="alert the owner channels about a job failure outside ingestion")
    notify.add_argument("--subject", required=True)
    notify.add_argument("--text", required=True)
    stale = sub.add_parser("staleness")
    stale.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "run":
        result = run_ingestion(args.root, args.cache_dir, models=args.model)
        report = result.to_json()
        if args.report:
            args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, sort_keys=True))
        if result.status == "failed":
            _env_alert("ingestion failed", f"{result.error}\nThe last qualified snapshot is retained.")
            return 1
        return 0
    if args.command == "promote":
        print(json.dumps(promote(args.root, args.snapshot_id, args.reviewer), sort_keys=True))
        return 0
    if args.command == "notify":
        failures = _env_alert(args.subject, args.text)
        print(json.dumps({"delivery_failures": failures}))
        return 0 if len(failures) < 2 else 1
    verdict = staleness(load_state(args.root), datetime.now(UTC))
    print(json.dumps(verdict, sort_keys=True))
    if verdict["level"] in ("warn", "stale", "never"):
        _env_alert(f"data {verdict['level']}", json.dumps(verdict, sort_keys=True))
    return {"ok": 0, "warn": 10, "stale": 11, "never": 11}[verdict["level"]]


if __name__ == "__main__":
    sys.exit(main())
