# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Send the Monday/Thursday PR digest; --dry-run previews without contacting Slack."""

import argparse
import json
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

PACIFIC = ZoneInfo("America/Los_Angeles")


def monday_date(value):
    """Validate dates used to anchor the rotation and weekly coverage."""
    if not isinstance(value, str):
        raise ValueError("Rotation dates must be YYYY-MM-DD Mondays")
    monday = date.fromisoformat(value)
    if monday.isoformat() != value or monday.weekday() != 0:
        raise ValueError("Rotation dates must be YYYY-MM-DD Mondays")
    return monday


def review_pic(now, rotation):
    """Resolve the PIC from editable Actions configuration, including OOO coverage."""
    config = json.loads(rotation)
    if not isinstance(config, dict):
        raise ValueError("SLACK_REVIEW_ROTATION must be a JSON object")
    start = monday_date(config.get("start_date"))
    roster = config.get("roster")
    if not isinstance(roster, list) or not roster:
        raise ValueError("Rotation roster must be a nonempty list of names and Slack IDs")
    users = {}
    for member in roster:
        if not isinstance(member, dict):
            raise ValueError("Each roster member needs a name and slack_id")
        name, user_id = member.get("name"), member.get("slack_id")
        if (
            not isinstance(name, str)
            or not name.strip()
            or name != name.strip()
            or name in users
            or not isinstance(user_id, str)
            or not re.fullmatch(r"[UW][A-Z0-9]{8,}", user_id)
            or user_id in users.values()
        ):
            raise ValueError("Roster names and Slack user IDs must be valid and unique")
        users[name] = user_id
    coverage = config.get("overrides", {})
    if not isinstance(coverage, dict):
        raise ValueError("Rotation overrides must map Monday dates to roster names")
    for week, name in coverage.items():
        monday_date(week)
        if not isinstance(name, str) or name not in users:
            raise ValueError("Rotation overrides must refer to roster names")
    # Subtract local wall-clock hours so the Monday 09:00 handoff survives DST changes.
    shift_date = (now.astimezone(PACIFIC) - timedelta(hours=9)).date()
    week_start = shift_date - timedelta(days=shift_date.weekday())
    index = max(0, (week_start - start).days // 7) % len(users)
    name = coverage.get(week_start.isoformat(), list(users)[index])
    return name, users[name]


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def pull_requests(repository, token, state, since=None):
    """Page through PRs; updated order permits stopping at the reporting window's boundary."""
    page = 1
    while True:
        query = urlencode(dict(state=state, sort="updated", direction="desc", per_page=100, page=page))
        request = Request(
            f"https://api.github.com/repos/{repository}/pulls?{query}",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
        )
        with urlopen(request, timeout=30) as response:
            batch = json.load(response)
        for pr in batch:
            if since is not None and timestamp(pr["updated_at"]) < since:
                return
            yield pr
        if len(batch) < 100:
            return
        page += 1


def messages(repository, open_prs, recent_prs, now):
    local = now.astimezone(PACIFIC)
    start = local - timedelta(days=7)
    ready = [pr for pr in open_prs if not pr["draft"]]
    stale = sorted(
        (pr for pr in ready if timestamp(pr["created_at"]) < now - timedelta(hours=120)),
        key=lambda pr: pr["created_at"],
    )

    def in_window(value):
        return value is not None and start < timestamp(value) <= now

    merged = sum(in_window(pr["merged_at"]) for pr in recent_prs)
    opened = sum(in_window(pr["created_at"]) for pr in recent_prs)
    lines = [
        f"AISim PR digest — {local:%Y-%m-%d, %I:%M %p %Z}",
        f":merged-2472: PRs merged in the past 7 days: {merged}",
        f":pr-opened: PRs opened in the past 7 days: {opened}",
        f":reminder-alarm: PRs waiting for review: {len(ready)}",
        "",
        f"Open non-draft PRs older than 5 days — {len(stale)}",
    ]
    summary = "\n".join(lines)
    lines = []
    for pr in stale:
        age = (now - timestamp(pr["created_at"])).total_seconds() / 86400
        title = " ".join(pr["title"].split())[:200]
        author = (pr.get("user") or {}).get("login", "deleted-user")
        lines.append(
            f"• #{pr['number']} {title} — {author} · {age:.1f} days\n"
            f"  https://github.com/{repository}/pull/{pr['number']}"
        )
    if not stale:
        lines.append("None.")
    details = "\n".join(lines)
    # Fail before triggering instead of silently dropping PRs in a large queue.
    if len(details) > 35000:
        raise ValueError("PR details exceed the single thread reply budget (35000 characters)")
    return {"message": summary, "pr_details": details}


def send_message(webhook, payload):
    """Trigger one workflow: channel summary followed by a thread reply."""
    request = Request(
        webhook,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=30) as response:
        result = json.load(response)
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise ValueError("Slack did not acknowledge the workflow trigger")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repository = os.environ.get("GITHUB_REPOSITORY", "ai-dynamo/aisimulate")
    token = os.environ["GH_TOKEN"]
    webhook = os.environ.get("SLACK_REVIEW_DIGEST_WEBHOOK_URL", "")
    if not args.dry_run and not webhook:
        raise ValueError("Set SLACK_REVIEW_DIGEST_WEBHOOK_URL before sending the digest")
    now = datetime.now(timezone.utc)
    pic_name, pic_id = review_pic(now, os.environ.get("SLACK_REVIEW_ROTATION", ""))
    start = now.astimezone(PACIFIC) - timedelta(days=7)
    open_prs = list(pull_requests(repository, token, "open"))
    recent_prs = list(pull_requests(repository, token, "all", since=start))
    payload = messages(repository, open_prs, recent_prs, now)
    payload["message"] += f"\nReview PIC: {pic_name} — please triage external-contributor PRs."
    # Workflow Builder must declare this as User ID; Text inputs do not parse mentions.
    payload["review_pic"] = pic_id
    if args.dry_run:
        print("CHANNEL MESSAGE:\n" + payload["message"])
        print("\nTHREAD REPLY:\n" + payload["pr_details"])
    else:
        send_message(webhook, payload)


if __name__ == "__main__":
    try:
        main()
    except HTTPError as error:
        # Do not print request URLs: Slack webhook URLs contain credentials.
        sys.exit(f"Digest request failed (HTTP {error.code})")
    except URLError:
        sys.exit("Digest request failed (network error)")
