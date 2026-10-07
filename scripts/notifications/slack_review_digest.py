# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Send the daily PR digest; --dry-run prints messages without contacting Slack."""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

PACIFIC = ZoneInfo("America/Los_Angeles")


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def pull_requests(repository, token, state, since=None):
    """Page through PRs; updated order permits stopping at the day's boundary."""
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
    start = local.replace(hour=0, minute=0, second=0, microsecond=0)
    ready = [pr for pr in open_prs if not pr["draft"]]
    stale = sorted(
        (pr for pr in ready if timestamp(pr["created_at"]) < now - timedelta(hours=120)),
        key=lambda pr: pr["created_at"],
    )

    def today(value):
        return value is not None and start <= timestamp(value) <= now

    merged = sum(today(pr["merged_at"]) for pr in recent_prs)
    opened = sum(today(pr["created_at"]) for pr in recent_prs)
    lines = [
        f"AISimulate PR digest — {local:%Y-%m-%d, %I:%M %p %Z}",
        f":merged-2472: PRs merged today: {merged}",
        f":pr-opened: PRs opened today: {opened}",
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
    start = now.astimezone(PACIFIC).replace(hour=0, minute=0, second=0, microsecond=0)
    open_prs = list(pull_requests(repository, token, "open"))
    recent_prs = list(pull_requests(repository, token, "all", since=start))
    payload = messages(repository, open_prs, recent_prs, now)
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
