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
        query = urlencode(
            dict(state=state, sort="updated", direction="desc", per_page=100, page=page)
        )
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


def escape(value):
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def messages(repository, open_prs, recent_prs, now):
    local = now.astimezone(PACIFIC)
    start = local.replace(hour=0, minute=0, second=0, microsecond=0)
    ready = [pr for pr in open_prs if not pr["draft"]]
    stale = sorted(
        (
            pr
            for pr in ready
            if timestamp(pr["created_at"]) < now - timedelta(hours=120)
        ),
        key=lambda pr: pr["created_at"],
    )

    def today(value):
        return value is not None and start <= timestamp(value) <= now

    merged = sum(today(pr["merged_at"]) for pr in recent_prs)
    opened = sum(today(pr["created_at"]) for pr in recent_prs)
    lines = [
        f"*AISimulate PR digest — {local:%Y-%m-%d, %I:%M %p %Z}*",
        f":merged-2472: Merged today: *{merged}*",
        f":eyes: New PRs opened today: *{opened}*",
        "",
        f":pr-opened: Open and ready for review: *{len(ready)}*",
        "",
        f"*Open non-draft PRs older than 5 days — {len(stale)}*",
    ]
    for pr in stale:
        age = (now - timestamp(pr["created_at"])).total_seconds() / 86400
        title = escape(" ".join(pr["title"].split())[:200])
        author = escape((pr.get("user") or {}).get("login", "deleted-user"))
        lines.append(
            f"• <https://github.com/{repository}/pull/{pr['number']}|#{pr['number']}> "
            f"{title} — {author} · {age:.1f} days"
        )
    if not stale:
        lines.append("None.")
    # Keep every stale PR, splitting large queues into comfortably sized messages.
    chunk = ""
    for line in lines:
        if len(chunk) + len(line) + 1 > 3500:
            yield chunk
            chunk = "*AISimulate PR digest — continued*\n"
        chunk += line + "\n"
    yield chunk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repository = os.environ.get("GITHUB_REPOSITORY", "ai-dynamo/aisimulate")
    token = os.environ["GH_TOKEN"]
    webhook = os.environ.get("SLACK_REVIEW_DIGEST_WEBHOOK_URL", "")
    if not args.dry_run and not webhook:
        raise ValueError(
            "Set SLACK_REVIEW_DIGEST_WEBHOOK_URL before sending the digest"
        )
    now = datetime.now(timezone.utc)
    start = now.astimezone(PACIFIC).replace(hour=0, minute=0, second=0, microsecond=0)
    open_prs = list(pull_requests(repository, token, "open"))
    recent_prs = list(pull_requests(repository, token, "all", since=start))
    for message in messages(repository, open_prs, recent_prs, now):
        if args.dry_run:
            print(message)
            continue
        request = Request(
            webhook,
            data=json.dumps(
                {"text": message, "unfurl_links": False, "unfurl_media": False}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=30) as response:
            if response.read().strip() != b"ok":
                raise ValueError("Slack did not acknowledge the digest")


if __name__ == "__main__":
    try:
        main()
    except HTTPError as error:
        # Do not print request URLs: Slack webhook URLs contain credentials.
        sys.exit(f"Digest request failed (HTTP {error.code})")
    except URLError:
        sys.exit("Digest request failed (network error)")
