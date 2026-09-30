# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline checks for digest boundaries, pagination, and Slack formatting."""

import io
import json
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from slack_review_digest import PACIFIC, messages, pull_requests


class DigestTests(unittest.TestCase):
    now = datetime(2026, 9, 30, 0, 7, tzinfo=timezone.utc)

    def pr(self, number=1, age=6, **changes):
        pr = dict(
            number=number,
            title="Fix <queue> & counters",
            user={"login": "author"},
            draft=False,
            created_at=(self.now - timedelta(days=age)).isoformat(),
            updated_at=self.now.isoformat(),
            merged_at=None,
        )
        pr.update(changes)
        return pr

    def test_stale_boundary_drafts_and_oldest_first(self):
        opened = [
            self.pr(1, age=5),
            self.pr(2, age=6),
            self.pr(3, age=8),
            self.pr(4, age=10, draft=True),
        ]
        text = "".join(messages("ai-dynamo/aisimulate", opened, [], self.now))
        self.assertIn(":pr-opened: Open and ready for review: *3*", text)
        self.assertLess(text.index(":merged-2472:"), text.index(":eyes:"))
        self.assertLess(text.index(":eyes:"), text.index(":pr-opened:"))
        self.assertIn("older than 120 hours — 2", text)
        self.assertLess(text.index("|#3>"), text.index("|#2>"))
        self.assertNotIn("|#1>", text)
        self.assertNotIn("|#4>", text)
        self.assertIn("&lt;queue&gt; &amp; counters", text)

    def test_pacific_day_includes_closed_and_draft_new_prs(self):
        recent = [
            self.pr(
                created_at="2026-09-29T07:00:00Z", merged_at="2026-09-30T00:00:00Z"
            ),
            self.pr(
                created_at="2026-09-29T06:59:59Z", merged_at="2026-09-29T06:59:59Z"
            ),
            self.pr(created_at="2026-09-29T12:00:00Z", draft=True, state="closed"),
            self.pr(created_at="2026-09-30T01:00:00Z"),
        ]
        text = "".join(messages("ai-dynamo/aisimulate", [], recent, self.now))
        self.assertIn(":merged-2472: Merged today: *1*", text)
        self.assertIn(":eyes: New PRs opened today: *2*", text)
        self.assertIn("2026-09-29, 05:07 PM PDT", text)

    def test_dst_day_uses_midnight_offset(self):
        now = datetime(2026, 11, 2, 1, 7, tzinfo=timezone.utc)
        self.assertEqual(
            now.astimezone(PACIFIC).replace(hour=0, minute=0).utcoffset(),
            timedelta(hours=-7),
        )
        recent = [self.pr(created_at="2026-11-01T07:30:00Z")]
        text = "".join(messages("ai-dynamo/aisimulate", [], recent, now))
        self.assertIn(":eyes: New PRs opened today: *1*", text)
        self.assertIn("05:07 PM PST", text)

    def test_large_queue_retains_all_prs(self):
        prs = [self.pr(i, title="x" * 200) for i in range(100)]
        chunks = list(messages("ai-dynamo/aisimulate", prs, [], self.now))
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 3500 for chunk in chunks))
        for i in range(100):
            self.assertEqual("".join(chunks).count(f"|#{i}>"), 1)

    @patch("slack_review_digest.urlopen")
    def test_pagination_and_updated_cutoff(self, urlopen):
        first = [self.pr(i) for i in range(100)]
        second = [self.pr(100), self.pr(101, updated_at="2026-09-28T00:00:00Z")]
        urlopen.side_effect = [
            io.BytesIO(json.dumps(page).encode()) for page in [first, second]
        ]
        result = list(
            pull_requests(
                "ai-dynamo/aisimulate",
                "test",
                "all",
                since=self.now - timedelta(hours=1),
            )
        )
        self.assertEqual(len(result), 101)
        self.assertEqual(urlopen.call_count, 2)
        self.assertIn("page=2", urlopen.call_args.args[0].full_url)


if __name__ == "__main__":
    unittest.main()
