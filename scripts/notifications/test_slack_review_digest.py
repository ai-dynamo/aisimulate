# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline checks for digest boundaries, pagination, and Slack formatting."""

import io
import json
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.notifications.slack_review_digest import (
    PACIFIC,
    main,
    messages,
    pull_requests,
    review_pic,
    send_message,
)


class DigestTests(unittest.TestCase):
    now = datetime(2026, 9, 30, 0, 7, tzinfo=timezone.utc)

    def rotation(self, **changes):
        config = {
            "start_date": "2026-10-19",
            "roster": [
                {"name": name, "slack_id": f"U0000000{i}"}
                for i, name in enumerate(
                    ["Simone Chen", "Harry Lee", "Jason Zhou", "Yongming Ding", "Hongkuan Zhou", "Yueqi Zhao"],
                    start=1,
                )
            ],
        }
        config.update(changes)
        return json.dumps(config)

    def test_pic_rotation_same_on_monday_and_thursday_and_wraps(self):
        monday = datetime(2026, 10, 19, 9, 0, tzinfo=PACIFIC)
        expected = [
            "Simone Chen",
            "Harry Lee",
            "Jason Zhou",
            "Yongming Ding",
            "Hongkuan Zhou",
            "Yueqi Zhao",
            "Simone Chen",
        ]
        for week, name in enumerate(expected):
            for day in (0, 3):
                with self.subTest(week=week, day=day):
                    now = monday + timedelta(weeks=week, days=day)
                    self.assertEqual(
                        review_pic(now, self.rotation()),
                        (name, f"U0000000{week % 6 + 1}"),
                    )

    def test_pic_handoff_at_local_nine_after_dst(self):
        before = datetime(2026, 11, 2, 16, 59, tzinfo=timezone.utc)
        after = datetime(2026, 11, 2, 17, 0, tzinfo=timezone.utc)
        self.assertEqual(review_pic(before, self.rotation())[0], "Harry Lee")
        self.assertEqual(review_pic(after, self.rotation())[0], "Jason Zhou")
        self.assertEqual(review_pic(self.now, self.rotation())[0], "Simone Chen")

    def test_pic_ooo_swap_does_not_shift_later_weeks(self):
        rotation = self.rotation(overrides={"2026-11-02": "Yongming Ding", "2026-11-09": "Jason Zhou"})
        for day, name, user_id in (
            (2, "Yongming Ding", "U00000004"),
            (5, "Yongming Ding", "U00000004"),
            (9, "Jason Zhou", "U00000003"),
            (12, "Jason Zhou", "U00000003"),
            (16, "Hongkuan Zhou", "U00000005"),
        ):
            with self.subTest(day=day):
                self.assertEqual(
                    review_pic(datetime(2026, 11, day, 9, 0, tzinfo=PACIFIC), rotation),
                    (name, user_id),
                )

    def test_roster_order_size_and_start_are_configurable(self):
        roster = [
            {"name": "New Reviewer", "slack_id": "U00000009"},
            {"name": "Another Reviewer", "slack_id": "U00000008"},
        ]
        rotation = self.rotation(start_date="2027-01-04", roster=roster)
        for day, index in ((4, 0), (11, 1), (18, 0)):
            self.assertEqual(
                review_pic(datetime(2027, 1, day, 9, tzinfo=PACIFIC), rotation),
                (roster[index]["name"], roster[index]["slack_id"]),
            )
        roster.reverse()
        self.assertEqual(
            review_pic(
                datetime(2027, 1, 4, 9, tzinfo=PACIFIC),
                self.rotation(start_date="2027-01-04", roster=roster),
            )[0],
            "Another Reviewer",
        )

    def test_invalid_pic_configuration_fails_before_delivery(self):
        roster = json.loads(self.rotation())["roster"]
        invalid = ["", "not json", "[]", "{}"]
        for changes in (
            {"start_date": "2026-10-20"},
            {"start_date": "not-a-date"},
            {"start_date": None},
            {"roster": []},
            {"roster": {}},
            {"roster": [None]},
            {"roster": [{"name": " "}]},
            {"roster": [{"name": "Test", "slack_id": "bad"}]},
            {"roster": roster + [roster[0]]},
            {"roster": roster + [{"name": "Other", "slack_id": "U00000001"}]},
            {"roster": roster + [{"name": "Simone Chen", "slack_id": "U00000009"}]},
            {"overrides": []},
            {"overrides": {"2026-11-03": "Jason Zhou"}},
            {"overrides": {"2026-11-02": "Unknown"}},
            {"overrides": {"2026-11-02": []}},
        ):
            invalid.append(self.rotation(**changes))
        for rotation in invalid:
            with self.subTest(rotation=rotation), self.assertRaises(ValueError):
                review_pic(self.now, rotation)

    @patch("scripts.notifications.slack_review_digest.send_message")
    @patch("scripts.notifications.slack_review_digest.pull_requests", return_value=[])
    def test_main_passes_pic_as_typed_workflow_input(self, pull_requests, send_message):
        env = {
            "GH_TOKEN": "test",
            "SLACK_REVIEW_DIGEST_WEBHOOK_URL": "https://hooks.slack.com/triggers/test",
            "SLACK_REVIEW_ROTATION": self.rotation(),
        }
        with (
            patch.dict("os.environ", env),
            patch.object(sys, "argv", ["slack_review_digest.py"]),
            patch("scripts.notifications.slack_review_digest.datetime") as clock,
        ):
            clock.now.return_value = datetime(2026, 11, 5, 9, 0, tzinfo=PACIFIC)
            main()
        payload = send_message.call_args.args[1]
        self.assertEqual(set(payload), {"message", "pr_details", "review_pic"})
        self.assertEqual(payload["review_pic"], "U00000003")
        self.assertIn("Review PIC: Jason Zhou", payload["message"])

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
        text = "".join(messages("ai-dynamo/aisimulate", opened, [], self.now).values())
        self.assertIn(":reminder-alarm: PRs waiting for review: 3", text)
        self.assertLess(text.index(":merged-2472:"), text.index(":pr-opened:"))
        self.assertLess(text.index(":pr-opened:"), text.index(":reminder-alarm:"))
        self.assertIn("older than 5 days — 2", text)
        self.assertLess(text.index("• #3 "), text.index("• #2 "))
        self.assertNotIn("• #1 ", text)
        self.assertNotIn("• #4 ", text)
        self.assertIn("<queue> & counters", text)

    def test_week_includes_closed_and_draft_new_prs(self):
        recent = [
            self.pr(created_at="2026-09-23T00:07:01Z", merged_at="2026-09-30T00:00:00Z"),
            self.pr(created_at="2026-09-23T00:07:00Z", merged_at="2026-09-23T00:07:00Z"),
            self.pr(created_at="2026-09-29T12:00:00Z", draft=True, state="closed"),
            self.pr(created_at="2026-09-30T01:00:00Z"),
        ]
        text = "".join(messages("ai-dynamo/aisimulate", [], recent, self.now).values())
        self.assertIn(":merged-2472: PRs merged in the past 7 days: 1", text)
        self.assertIn(":pr-opened: PRs opened in the past 7 days: 2", text)
        self.assertIn("2026-09-29, 05:07 PM PDT", text)

    def test_dst_week_uses_local_calendar_boundary(self):
        now = datetime(2026, 11, 3, 1, 7, tzinfo=timezone.utc)
        start = now.astimezone(PACIFIC) - timedelta(days=7)
        self.assertEqual(start.utcoffset(), timedelta(hours=-7))
        recent = [
            self.pr(created_at="2026-10-27T00:07:00Z"),
            self.pr(created_at="2026-10-27T00:07:01Z"),
        ]
        text = messages("ai-dynamo/aisimulate", [], recent, now)["message"]
        self.assertIn("PRs opened in the past 7 days: 1", text)
        self.assertIn("05:07 PM PST", text)

    def test_large_queue_retains_all_prs(self):
        prs = [self.pr(i, title="x" * 200) for i in range(100)]
        payload = messages("ai-dynamo/aisimulate", prs, [], self.now)
        self.assertNotIn("https://", payload["message"])
        self.assertNotIn("• #", payload["message"])
        self.assertIn("older than 5 days — 100", payload["message"])
        for i in range(100):
            self.assertEqual(payload["pr_details"].count(f"• #{i} "), 1)
        with self.assertRaises(ValueError):
            messages("ai-dynamo/aisimulate", prs * 2, [], self.now)

    def test_empty_queue_thread_reply(self):
        payload = messages("ai-dynamo/aisimulate", [], [], self.now)
        self.assertEqual(payload["pr_details"], "None.")

    @patch("scripts.notifications.slack_review_digest.urlopen")
    def test_workflow_payload_and_acknowledgement(self, urlopen):
        urlopen.return_value = io.BytesIO(b'{"ok":true}')
        send_message(
            "https://hooks.slack.com/triggers/test",
            {"message": "Digest", "pr_details": "PRs"},
        )
        request = urlopen.call_args.args[0]
        self.assertEqual(json.loads(request.data), {"message": "Digest", "pr_details": "PRs"})
        urlopen.return_value = io.BytesIO(b'{"ok":false}')
        with self.assertRaises(ValueError):
            send_message(
                "https://hooks.slack.com/triggers/test",
                {"message": "Digest", "pr_details": "PRs"},
            )

    @patch("scripts.notifications.slack_review_digest.urlopen")
    def test_pagination_and_updated_cutoff(self, urlopen):
        first = [self.pr(i) for i in range(100)]
        second = [self.pr(100), self.pr(101, updated_at="2026-09-28T00:00:00Z")]
        urlopen.side_effect = [io.BytesIO(json.dumps(page).encode()) for page in [first, second]]
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
