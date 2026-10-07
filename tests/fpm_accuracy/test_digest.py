# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Notification policy, reductions, timing, and delivery with deterministic fakes."""

import hashlib
import io
import json
import sys
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
import yaml

import scripts.notifications.accuracy_digest as digest
import scripts.notifications.notify_accuracy as notify
import scripts.pages.prepare_e2e_accuracy_pages as pages


def snapshot(errors, *, kind="e2e", dataset="data", rules="rules"):
    return {
        "kind": kind,
        "branch": "main",
        "commit": "a" * 40,
        "dataset": dataset,
        "rules": rules,
        "url": "https://github.com/ai-dynamo/aisimulate/actions/runs/1",
        "groups": {
            "config": {
                "label": "model / GPU / config",
                "model": "model",
                "gpu": "GPU",
                "framework": "vllm",
                "points": digest.encode_points(None if v is None else v[0] for v in errors.values())
                if kind == "fpm" and errors is not None
                else errors,
            }
        },
    }


@pytest.mark.parametrize(
    "before,after,alert",
    [
        (10, 12, True),
        (30, 32, False),
        (10, 11.99, False),
        (20, 22, True),
        (0, 2, True),
        (0, 1, False),
        (12, 10, False),
    ],
)
def test_regression_requires_both_thresholds(before, after, alert):
    previous = snapshot({"p": [before, 0]})
    current = snapshot({"p": [after, 0]})
    alerts, notes = digest.compare(current, previous)
    assert bool(alerts) is alert
    assert not notes


def test_only_successful_common_points_compared_and_lost_successes_alert():
    previous = snapshot({"same": [10, 10], "lost": [1000, 1000]})
    current = snapshot({"same": [15, 15], "lost": None, "new": [0, 0]})
    alerts, _ = digest.compare(current, previous)
    assert len(alerts) == 3
    assert "1 previously predicted" in alerts[0]
    assert "10.00% -> 15.00%" in alerts[1]
    assert "1 common points" in alerts[1]


def test_exchanging_failed_points_still_alerts_with_unchanged_coverage():
    previous = snapshot({"a": [1], "b": None}, kind="fpm")
    current = snapshot({"a": None, "b": [1]}, kind="fpm")
    alerts, _ = digest.compare(current, previous)
    assert len(alerts) == 1 and "previously predicted" in alerts[0]


def test_missing_configuration_alerts():
    previous = snapshot({"a": [1, 1]})
    current = snapshot({})
    current["groups"] = {}
    assert "previously predicted points missing" in digest.compare(current, previous)[0][0]


@pytest.mark.parametrize("field", ["dataset", "rules"])
def test_changed_inputs_are_not_regressions(field):
    previous = snapshot({"a": [1, 1]})
    current = snapshot({"a": [99, 99]})
    current[field] = "different"
    alerts, notes = digest.compare(current, previous)
    assert not alerts and "baseline reset" in notes[0]


def test_legacy_fpm_without_points_does_not_claim_comparable():
    previous = snapshot(None, kind="fpm")
    current = snapshot({"a": [99]}, kind="fpm")
    alerts, notes = digest.compare(current, previous)
    assert not alerts and "Point evidence unavailable" in notes[0]


def test_reduction_uses_points_not_mean_of_means():
    groups = [
        {"points": {"1": [0, 10]}},
        {"points": {"2": [30, 40], "3": [30, 40], "4": None}},
    ]
    assert digest.reduce_e2e(groups) == ("20.00%/30.00%", "3/4")
    assert (
        digest.reduce_fpm(
            [
                {"metric": {"predicted_count": 1, "measured_count": 1, "mape_pct": 0}},
                {"metric": {"predicted_count": 2, "measured_count": 3, "mape_pct": 30}},
            ]
        )
        == "20.00% (3/4)"
    )


def test_two_separate_main_message_dimensions_and_framework_columns():
    current = snapshot({"a": [24, 43]})
    current["groups"]["config2"] = {
        **current["groups"]["config"],
        "gpu": "GPU2",
        "points": {"b": [26, 45]},
    }
    pipelines = {kind: {"url": "https://example.com/" + kind, "status": "success"} for kind in ("e2e", "fpm")}
    root, replies = digest.messages("2026-09-29", pipelines, {"e2e": {"main": current}}, [], [])
    assert all(name in root for name in ("Overall", "vLLM", "SGLang", "TRT-LLM", "Coverage"))
    assert "https://example.com/e2e" in root and "https://example.com/fpm" in root
    assert not replies
    assert "per model" in root and "25.00%/44.00%" in root
    assert "per gpu" in root and "GPU2" in root
    assert root.index("per model") < root.index("per gpu") < root.index("*FPM")


@pytest.mark.parametrize(
    "now,expected",
    [
        ("2026-09-29T15:59:59+00:00", False),
        ("2026-09-29T16:00:00+00:00", True),
        ("2026-12-29T16:59:59+00:00", False),
        ("2026-12-29T17:00:00+00:00", True),
    ],
)
def test_deadline_follows_los_angeles_dst(now, expected):
    now = datetime.fromisoformat(now)
    assert notify.ready({"e2e": {"status": "completed"}, "fpm": None}, now.date(), now) is expected


def test_failures_are_finished_and_do_not_delay_digest():
    now = datetime(2026, 9, 29, 12, tzinfo=UTC)
    assert notify.ready(
        {
            "e2e": {"status": "completed", "conclusion": "failure"},
            "fpm": {"status": "completed", "conclusion": "success"},
        },
        now.date(),
        now,
    )


def test_select_scheduled_run_by_local_day(monkeypatch):
    runs = [
        {"id": 1, "created_at": "2026-09-29T00:00:00Z"},
        {"id": 2, "created_at": "2026-09-29T10:30:00Z", "run_attempt": 2},
    ]
    monkeypatch.setattr(notify, "scheduled_runs", lambda *args: iter(runs))
    assert notify.select_run("e2e", date(2026, 9, 29))["id"] == 2
    assert notify.report_day(runs[0]) == date(2026, 9, 28)


def test_pipeline_failure_reports_step_and_link(monkeypatch):
    run = {
        "id": 7,
        "path": ".github/workflows/e2e-accuracy.yml",
        "event": "schedule",
        "head_branch": "main",
        "repository": {"full_name": notify.REPO},
        "head_repository": {"full_name": notify.REPO},
        "status": "completed",
        "conclusion": "failure",
        "run_attempt": 2,
        "html_url": "https://github.com/run/7",
        "run_started_at": "2026-09-29T10:00:00Z",
    }
    monkeypatch.setattr(
        notify,
        "api_items",
        lambda *args: [
            {
                "name": "Qualify",
                "conclusion": "failure",
                "html_url": "https://github.com/job/8",
                "completed_at": "2026-09-29T10:10:00Z",
                "steps": [{"name": "Install", "conclusion": "failure"}],
            }
        ],
    )
    status, alerts = notify.pipeline_status("e2e", run, datetime.now(UTC))
    assert "10 min" in status["status"]
    assert "Install" in alerts[1] and "https://github.com/job/8" in alerts[1]


def test_recovery_compares_against_pre_alert_baseline(monkeypatch):
    day, now = date(2026, 9, 29), datetime(2026, 9, 29, 16, tzinfo=UTC)
    healthy = snapshot({"a": [10, 10], "existing-failure": None})
    regressed = snapshot({"a": [20, 20]})
    monkeypatch.setattr(
        notify,
        "pipeline_status",
        lambda *args, **kwargs: (
            {"url": "https://example.com", "status": "success"},
            [],
        ),
    )
    monkeypatch.setattr(notify, "load_snapshots", lambda *args, **kwargs: ({"main": regressed}, []))
    state = {"baselines": {"e2e:main": healthy}, "active": []}
    report = notify.build_report(day, {"e2e": None}, state, now, historical=False)
    assert report["state"]["baselines"]["e2e:main"] == healthy
    assert report["state"]["active"] == ["e2e:main"]
    monkeypatch.setattr(notify, "load_snapshots", lambda *args, **kwargs: ({"main": healthy}, []))
    recovered = notify.build_report(day, {"e2e": None}, report["state"], now, historical=False)
    assert "Recovered: e2e:main" in recovered["root"]


def test_new_dataset_is_not_announced_as_recovery(monkeypatch):
    current = snapshot({"a": [1, 1]}, dataset="new")
    monkeypatch.setattr(
        notify,
        "pipeline_status",
        lambda *args, **kwargs: (
            {"url": "https://example.com", "status": "success"},
            [],
        ),
    )
    monkeypatch.setattr(notify, "load_snapshots", lambda *args, **kwargs: ({"main": current}, []))
    state = {
        "baselines": {"e2e:main": snapshot({"a": [10, 10]})},
        "active": ["e2e:main"],
    }
    report = notify.build_report(date(2026, 9, 29), {"e2e": None}, state, datetime.now(UTC), historical=False)
    assert "Recovered" not in report["root"] and report["notes"]


class FakeSlack(notify.Slack):
    def __init__(self):
        super().__init__("fake", "C123")
        self.posts = []
        self.fail_part = None

    def call(self, method, **payload):
        if method == "conversations.history":
            return {"messages": [p for p in self.posts if "thread_ts" not in p]}
        if method == "conversations.replies":
            return {"messages": self.posts}
        assert method == "chat.postMessage"
        self.posts.append({**payload, "ts": str(len(self.posts) + 1)})
        part = payload.get("metadata", {}).get("event_payload", {}).get("part")
        if part is not None and part == self.fail_part:
            self.fail_part = None
            raise TimeoutError("response lost after acceptance")
        return {"ts": self.posts[-1]["ts"]}


def test_one_root_and_resume_partial_thread_after_ambiguous_timeout():
    slack = FakeSlack()
    slack.fail_part = "0"
    report = {"day": "2026-09-29", "root": "Daily", "replies": ["models", "GPUs"]}
    with pytest.raises(TimeoutError):
        slack.send(report)
    slack.send(report)
    slack.send(report)
    assert len(slack.posts) == 3
    assert sum("thread_ts" not in p for p in slack.posts) == 1
    assert all(p.get("link_names") is False and p["parse"] == "none" and p["mrkdwn"] is True for p in slack.posts)


def test_test_message_not_production_dedup():
    slack = FakeSlack()
    report = {"day": "2026-09-29", "root": "Daily", "replies": []}
    slack.send(report, test=True)
    slack.send(report)
    assert len(slack.posts) == 2 and slack.posts[0]["text"].startswith("[TEST]")


def test_slack_api_ok_false_is_failure(monkeypatch):
    response = io.BytesIO(b'{"ok": false, "error": "not_in_channel"}')
    monkeypatch.setattr(notify.urllib.request, "urlopen", lambda *args, **kwargs: response)
    with pytest.raises(RuntimeError, match="not_in_channel"):
        notify.Slack("fake", "C123").call("chat.postMessage", text="test")


def archive(name, data):
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w") as bundle:
        bundle.writestr(name, json.dumps(data))
    return target.getvalue()


def test_point_sidecar_must_match_qualified_summary():
    row = {
        "configuration_id": "config",
        "snapshot_id": "s",
        "results": {"warmup": {"metrics": {"all": {"measured_count": 2, "predicted_count": 1, "mape_pct": 10}}}},
    }
    summary = {"snapshot": {"run_id": "1"}, "rows": [row]}
    public = archive("summary.json", summary)
    evidence = {
        "schema_version": 1,
        "snapshot": summary["snapshot"],
        "summary_sha256": hashlib.sha256(json.dumps(summary).encode()).hexdigest(),
        "points": {"config/s": {"order_sha256": "a" * 64, "methods": {"warmup": digest.encode_points([10, None])}}},
    }
    assert notify.fpm_points(archive("comparison.json", evidence), summary, public) == evidence["points"]
    evidence["points"]["config/s"]["methods"]["warmup"] = digest.encode_points([99, None])
    with pytest.raises(ValueError, match="aggregate mismatch"):
        notify.fpm_points(archive("comparison.json", evidence), summary, public)


def test_read_only_dry_run_and_guarded_secret_scope():
    workflow = yaml.safe_load((Path(__file__).parents[2] / ".github/workflows/accuracy-digest.yml").read_text())
    triggers = workflow.get("on", workflow.get(True))
    assert triggers["workflow_dispatch"]["inputs"]["mode"]["default"] == "dry-run"
    assert triggers["workflow_dispatch"]["inputs"]["mode"]["options"] == [
        "dry-run",
        "test",
    ]
    assert workflow["permissions"] == {"contents": "read", "actions": "read"}
    assert workflow["concurrency"]["cancel-in-progress"] is False
    steps = workflow["jobs"]["report"]["steps"]
    assert "workflow_run.head_sha" not in json.dumps(steps)
    prepare = next(s for s in steps if s.get("name") == "Prepare report and preview")
    assert "env.MODE != 'dry-run'" in prepare["env"]["SLACK_ACCURACY_BOT_TOKEN"]


def test_corrupt_or_oversized_compressed_points_rejected():
    packed = digest.encode_points([1, None, 2])
    assert list(digest.decode_points(packed)) == [1, -1, 2]
    with pytest.raises(ValueError):
        digest.decode_points({**packed, "count": 1})
    with pytest.raises(ValueError):
        digest.decode_points({"count": 10_000_001, "data": ""})


def test_fpm_observation_order_change_does_not_compare_positions():
    old = snapshot({"a": [1]}, kind="fpm")
    new = snapshot({"a": [90]}, kind="fpm")
    old["groups"]["config"]["point_order"] = "old"
    new["groups"]["config"]["point_order"] = "new"
    alerts, notes = digest.compare(new, old)
    assert not alerts and "not comparable" in notes[0]


def test_explicit_manual_branch_only_allowed_for_preview_and_test():
    run = {
        "path": ".github/workflows/e2e-accuracy.yml",
        "head_branch": "simonec/fix",
        "event": "workflow_dispatch",
        "repository": {"full_name": notify.REPO},
        "head_repository": {"full_name": notify.REPO},
    }
    with pytest.raises(ValueError):
        notify.validate_run(run, "e2e")
    notify.validate_run(run, "e2e", allow_manual_branch=True)
    run["head_repository"]["full_name"] = "someone/fork"
    with pytest.raises(ValueError):
        notify.validate_run(run, "e2e", allow_manual_branch=True)


def test_dry_run_never_instantiates_slack_or_writes_delivery_state(monkeypatch, tmp_path):
    report = {"root": "preview", "replies": [], "alerts": [], "state": {"production_sent": False}}
    monkeypatch.setattr(
        sys, "argv", ["notify_accuracy.py", "--e2e-run-id", "1", "--fpm-run-id", "2", "--output", str(tmp_path)]
    )
    monkeypatch.setattr(notify, "api", lambda path: {"id": path.rsplit("/", 1)[-1]})
    monkeypatch.setattr(notify, "prior_state", lambda day: {})
    monkeypatch.setattr(notify, "build_report", lambda *args, **kwargs: report)

    def no_slack(*args, **kwargs):
        pytest.fail("dry run must not instantiate Slack")

    monkeypatch.setattr(notify, "Slack", no_slack)
    notify.main()
    assert (tmp_path / "report.json").exists()
    assert not (tmp_path / "delivery").exists()


def test_artifact_pagination_preserves_name_filter(monkeypatch):
    paths = []

    def fake_api(path):
        paths.append(path)
        return {"artifacts": [{}] * 100 if len(paths) == 1 else []}

    monkeypatch.setattr(pages, "api", fake_api)
    assert len(pages.api_items("actions/artifacts?name=accuracy-attempt-2026-09-29", "artifacts")) == 100
    assert paths == [f"actions/artifacts?name=accuracy-attempt-2026-09-29&per_page=100&page={page}" for page in (1, 2)]


@pytest.mark.parametrize("method", ["conversations.history", "conversations.replies"])
def test_slack_reads_use_query_parameters_without_empty_cursor(monkeypatch, method):
    calls = []

    def respond(request, **kwargs):
        calls.append(request)
        return io.BytesIO(b'{"ok":true,"messages":[]}')

    monkeypatch.setattr(notify.urllib.request, "urlopen", respond)
    notify.Slack("fake", "C123").call(method, channel="C123", cursor="", include_all_metadata=True)
    assert calls[0].get_method() == "GET"
    assert calls[0].data is None
    assert parse_qs(urlsplit(calls[0].full_url).query) == {"channel": ["C123"], "include_all_metadata": ["true"]}


@pytest.mark.parametrize("kind", ["e2e", "fpm"])
@pytest.mark.parametrize("has_target", [False, True])
def test_baseline_branch_filter_skips_unrelated_downloads(monkeypatch, kind, has_target):
    target = f"{kind}-accuracy-web-{pages.artifact_key('main')}"
    artifacts = [{"id": 1, "name": f"{kind}-accuracy-web-{pages.artifact_key('release/0.12.0')}", "expired": False}]
    if has_target:
        artifacts.append({"id": 2, "name": target, "expired": False})
    downloads = []
    monkeypatch.setattr(notify, "validate_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(notify, "api_items", lambda *args: artifacts)

    def download(path, **kwargs):
        downloads.append(path)
        raise ValueError("target artifact validation failed")

    monkeypatch.setattr(notify, "api", download)
    found, warnings = notify.load_snapshots(kind, {"id": 7, "status": "completed"}, only_branch="main")
    assert not found
    assert downloads == (["actions/artifacts/2/zip"] if has_target else [])
    assert bool(warnings) is has_target


def test_initial_baseline_filters_branch_and_stops_at_newest_match(monkeypatch):
    runs = [
        {"id": 3, "status": "completed", "created_at": "2026-09-28T10:00:00Z"},
        {"id": 2, "status": "completed", "created_at": "2026-09-27T10:00:00Z"},
        {"id": 1, "status": "completed", "created_at": "2026-09-26T10:00:00Z"},
    ]
    calls = []
    monkeypatch.setattr(notify, "scheduled_runs", lambda *args: iter(runs))
    monkeypatch.setattr(notify, "api", lambda path: next(r for r in runs if path.endswith(str(r["id"]))))

    def snapshots(kind, run, *, only_branch):
        calls.append((run["id"], only_branch))
        return ({"main": "latest qualified"} if run["id"] == 2 else {}), []

    monkeypatch.setattr(notify, "load_snapshots", snapshots)
    assert notify.initial_baseline("e2e", "main", date(2026, 9, 29)) == "latest qualified"
    assert calls == [(3, "main"), (2, "main")]


def test_slack_history_follows_populated_cursor(monkeypatch):
    queries = []

    def respond(request, **kwargs):
        assert request.get_method() == "GET"
        queries.append(parse_qs(urlsplit(request.full_url).query))
        result = {"ok": True, "messages": [{"ts": str(len(queries))}]}
        if len(queries) == 1:
            result["response_metadata"] = {"next_cursor": "history+cursor="}
        return io.BytesIO(json.dumps(result).encode())

    monkeypatch.setattr(notify.urllib.request, "urlopen", respond)
    messages = list(notify.Slack("fake", "C123").history(date(2026, 9, 29)))
    assert [message["ts"] for message in messages] == ["1", "2"]
    assert "cursor" not in queries[0]
    assert queries[1]["cursor"] == ["history+cursor="]
    assert queries[0]["oldest"] == queries[1]["oldest"]


def test_slack_resume_reads_all_reply_pages_before_posting(monkeypatch):
    queries = []
    slack = notify.Slack("fake", "C123")
    monkeypatch.setattr(slack, "existing", lambda day: {"ts": "1"})

    def respond(request, **kwargs):
        assert request.get_method() == "GET", "both existing replies must be discovered without reposting"
        assert urlsplit(request.full_url).path.endswith("conversations.replies")
        queries.append(parse_qs(urlsplit(request.full_url).query))
        part = str(len(queries) - 1)
        result = {
            "ok": True,
            "messages": [{"metadata": {"event_type": "aisim_accuracy_detail", "event_payload": {"part": part}}}],
        }
        if len(queries) == 1:
            result["response_metadata"] = {"next_cursor": "reply+cursor="}
        return io.BytesIO(json.dumps(result).encode())

    monkeypatch.setattr(notify.urllib.request, "urlopen", respond)
    assert slack.send({"day": "2026-09-29", "root": "Daily", "replies": ["first", "second"]})
    assert len(queries) == 2
    assert "cursor" not in queries[0]
    assert queries[1]["cursor"] == ["reply+cursor="]
    assert queries[1]["ts"] == ["1"]


def test_actual_reusable_job_names_discover_missing_branch(monkeypatch):
    # Names observed on runs 36643620072 and 36558333115 include the called job suffix.
    monkeypatch.setattr(
        notify, "pipeline_status", lambda *args, **kwargs: ({"url": "https://example.com", "status": "success"}, [])
    )
    monkeypatch.setattr(notify, "load_snapshots", lambda *args, **kwargs: ({"main": snapshot({"a": [1, 1]})}, []))
    monkeypatch.setattr(
        notify,
        "api_items",
        lambda *args: [
            {"name": "E2E accuracy (main) / Prepare exact accuracy wheel"},
            {"name": "E2E accuracy (release/0.12.0) / Qualify E2E accuracy (cb21c32489602c02)"},
        ],
    )
    report = notify.build_report(
        date(2026, 9, 29),
        {"e2e": {"id": 1, "status": "completed", "conclusion": "success"}},
        {},
        datetime.now(UTC),
        historical=False,
    )
    assert report["alerts"] == ["E2E release/0.12.0: no qualified result in today's run."]
