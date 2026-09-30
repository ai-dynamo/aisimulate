# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Notification policy, reductions, timing, and delivery with deterministic fakes."""

import hashlib
import io
import json
import sys
import urllib.error
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path

import accuracy_digest as digest
import notify_accuracy as notify
import prepare_e2e_accuracy_pages as pages
import pytest
import yaml


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
    assert digest.reduce_e2e(groups) == ("20.00 / 30.00", "3/4")
    assert (
        digest.reduce_fpm(
            [
                {"metric": {"predicted_count": 1, "measured_count": 1, "mape_pct": 0}},
                {"metric": {"predicted_count": 2, "measured_count": 3, "mape_pct": 30}},
            ]
        )
        == "20.00% (3/4)"
    )


def test_two_separate_thread_dimensions_and_framework_columns():
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
    assert len(replies) == 2
    assert "per model" in replies[0] and "25.00 / 44.00" in replies[0]
    assert "per gpu" in replies[1] and "GPU2" in replies[1]


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


def test_workflow_payload_has_one_parent_and_one_detail_reply():
    report = {"root": "Daily", "replies": ["per model", "per gpu"]}
    assert notify.webhook_payload(report) == {"message": "Daily", "accuracy_details": "per model\n\nper gpu"}
    assert notify.webhook_payload(report, test=True)["message"] == "[TEST] Daily"
    with pytest.raises(ValueError, match="35000"):
        notify.webhook_payload({"root": "Daily", "replies": ["x" * 20000, "y" * 20000]})


@pytest.mark.parametrize("response", [b'{"ok":false}', b"{}", b"[]", b"not json"])
def test_webhook_rejects_unacknowledged_or_invalid_response(monkeypatch, response):
    monkeypatch.setattr(notify.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(response))
    with pytest.raises(RuntimeError):
        notify.send_webhook("https://hooks.slack.com/triggers/test", {"message": "test"})


def test_webhook_posts_once_and_accepts_ok_true(monkeypatch):
    requests = []

    def respond(request, **kwargs):
        requests.append(request)
        return io.BytesIO(b'{"ok":true}')

    monkeypatch.setattr(notify.urllib.request, "urlopen", respond)
    payload = {"message": "Daily", "accuracy_details": "Details"}
    notify.send_webhook("https://hooks.slack.com/triggers/test", payload)
    assert len(requests) == 1 and json.loads(requests[0].data) == payload


@pytest.mark.parametrize("http", [False, True])
def test_webhook_never_retries_or_exposes_secret_in_errors(monkeypatch, http):
    url = "https://hooks.slack.com/triggers/secret-value"
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        if http:
            raise urllib.error.HTTPError(url, 400, url, {}, None)
        raise urllib.error.URLError(url)

    monkeypatch.setattr(notify.urllib.request, "urlopen", fail)
    with pytest.raises(RuntimeError) as error:
        notify.send_webhook(url, {})
    assert len(calls) == 1 and "secret-value" not in str(error.value)


def test_reservation_from_failed_production_run_blocks_resend_but_manual_does_not(monkeypatch):
    day = date(2026, 9, 29)
    artifact = {"id": 10, "name": f"accuracy-attempt-{day}", "expired": False, "workflow_run": {"id": 8}}
    run = {
        "id": 8,
        "path": notify.REPORT_WORKFLOW,
        "head_branch": "main",
        "event": "workflow_run",
        "head_repository": {"full_name": notify.REPO},
        "conclusion": "failure",
    }
    claim = {"day": str(day), "run_id": "8", "run_attempt": "1", "report_sha256": "hash"}
    monkeypatch.setattr(notify, "api_items", lambda *args: [artifact])
    monkeypatch.setattr(notify, "api", lambda path, **kwargs: archive("attempt.json", claim) if kwargs else run)
    assert notify.delivery_attempt(day) == claim
    run["event"] = "workflow_dispatch"
    assert notify.delivery_attempt(day) is None


def test_daily_prepare_skips_reserved_day_without_webhook_post(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys, "argv", ["notify_accuracy.py", "--mode", "daily", "--prepare-only", "--output", str(tmp_path)]
    )
    monkeypatch.setattr(notify, "webhook_url", lambda: "https://hooks.slack.com/triggers/test")
    monkeypatch.setattr(notify, "delivery_attempt", lambda day: {"day": str(day)})
    monkeypatch.setattr(notify, "send_webhook", lambda *args: pytest.fail("must not resend"))
    notify.main()
    assert not (tmp_path / "report.json").exists()


def test_daily_deliver_requires_matching_persisted_reservation(monkeypatch, tmp_path):
    report = {"day": str(datetime.now(notify.LA).date()), "root": "Daily", "replies": [], "alerts": [], "state": {}}
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps(report))
    monkeypatch.setattr(
        sys, "argv", ["notify_accuracy.py", "--mode", "daily", "--deliver", str(frozen), "--output", str(tmp_path)]
    )
    monkeypatch.setattr(notify, "webhook_url", lambda: "https://hooks.slack.com/triggers/test")
    monkeypatch.setattr(notify, "delivery_attempt", lambda day: None)
    monkeypatch.setattr(notify, "send_webhook", lambda *args: pytest.fail("must not send without reservation"))
    with pytest.raises(ValueError, match="persisted reservation"):
        notify.main()


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
    assert "env.MODE != 'dry-run'" in prepare["env"]["SLACK_ACCURACY_WEBHOOK_URL"]
    reservation = next(
        i for i, step in enumerate(steps) if step.get("name") == "Reserve the daily trigger before posting"
    )
    delivery = next(i for i, step in enumerate(steps) if step.get("name", "").startswith("Trigger Slack workflow"))
    assert reservation < delivery
    assert steps[reservation]["with"].get("overwrite", False) is False
    assert "SLACK_ACCURACY_BOT_TOKEN" not in json.dumps(workflow)


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


def test_dry_run_never_reads_webhook_or_writes_delivery_state(monkeypatch, tmp_path):
    report = {"root": "preview", "replies": [], "alerts": [], "state": {"trigger_accepted": False}}
    monkeypatch.setattr(
        sys, "argv", ["notify_accuracy.py", "--e2e-run-id", "1", "--fpm-run-id", "2", "--output", str(tmp_path)]
    )
    monkeypatch.setattr(notify, "api", lambda path: {"id": path.rsplit("/", 1)[-1]})
    monkeypatch.setattr(notify, "prior_state", lambda day: {})
    monkeypatch.setattr(notify, "build_report", lambda *args, **kwargs: report)

    def no_slack(*args, **kwargs):
        pytest.fail("dry run must not read webhook credentials")

    monkeypatch.setattr(notify, "webhook_url", no_slack)
    notify.main()
    assert (tmp_path / "report.json").exists()
    assert not (tmp_path / "delivery").exists()


@pytest.mark.parametrize("failure", [False, True])
def test_reserved_delivery_persists_baseline_only_after_acceptance(monkeypatch, tmp_path, failure):
    day = str(datetime.now(notify.LA).date())
    report = {"day": day, "root": "Daily", "replies": ["Details"], "alerts": [], "state": {"trigger_accepted": False}}
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps(report))
    monkeypatch.setattr(
        sys, "argv", ["notify_accuracy.py", "--mode", "daily", "--deliver", str(frozen), "--output", str(tmp_path)]
    )
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setattr(notify, "webhook_url", lambda: "https://hooks.slack.com/triggers/test")
    claim = {"day": day, "run_id": "123", "run_attempt": "1", "report_sha256": notify.report_hash(report)}
    monkeypatch.setattr(notify, "delivery_attempt", lambda day: claim)
    calls = []

    def send(*args):
        calls.append(args)
        if failure:
            raise RuntimeError("outcome unknown")

    monkeypatch.setattr(notify, "send_webhook", send)
    if failure:
        with pytest.raises(RuntimeError):
            notify.main()
        assert not (tmp_path / "delivery").exists()
    else:
        notify.main()
        assert json.loads((tmp_path / "delivery/state.json").read_text())["trigger_accepted"] is True
    assert len(calls) == 1


def test_webhook_message_uses_plain_text_and_complete_links():
    root, replies = digest.messages(
        "2026-09-29",
        {"e2e": {"url": "https://example.com/run", "status": "success"}},
        {"e2e": {"main": snapshot({"a": [1, 2]})}},
        [],
        [],
    )
    payload = notify.webhook_payload({"root": root, "replies": replies})
    assert "E2E run: https://example.com/run" in payload["message"]
    assert "Overall | vLLM | SGLang | TRT-LLM" in payload["message"]
    assert all("```" not in value and "<https://" not in value and "*E2E" not in value for value in payload.values())


def test_artifact_pagination_preserves_name_filter(monkeypatch):
    paths = []

    def fake_api(path):
        paths.append(path)
        return {"artifacts": [{}] * 100 if len(paths) == 1 else []}

    monkeypatch.setattr(pages, "api", fake_api)
    assert len(pages.api_items("actions/artifacts?name=accuracy-attempt-2026-09-29", "artifacts")) == 100
    assert paths == [f"actions/artifacts?name=accuracy-attempt-2026-09-29&per_page=100&page={page}" for page in (1, 2)]
