# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from readme_report import PROFILES, summarize, transition

RUN = {
    "id": 123,
    "head_sha": "abc",
    "html_url": "https://github.com/ai-dynamo/aisimulate/actions/runs/123",
}


def reports():
    return [
        {
            "profile": p,
            "sha": "abc",
            "checks": [{"id": "command", "status": "passed"}],
            "expected": ["command"],
            "complete": True,
            "platform": "darwin" if p == "macos" else "linux",
            "run_attempt": 1,
        }
        for p in PROFILES
    ]


def test_healthy_is_quiet_and_repeated_failure_is_suppressed():
    assert transition({}, [], True, RUN) == (None, {})
    event, state = transition({}, ["dynamo/install: failed"], False, RUN)
    assert event["event"] == "failure"
    assert transition(state, ["dynamo/install: failed"], False, RUN)[0] is None
    update, updated = transition(state, ["source/predict: timeout"], False, RUN)
    assert update["event"] == "update"
    assert updated["incident"]["id"] == state["incident"]["id"]


def test_recovery_reuses_original_incident():
    _, state = transition({}, ["dynamo/install: failed"], False, RUN)
    failures, healthy = summarize(reports(), "success", "abc")
    event, state = transition(state, failures, healthy, {**RUN, "id": 124})
    assert event["event"] == "recovery"
    assert event["incident_id"] == "readme-123"
    assert state == {}


@pytest.mark.parametrize("status", ["failed", "timeout", "blocked", "skipped"])
def test_partial_or_failed_evidence_cannot_recover(status):
    evidence = reports()
    evidence[0]["checks"][0]["status"] = status
    assert summarize(evidence, "success", "abc")[1] is False


def test_missing_canceled_and_wrong_sha_cannot_recover():
    assert not summarize(reports()[:-1], "success", "abc")[1]
    assert not summarize(reports(), "cancelled", "abc")[1]
    with pytest.raises(ValueError, match="wrong-SHA"):
        summarize(reports(), "success", "different")


def test_incident_remembers_all_previously_failed_checks():
    _, state = transition({}, ["development/root-tests: failed"], False, RUN)
    _, state = transition(state, ["dynamo/dynamo-install: failed"], False, RUN)
    assert state["incident"]["required_checks"] == ["development/root-tests", "dynamo/dynamo-install"]


def test_recovery_requires_all_profiles_from_the_current_attempt():
    evidence = reports()
    evidence[0]["run_attempt"] = 2
    assert not summarize(evidence, "success", "abc", attempt=2)[1]
    for report in evidence:
        report["run_attempt"] = 2
    assert summarize(evidence, "success", "abc", attempt=2)[1]
    next(report for report in evidence if report["profile"] == "macos")["platform"] = "linux"
    assert not summarize(evidence, "success", "abc", attempt=2)[1]
