# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The internal GitLab job definition keeps its safety properties (it is never executed here)."""

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

JOB = Path(__file__).resolve().parents[5] / "scripts/ci/gitlab/aisim-infx-ingest.gitlab-ci.yml"


@pytest.fixture(scope="module")
def definition():
    return yaml.safe_load(JOB.read_text())


def _text(job):
    return "\n".join(str(line) for line in job.get("script", []))


def test_runs_only_on_schedule_or_manual_and_is_pinned(definition):
    assert {r.get("if") for r in definition["workflow"]["rules"] if "if" in r} == {
        '$CI_PIPELINE_SOURCE == "schedule"',
        '$CI_PIPELINE_SOURCE == "web"',
    }
    pinned = "\n".join(definition[".pinned"]["before_script"])
    assert "AISIM_REVIEWED_COMMIT" in pinned and "-eq 40" in pinned and "rev-parse HEAD" in pinned


def test_ingestion_runs_without_the_cache_token_and_never_pushes_to_the_default_branch(definition):
    job = definition["publish_candidate"]
    text = _text(job)
    assert "env -u INFX_CACHE_WRITE_TOKEN" in text and 'push -q origin "candidate/' in text
    assert "main" not in text and "unchanged refresh: no commit" in text and "git status --porcelain" in text
    assert "qualified" not in text.replace("not qualified", "")


def test_promotion_is_manual_with_a_named_reviewer_and_staleness_is_alert_only(definition):
    assert [r["when"] for r in definition["promote"]["rules"]] == ["manual"]
    assert '--reviewer "$GITLAB_USER_NAME"' in _text(definition["promote"])
    assert definition["staleness"]["allow_failure"] == {"exit_codes": [10, 11]}


def test_failure_alert_job_exists(definition):
    assert definition["alert_failure"]["when"] == "on_failure" and "notify" in _text(definition["alert_failure"])
