# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import json
import zipfile
from pathlib import Path

import prepare_fpm_accuracy_pages as publish
import pytest
import yaml
from fpm_accuracy.contract import artifact_key, eligible_branch, strict_json, validate_summary

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def summary():
    return json.loads((Path(__file__).parent / "fixtures/summary.json").read_text())


def archive(summary):
    data = json.dumps(summary).encode()
    qualification = {
        "schema_version": 1,
        "snapshot": summary["snapshot"],
        "summary_sha256": hashlib.sha256(data).hexdigest(),
    }
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as bundle:
        bundle.writestr("summary.json", data)
        bundle.writestr("qualification.json", json.dumps(qualification))
    return out.getvalue()


def run(summary, **updates):
    return dict(
        id=123,
        run_attempt=1,
        head_sha=summary["snapshot"]["evaluator_sha"],
        head_branch="main",
        path=publish.WORKFLOW,
        status="completed",
        conclusion="failure",
        event="schedule",
        repository={"full_name": publish.REPO},
        head_repository={"full_name": publish.REPO},
        **updates,
    )


def job(producer, branch="main", success=True):
    return {
        "name": f"FPM accuracy ({branch}) / Qualify FPM accuracy ({artifact_key(branch)})",
        "status": "completed",
        "conclusion": "success" if success else "failure",
        "head_sha": producer["head_sha"],
        "run_id": producer["id"],
    }


@pytest.mark.parametrize(
    "branch,expected",
    [
        ("main", True),
        ("release/0.11.9", False),
        ("release/0.12.0", True),
        ("release/0.100.0", True),
        ("release/1.0.0", True),
        ("release/0.12.0-rc1", False),
        ("release/00.12.0", False),
        ("simonec/test", False),
        ("release/../../x", False),
    ],
)
def test_release_policy(branch, expected):
    assert eligible_branch(branch) is expected


def test_successful_branch_can_publish_from_failed_matrix(summary):
    producer = run(summary)
    assert (
        publish.validate_artifact(
            archive(summary), producer, "fpm-accuracy-web-" + artifact_key("main"), [job(producer)]
        )
        == summary
    )
    with pytest.raises(ValueError, match="did not qualify"):
        publish.validate_artifact(
            archive(summary), producer, "fpm-accuracy-web-" + artifact_key("main"), [job(producer, success=False)]
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("commit_sha", "invalid"),
        ("complete", False),
        ("configuration_count", 3),
        ("wheel_sha256", ""),
        ("hf_repo", "private/dataset"),
        ("completed_at", "2026-09-17"),
    ],
)
def test_reject_invalid_campaign(summary, field, value):
    summary["snapshot"][field] = value
    with pytest.raises(ValueError):
        validate_summary(summary)


def test_reject_raw_fields_bad_coverage_and_internal_links(summary):
    summary["rows"][0]["raw_latency"] = [123]
    with pytest.raises(ValueError):
        validate_summary(summary)
    del summary["rows"][0]["raw_latency"]
    metric = summary["rows"][0]["results"]["warmup"]["metrics"]["all"]
    metric["predicted_count"] = 101
    with pytest.raises(ValueError):
        validate_summary(summary)
    metric["predicted_count"] = 80
    summary["rows"][0]["configuration_manifest"] = "https://gitlab-master.nvidia.com/private"
    with pytest.raises(ValueError):
        validate_summary(summary)


@pytest.mark.parametrize("data", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_reject_ambiguous_json(data):
    with pytest.raises(ValueError):
        strict_json(data)


def test_reject_wrong_attempt_evaluator_and_branch(summary):
    producer = run(summary)
    for name in ["run_id", "run_attempt", "evaluator_sha"]:
        value = summary["snapshot"][name]
        summary["snapshot"][name] = "e" * 40 if name == "evaluator_sha" else "2"
        with pytest.raises(ValueError):
            publish.validate_artifact(
                archive(summary), producer, "fpm-accuracy-web-" + artifact_key("main"), [job(producer)]
            )
        summary["snapshot"][name] = value
    with pytest.raises(ValueError, match="wrong branch"):
        publish.validate_artifact(
            archive(summary), producer, "fpm-accuracy-web-" + artifact_key("release/0.12.0"), [job(producer)]
        )


def test_invalid_or_expired_new_result_retains_previous(summary, monkeypatch, tmp_path):
    producer = run(summary)
    responses = {
        "actions/runs/123": producer,
        "actions/artifacts/1/zip": b"broken archive",
        "actions/artifacts/2/zip": archive(summary),
    }
    monkeypatch.setattr(
        publish,
        "api",
        lambda path, **kwargs: (
            {"workflow_runs": [producer]} if path.startswith("actions/workflows/") else responses[path]
        ),
    )
    monkeypatch.setattr(
        publish,
        "api_items",
        lambda path, key: (
            [
                {"id": 3, "name": "fpm-accuracy-web-" + artifact_key("main"), "expired": True},
                {"id": 1, "name": "fpm-accuracy-web-" + artifact_key("main"), "expired": False},
                {"id": 2, "name": "fpm-accuracy-web-" + artifact_key("main"), "expired": False},
            ]
            if key == "artifacts"
            else [job(producer)]
        ),
    )
    monkeypatch.setattr(publish, "ancestor", lambda *args: True)
    monkeypatch.setattr(publish.subprocess, "check_output", lambda *args, **kwargs: "release/0.11.0\nrelease/0.12.0\n")
    publish.prepare(ROOT, tmp_path / "result")
    assert json.loads((tmp_path / "result" / f"{artifact_key('main')}.json").read_text()) == summary


def test_campaign_discovery_includes_older_retained_pages(monkeypatch):
    requests = []

    def api(path):
        requests.append(path)
        return {"workflow_runs": list(range(100)) if path.endswith("page=1") else ["older-success"]}

    monkeypatch.setattr(publish, "api", api)
    assert list(publish.completed_runs())[-1] == "older-success"
    assert len(requests) == 2 and all("&created=%3E%3D" in path for path in requests)


def test_workflow_and_pages_contract():
    workflow = yaml.load((ROOT / ".github/workflows/fpm-accuracy.yml").read_text(), Loader=yaml.BaseLoader)
    assert workflow["on"]["schedule"] == [{"cron": "47 10 * * *"}]
    assert workflow["jobs"]["branches"]["strategy"]["max-parallel"] == "2"
    assert "pull_request" not in workflow["on"]
    branch = yaml.load((ROOT / ".github/workflows/fpm-accuracy-branch.yml").read_text(), Loader=yaml.BaseLoader)
    campaign = branch["jobs"]["campaign"]
    assert campaign["needs"] == "wheel"
    upload = next(s for s in campaign["steps"] if "upload-artifact@" in s.get("uses", ""))
    assert upload["with"]["retention-days"] == "90" and "if" not in upload
    # Container hooks must remap one directory, not multiline host paths.
    assert upload["with"]["path"] == "${{ runner.temp }}/accuracy-public/"
    assert "gitlab" not in json.dumps(workflow) + json.dumps(branch)
    pages = yaml.load((ROOT / ".github/workflows/pages.yml").read_text(), Loader=yaml.BaseLoader)
    assert "FPM Accuracy Matrix" in pages["on"]["workflow_run"]["workflows"]
    preview = next(
        s for s in pages["jobs"]["build"]["steps"] if s.get("name") == "Build pull request preview from repository data"
    )
    assert "--fpm-artifacts" not in preview["run"]
