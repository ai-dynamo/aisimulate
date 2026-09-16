# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import zipfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_pages_site as pages
import fetch_accuracy_measurements as fetch
import prepare_e2e_accuracy_pages as publish
import run_e2e_accuracy as campaign


def tables():
    return {
        "configs": [{"id": 1, "is_multinode": False}],
        "workflow_runs": [
            {
                "id": 1,
                "run_started_at": "2026-09-12T12:00:00Z",
                "status": "completed",
                "conclusion": "success",
            },
            {
                "id": 2,
                "run_started_at": "2026-09-13T12:00:00Z",
                "status": "completed",
                "conclusion": "success",
            },
        ],
        "benchmark_results": [
            {
                "id": 1,
                "config_id": 1,
                "workflow_run_id": 1,
                "benchmark_type": "single_turn",
                "date": "2026-09-12",
                "isl": 1024,
                "osl": 128,
                "conc": 8,
                "metrics": {"mean_ttft": 1.0, "mean_tpot": 0.01},
                "image": "old",
            },
            {
                "id": 2,
                "config_id": 1,
                "workflow_run_id": 2,
                "benchmark_type": "single_turn",
                "date": "2026-09-13",
                "isl": 1024,
                "osl": 128,
                "conc": 1,
                "metrics": {"mean_ttft": 0.1, "mean_tpot": 0.02},
                "image": "new",
            },
            {
                "id": 3,
                "config_id": 1,
                "workflow_run_id": 2,
                "benchmark_type": "single_turn",
                "date": "2026-09-13",
                "isl": 1024,
                "osl": 128,
                "conc": 2,
                "metrics": {"mean_ttft": 0.2, "mean_tpot": 0.03},
                "image": "new",
            },
        ],
    }


def test_selection_keeps_one_complete_run_and_never_fills_missing_concurrency():
    points, stats = campaign.select_points(tables(), 30)
    assert {point["benchmark"]["id"] for point in points} == {2, 3}
    assert stats["selected"] == 2
    assert stats["measurement_date_through"] == "2026-09-13"


def test_family_without_recent_measurements_retains_its_latest_evidence():
    data = tables()
    data["configs"].append({"id": 2, "model": "older-model", "is_multinode": False})
    old = deepcopy(data["benchmark_results"][0])
    old.update(id=4, config_id=2, date="2026-01-01")
    data["benchmark_results"].append(old)
    points, _ = campaign.select_points(data, 30)
    assert {point["benchmark"]["id"] for point in points} == {2, 3, 4}


@pytest.mark.parametrize(
    "status,conclusion",
    [("completed", "failure"), ("completed", "cancelled"), ("in_progress", None)],
)
def test_incomplete_new_source_run_preserves_the_previous_successful_curve(status, conclusion):
    data = tables()
    data["workflow_runs"][1].update(status=status, conclusion=conclusion)
    points, stats = campaign.select_points(data, 30)
    assert {point["benchmark"]["id"] for point in points} == {1}
    assert stats["excluded"]["incomplete_measurement_run"] == 2


@pytest.mark.parametrize(
    "change,reason",
    [
        ("offload", "nonstandard_or_error"),
        ("multinode", "multinode"),
        ("gpu_limit", "multinode"),
        ("missing_latency", "missing_mean_latency"),
    ],
)
def test_new_ineligible_measurements_do_not_age_out_eligible_evidence(change, reason):
    data = tables()
    data["configs"][0].update(hardware="h200", num_decode_gpu=1)
    newer = {**data["configs"][0], "id": 2}
    data["configs"].append(newer)
    data["benchmark_results"][0]["date"] = "2026-07-01"
    for row in data["benchmark_results"][1:]:
        row["config_id"] = 2
        if change == "offload":
            row["offload_mode"] = "cpu"
        elif change == "missing_latency":
            row["metrics"]["mean_ttft"] = 0
    if change == "multinode":
        newer["is_multinode"] = True
    elif change == "gpu_limit":
        newer["num_decode_gpu"] = 9
    points, stats = campaign.select_points(data, 30)
    assert {point["benchmark"]["id"] for point in points} == {1}
    assert stats["excluded"] == {reason: 2}


@pytest.mark.parametrize(
    "change",
    ["duplicate_id", "duplicate_concurrency", "mixed_image", "multinode", "nonfinite"],
)
def test_bad_or_out_of_scope_measurements_cannot_be_published(change):
    data = tables()
    if change == "duplicate_id":
        data["benchmark_results"].append(deepcopy(data["benchmark_results"][-1]))
    elif change == "duplicate_concurrency":
        data["benchmark_results"][-1]["conc"] = 1
    elif change == "mixed_image":
        data["benchmark_results"][-1]["image"] = "another"
    elif change == "multinode":
        data["configs"][0]["is_multinode"] = True
    else:
        for row in data["benchmark_results"]:
            row["metrics"]["mean_ttft"] = float("nan")
    with pytest.raises(ValueError):
        campaign.select_points(data, 30)


def copy_fixture():
    return "\n".join(
        [
            "SELECT malicious_function();",  # Parser never evaluates SQL.
            'COPY public.configs (id, model, disagg, "precision") FROM stdin;',
            "1\tname\\twith\\nwhitespace\\\\end\tf\tfp8",
            r"\.",
            "COPY public.benchmark_results (id, metrics, error) FROM stdin;",
            '1\t{"mean_ttft": 0.1}\t\\N',
            r"\.",
            "COPY public.workflow_runs (id, date) FROM stdin;",
            "1\t2026-09-13",
            r"\.",
            "",
        ]
    )


def test_copy_reader_decodes_data_without_executing_sql():
    data = fetch.read_copy(io.StringIO(copy_fixture()))
    assert data["configs"][0] == {
        "id": 1,
        "model": "name\twith\nwhitespace\\end",
        "disagg": False,
        "precision": "fp8",
    }
    assert data["benchmark_results"][0]["error"] is None
    assert data["benchmark_results"][0]["metrics"] == {"mean_ttft": 0.1}


@pytest.mark.parametrize(
    "text",
    [
        copy_fixture().rsplit(r"\.", 1)[0],
        copy_fixture().replace("model, disagg", "model, model"),
        copy_fixture().replace("\tf\tfp8", "\tunknown\tfp8"),
        copy_fixture().replace("public.configs", "public.private_table"),
        copy_fixture().replace('"precision"', '"id"'),
        copy_fixture().replace('"precision"', '"unsupported,column"'),
    ],
)
def test_copy_reader_rejects_incomplete_or_ambiguous_data(text):
    with pytest.raises(ValueError):
        fetch.read_copy(io.StringIO(text))


def test_missing_overlapping_and_crashed_points_fail_qualification():
    points = [{"id": "a"}, {"id": "b"}]
    success = {
        "id": "a",
        "outcome": "evaluated",
        "row": {"aisimulate_status": "success"},
    }
    failed = {"id": "b", "outcome": "evaluated", "row": {"aisimulate_status": "failed"}}
    assert len(campaign.qualify_results(points, [success, failed])) == 2
    for outcomes in (
        [success],
        [success, success],
        [success, {"id": "b", "outcome": "worker_failed"}],
        [
            {"id": "a", "outcome": "unsupported"},
            {"id": "b", "outcome": "baseline_failed"},
        ],
    ):
        with pytest.raises(ValueError):
            campaign.qualify_results(points, outcomes)


def test_point_timeout_remains_an_explicit_incomplete_campaign(monkeypatch):
    def timed_out(*args, **kwargs):
        raise subprocess.TimeoutExpired("point", 5)

    monkeypatch.setattr(campaign.subprocess, "run", timed_out)
    assert campaign.run_child({"id": "a"}, 5) == {
        "id": "a",
        "outcome": "worker_failed",
        "reason": "worker_failed",
    }


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    source = tmp_path / "tables.json"
    source.write_text(json.dumps(tables()))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "release_tag": "db-dump/2026-09-14",
                "selection_policy": campaign.POLICY,
                "max_age_days": 30,
            }
        )
    )
    monkeypatch.setattr(
        campaign,
        "wheel_identity",
        lambda path: {"wheel_sha256": "a" * 64, "packages": {"aisimulate": "0.12.0"}},
    )

    def predictor(point, timeout):
        b = point["benchmark"]
        row = {
            "config_id": "opaque",
            "silicon_model": "Alpha",
            "display_name": "Alpha",
            "hf_model_path": "example/Alpha",
            "hardware": "h200",
            "framework": "vllm",
            "precision": "fp8",
            "spec_method": "none",
            "disagg": False,
            "isl": b["isl"],
            "osl": b["osl"],
            "conc": b["conc"],
            "tp_size": 1,
            "pp_size": 1,
            "attention_dp_size": 1,
            "moe_tp_size": None,
            "moe_ep_size": None,
            "silicon_ttft_ms": b["metrics"]["mean_ttft"] * 1000,
            "silicon_tpot_ms": 20,
            "aic_ttft_ms": 100,
            "aic_tpot_ms": 10,
            "dynamo_ttft_ms": 110,
            "dynamo_tpot_ms": 12,
            "aisimulate_status": "success",
            "aisimulate_runner": "aisimulate.engine_replay",
        }
        return {
            "id": point["id"],
            "outcome": "evaluated",
            "row": row,
            "backend_version": "0.10.0",
        }

    monkeypatch.setattr(campaign, "run_child", predictor)
    out = tmp_path / "public"
    campaign.campaign(
        SimpleNamespace(
            tables=source,
            manifest=manifest,
            wheel=tmp_path / "wheel.whl",
            output=out,
            branch="main",
            commit="d" * 40,
            run_id="123",
            run_attempt="1",
            workers=2,
            point_timeout=10,
        )
    )
    summary = json.loads((out / "summary.json").read_text())
    run = {
        "id": 123,
        "run_attempt": 1,
        "status": "completed",
        "head_sha": "a" * 40,
        "event": "schedule",
        "head_branch": "main",
        "path": publish.WORKFLOW,
        "conclusion": "success",
        "repository": {"full_name": publish.REPO},
        "head_repository": {"full_name": publish.REPO},
    }
    return summary, run


def archive(summary, *, qualification=None, extra=None):
    data = campaign.encoded(summary)
    q = qualification or {
        **summary["snapshot"]["campaign"],
        "summary_sha256": hashlib.sha256(data).hexdigest(),
    }
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as z:
        z.writestr("summary.json", data)
        z.writestr("qualification.json", campaign.encoded(q))
        if extra:
            z.writestr(extra, "private")
    return stream.getvalue()


def test_complete_artifact_contains_only_derived_accuracy_and_provenance(artifact):
    summary, run = artifact
    assert publish.validate_artifact(archive(summary), run) == summary
    serialized = json.dumps(summary)
    assert "silicon_ttft_ms" not in serialized
    assert "workflow_run_id" not in serialized
    assert "opaque" not in serialized
    assert summary["snapshot"]["campaign"]["selected"] == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("event", "pull_request"),
        ("head_branch", "feature"),
        ("conclusion", "failure"),
        ("path", ".github/workflows/other.yml"),
        ("run_attempt", 2),
        ("head_repository", {"full_name": "other/repo"}),
    ],
)
def test_untrusted_wrong_attempt_or_failed_producer_rejected(artifact, field, value):
    summary, run = artifact
    run[field] = value
    with pytest.raises(ValueError):
        publish.validate_artifact(archive(summary), run)


@pytest.mark.parametrize(
    "change",
    [
        "raw",
        "raw_nested",
        "incomplete",
        "mixed_revision",
        "unknown_outcome",
        "digest",
        "extra_file",
    ],
)
def test_unqualified_or_unsanitized_artifact_rejected(artifact, change):
    summary, run = artifact
    extra = None
    q = None
    if change == "raw":
        summary["raw_predictions"] = [1]
    elif change == "raw_nested":
        summary["models"][0]["workloads"][0]["gpus"][0]["topologies"][0]["points"][0]["measured"]["raw_ms"] = 42
    elif change == "incomplete":
        summary["snapshot"]["campaign"]["selected"] += 1
    elif change == "mixed_revision":
        summary["snapshot"]["campaign"]["commit_sha"] = "e" * 40
    elif change == "unknown_outcome":
        summary["snapshot"]["campaign"]["outcomes"]["worker_failed"] = 0
    elif change == "digest":
        q = {**summary["snapshot"]["campaign"], "summary_sha256": "0" * 64}
    else:
        extra = "../raw.json"
    with pytest.raises(ValueError):
        publish.validate_artifact(archive(summary, qualification=q, extra=extra), run)


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_ambiguous_json_rejected(text):
    with pytest.raises(ValueError):
        publish.strict_json(text)


def test_qualified_main_updates_catalog_and_legacy_download_together(artifact, tmp_path):
    summary, _ = artifact
    artifacts = tmp_path / "qualified"
    artifacts.mkdir()
    key = hashlib.sha256(b"main").hexdigest()[:16]
    (artifacts / (key + ".json")).write_bytes(campaign.encoded(summary))
    output = tmp_path / "site"
    fpe = tmp_path / "qualified-fpe"
    fpe.mkdir()
    fpe_index = {"files": ["example.csv"], "snapshot": {"source_sha": "a" * 40}}
    (fpe / "index.json").write_text(json.dumps(fpe_index))
    (fpe / "example.csv").write_text("System,Status\nh200_sxm,PASS\n")
    pages.build_site(ROOT, output, fpe_data_dir=fpe, accuracy_artifacts=artifacts)
    published_fpe = output / "data/fpe-support-matrix"
    assert json.loads((published_fpe / "index.json").read_text()) == fpe_index
    assert (published_fpe / "example.csv").read_bytes() == (fpe / "example.csv").read_bytes()
    assert json.loads((output / "e2e-accuracy/summary.json").read_text()) == summary
    catalog = json.loads((output / "e2e-accuracy/branches.json").read_text())
    entry = catalog["branches"][0]
    assert entry["status"] == "evaluated"
    assert entry["published_from_commit"] is None
    assert entry["evaluated_revision"] == summary["snapshot"]["evaluated_revision"]
    assert json.loads((output / "e2e-accuracy" / entry["summary_path"]).read_text()) == summary


def test_nightly_accuracy_is_independent_from_release_staging_and_has_no_public_raw_artifacts():
    workflow = yaml.load(
        (ROOT / ".github/workflows/e2e-accuracy.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    assert workflow["on"]["schedule"] == [{"cron": "17 10 * * *"}]
    assert "workflow_run" not in workflow["on"]
    assert "pull_request" not in workflow["on"]
    matrix = workflow["jobs"]["branches"]
    assert matrix["strategy"]["fail-fast"] == "false"
    assert matrix["strategy"]["max-parallel"] == "2"
    assert matrix["strategy"]["matrix"] == "${{ fromJSON(needs.resolve.outputs.matrix) }}"
    assert matrix["uses"] == "./.github/workflows/e2e-accuracy-branch.yml"
    workflow = yaml.load((ROOT / matrix["uses"]).read_text(), Loader=yaml.BaseLoader)
    assert set(workflow["on"]) == {"workflow_call"}
    assert "continue-on-error" not in workflow["jobs"]["campaign"]
    assert workflow["jobs"]["campaign"]["needs"] == "wheel"
    assert workflow["jobs"]["campaign"]["name"] == "Qualify E2E accuracy (${{ inputs.artifact_key }})"
    uploads = [s for s in workflow["jobs"]["campaign"]["steps"] if "upload-artifact@" in s.get("uses", "")]
    assert len(uploads) == 1 and "if" not in uploads[0]
    assert uploads[0]["with"]["overwrite"] == "true"
    assert uploads[0]["with"]["name"] == "e2e-accuracy-web-${{ inputs.artifact_key }}"
    wheel_upload = next(s for s in workflow["jobs"]["wheel"]["steps"] if "upload-artifact@" in s.get("uses", ""))
    assert wheel_upload["with"]["overwrite"] == "true"
    assert wheel_upload["with"]["name"] == "e2e-accuracy-wheel-${{ inputs.artifact_key }}"
    assert set(uploads[0]["with"]["path"].splitlines()) == {
        "${{ runner.temp }}/accuracy-public/summary.json",
        "${{ runner.temp }}/accuracy-public/qualification.json",
    }
    assert "actions: write" not in (ROOT / ".github/workflows/e2e-accuracy.yml").read_text()
    pages_workflow = yaml.load((ROOT / ".github/workflows/pages.yml").read_text(), Loader=yaml.BaseLoader)
    assert set(pages_workflow["on"]["workflow_run"]["workflows"]) == {
        "FPE Support Matrix",
        "Nightly CI",
        "E2E Accuracy Matrix",
    }
    deploy_build = next(s for s in pages_workflow["jobs"]["build"]["steps"] if "--fpe-data-dir" in s.get("run", ""))
    assert "--accuracy-artifacts" in deploy_build["run"]
    assert deploy_build["if"] == "github.event_name != 'pull_request'"
    checkout = pages_workflow["jobs"]["build"]["steps"][0]
    assert "'main'" in checkout["with"]["ref"]


@pytest.mark.parametrize(
    "previous_time,current_time,should_publish",
    [
        ("2026-09-15T09:00:00Z", "2026-09-15T09:00:00.500000+00:00", True),
        ("2026-09-15T09:00:00.500000+00:00", "2026-09-15T09:00:00Z", False),
        ("2026-09-15T11:00:00+02:00", "2026-09-15T09:00:00.500000+00:00", True),
        ("2026-09-15T09:00:00.000000+00:00", "2026-09-15T09:00:00+00:00", False),
        (None, "2026-09-15T09:00:00+00:00", True),
    ],
)
def test_same_commit_publication_compares_completion_times(
    artifact,
    tmp_path,
    monkeypatch,
    previous_time,
    current_time,
    should_publish,
):
    summary, run = artifact
    run["head_sha"] = "a" * 40
    summary["snapshot"]["campaign"]["completed_at"] = current_time
    summary["snapshot"]["aisimulate_completed_at"] = current_time
    previous = deepcopy(summary)
    previous["snapshot"]["aisimulate_completed_at"] = previous_time
    responses = {
        "actions/workflows/e2e-accuracy.yml/runs?status=completed&branch=main&per_page=100": {"workflow_runs": [run]},
        "actions/runs/123": run,
        "actions/runs/123/artifacts?per_page=100&page=1": {
            "artifacts": [{"id": 7, "name": "e2e-accuracy-web", "expired": False}]
        },
        "actions/artifacts/7/zip": archive(summary),
    }
    monkeypatch.setattr(publish, "api", lambda path, **kwargs: responses[path])
    monkeypatch.setattr(publish, "ancestor", lambda *args: True)
    monkeypatch.setattr(publish.subprocess, "check_output", lambda *args, **kwargs: "")
    monkeypatch.setattr(
        publish.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=json.dumps(previous))
    )
    output = tmp_path / "prepared"
    publish.prepare(ROOT, output)
    files = list(output.glob("*.json"))
    assert bool(files) == should_publish
    if should_publish:
        assert json.loads(files[0].read_text()) == summary


def branch_snapshot(summary, branch, commit, *, attempt=1):
    result = deepcopy(summary)
    snapshot = result["snapshot"]
    revision = {"branch": branch, "commit_sha": commit}
    snapshot["evaluated_revision"] = revision
    snapshot["aic_source"].update(revision)
    snapshot["aic_commit_sha"] = commit
    snapshot["campaign"].update(revision, run_attempt=str(attempt))
    return result


def qualification_job(run, branch, **changes):
    return {
        "name": f"E2E accuracy ({branch}) / Qualify E2E accuracy ({publish.artifact_key(branch)})",
        "run_id": run["id"],
        "head_sha": run["head_sha"],
        "status": "completed",
        "conclusion": "success",
        **changes,
    }


def publication_api(monkeypatch, run, snapshots, *, jobs=None, earlier=None):
    artifacts = [
        {
            "id": index,
            "name": "e2e-accuracy-web-" + publish.artifact_key(s["snapshot"]["campaign"]["branch"]),
            "expired": False,
        }
        for index, s in enumerate(snapshots, 1)
    ]
    responses = {
        "actions/workflows/e2e-accuracy.yml/runs?status=completed&branch=main&per_page=100": {"workflow_runs": [run]},
        f"actions/runs/{run['id']}": run,
        f"actions/runs/{run['id']}/artifacts?per_page=100&page=1": {"artifacts": artifacts},
    }
    responses.update({f"actions/artifacts/{index}/zip": archive(s) for index, s in enumerate(snapshots, 1)})
    attempts = {str(run["run_attempt"]): run, **(earlier or {})}
    for number, attempt in attempts.items():
        path = f"actions/runs/{run['id']}/attempts/{number}"
        responses[path] = attempt
        responses[path + "/jobs?per_page=100&page=1"] = {
            "jobs": (
                jobs[number]
                if jobs is not None
                else [
                    qualification_job(run, s["snapshot"]["campaign"]["branch"])
                    for s in snapshots
                    if s["snapshot"]["campaign"]["run_attempt"] == number
                ]
            )
        }
    monkeypatch.setattr(publish, "api", lambda path, **kwargs: responses[path])
    monkeypatch.setattr(publish, "ancestor", lambda *args: True)
    monkeypatch.setattr(publish.subprocess, "check_output", lambda *args, **kwargs: "release/0.12.0\n")
    monkeypatch.setattr(publish.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    return responses


def prepared_snapshots(tmp_path):
    return {s["snapshot"]["campaign"]["branch"]: s for p in tmp_path.glob("*.json") if (s := json.loads(p.read_text()))}


def test_one_run_publishes_main_and_release_independently(artifact, tmp_path, monkeypatch):
    summary, run = artifact
    release = branch_snapshot(summary, "release/0.12.0", "e" * 40)
    output = tmp_path / "prepared"
    with monkeypatch.context() as scoped:
        publication_api(scoped, run, [summary, release])
        publish.prepare(ROOT, output)
    expected = {"main": summary, "release/0.12.0": release}
    assert prepared_snapshots(output) == expected
    original_git = pages._git
    monkeypatch.setattr(
        pages,
        "_git",
        lambda repo, *args: "refs/remotes/origin/release/0.12.0"
        if args[0] == "for-each-ref"
        else original_git(repo, *args),
    )
    site = tmp_path / "site"
    pages.build_site(ROOT, site, accuracy_refs=True, accuracy_artifacts=output)
    catalog = json.loads((site / "e2e-accuracy/branches.json").read_text())
    assert {entry["branch"] for entry in catalog["branches"]} == set(expected)
    for entry in catalog["branches"]:
        assert entry["status"] == "evaluated"
        assert entry["published_from_commit"] is None
        published = json.loads((site / "e2e-accuracy" / entry["summary_path"]).read_text())
        assert published == expected[entry["branch"]]
    assert json.loads((site / "e2e-accuracy/summary.json").read_text()) == summary


def test_failed_release_does_not_block_successful_main_publication(artifact, tmp_path, monkeypatch):
    summary, run = artifact
    run["conclusion"] = "failure"
    jobs = {"1": [qualification_job(run, "main"), qualification_job(run, "release/0.12.0", conclusion="failure")]}
    publication_api(monkeypatch, run, [summary], jobs=jobs)
    output = tmp_path / "prepared"
    publish.prepare(ROOT, output)
    assert prepared_snapshots(output) == {"main": summary}


def test_failed_job_retry_preserves_successful_prior_attempt_with_exact_provenance(artifact, tmp_path, monkeypatch):
    summary, run = artifact
    earlier = {**run, "conclusion": "failure"}
    run["run_attempt"] = 2
    release = branch_snapshot(summary, "release/0.12.0", "e" * 40, attempt=2)
    publication_api(monkeypatch, run, [summary, release], earlier={"1": earlier})
    output = tmp_path / "prepared"
    publish.prepare(ROOT, output)
    assert prepared_snapshots(output) == {"main": summary, "release/0.12.0": release}
    assert prepared_snapshots(output)["main"]["snapshot"]["campaign"]["run_attempt"] == "1"


@pytest.mark.parametrize("change", ["failed", "missing", "duplicate", "wrong_run", "wrong_sha", "in_progress"])
def test_artifact_requires_its_own_successful_qualification_job(artifact, tmp_path, monkeypatch, change):
    summary, run = artifact
    job = qualification_job(run, "main")
    jobs = [job]
    if change == "failed":
        job["conclusion"] = "failure"
    elif change == "missing":
        jobs = [qualification_job(run, "release/0.12.0")]
    elif change == "duplicate":
        jobs.append(deepcopy(job))
    elif change == "wrong_run":
        job["run_id"] = 999
    elif change == "wrong_sha":
        job["head_sha"] = "f" * 40
    else:
        job["status"] = "in_progress"
    publication_api(monkeypatch, run, [summary], jobs={"1": jobs})
    output = tmp_path / "prepared"
    publish.prepare(ROOT, output)
    assert prepared_snapshots(output) == {}


@pytest.mark.parametrize(
    "change", ["branch_name", "duplicate_branch", "future_attempt", "wrong_run", "wrong_attempt_source"]
)
def test_matrix_artifacts_reject_ambiguous_or_misattributed_provenance(artifact, tmp_path, monkeypatch, change):
    summary, run = artifact
    if change == "future_attempt":
        summary["snapshot"]["campaign"]["run_attempt"] = "2"
    elif change == "wrong_run":
        summary["snapshot"]["campaign"]["run_id"] = "999"
    elif change == "wrong_attempt_source":
        run["run_attempt"] = 2
    responses = publication_api(
        monkeypatch, run, [summary], earlier={"1": {**run, "run_attempt": 1, "head_sha": "f" * 40}}
    )
    artifacts = responses["actions/runs/123/artifacts?per_page=100&page=1"]["artifacts"]
    if change == "branch_name":
        artifacts[0]["name"] = "e2e-accuracy-web-" + publish.artifact_key("release/0.12.0")
    elif change == "duplicate_branch":
        artifacts.append(deepcopy(artifacts[0]))
    with pytest.raises(ValueError):
        publish.prepare(ROOT, tmp_path / "prepared")


def test_matrix_publication_paginates_artifacts_and_attempt_jobs(artifact, tmp_path, monkeypatch):
    summary, run = artifact
    responses = publication_api(monkeypatch, run, [summary])
    artifacts = "actions/runs/123/artifacts?per_page=100&page="
    jobs = "actions/runs/123/attempts/1/jobs?per_page=100&page="
    responses[artifacts + "2"] = responses[artifacts + "1"]
    responses[artifacts + "1"] = {"artifacts": [{"name": "unrelated", "expired": False}] * 100}
    responses[jobs + "2"] = responses[jobs + "1"]
    responses[jobs + "1"] = {"jobs": [{"name": "unrelated"}] * 100}
    output = tmp_path / "prepared"
    publish.prepare(ROOT, output)
    assert prepared_snapshots(output) == {"main": summary}


def test_failed_prior_attempt_requires_that_attempts_successful_branch_job(artifact, tmp_path, monkeypatch):
    summary, run = artifact
    earlier = {**run, "conclusion": "failure"}
    run["run_attempt"] = 2
    publication_api(
        monkeypatch,
        run,
        [summary],
        earlier={"1": earlier},
        jobs={"1": [qualification_job(run, "main", conclusion="failure")], "2": [qualification_job(run, "main")]},
    )
    output = tmp_path / "prepared"
    publish.prepare(ROOT, output)
    assert prepared_snapshots(output) == {}


@pytest.mark.parametrize("reverse", [False, True])
def test_same_commit_campaign_selection_uses_completion_time_not_run_order(artifact, tmp_path, monkeypatch, reverse):
    summary, run = artifact
    summary["snapshot"]["campaign"]["completed_at"] = "2026-09-15T09:00:00Z"
    summary["snapshot"]["aisimulate_completed_at"] = "2026-09-15T09:00:00Z"
    newer = deepcopy(summary)
    newer["snapshot"]["campaign"].update(run_id="124", completed_at="2026-09-15T11:00:00.500000+02:00")
    newer["snapshot"]["aisimulate_completed_at"] = newer["snapshot"]["campaign"]["completed_at"]
    newer_run = {**run, "id": 124}
    responses = publication_api(monkeypatch, run, [summary])
    runs = [run, newer_run]
    responses["actions/workflows/e2e-accuracy.yml/runs?status=completed&branch=main&per_page=100"]["workflow_runs"] = (
        runs[::-1] if reverse else runs
    )
    responses.update(
        {
            "actions/runs/124": newer_run,
            "actions/runs/124/artifacts?per_page=100&page=1": {
                "artifacts": [{"id": 2, "name": "e2e-accuracy-web-" + publish.artifact_key("main"), "expired": False}],
            },
            "actions/artifacts/2/zip": archive(newer),
            "actions/runs/124/attempts/1/jobs?per_page=100&page=1": {"jobs": [qualification_job(newer_run, "main")]},
        }
    )
    output = tmp_path / "prepared"
    publish.prepare(ROOT, output)
    assert prepared_snapshots(output) == {"main": newer}
