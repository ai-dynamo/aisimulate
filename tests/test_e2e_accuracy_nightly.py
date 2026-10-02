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
    assert stats["excluded"] == {"superseded_curve": 1}


@pytest.mark.parametrize("reverse", [False, True])
def test_selection_accounts_for_every_input_row(reverse):
    data = tables()
    if reverse:
        data["benchmark_results"].reverse()
    points, stats = campaign.select_points(data, 30)
    assert len(points) + sum(stats["excluded"].values()) == len(data["benchmark_results"])


@pytest.mark.parametrize("tp", [2, 4, 8])
def test_selection_preserves_single_node_expert_parallel_curves(tp):
    data = tables()
    data["configs"][0].update(hardware="b200", decode_tp=tp, decode_ep=tp, decode_num_workers=0, num_decode_gpu=tp * tp)
    original = deepcopy(data)
    points, stats = campaign.select_points(data, 30)
    assert len(points) == 2
    assert "multinode" not in stats["excluded"]
    assert data == original


@pytest.mark.parametrize(
    "overrides",
    [{"is_multinode": True}, {"disagg": True}, {"decode_num_workers": 2}, {"decode_ep": 3}],
)
def test_gpu_selection_does_not_guess_other_topologies(overrides):
    config = dict(is_multinode=False, decode_tp=4, decode_ep=4, decode_num_workers=0, num_decode_gpu=16)
    config.update(overrides)
    assert campaign.measurement_gpu_count(config) == 16


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


@pytest.mark.parametrize("field", ["num_decode_gpu", "num_prefill_gpu"])
@pytest.mark.parametrize("value", [None, True, 1.5, -1])
def test_invalid_gpu_counts_retain_the_older_eligible_curve(field, value):
    data = tables()
    data["configs"][0].update(hardware="h200", num_decode_gpu=1, num_prefill_gpu=1, disagg=True)
    data["configs"].append({**data["configs"][0], "id": 2, field: value})
    data["benchmark_results"][0]["date"] = "2026-07-01"
    for row in data["benchmark_results"][1:]:
        row["config_id"] = 2
    points, stats = campaign.select_points(data, 30)
    assert {point["benchmark"]["id"] for point in points} == {1}
    assert stats["excluded"] == {"invalid_gpu_count": 2}


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


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"schema_version": None},
        {"unexpected": 1},
        {"selection_policy": "unknown"},
        {"release_tag": "../other"},
        {"max_age_days": -1},
        {"max_age_days": True},
        {"minimum_free_bytes": 0},
        {"parts": []},
        {"parts": {}},
    ],
)
def test_manifest_rejected_before_downloading_or_loading_runtime(tmp_path, change):
    manifest = json.loads((ROOT / ".github/e2e-accuracy-dataset.json").read_text())
    manifest.update(change)
    output = tmp_path / "download"
    with pytest.raises(ValueError):
        fetch.fetch(manifest, output)
    assert not output.exists()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        campaign.campaign(SimpleNamespace(branch="main", commit="a" * 40, workers=1, manifest=path))


@pytest.mark.parametrize("change", [{"name": "../other"}, {"sha256": "bad"}, {"size": True}, {"size": -1}])
def test_manifest_validates_every_part_before_starting_downloads(tmp_path, change):
    manifest = json.loads((ROOT / ".github/e2e-accuracy-dataset.json").read_text())
    manifest["parts"][-1].update(change)
    with pytest.raises(ValueError, match="pinned dump part"):
        fetch.fetch(manifest, tmp_path / "download")
    assert not (tmp_path / "download").exists()


def test_fetch_verifies_pinned_bytes_and_extracts_only_measurement_tables(tmp_path, monkeypatch):
    manifest = json.loads((ROOT / ".github/e2e-accuracy-dataset.json").read_text())
    payload = b"pinned compressed fixture"
    manifest["parts"] = [dict(manifest["parts"][0], size=len(payload), sha256=hashlib.sha256(payload).hexdigest())]
    manifest["minimum_free_bytes"] = 1
    calls = []

    def download(url, *, timeout):
        calls.append(url)
        return io.BytesIO(payload)

    class Decompressor:
        def __init__(self, command, **kwargs):
            assert command == ["zstd", "-dc", str(tmp_path / "measurements.dump.zst")]
            self.stdout = io.BytesIO(b"archive fixture")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def wait(self):
            return 0

    def restore(command, **kwargs):
        assert command[:4] == ["pg_restore", "--data-only", "--no-owner", "--no-privileges"]
        assert command[4:-2] == [flag for table in sorted(fetch.TABLES) for flag in ("--table", table)]
        assert kwargs["check"] is True
        Path(command[-1]).write_text(
            "".join(f"COPY public.{table} (id) FROM stdin;\n1\n\\.\n" for table in sorted(fetch.TABLES))
        )

    monkeypatch.setattr(fetch.urllib.request, "urlopen", download)
    monkeypatch.setattr(fetch.subprocess, "Popen", Decompressor)
    monkeypatch.setattr(fetch.subprocess, "run", restore)
    result = fetch.fetch(manifest, tmp_path)
    assert json.loads(result.read_text()) == {table: [{"id": 1}] for table in fetch.TABLES}
    assert calls == [fetch.RELEASE_ROOT + manifest["release_tag"] + "/" + manifest["parts"][0]["name"]]
    assert not (tmp_path / "measurements.dump.zst").exists()
    assert not (tmp_path / "measurements.copy").exists()


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


@pytest.fixture(params=["aisimulate", "aiconfigurator"])
def artifact(tmp_path, monkeypatch, request):
    source = tmp_path / "tables.json"
    source.write_text(json.dumps(tables()))
    manifest = tmp_path / "manifest.json"
    legacy_manifest = json.loads((ROOT / ".github/e2e-accuracy-dataset.json").read_text())
    legacy_manifest.update(selection_policy=fetch.POLICY, max_age_days=30)
    manifest.write_text(json.dumps(legacy_manifest))
    monkeypatch.setattr(
        campaign,
        "wheel_identity",
        lambda path: {
            "wheel_sha256": "a" * 64,
            "packages": {"aisimulate": "0.12.0"},
            "baseline_api": "aiconfigurator.cli.api"
            if request.param == "aiconfigurator"
            else "aisimulate.legacy_cli.api",
            "config_adapter": request.param + ".sdk.config_adapter",
            "cli_entry_point": "aiconfigurator.main:main"
            if request.param == "aiconfigurator"
            else "aisimulate.legacy_cli.entrypoint:main",
        },
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


def test_campaign_hash_includes_nested_source_manifests(artifact, tmp_path, monkeypatch):
    before, _ = artifact
    original_digest = campaign.digest
    monkeypatch.setattr(
        campaign, "digest", lambda path: "0" * 64 if path.parent.name == "manifests" else original_digest(path)
    )
    output = tmp_path / "changed-manifests"
    campaign.campaign(
        SimpleNamespace(
            tables=tmp_path / "tables.json",
            manifest=tmp_path / "manifest.json",
            wheel=tmp_path / "wheel.whl",
            output=output,
            branch="main",
            commit="d" * 40,
            run_id="123",
            run_attempt="1",
            workers=2,
            point_timeout=10,
        )
    )
    after = json.loads((output / "summary.json").read_text())
    assert after["snapshot"]["campaign"]["driver_sha256"] != before["snapshot"]["campaign"]["driver_sha256"]


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
    fpe_catalog = {
        "schema_version": 1,
        "default": "main",
        "branches": [{"name": "main", "path": ".", "status": "available"}],
    }
    (fpe / "index.json").write_text(json.dumps(fpe_index))
    (fpe / "branches.json").write_text(json.dumps(fpe_catalog))
    (fpe / "example.csv").write_text("System,Status\nh200_sxm,PASS\n")
    pages.build_site(ROOT, output, fpe_data_dir=fpe, accuracy_artifacts=artifacts)
    published_fpe = output / "data/fpe-support-matrix"
    assert json.loads((published_fpe / "index.json").read_text()) == fpe_index
    assert json.loads((published_fpe / "branches.json").read_text()) == fpe_catalog
    assert (published_fpe / "example.csv").read_bytes() == (fpe / "example.csv").read_bytes()
    assert json.loads((output / "e2e-accuracy/summary.json").read_text()) == summary
    catalog = json.loads((output / "e2e-accuracy/branches.json").read_text())
    entry = catalog["branches"][0]
    assert entry["status"] == "evaluated"
    assert entry["published_from_commit"] is None
    assert entry["evaluated_revision"] == summary["snapshot"]["evaluated_revision"]
    assert json.loads((output / "e2e-accuracy" / entry["summary_path"]).read_text()) == summary


@pytest.mark.parametrize("dangling", [False, True])
def test_qualified_artifact_symlinks_never_fall_back_to_committed_data(tmp_path, dangling):
    artifacts = tmp_path / "qualified"
    artifacts.mkdir()
    target = tmp_path / "target.json"
    if not dangling:
        target.write_text("{}")
    (artifacts / (publish.artifact_key("main") + ".json")).symlink_to(target)
    with pytest.raises(pages.PagesBuildError, match="qualified accuracy summary cannot be a symlink"):
        pages.build_site(ROOT, tmp_path / "site", accuracy_artifacts=artifacts)


@pytest.mark.parametrize("content", ["not-json", "{}", "directory", "bad-models"])
def test_invalid_qualified_artifacts_use_pages_error_contract(artifact, tmp_path, content):
    summary, _ = artifact
    artifacts = tmp_path / "qualified"
    artifacts.mkdir()
    path = artifacts / (publish.artifact_key("main") + ".json")
    if content == "directory":
        path.mkdir()
    else:
        if content == "bad-models":
            summary["models"] = None
            content = json.dumps(summary)
        path.write_text(content)
    with pytest.raises(pages.PagesBuildError, match="invalid qualified accuracy artifact") as caught:
        pages.build_site(ROOT, tmp_path / "site", accuracy_artifacts=artifacts)
    assert isinstance(caught.value.__cause__, (OSError, ValueError, KeyError, TypeError))


def test_invalid_gpu_count_exclusions_survive_publication_validation(artifact):
    summary, run = artifact
    summary["snapshot"]["campaign"]["measurement_filter_counts"]["invalid_gpu_count"] = 2
    summary["snapshot"]["campaign"]["exclusion_reasons"]["adapter_topology_mismatch"] = 0
    assert publish.validate_artifact(archive(summary), run) == summary


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
    assert workflow["permissions"] == {"contents": "read", "actions": "read"}
    for job in workflow["jobs"].values():
        assert job.get("permissions", workflow["permissions"]) == {"contents": "read", "actions": "read"}
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
    # The container hook remaps one directory, not multiline absolute paths.
    assert uploads[0]["with"]["path"] == "${{ runner.temp }}/accuracy-public/"
    assert "actions: write" not in (ROOT / ".github/workflows/e2e-accuracy.yml").read_text()
    pages_workflow = yaml.load((ROOT / ".github/workflows/pages.yml").read_text(), Loader=yaml.BaseLoader)
    assert set(pages_workflow["on"]["workflow_run"]["workflows"]) == {
        "FPE Support Matrix",
        "Main branch nightly CI",
        "Release branch nightly CI",
        "Nightly CI",
        "E2E Accuracy Matrix",
        "FPM Accuracy Matrix",
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
        ("2026-09-15", "2026-09-15T09:00:00+00:00", False),
        ("2026-09-15T09:00:00", "2026-09-15T09:00:00+00:00", False),
        ("not-a-date", "2026-09-15T09:00:00+00:00", False),
        ("", "2026-09-15T09:00:00+00:00", False),
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
        publish, "committed_accuracy", lambda *args: (json.dumps(previous), "pages/e2e-accuracy/summary.json")
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
    monkeypatch.setattr(publish, "committed_accuracy", lambda *args: None)
    return responses


def prepared_snapshots(tmp_path):
    return {s["snapshot"]["campaign"]["branch"]: s for p in tmp_path.glob("*.json") if (s := json.loads(p.read_text()))}


@pytest.mark.parametrize("missing_ref", [False, True])
def test_committed_summary_distinguishes_absent_path_from_broken_ref(artifact, tmp_path, monkeypatch, missing_ref):
    summary, run = artifact
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--allow-empty",
            "--signoff",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    if not missing_ref:
        subprocess.run(["git", "-C", str(repo), "update-ref", "refs/remotes/origin/main", "HEAD"], check=True)
    real_check_output = subprocess.check_output
    real_committed_accuracy = publish.committed_accuracy
    publication_api(monkeypatch, run, [summary])
    monkeypatch.setattr(publish.subprocess, "check_output", real_check_output)
    monkeypatch.setattr(publish, "committed_accuracy", real_committed_accuracy)
    output = tmp_path / "prepared"
    if missing_ref:
        with pytest.raises(pages.PagesBuildError, match="cannot read accuracy branch evidence"):
            publish.prepare(repo, output)
        assert not list(output.glob("*.json"))
    else:
        publish.prepare(repo, output)
        assert prepared_snapshots(output) == {"main": summary}


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
        lambda repo, *args: (
            "refs/remotes/origin/release/0.12.0" if args[0] == "for-each-ref" else original_git(repo, *args)
        ),
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


@pytest.mark.parametrize("previous_time", ["not-a-date", "2026-09-15T09:00:00"])
def test_invalid_committed_timestamp_does_not_block_other_branches(artifact, tmp_path, monkeypatch, previous_time):
    summary, run = artifact
    release = branch_snapshot(summary, "release/0.12.0", "e" * 40)
    publication_api(monkeypatch, run, [summary, release])
    previous = deepcopy(summary)
    previous["snapshot"]["aisimulate_completed_at"] = previous_time
    monkeypatch.setattr(
        publish,
        "committed_accuracy",
        lambda repo, ref: ((json.dumps(previous), "pages/e2e-accuracy/summary.json") if ref == "origin/main" else None),
    )
    output = tmp_path / "prepared"
    publish.prepare(ROOT, output)
    assert prepared_snapshots(output) == {"release/0.12.0": release}


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


@pytest.mark.parametrize(
    "counts", [[], {"adapter_unsupported": -1}, {"adapter_unsupported": "1"}, {"adapter_unsupported": True}]
)
def test_invalid_exclusion_counts_cannot_publish(artifact, counts):
    summary, run = artifact
    summary["snapshot"]["campaign"]["exclusion_reasons"] = counts
    with pytest.raises(ValueError):
        publish.validate_artifact(archive(summary), run)


@pytest.fixture
def latest_dump(monkeypatch):
    template = json.loads((ROOT / ".github/e2e-accuracy-dataset.json").read_text())
    tag = "db-dump/2026-09-28"
    name = "inferencex-2026-09-28.dump.zst.part00"
    checksum = "a" * 64
    payload = f"{checksum}  {name}\n".encode()
    releases = [
        {"id": 1, "tag_name": template["release_tag"]},
        {"id": 2, "tag_name": tag},
        {"id": 3, "tag_name": "inferencex-skills-v1.0.0"},
        {"id": 4, "tag_name": "db-dump/2026-09-30", "draft": True},
        {"id": 5, "tag_name": "db-dump/2026-09-29", "prerelease": True},
    ]
    assets = [
        {"name": name, "size": 40_000_000_000, "digest": "sha256:" + checksum, "state": "uploaded"},
        {
            "name": "SHA256SUMS",
            "size": len(payload),
            "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
            "state": "uploaded",
        },
    ]

    def items(path):
        if path == "releases":
            return releases
        assert path == "releases/2/assets"
        return assets

    def download(url, timeout):
        assert url == fetch.RELEASE_ROOT + tag + "/SHA256SUMS"
        assert timeout == 120
        return io.BytesIO(payload)

    monkeypatch.setattr(fetch, "release_items", items)
    monkeypatch.setattr(fetch.urllib.request, "urlopen", download)
    return template, releases, assets


def test_latest_dump_freezes_dated_release_checksums_and_disk_budget(latest_dump):
    template, _, _ = latest_dump
    original = deepcopy(template)
    manifest = fetch.resolve_latest_manifest(template)
    assert template == original
    assert manifest["release_tag"] == "db-dump/2026-09-28"
    assert manifest["parts"] == [
        {"name": "inferencex-2026-09-28.dump.zst.part00", "size": 40_000_000_000, "sha256": "a" * 64}
    ]
    assert manifest["minimum_free_bytes"] == 50_000_000_000
    assert manifest["selection_policy"] == template["selection_policy"]
    assert manifest["max_age_days"] == template["max_age_days"]


@pytest.mark.parametrize(
    "damage",
    ["missing_part", "missing_checksums", "digest", "checksum_digest", "uploading", "extra_part", "duplicate", "gap"],
)
def test_incomplete_latest_dump_fails_without_falling_back(latest_dump, damage):
    template, _, assets = latest_dump
    if damage == "missing_part":
        assets.pop(0)
    elif damage == "missing_checksums":
        assets.pop()
    elif damage == "digest":
        assets[0]["digest"] = None
    elif damage == "checksum_digest":
        assets[1]["digest"] = "sha256:" + "b" * 64
    elif damage == "uploading":
        assets[0]["state"] = "new"
    elif damage == "extra_part":
        assets.append({**assets[0], "name": assets[0]["name"].replace("part00", "part01")})
    elif damage == "duplicate":
        assets.append(assets[0])
    else:
        assets[0]["name"] = assets[0]["name"].replace("part00", "part01")
    with pytest.raises(ValueError):
        fetch.resolve_latest_manifest(template)


def test_latest_dump_requires_a_published_database_release(latest_dump):
    template, releases, _ = latest_dump
    releases[:] = [release for release in releases if release["id"] > 2]
    with pytest.raises(ValueError, match="no published database dump"):
        fetch.resolve_latest_manifest(template)


def test_release_metadata_paginates_and_authenticates_only_api_requests(monkeypatch):
    requests = []
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")

    def download(request, timeout):
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer test-token"
        assert request.full_url.startswith("https://api.github.com/repos/SemiAnalysisAI/InferenceX-app/releases?")
        assert timeout == 120
        return io.BytesIO(json.dumps([{"id": 1}] * 100 if len(requests) == 1 else [{"id": 2}]).encode())

    monkeypatch.setattr(fetch.urllib.request, "urlopen", download)
    assert len(fetch.release_items("releases")) == 101
    assert requests[0].full_url.endswith("page=1")
    assert requests[1].full_url.endswith("page=2")


@pytest.mark.parametrize("first", [b"short", b"broken", OSError("connection reset")])
def test_dump_part_retry_preserves_previous_parts(monkeypatch, first):
    payload = b"valid!"
    part = {"name": "part00", "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    responses = iter([first, payload])

    def download(url, timeout):
        value = next(responses)
        if isinstance(value, Exception):
            raise value
        return io.BytesIO(value)

    monkeypatch.setattr(fetch.urllib.request, "urlopen", download)
    monkeypatch.setattr(fetch.time, "sleep", lambda _: None)
    target = io.BytesIO(b"previous part")
    target.seek(0, 2)
    fetch.download_part("https://example.invalid/part", part, target)
    assert target.getvalue() == b"previous part" + payload


def test_dump_part_retry_fails_closed_after_three_attempts(monkeypatch):
    calls = []

    def download(url, timeout):
        calls.append(url)
        return io.BytesIO(b"corrupt")

    monkeypatch.setattr(fetch.urllib.request, "urlopen", download)
    monkeypatch.setattr(fetch.time, "sleep", lambda _: None)
    target = io.BytesIO(b"previous part")
    target.seek(0, 2)
    with pytest.raises(ValueError, match="received 7/7 bytes"):
        fetch.download_part("https://example.invalid/part", {"name": "part00", "size": 7, "sha256": "a" * 64}, target)
    assert len(calls) == 3
    assert target.getvalue() == b"previous part"


@pytest.mark.parametrize(
    "api,adapter",
    [
        ("aisimulate.legacy_cli.api", "aisimulate.sdk.config_adapter"),
        ("aiconfigurator.cli.api", "aiconfigurator.sdk.config_adapter"),
    ],
)
def test_wheel_identity_accepts_both_packaged_predictor_layouts(tmp_path, monkeypatch, api, adapter):
    members = {
        "aisimulate/_runtime.py": b"runtime",
        "aisimulate/runner.py": b"runner",
        api.replace(".", "/") + ".py": b"baseline",
        adapter.replace(".", "/") + "/__init__.py": b"adapter",
    }
    wheel = tmp_path / "test.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    dist = SimpleNamespace(locate_file=lambda name: tmp_path / name, version="test", files=list(members))
    monkeypatch.setattr(campaign.importlib.metadata, "distribution", lambda _: dist)
    imported = []

    def load(name):
        imported.append(name)
        path = name.replace(".", "/") + ("/__init__.py" if name == adapter else ".py")
        return SimpleNamespace(__file__=str(tmp_path / path))

    monkeypatch.setattr(campaign.importlib, "import_module", load)
    assert campaign.wheel_identity(wheel)["packages"] == {"aisimulate": "test"}
    assert imported == ["aisimulate._runtime", "aisimulate.runner", api, adapter]
    (tmp_path / (api.replace(".", "/") + ".py")).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="differs from qualified wheel"):
        campaign.wheel_identity(wheel)


def test_predictor_layout_rejects_missing_or_mixed_namespaces():
    with pytest.raises(ValueError, match="no supported baseline"):
        campaign.predictor_module_names([])
    with pytest.raises(ValueError, match="matching config adapter"):
        campaign.predictor_module_names(
            ["aisimulate/legacy_cli/api.py", "aiconfigurator/sdk/config_adapter/__init__.py"]
        )


@pytest.mark.parametrize("adapters", [[], ["aisimulate"], ["aiconfigurator"], ["aisimulate", "aiconfigurator"]])
def test_predictor_layout_rejects_multiple_baseline_apis(adapters):
    files = ["aisimulate/legacy_cli/api.py", "aiconfigurator/cli/api.py"]
    files += [name + "/sdk/config_adapter/__init__.py" for name in adapters]
    with pytest.raises(ValueError, match="ambiguous"):
        campaign.predictor_module_names(files)


@pytest.mark.parametrize(
    "api,adapter_name",
    [
        ("aisimulate.legacy_cli.api", "aisimulate.sdk.config_adapter"),
        ("aiconfigurator.cli.api", "aiconfigurator.sdk.config_adapter"),
    ],
)
def test_predict_point_calls_selected_api_and_adapter(monkeypatch, api, adapter_name):
    files = [api.replace(".", "/") + ".py", adapter_name.replace(".", "/") + "/__init__.py"]
    monkeypatch.setattr(campaign.importlib.metadata, "distribution", lambda _: SimpleNamespace(files=files))
    monkeypatch.setitem(sys.modules, "aisimulate.runner", SimpleNamespace(EngineReplayRunnerFactory=object))
    calls = []
    request = SimpleNamespace(
        topology=SimpleNamespace(kind="agg", worker=SimpleNamespace(replicas=1, gpus_per_replica=0))
    )

    def adapt(source):
        calls.append(("adapt", source))
        return SimpleNamespace(requests=[request])

    def kwargs(value):
        assert value is request
        return {"test_input": 42}

    def estimate(**values):
        calls.append(("estimate", values))
        raise ValueError("test baseline failure")

    modules = {
        api: SimpleNamespace(cli_estimate=estimate),
        adapter_name: SimpleNamespace(
            InferenceXSource=lambda **values: values, adapt_config=adapt, to_cli_estimate_kwargs=kwargs
        ),
    }
    monkeypatch.setattr(campaign.importlib, "import_module", modules.__getitem__)
    result = campaign.predict_point({"id": "point", "config": {}, "benchmark": {}})
    assert result["outcome"] == "baseline_failed"
    assert calls == [("adapt", {"config": {}, "benchmark": {}}), ("estimate", {"test_input": 42})]


def test_publication_rejects_unknown_baseline_entry_point(artifact):
    summary, run = artifact
    summary["snapshot"]["aic_source"]["cli_entry_point"] = "foreign.main:main"
    with pytest.raises(ValueError, match="baseline entry point"):
        publish.validate_artifact(archive(summary), run)


@pytest.mark.parametrize(
    "entry", ["aiconfigurator.main:main", "aisimulate.legacy_cli.entrypoint:main", None, "foreign.main:main", 42]
)
def test_site_builder_validates_baseline_entry_point(artifact, tmp_path, entry):
    summary, _ = artifact
    summary["snapshot"]["aic_source"]["cli_entry_point"] = entry
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    if entry in ("aiconfigurator.main:main", "aisimulate.legacy_cli.entrypoint:main"):
        pages._accuracy_summary(path.read_text())
    else:
        with pytest.raises(pages.PagesBuildError, match="legacy AIC CLI source"):
            pages._accuracy_summary(path.read_text())
    del summary["snapshot"]["aic_source"]["cli_entry_point"]
    path.write_text(json.dumps(summary))
    pages._accuracy_summary(path.read_text())


def expert_parallel_point():
    return {
        "id": "ep-point",
        "config": {
            "id": 202,
            "hardware": "b200",
            "framework": "vllm",
            "model": "dsr1",
            "precision": "fp4",
            "spec_method": "none",
            "disagg": False,
            "is_multinode": False,
            "decode_tp": 4,
            "decode_ep": 4,
            "decode_dp_attention": False,
            "decode_num_workers": 0,
            "num_decode_gpu": 16,
        },
        "benchmark": {
            "id": 1,
            "isl": 1024,
            "osl": 128,
            "conc": 64,
            "metrics": {"mean_ttft": 0.5, "mean_tpot": 0.02},
        },
    }


@pytest.fixture
def source_config_adapter(monkeypatch):
    # Pages installs only its pinned Python dependencies, not an AISim wheel.
    # The adapter/replay contracts are pure Python; the runner is stubbed below.
    existing_modules = set(sys.modules)
    monkeypatch.syspath_prepend(str(ROOT / "python" / "aisimulate" / "src"))
    try:
        from aisimulate.sdk import config_adapter

        yield config_adapter
    finally:
        # Do not make later Pages tests mistake source imports for an installed
        # wheel with a native runtime, after monkeypatch restores sys.path.
        for name in set(sys.modules) - existing_modules:
            if name.split(".")[0] in {"aisimulate", "aisimulate_core"}:
                sys.modules.pop(name, None)


@pytest.mark.parametrize("replay_fails", [False, True])
@pytest.mark.parametrize("framework", ["vllm", "sglang", "trt"])
def test_ep_prediction_preserves_physical_gpus_through_publication(
    monkeypatch, source_config_adapter, replay_fails, framework
):
    point = expert_parallel_point()
    point["config"]["framework"] = framework
    calls = []

    def estimate(**kwargs):
        assert kwargs["tp_size"] == 4
        assert kwargs["attention_dp_size"] == 1
        assert kwargs["batch_size"] == 64
        return SimpleNamespace(ttft=500.0, tpot=20.0, backend_version="0.25.0")

    class Runner:
        def run(self, spec):
            args = spec.backend_deployment.agg_engine_args
            assert args["aic_tp_size"] == 4
            assert args["aic_attention_dp_size"] == 1
            assert args["aic_moe_ep_size"] == 4
            assert spec.concurrency == 64
            calls.append("run")
            if replay_fails:
                raise ValueError("missing performance data")
            return SimpleNamespace(metrics={"completed_requests": 640, "mean_ttft_ms": 600, "mean_tpot_ms": 22})

        def close(self):
            calls.append("close")

    monkeypatch.setitem(
        sys.modules,
        "aisimulate.runner",
        SimpleNamespace(EngineReplayRunnerFactory=lambda: SimpleNamespace(create=lambda _: Runner())),
    )
    modules = {
        "aisimulate.legacy_cli.api": SimpleNamespace(cli_estimate=estimate),
        "aisimulate.sdk.config_adapter": source_config_adapter,
    }
    files = ["aisimulate/legacy_cli/api.py", "aisimulate/sdk/config_adapter/__init__.py"]
    monkeypatch.setattr(campaign.importlib.metadata, "distribution", lambda _: SimpleNamespace(files=files))
    monkeypatch.setattr(campaign.importlib, "import_module", modules.__getitem__)
    result = campaign.predict_point(point)
    row = result["row"]
    assert row["aisimulate_total_gpus"] == 4
    assert row["silicon_ttft_ms"] == 500
    assert row["silicon_tpot_ms"] == 20
    assert row["aisimulate_status"] == ("failed" if replay_fails else "success")
    assert calls == ["run", "close"]
    from build_e2e_accuracy_overview import _is_multinode

    assert not _is_multinode(row)
    assert point["config"]["num_decode_gpu"] == 16


@pytest.mark.parametrize(
    ("framework", "memory_field"),
    [("vllm", "gpu_memory_utilization"), ("sglang", "mem_fraction_static"), ("trt", "free_gpu_memory_fraction")],
)
@pytest.mark.parametrize(("disagg", "role_overrides"), [(False, False), (True, False), (True, True)])
def test_replay_preserves_resolved_runtime_limits(
    source_config_adapter, framework, memory_field, disagg, role_overrides
):
    point = expert_parallel_point()
    point["config"]["framework"] = framework
    adapter = source_config_adapter
    overrides = {"free_gpu_memory_fraction": 0.83, "max_seq_len": 4096}
    if disagg:
        point["config"].update(
            disagg=True,
            num_decode_gpu=4,
            decode_num_workers=1,
            prefill_tp=2,
            prefill_ep=2,
            prefill_num_workers=1,
            num_prefill_gpu=2,
        )
    if role_overrides:
        overrides.update(
            prefill_free_gpu_memory_fraction=0.76,
            decode_free_gpu_memory_fraction=0.91,
            prefill_max_seq_len=2048,
            decode_max_seq_len=8192,
            decode_system_name="h200_sxm",
        )
    request = adapter.adapt_config(
        adapter.InferenceXSource(point["config"], point["benchmark"]), adapter.AdapterOverrides(**overrides)
    ).requests[0]
    request = adapter.EstimateRequestV1.model_validate_json(request.model_dump_json())
    if disagg and not role_overrides:
        # The canonical schema permits omitting decode hardware when shared.
        request = request.model_copy(update={"systems": request.systems.model_copy(update={"decode": None})})
    spec = campaign.replay_spec(request, "0.25.0")
    deployment = spec.backend_deployment
    if disagg:
        expected = [(0.76, 2048), (0.91, 8192)] if role_overrides else [(0.83, 4096)] * 2
        engines = [deployment.prefill_engine_args, deployment.decode_engine_args]
        assert engines[1]["aic_system"] == ("h200_sxm" if role_overrides else "b200_sxm")
    else:
        expected = [(0.83, 4096)]
        engines = [deployment.agg_engine_args]
    for args, (memory, context) in zip(engines, expected, strict=True):
        assert args[memory_field] == memory
        assert args["max_model_len"] == context
        assert set(args).intersection(
            {"gpu_memory_utilization", "mem_fraction_static", "free_gpu_memory_fraction"}
        ) == {memory_field}


@pytest.mark.parametrize("dp_attention", [False, True])
def test_old_wheel_incorrect_topology_cannot_publish(monkeypatch, dp_attention):
    point = expert_parallel_point()
    if dp_attention:
        point["config"].update(decode_ep=1, decode_dp_attention=True, num_decode_gpu=4)
    worker = SimpleNamespace(replicas=1, gpus_per_replica=4 if dp_attention else 16, tp_size=4)
    request = SimpleNamespace(topology=SimpleNamespace(kind="agg", worker=worker))
    adapter = SimpleNamespace(
        InferenceXSource=lambda **values: values,
        adapt_config=lambda _: SimpleNamespace(requests=[request]),
    )
    modules = {
        "aisimulate.legacy_cli.api": SimpleNamespace(cli_estimate=None),
        "aisimulate.sdk.config_adapter": adapter,
    }
    files = ["aisimulate/legacy_cli/api.py", "aisimulate/sdk/config_adapter/__init__.py"]
    monkeypatch.setattr(campaign.importlib.metadata, "distribution", lambda _: SimpleNamespace(files=files))
    monkeypatch.setattr(campaign.importlib, "import_module", modules.__getitem__)
    monkeypatch.setitem(sys.modules, "aisimulate.runner", SimpleNamespace(EngineReplayRunnerFactory=object))
    assert campaign.predict_point(point) == {
        "id": "ep-point",
        "outcome": "unsupported",
        "reason": "adapter_topology_mismatch",
    }


def resolved_point(framework="vllm", disagg=False):
    point = expert_parallel_point()
    point["config"].update(framework=framework, disagg=disagg)
    backend = "trtllm" if framework == "trt" else framework
    memory_key = {
        "vllm": "gpu_memory_utilization",
        "sglang": "mem_fraction_static",
        "trtllm": "free_gpu_memory_fraction",
    }[backend]
    role = {
        "framework_version": "source-1.0",
        "topology": {"tp": 4, "pp": 1, "attention_dp": 1, "moe_tp": 1, "moe_ep": 4, "workers": 1},
        "args": {
            "block_size": 16,
            "max_num_seqs": 48,
            "max_num_batched_tokens": 16384,
            memory_key: 0.85,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": False,
            "max_model_len": 32768,
            "kv_cache_dtype": "fp8_e4m3",
        },
        "quantization": {"gemm": "nvfp4", "moe": "nvfp4", "evidence": {"gemm_profile_is_explicit": True}},
    }
    point["source_row"] = {"head_sha": "a" * 40}
    point["deployment"] = {
        "schema_version": "resolved-deployment/1",
        "backend": backend,
        "system": "b200_sxm",
        "model_path": "source/checkpoint",
        "roles": {"prefill": deepcopy(role), "decode": role} if disagg else {"aggregated": role},
        "workload": {
            "isl": 1024,
            "osl": 128,
            "concurrency": 64,
            "request_count": 192,
            "random_range_ratio": 0.5,
            "benchmark_controls": {"seed": 47},
        },
    }
    return point


def resolved_tables():
    point = resolved_point()
    config = point["config"]
    config.update(prefill_tp=1, prefill_ep=1, prefill_dp_attention=False, prefill_num_workers=0, num_prefill_gpu=0)
    bench = point["benchmark"]
    bench.update(
        config_id=config["id"], workflow_run_id=1, date="2026-09-28", image="source-image", benchmark_type="single_turn"
    )
    return {
        "configs": [config],
        "benchmark_results": [bench],
        "workflow_runs": [{"id": 1, "head_sha": "a" * 40, "github_run_id": 100, "run_attempt": 2}],
    }


def test_resolved_campaign_publishes_replay_when_baseline_fails(artifact, tmp_path, monkeypatch):
    _, run = artifact
    data = resolved_tables()
    tables_path = tmp_path / "resolved-tables.json"
    tables_path.write_text(json.dumps(data))
    manifest = tmp_path / "resolved-manifest.json"
    manifest.write_text((ROOT / ".github/e2e-accuracy-dataset.json").read_text())
    predictor = campaign.run_child

    def predict(point, timeout):
        result = predictor(point, timeout)
        result["row"].update(aic_status="failed", aic_ttft_ms=None, aic_tpot_ms=None)
        return result

    monkeypatch.setattr(campaign, "run_child", predict)
    monkeypatch.setattr(
        campaign,
        "resolve_points",
        lambda points, *_: [
            {**point, "deployment": resolved_point()["deployment"], "evidence": {"recipe_sha256": "f" * 64}}
            for point in points
        ],
    )
    output, evidence = tmp_path / "resolved-public", tmp_path / "evidence"
    campaign.campaign(
        SimpleNamespace(
            tables=tables_path,
            manifest=manifest,
            wheel=tmp_path / "wheel.whl",
            output=output,
            branch="main",
            commit="d" * 40,
            run_id="123",
            run_attempt="1",
            workers=2,
            point_timeout=10,
            evidence=evidence,
            source_cache=tmp_path / "source-cache",
        )
    )
    summary = json.loads((output / "summary.json").read_text())
    assert publish.validate_artifact(archive(summary), run) == summary
    resolved = json.loads((evidence / "resolved-points.json").read_text())
    qualification = json.loads((output / "qualification.json").read_text())
    assert qualification["cohort_sha256"] == campaign.sha(resolved)
    results = json.loads((evidence / "results.json").read_text())
    assert results[0]["row"]["aic_status"] == "failed"
    assert summary["totals"]["rows"] == 1


def test_resolved_cohort_joins_provenance_and_accounts_for_every_row():
    from e2e_accuracy_source.cohort import select_points

    data = resolved_tables()
    config = data["configs"][0]
    config.update(disagg=True, is_multinode=True)
    bench = data["benchmark_results"][0]
    data["configs"].append({**config, "id": 300})
    data["benchmark_results"].extend(
        [
            {**bench, "id": 2, "date": "2026-09-27"},
            {**bench, "id": 3, "conc": 128, "image": "old", "date": "2026-09-27"},
            {**bench, "id": 4, "config_id": 300, "date": "2026-01-01"},
            {**bench, "id": 5, "config_id": 999},
            {**bench, "id": 6, "benchmark_type": "multi_turn"},
        ]
    )
    points, stats = select_points(data)
    assert [point["benchmark"]["id"] for point in points] == [1]
    assert points[0]["source_row"]["head_sha"] == "a" * 40
    assert points[0]["source_row"]["run_attempt"] == 2
    assert stats["excluded"] == {
        "orphaned_measurement": 1,
        "source_filter": 1,
        "stale": 1,
        "superseded_image": 1,
        "superseded_row": 1,
    }
    assert len(points) + sum(stats["excluded"].values()) == len(data["benchmark_results"])


@pytest.mark.parametrize("framework", ["vllm", "sglang", "trt"])
@pytest.mark.parametrize("disagg", [False, True])
def test_historical_wheel_estimate_projection_matches_public_adapter(source_config_adapter, framework, disagg):
    from e2e_accuracy_source.cohort import select_points
    from e2e_accuracy_source.estimate import estimate_kwargs
    from e2e_accuracy_source.schema import SiliconRow

    point = resolved_point(framework, disagg)
    points, _ = select_points(resolved_tables())
    row = SiliconRow(**{**points[0]["source_row"], "disagg": disagg, "framework": framework})
    expected = estimate_kwargs(row, point["deployment"]).to_call_kwargs()
    adapter = source_config_adapter
    report = adapter.adapt_config(
        adapter.ResolvedInferenceXSource(
            point["deployment"], point["config"], point["benchmark"], "https://example.com/source"
        )
    )
    actual = adapter.to_cli_estimate_kwargs(report.requests[0])
    for key, value in expected.items():
        assert actual[key] == value, key


@pytest.mark.parametrize("problem", ["workload", "topology", "kv_dtype", "override"])
def test_resolved_adapter_rejects_unrepresentable_inputs(source_config_adapter, problem):
    adapter = source_config_adapter
    point = resolved_point(disagg=True)
    deployment = point["deployment"]
    overrides = adapter.AdapterOverrides()
    if problem == "workload":
        deployment["workload"]["concurrency"] += 1
    elif problem == "topology":
        deployment["roles"]["decode"]["topology"]["attention_dp"] = 0
    elif problem == "kv_dtype":
        deployment["roles"]["decode"]["args"]["kv_cache_dtype"] = "bf16"
    else:
        overrides = adapter.AdapterOverrides(batch_size=1)
    report = adapter.adapt_config(
        adapter.ResolvedInferenceXSource(deployment, point["config"], point["benchmark"], "https://example.com/source"),
        overrides,
    )
    assert not report.requests
    assert report.outcomes[0].status == "rejected"


@pytest.mark.parametrize("framework", ["vllm", "sglang", "trt"])
@pytest.mark.parametrize("disagg", [False, True])
@pytest.mark.parametrize("baseline_fails", [False, True])
def test_source_resolved_prediction_preserves_settings_and_independent_outcomes(
    monkeypatch, source_config_adapter, framework, disagg, baseline_fails
):
    point = resolved_point(framework, disagg)
    calls = []

    def estimate(**kwargs):
        calls.append("estimate")
        assert kwargs["model_path"] == "source/checkpoint"
        assert kwargs["kvcache_quant_mode"] == "fp8"
        assert kwargs["backend_version"] == "database-2.0"
        assert kwargs["prefill_free_gpu_memory_fraction" if disagg else "free_gpu_memory_fraction"] == 0.85
        if baseline_fails:
            raise ValueError("baseline does not support this model")
        return SimpleNamespace(ttft=500, tpot=20)

    class Runner:
        def run(self, spec):
            calls.append("replay")
            deployment = spec.backend_deployment
            args = deployment.decode_engine_args if disagg else deployment.agg_engine_args
            assert args["max_num_seqs"] == 48
            assert args["max_num_batched_tokens"] == 16384
            assert args["enable_prefix_caching"] is True
            assert args["enable_chunked_prefill"] is False
            assert args["aic_backend_version"] == "database-2.0"
            assert args["aic_kv_cache_dtype"] == "fp8"
            assert spec.workload["length_sampler"] == "numpy_random_state"
            assert spec.workload["random_range_ratio"] == 0.5
            assert spec.workload["random_seed"] == 47
            assert spec.workload["request_count"] == 192
            return SimpleNamespace(metrics={"completed_requests": 192, "mean_ttft_ms": 510, "mean_tpot_ms": 21})

        def close(self):
            calls.append("close")

    monkeypatch.setitem(
        sys.modules,
        "aisimulate.runner",
        SimpleNamespace(EngineReplayRunnerFactory=lambda: SimpleNamespace(create=lambda _: Runner())),
    )
    modules = {
        "aisimulate.legacy_cli.api": SimpleNamespace(cli_estimate=estimate),
        "aisimulate.sdk.config_adapter": source_config_adapter,
        "aisimulate.sdk.perf_database": SimpleNamespace(get_latest_database_version=lambda **_: "database-2.0"),
    }
    files = ["aisimulate/legacy_cli/api.py", "aisimulate/sdk/config_adapter/__init__.py"]
    monkeypatch.setattr(campaign.importlib.metadata, "distribution", lambda _: SimpleNamespace(files=files))
    original_import = campaign.importlib.import_module
    monkeypatch.setattr(
        campaign.importlib,
        "import_module",
        lambda name, *a, **kw: modules[name] if name in modules else original_import(name, *a, **kw),
    )
    result = campaign.predict_point(point)
    assert result["outcome"] == "evaluated"
    assert result["row"]["aic_status"] == ("failed" if baseline_fails else "success")
    assert result["row"]["aisimulate_status"] == "success"
    assert calls == ["estimate", "replay", "close"]
    assert result["row"]["aisimulate_total_gpus"] == (8 if disagg else 4)
