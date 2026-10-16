# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib
import io
import json
import zipfile
from itertools import pairwise
from pathlib import Path

import pytest
from test_evaluation import Predictor
from test_evaluation import case as case_fixture

import scripts.pages.prepare_e2e_accuracy_pages as transport
from scripts.fpm_accuracy.dashboard.measurement_heatmaps import _bin_index, _bins
from scripts.fpm_accuracy.dashboard.visualization import VisualizationWriter
from scripts.fpm_accuracy.dashboard_contract import archive_files, population, validate_details, validate_visualization
from scripts.fpm_accuracy.evaluate import evaluate_case
from scripts.pages.prepare_fpm_accuracy_pages import unpack


@pytest.fixture
def case(tmp_path):
    return case_fixture.__wrapped__(tmp_path)


def evaluated(case):
    details = []
    row = evaluate_case(case, factory=lambda m, c: Predictor(m, c, []), details=details)
    summary = json.loads((Path(__file__).parent / "fixtures/summary.json").read_text())
    summary["rows"] = [row]
    summary["snapshot"]["configuration_count"] = 1
    return summary, dict(schema_version=1, snapshot=summary["snapshot"], rows=details)


def test_detail_scoring_and_qualification(case):
    summary, detail = evaluated(case)
    validate_details(detail, summary)
    raw, detail_raw = json.dumps(summary).encode(), json.dumps(detail).encode()
    qualification = dict(
        schema_version=2,
        snapshot=summary["snapshot"],
        summary_sha256=hashlib.sha256(raw).hexdigest(),
        details_sha256=hashlib.sha256(detail_raw).hexdigest(),
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, data in [
            ("summary.json", raw),
            ("details.json", detail_raw),
            ("qualification.json", json.dumps(qualification)),
        ]:
            archive.writestr(name, data)
    assert unpack(output.getvalue()) == summary
    broken = copy.deepcopy(detail)
    broken["rows"][0]["methods"]["regression"][0]["_heatmaps"]["decode"]["cells"][0]["measured_count"] += 1
    with pytest.raises(ValueError):
        validate_details(broken, summary)
    assert summary["rows"][0]["results"]["regression"]["metrics"]["all"]["predicted_count"] == 7


def test_population_changes_with_same_code(case):
    summary, _ = evaluated(case)
    newer = copy.deepcopy(summary)
    newer["snapshot"]["completed_at"] = "2026-10-02T00:00:00+00:00"
    assert population(summary) == population(newer)
    newer["rows"][0]["membership_sha256"] = "f" * 64
    assert population(summary) != population(newer)
    newer = copy.deepcopy(summary)
    newer["rows"][0]["results"]["warmup"]["artifact"]["sha256"] = "e" * 64
    assert population(summary) != population(newer)


def test_bins_have_no_overlaps_or_dropped_boundaries():
    values = list(range(100))
    bins = _bins(values)
    assert len(bins) == 8
    assert {_bin_index(v, bins) for v in values} == set(range(8))
    assert all(a.upper < b.lower for a, b in pairwise(bins))


@pytest.mark.parametrize("name", ["../escape.json", "/absolute.json", "dir/x.json"])
def test_archive_rejects_unsafe_paths(name):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(name, "{}")
    with pytest.raises(ValueError):
        archive_files(output.getvalue())


def test_visualization_integrity(case, tmp_path):
    writer = VisualizationWriter(tmp_path / "points", repo_id="nvidia/aisimulate-fpm-dataset", revision="a" * 40)
    writer.add_case(case)
    writer.finish()
    files = {p.name: p.read_bytes() for p in writer.directory.iterdir()}
    files["qualification.json"] = b"{}"
    assert validate_visualization(files, "a" * 40)["catalog"][0]["measured"] == 12
    with pytest.raises(ValueError):
        validate_visualization(files, "b" * 40)
    chunk = writer.groups[0]["all_files"][0]
    files[chunk] = b"bad"
    with pytest.raises(ValueError):
        validate_visualization(files, "a" * 40)


def test_history_keeps_dataset_changes_and_applies_baseline(tmp_path, monkeypatch):
    from test_publication import archive, job, run

    import scripts.pages.prepare_fpm_accuracy_pages as publish
    from scripts.fpm_accuracy.contract import artifact_key
    from scripts.fpm_accuracy.dashboard_contract import BASELINE

    original = json.loads((Path(__file__).parent / "fixtures/summary.json").read_text())
    original["snapshot"]["commit_sha"] = BASELINE
    records = {}
    for number in (123, 124, 125, 126):
        summary = copy.deepcopy(original)
        summary["snapshot"]["run_id"] = str(number)
        summary["snapshot"]["completed_at"] = f"2026-10-02T{number - 120:02}:00:00+00:00"
        if number >= 125:
            summary["snapshot"]["hf_revision"] = "f" * 40
            summary["rows"][0]["membership_sha256"] = "f" * 64
        if number == 126:
            summary["snapshot"]["commit_sha"] = "c" * 40
        producer = run(summary)
        producer["id"] = number
        records[number] = (summary, producer)
    monkeypatch.setattr(publish, "completed_runs", lambda: [p for _, p in records.values()])
    monkeypatch.setattr(publish.subprocess, "check_output", lambda *a, **k: "")
    monkeypatch.setattr(publish, "ancestor", lambda repo, a, b: not (a == BASELINE and b == "c" * 40))

    def api(path, **kwargs):
        number = int(path.split("/")[2])
        return archive(records[number][0]) if path.startswith("actions/artifacts/") else records[number][1]

    def items(path, key):
        number = int(path.split("/")[2])
        return (
            [dict(id=number, name="fpm-accuracy-web-" + artifact_key("main"), expired=False)]
            if key == "artifacts"
            else [job(records[number][1])]
        )

    monkeypatch.setattr(publish, "api", api)
    monkeypatch.setattr(publish, "api_items", items)
    publish.prepare(tmp_path, tmp_path / "published")
    history = json.loads((tmp_path / "published/dashboard/history.json").read_text())["entries"]
    assert len(history) == 3
    assert {e["snapshot"]["run_id"] for e in history if e["trend"]} == {"124", "125"}
    assert all(e["details_path"] is None for e in history)


@pytest.mark.parametrize("successful", [True, False])
@pytest.mark.parametrize("evaluation_attempt,measurement_attempt", [(1, 1), (1, 2), (2, 1)])
def test_measurement_publication_keeps_last_good_snapshot(
    case, tmp_path, monkeypatch, successful, evaluation_attempt, measurement_attempt
):
    from test_publication import archive, job, run

    import scripts.pages.prepare_fpm_accuracy_pages as publish
    from scripts.fpm_accuracy.contract import artifact_key

    summary, _ = evaluated(case)
    summary["snapshot"]["hf_revision"] = "a" * 40
    summary["snapshot"]["run_attempt"] = str(evaluation_attempt)
    producer = run(summary)
    producer["run_attempt"] = max(evaluation_attempt, measurement_attempt)
    writer = VisualizationWriter(tmp_path / "points", repo_id="nvidia/aisimulate-fpm-dataset", revision="a" * 40)
    writer.add_case(case)
    writer.finish()
    files = {p.name: p.read_bytes() for p in writer.directory.iterdir()}
    # A valid stored ZIP above the default API cap, with each asset below 64 MiB.
    padding = b" " * (33 * 1024 * 1024)
    files["catalog.json"] += padding
    manifest = json.loads(files["manifest.json"])
    manifest["files"]["catalog.json"] = hashlib.sha256(files["catalog.json"]).hexdigest()
    files["manifest.json"] = json.dumps(manifest).encode() + padding
    files["qualification.json"] = json.dumps(
        dict(
            schema_version=1,
            evaluator_sha=producer["head_sha"],
            run_id="123",
            run_attempt=str(measurement_attempt),
            completed_at=summary["snapshot"]["completed_at"],
            hf_revision="a" * 40,
            manifest_sha256=hashlib.sha256(files["manifest.json"]).hexdigest(),
        )
    ).encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as bundle:
        for name, content in files.items():
            bundle.writestr(name, content)
    newer = copy.deepcopy(summary)
    newer["snapshot"].update(run_id="124", hf_revision="b" * 40, completed_at="2026-10-02T09:00:00+00:00")
    new_run = {**producer, "id": 124}
    monkeypatch.setattr(publish, "completed_runs", lambda: [new_run, producer])
    monkeypatch.setattr(publish.subprocess, "check_output", lambda *a, **k: "")
    monkeypatch.setattr(publish, "ancestor", lambda *a: True)
    responses = {
        "actions/runs/123": producer,
        "actions/runs/123/attempts/1": {**producer, "run_attempt": 1},
        "actions/runs/124/attempts/1": {**new_run, "run_attempt": 1},
        "actions/runs/124": new_run,
        "actions/artifacts/1/zip": archive(summary),
        "actions/artifacts/2/zip": output.getvalue(),
        "actions/artifacts/3/zip": archive(newer),
    }
    monkeypatch.setenv("GH_TOKEN", "test-token")

    class Opener:
        def open(self, request, timeout):
            return io.BytesIO(output.getvalue())

    monkeypatch.setattr(transport.urllib.request, "build_opener", lambda *args: Opener())
    assert len(output.getvalue()) > 64 * 1024 * 1024
    with pytest.raises(ValueError, match="oversized Actions response"):
        transport.api("actions/artifacts/2/zip", binary=True)

    def api(path, **kwargs):
        if path == "actions/artifacts/2/zip":
            return transport.api(path, **kwargs)
        return responses[path]

    monkeypatch.setattr(publish, "api", api)

    def items(path, key):
        if key == "artifacts":
            result = [
                dict(id=1 if "/123/" in path else 3, name="fpm-accuracy-web-" + artifact_key("main"), expired=False)
            ]
            if "/123/" in path:
                result.append(dict(id=2, name="fpm-accuracy-measurements", expired=False))
            return result
        result = [job(producer if "/123/" in path else new_run)]
        if "/123/" in path and f"/attempts/{measurement_attempt}/" in path:
            result.append(
                dict(
                    name="Qualify FPM measurements",
                    status="completed",
                    conclusion="success" if successful else "failure",
                    run_id=123,
                    head_sha=producer["head_sha"],
                )
            )
        return result

    monkeypatch.setattr(publish, "api_items", items)
    publish.prepare(tmp_path, tmp_path / "published")
    publication = tmp_path / "published/dashboard/visualization/publication.json"
    assert publication.exists() is successful
    if successful:
        status = json.loads(publication.read_text())
        assert status["hf_revision"] == "a" * 40 and status["current_hf_revision"] == "b" * 40


@pytest.mark.parametrize("size", [8, 9])
def test_actions_download_enforces_selected_limit(monkeypatch, size):
    monkeypatch.setenv("GH_TOKEN", "test-token")

    class Opener:
        def open(self, request, timeout):
            return io.BytesIO(b"x" * size)

    monkeypatch.setattr(transport.urllib.request, "build_opener", lambda *args: Opener())
    if size == 8:
        assert transport.api("actions/artifacts/1/zip", binary=True, max_bytes=8) == b"x" * 8
    else:
        with pytest.raises(ValueError, match="oversized Actions response"):
            transport.api("actions/artifacts/1/zip", binary=True, max_bytes=8)
