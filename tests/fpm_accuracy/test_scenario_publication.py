# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import json
import sys
import zipfile
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_scenarios as scenario_fixtures
from test_evaluation import Predictor
from test_evaluation import case as source_case_fixture

import scripts.pages.prepare_fpm_accuracy_pages as publish
from scripts.fpm_accuracy.contract import METHODS, artifact_key
from scripts.fpm_accuracy.evaluate import evaluate_case
from scripts.fpm_accuracy.hf.models import CaseStatus
from scripts.fpm_accuracy.scenario_contract import validate_scenarios
from scripts.fpm_accuracy.scenarios import evaluate_scenario_case, scenario_document
from scripts.pages.build_pages_site import PagesBuildError, _build_fpm_catalog
from scripts.pages.prepare_fpm_accuracy_pages import unpack

public_scenarios = scenario_fixtures.public_scenarios
recording = scenario_fixtures.recording
scenario_case = scenario_fixtures.scenario_case


@pytest.fixture
def source_case(tmp_path):
    return source_case_fixture.__wrapped__(tmp_path)


def _non_ready_cases(source_case):
    return [
        replace(
            source_case,
            configuration=replace(
                source_case.configuration,
                configuration_id=f"{source_case.configuration_id}-{status}",
                configuration_path=f"{source_case.configuration.configuration_path}-{status}",
            ),
            status=status,
            observations=(),
            measurement_membership_sha256=hashlib.sha256(status.encode()).hexdigest(),
        )
        for status in CaseStatus
        if status != CaseStatus.READY
    ]


@pytest.fixture(params=[status for status in CaseStatus if status != CaseStatus.READY])
def mixed_status_campaign(request, source_case, recording):
    _, options = recording
    unavailable = next(case for case in _non_ready_cases(source_case) if case.status == request.param)
    cases = [source_case, unavailable]
    details = []
    rows = [
        evaluate_case(case, factory=lambda method, context: Predictor(method, context, []), details=details)
        for case in cases
    ]
    snapshot = {
        **json.loads((Path(__file__).parent / "fixtures" / "summary.json").read_text())["snapshot"],
        "configuration_count": len(cases),
    }
    summary = {"schema_version": 2, "snapshot": snapshot, "methods": list(METHODS), "rows": rows}
    document = scenario_document([evaluate_scenario_case(case, **options) for case in cases], snapshot=snapshot)
    return document, summary, {"schema_version": 2, "snapshot": snapshot, "rows": details}


def _details(summary):
    rows = []
    for row in summary["rows"]:
        measurement_maps = {}
        for kind in ("prefill", "decode", "mixed"):
            measured = row["results"]["regression"]["metrics"][kind]["measured_count"]
            measurement_maps[kind] = {
                "x_label": "test x",
                "y_label": "test y",
                "x_bins": [{"label": "all", "lower": 0, "upper": 1}],
                "y_bins": [{"label": "all", "lower": 0, "upper": 1}],
                "cells": [
                    {
                        "x_index": 0,
                        "y_index": 0,
                        "measured_count": measured,
                        "predicted_count": 0,
                        "mape_pct": None,
                    }
                ],
            }
        methods = {}
        for method, result in row["results"].items():
            heatmaps = {
                kind: {
                    **geometry,
                    "cells": [{"x_index": 0, "y_index": 0, **result["metrics"][kind]}],
                }
                for kind, geometry in measurement_maps.items()
            }
            methods[method] = [{**result, "_heatmaps": heatmaps}]
        rows.append(
            {
                "configuration_id": row["configuration_id"],
                "snapshot_id": row["snapshot_id"],
                "membership_sha256": row["membership_sha256"],
                "workload_heatmaps": measurement_maps,
                "methods": methods,
            }
        )
    return {"schema_version": 1, "snapshot": summary["snapshot"], "rows": rows}


def _files(document, summary, version=3, *, details=None):
    files = {"summary.json": json.dumps(summary).encode()}
    qualification = {
        "schema_version": version,
        "snapshot": summary["snapshot"],
        "summary_sha256": hashlib.sha256(files["summary.json"]).hexdigest(),
    }
    if version >= 2:
        files["details.json"] = json.dumps(_details(summary) if details is None else details).encode()
        qualification["details_sha256"] = hashlib.sha256(files["details.json"]).hexdigest()
    if version >= 3:
        files["scenarios.json"] = json.dumps(document).encode()
        qualification["scenarios_sha256"] = hashlib.sha256(files["scenarios.json"]).hexdigest()
    files["qualification.json"] = json.dumps(qualification).encode()
    return files


def _archive(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


@pytest.mark.parametrize("version", [1, 2, 3])
def test_qualified_scenarios_and_legacy_campaigns_are_accepted(public_scenarios, version):
    document, summary = public_scenarios
    assert unpack(_archive(_files(document, summary, version))) == summary


def test_scenario_validation_preserves_non_ready_source_rows(mixed_status_campaign):
    document, summary, _ = mixed_status_campaign
    assert summary["rows"][1]["results"] == {}
    assert validate_scenarios(document, summary) is document
    row = document["rows"][1]
    for methods in [
        *row["results"].values(),
        *(methods for split in row["split_results"].values() for methods in split.values()),
    ]:
        for result in methods.values():
            if result is not None:
                assert result["status"] == row["status"]
                assert result["metrics"]["all"]["measured_count"] == 0
                assert result["cost"]["predict_count"] == result["cost"]["update_count"] == 0


def test_v3_publication_preserves_ready_and_non_ready_rows(mixed_status_campaign):
    document, summary, details = mixed_status_campaign
    assert unpack(_archive(_files(document, summary, details=details))) == summary
    assert details["rows"][1]["methods"] == {}


def test_ready_source_row_still_requires_regression_results(public_scenarios):
    document, summary = deepcopy(public_scenarios)
    del summary["rows"][0]["results"]["regression"]
    with pytest.raises(ValueError):
        validate_scenarios(document, summary)


def test_scenario_checksum_is_verified_before_publication(public_scenarios):
    files = _files(*public_scenarios)
    files["scenarios.json"] += b"\n"
    with pytest.raises(ValueError, match="scenario checksum mismatch"):
        unpack(_archive(files))


def test_old_local_protocol_cannot_be_published_as_strict_c(public_scenarios):
    document, summary = deepcopy(public_scenarios)
    document.update(schema_version=12, protocol_id="fpm-scenarios-v12")
    with pytest.raises(ValueError):
        unpack(_archive(_files(document, summary)))


@pytest.mark.parametrize("method,scenario", [("gbdt", "a"), ("gbdt", "c"), ("hybrid", "ac"), ("selector", "bc")])
@pytest.mark.parametrize("field", ["method", "adaptation", "model_scope", "recipe"])
@pytest.mark.parametrize("missing", [False, True])
def test_recomputed_checksum_cannot_hide_native_model_identity(public_scenarios, method, scenario, field, missing):
    document, summary = deepcopy(public_scenarios)
    evidence = "offline_model" if scenario == "a" else "online_model"
    diagnostics = document["rows"][0]["results"][scenario][method][evidence]["diagnostics"]
    if missing:
        del diagnostics[field]
    else:
        diagnostics[field] = {"id": "different-recipe"} if field == "recipe" else "different-model-policy"
    with pytest.raises(ValueError, match="native scenario .* mismatch"):
        unpack(_archive(_files(document, summary)))


def test_native_diagnostic_extensions_remain_publishable(public_scenarios):
    document, summary = deepcopy(public_scenarios)
    diagnostics = document["rows"][0]["results"]["c"]["gbdt"]["online_model"]["diagnostics"]
    diagnostics["scopes"] = [{"native": {"future_counter": 3, "future_setting": "native default"}}]
    assert unpack(_archive(_files(document, summary))) == summary


@pytest.mark.parametrize(
    "change",
    [
        "missing_ratio",
        "temporal_boundary",
        "wrong_default",
        "different_population",
        "ratio_type",
    ],
)
def test_recomputed_checksum_cannot_hide_invalid_split_grid(public_scenarios, change):
    original, summary = public_scenarios
    document = deepcopy(original)
    assert document["schema_version"] == 13
    row = document["rows"][0]
    splits = row["split_results"]
    if change == "missing_ratio":
        del splits["10"]
    elif change == "temporal_boundary":
        cell = splits["10"]["bc"]["linear"]
        cell["training_count"] += 1
        cell["test_count"] -= 1
    elif change == "wrong_default":
        # JSON serialization separates the default and split-grid objects.
        row["results"] = deepcopy(row["results"])
        row["results"]["b"]["linear"]["cost"]["fit_ms"] += 1
    elif change == "different_population":
        splits["20"]["b"]["spline"]["scoring_population_sha256"] = "f" * 64
    else:
        splits["10.0"] = splits.pop("10")
    with pytest.raises(ValueError):
        unpack(_archive(_files(document, summary)))


@pytest.mark.parametrize("change", ["snapshot", "configuration", "membership"])
def test_recomputed_checksum_cannot_hide_wrong_scenario_identity(public_scenarios, change):
    original, summary = public_scenarios
    document = deepcopy(original)
    if change == "snapshot":
        document["snapshot"]["hf_revision"] = "9" * 40
    elif change == "configuration":
        document["rows"][0]["configuration_id"] = "different-configuration"
    else:
        document["rows"][0]["membership_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="scenario .*mismatch|scenario configuration identity"):
        unpack(_archive(_files(document, summary)))


def test_scenario_qualification_requires_exact_public_asset_set(public_scenarios):
    files = _files(*public_scenarios)
    files["training.json"] = b"[]"
    with pytest.raises(ValueError, match="unexpected FPM artifact files"):
        unpack(_archive(files))


def _stage(tmp_path, public_scenarios):
    document, summary = public_scenarios
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    key = artifact_key("main")
    (artifacts / f"{key}.json").write_text(json.dumps(summary))
    (artifacts / f"{key}-scenarios.json").write_text(json.dumps(document))
    output = tmp_path / "site"
    (output / "fpm-accuracy").mkdir(parents=True)
    return artifacts, output, key


def test_pages_builder_copies_scenarios_without_changing_evidence(tmp_path, public_scenarios):
    artifacts, output, key = _stage(tmp_path, public_scenarios)
    _build_fpm_catalog(Path(__file__).resolve().parents[2], output, False, artifacts)
    emitted = output / "fpm-accuracy" / "branches" / key / "scenarios.json"
    assert json.loads(emitted.read_text()) == public_scenarios[0]
    published = json.loads(emitted.read_text())
    assert published["schema_version"] == 13
    assert set(published["rows"][0]["split_results"]) == {str(value) for value in range(10, 100, 10)}
    catalog = json.loads((output / "fpm-accuracy" / "branches.json").read_text())
    assert catalog["branches"][0]["status"] == "available"


@pytest.mark.parametrize("broken", [False, True])
def test_pages_builder_rejects_scenario_symlinks(tmp_path, public_scenarios, broken):
    artifacts, output, key = _stage(tmp_path, public_scenarios)
    source = artifacts / f"{key}-scenarios.json"
    target = tmp_path / "actual-scenarios.json"
    source.replace(target)
    source.symlink_to(target)
    if broken:
        target.unlink()
    with pytest.raises(PagesBuildError, match="scenarios cannot be a symlink"):
        _build_fpm_catalog(Path(__file__).resolve().parents[2], output, False, artifacts)


@pytest.mark.parametrize("newest_has_scenarios", [True, False])
def test_prepare_keeps_scenarios_with_the_selected_summary_snapshot(
    tmp_path, public_scenarios, monkeypatch, newest_has_scenarios
):
    documents = []
    for hour in (10, 12):
        document, summary = deepcopy(public_scenarios)
        summary["snapshot"]["completed_at"] = f"2026-09-17T{hour}:00:00+00:00"
        document["snapshot"] = deepcopy(summary["snapshot"])
        documents.append((document, summary))
    snapshots = [pair[1]["snapshot"] for pair in documents]
    producer = {
        "id": 123,
        "run_attempt": 1,
        "head_sha": snapshots[0]["evaluator_sha"],
        "head_branch": "main",
        "path": publish.WORKFLOW,
        "status": "completed",
        "conclusion": "success",
        "event": "schedule",
        "repository": {"full_name": publish.REPO},
        "head_repository": {"full_name": publish.REPO},
    }
    versions = [2 if newest_has_scenarios else 3, 3 if newest_has_scenarios else 2]
    payloads = {
        f"actions/artifacts/{index}/zip": _archive(_files(*pair, version))
        for index, (pair, version) in enumerate(zip(documents, versions, strict=True), start=1)
    }
    key = artifact_key("main")
    monkeypatch.setattr(publish, "completed_runs", lambda: [producer])
    monkeypatch.setattr(
        publish,
        "api",
        lambda endpoint, **kwargs: producer if endpoint == "actions/runs/123" else payloads[endpoint],
    )
    monkeypatch.setattr(
        publish,
        "api_items",
        lambda endpoint, kind: (
            [{"id": index, "name": f"fpm-accuracy-web-{key}", "expired": False} for index in (2, 1)]
            if kind == "artifacts"
            else [
                {
                    "name": f"FPM accuracy (main) / Qualify FPM accuracy ({key})",
                    "status": "completed",
                    "conclusion": "success",
                    "head_sha": producer["head_sha"],
                    "run_id": 123,
                }
            ]
        ),
    )
    monkeypatch.setattr(publish, "ancestor", lambda *args: True)
    monkeypatch.setattr(publish.subprocess, "check_output", lambda *args, **kwargs: "")
    output = tmp_path / "prepared"
    publish.prepare(Path(__file__).resolve().parents[2], output)
    assert json.loads((output / f"{key}.json").read_text())["snapshot"] == snapshots[1]
    scenario_path = output / f"{key}-scenarios.json"
    assert scenario_path.exists() == newest_has_scenarios
    if newest_has_scenarios:
        assert json.loads(scenario_path.read_text())["snapshot"] == snapshots[1]


@pytest.mark.parametrize("with_scenarios", [False, True])
@pytest.mark.parametrize(
    "mode", ["hub", "local", "smoke", "hf_revision_mismatch", "evaluation_failure", "oversized", "mixed_statuses"]
)
def test_cli_qualifies_only_complete_pinned_snapshots(
    tmp_path, source_case, recording, monkeypatch, with_scenarios, mode
):
    from scripts.fpm_accuracy import run_fpm_accuracy as cli

    if mode == "oversized":
        monkeypatch.setattr(cli, "MAX_FILE", 1)
    _, options = recording
    wheel = tmp_path / "test.whl"
    wheel.write_bytes(b"test wheel for exporter contract")
    wheel_hash = hashlib.sha256(wheel.read_bytes()).hexdigest()
    output = tmp_path / "export"
    revision = "a" * 40
    cases = [source_case, *_non_ready_cases(source_case)] if mode == "mixed_statuses" else [source_case]
    cases_by_path = {case.configuration.configuration_path: case for case in cases}
    dataset = SimpleNamespace(
        revision="b" * 40 if mode == "hf_revision_mismatch" else revision,
        configurations=lambda: [case.configuration for case in cases],
        measurement_case=lambda configuration_path, **kwargs: cases_by_path[configuration_path],
    )
    source_calls = []

    def load_dataset(*args, **kwargs):
        source_calls.append((args, kwargs))
        return dataset

    monkeypatch.setattr(cli.HfDataset, "from_local" if mode == "local" else "from_hub", load_dataset)
    original_import = cli.importlib.import_module
    monkeypatch.setattr(
        cli.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(RustForwardPassPerfModel=object())
            if name == "aisimulate_core.sdk"
            else original_import(name)
        ),
    )
    monkeypatch.setattr(
        cli.importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(
            read_text=lambda filename: json.dumps({"archive_info": {"hashes": {"sha256": wheel_hash}}})
        ),
    )
    monkeypatch.setattr(
        cli,
        "evaluate_case",
        lambda case, **kwargs: evaluate_case(
            case,
            factory=lambda method, context: Predictor(method, context, []),
            **kwargs,
        ),
    )

    def evaluate_scenarios(case):
        if mode == "evaluation_failure":
            raise RuntimeError("scenario evaluation failed")
        return evaluate_scenario_case(case, **options)

    monkeypatch.setattr(cli, "evaluate_scenario_case", evaluate_scenarios)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_fpm_accuracy",
            "--branch",
            "main",
            "--commit",
            revision,
            "--evaluator-sha",
            revision,
            "--hf-revision",
            revision,
            "--wheel",
            str(wheel),
            "--run-id",
            "123",
            "--run-attempt",
            "1",
            "--output",
            str(output),
            *(["--local-dataset", str(tmp_path)] if mode == "local" else []),
            *(["--configuration", source_case.configuration.configuration_path] if mode == "smoke" else []),
            *(["--scenarios"] if with_scenarios else []),
        ],
    )
    if mode == "hf_revision_mismatch" or (mode in {"evaluation_failure", "oversized"} and with_scenarios):
        with pytest.raises((ValueError, RuntimeError), match="revision differs|scenario evaluation failed|size limit"):
            cli.main()
        assert not (output / "qualification.json").exists()
        return
    cli.main()
    assert source_calls == [((tmp_path,) if mode == "local" else (), {"revision": revision})]
    files = {file.name: file.read_bytes() for file in output.iterdir()}
    if mode == "smoke":
        assert "SMOKE_ONLY.txt" in files
        assert "qualification.json" not in files
        return
    assert json.loads(files["qualification.json"])["schema_version"] == (3 if with_scenarios else 2)
    assert ("scenarios.json" in files) is with_scenarios
    summary = unpack(_archive(files))
    assert [row["status"] for row in summary["rows"]] == [str(case.status) for case in cases]
    if mode == "mixed_statuses":
        assert all(row["results"] == {} for row in summary["rows"][1:])
    if with_scenarios:
        document = json.loads(files["scenarios.json"])
        assert document["snapshot"] == summary["snapshot"]
        assert [row["membership_sha256"] for row in document["rows"]] == [
            row["membership_sha256"] for row in summary["rows"]
        ]
