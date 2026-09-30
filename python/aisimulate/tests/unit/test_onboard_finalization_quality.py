# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finalization exports rechecked quality, separately from serving accuracy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from collector.fpm_forward import cli as collector_cli
from collector.fpm_forward import planner, repeatability
from collector.fpm_forward.config import FPMCollectionOptions
from collector.fpm_forward.database import aggregate_cell, write_formal_database

from aisimulate import main as cli
from aisimulate.config import CorePredictionConfig, CoreRecommendationConfig
from aisimulate.support import finalization, validation_workflow
from aisimulate.support.plan import check_plan, create_plan
from aisimulate.support.schema import SupportRequest

from .collector.test_fpm_repeatability import _runtime_config, _write_campaign
from .test_onboard_finalization import _memory, _prepare_collection, _write, build_completed_collection

pytestmark = pytest.mark.unit


@pytest.fixture
def assessed_collection(tmp_path, monkeypatch):
    monkeypatch.setattr(planner, "_git_revision", lambda: "synthetic-quality-test-revision")
    original, _ = _prepare_collection(tmp_path)
    payload = original.model_dump(mode="json")
    payload["search"]["context_length"] = 2048
    payload["fpm_profile"]["context_length"] = 2048
    payload["fpm_profile"]["deployments"][0]["resources"]["max_batch_size"] = 32
    original = SupportRequest.model_validate(payload)
    root = tmp_path / "quality-source"
    saved = create_plan(original, root)
    args = collector_cli._parser().parse_args(saved["fpm"]["plan_command"][3:])
    plan = planner.build_collection_plan(
        backend="vllm",
        model_path=original.identity.model,
        system=original.identity.gpu,
        selected_ops=set(),
        options=FPMCollectionOptions.from_args(args),
        model_architecture=original.fpm_profile.architecture,
        model_config_path=str(tmp_path / "config.json"),
        fpm_profile=original.fpm_profile,
    )
    campaign = root / "fpm-artifacts" / plan.sha256[:16]
    checkpoint = root / "fpm-checkpoint/fpm_forward.json"
    points = {
        phase: [
            {
                "batch_size": batch,
                "total_prefill_tokens": batch if phase == "prefill" else 0,
                "total_kv_read_tokens": batch * context,
            }
            for batch in (1, 2, 4, 8, 16, 32)
            for context in (16, 64, 256, 1024)
        ]
        for phase in ("prefill", "decode")
    }
    _write_campaign(campaign, checkpoint, plan, attempt_id="source-quality", native_points=points)
    rows = []
    for cell in plan.cells:
        directory = campaign / "cells" / cell.cell_id
        _write(directory / "cell.json", cell.to_dict())
        raw = directory / "raw/pod"
        provenance = json.loads((raw / "collector-provenance.json").read_text())
        worker, scheduler = _memory(
            provenance, cell.workload_kind, original.identity.model, original.identity.model_revision
        )
        for record in (worker, scheduler):
            record["resolved_config"] = {
                **_runtime_config(plan, cell),
                "offload_config": record["resolved_config"]["offload_config"],
            }
        _write(raw / "fpm-memory-worker-dp0-tp0-pp0.json", worker)
        _write(raw / "fpm-memory-scheduler-dp0.json", scheduler)
        rows.extend(aggregate_cell(plan, cell, directory, expected_attempt_id="source-quality"))
    parquet, metadata, skipped = write_formal_database(plan, rows, systems_root=root / "systems/data")
    assert not skipped
    saved = json.loads(checkpoint.read_text())
    saved["database"] = {
        "status": "passed",
        "missing_cells": [],
        "skipped_first_publisher_wins": [],
        "plan_cells": 2,
        "published_cells": 2,
        "parquet": str(parquet),
        "metadata": str(metadata),
    }
    _write(checkpoint, saved)
    case = {
        "request": original,
        "root": root,
        "plan": plan,
        "repeat_factor": 1.0,
        "execution_evidence": True,
        "calls": [],
    }

    def synthesize_repeat(sample_plan, **kwargs):
        case["calls"].append(kwargs)
        _write_campaign(
            Path(kwargs["artifact_root"]) / sample_plan.sha256[:16],
            Path(kwargs["checkpoint_dir"]) / "fpm_forward.json",
            sample_plan,
            attempt_id=f"repeat-{len(case['calls'])}",
            factor=case["repeat_factor"],
            generator_overrides=kwargs["generator_overrides"],
            native_points=points,
            execution_evidence=case["execution_evidence"],
        )
        return 0

    monkeypatch.setattr(repeatability, "run_collection", synthesize_repeat)
    case["validation_args"] = [
        "onboard",
        "validate-collection",
        "--config",
        str(root / "request.yaml"),
        "--output-dir",
        str(root),
        "--validation-output-dir",
        str(tmp_path / "quality"),
    ]
    case["report"] = tmp_path / "quality/collection-validation.json"
    return case


def _assess(case, status):
    if status == "not_assessed":
        return
    case["repeat_factor"] = 1.5 if status == "failed" else 1.0
    assert cli.main([*case["validation_args"], *([] if status == "incomplete" else ["--execute"])]) == (
        1 if status == "failed" else 0
    )
    assert json.loads(case["report"].read_text())["status"] == status


def _finalize_args(case, target, *, include_report=True):
    return [
        "onboard",
        "finalize",
        "--config",
        str(case["root"] / "request.yaml"),
        "--output-dir",
        str(case["root"]),
        "--resolved-output-dir",
        str(target),
        *(["--collection-report", str(case["report"])] if include_report else []),
    ]


@pytest.mark.parametrize("mutation", ["stale_report", "forged_status"])
def test_checkpoint_acceptance_verifies_quality_without_prior_runtime_probe(assessed_collection, tmp_path, mutation):
    case = assessed_collection
    _assess(case, "failed" if mutation == "forged_status" else "passed")
    target = tmp_path / "resolved"
    finalization.finalize(case["request"], case["root"], target, collection_report=case["report"])
    request = SupportRequest.from_yaml(target / "request.yaml")
    payload = request.model_dump(mode="json")
    manifest = finalization.finalization_manifest(request)
    assert "runtime_probe" not in manifest
    if mutation == "stale_report":
        case["report"].write_text(case["report"].read_text() + "\n")
    else:
        manifest["collection_quality"]["status"] = "passed"
        payload["fpm_profile"]["deployments"][0]["resources"]["runtime_memory"]["provenance"] = json.dumps(manifest)
        profile = json.loads(payload["fpm_profile"]["provenance"])
        profile["collection_quality"] = manifest["collection_quality"]
        payload["fpm_profile"]["provenance"] = json.dumps(profile)
    patch = tmp_path / "checkpoint-update.json"
    _write(patch, {"configurations": {"tp1": {"draft_request": payload}}})
    checkpoint = tmp_path / "checkpoint.json"
    with pytest.raises(SystemExit, match="2"):
        cli.main(
            ["onboard", "checkpoint", "--file", str(checkpoint), "--update", str(patch), "--accept-profile", "tp1"]
        )
    assert not checkpoint.exists()


@pytest.mark.parametrize("status", ["passed", "failed", "incomplete", "not_assessed", "historical"])
def test_checkpoint_acceptance_and_resume_preserve_verified_exploratory_quality(
    assessed_collection, tmp_path, capsys, status
):
    case = assessed_collection
    _assess(case, "not_assessed" if status == "historical" else status)
    target = tmp_path / "resolved"
    finalization.finalize(
        case["request"],
        case["root"],
        target,
        collection_report=case["report"] if status not in {"not_assessed", "historical"} else None,
    )
    request = SupportRequest.from_yaml(target / "request.yaml")
    payload = request.model_dump(mode="json")
    manifest = finalization.finalization_manifest(request)
    assert "runtime_probe" not in manifest
    if status == "historical":
        manifest.pop("collection_quality")
        payload["fpm_profile"]["deployments"][0]["resources"]["runtime_memory"]["provenance"] = json.dumps(manifest)
        profile = json.loads(payload["fpm_profile"]["provenance"])
        profile.pop("collection_quality")
        payload["fpm_profile"]["provenance"] = json.dumps(profile)
    patch = tmp_path / "checkpoint-update.json"
    _write(patch, {"configurations": {"tp1": {"draft_request": payload}}})
    checkpoint = tmp_path / "checkpoint.json"
    capsys.readouterr()
    assert (
        cli.main(
            ["onboard", "checkpoint", "--file", str(checkpoint), "--update", str(patch), "--accept-profile", "tp1"]
        )
        == 0
    )
    capsys.readouterr()
    assert cli.main(["onboard", "resume", "--checkpoint", str(checkpoint)]) == 0
    resumed = json.loads(capsys.readouterr().out)
    assert resumed["configurations"]["tp1"]["profile_accepted"]
    assert resumed["integrity_issues"] == []


@pytest.mark.parametrize("status", ["not_assessed", "incomplete", "failed", "passed"])
def test_public_finalization_exports_quality_without_claiming_serving_accuracy(assessed_collection, tmp_path, status):
    case = assessed_collection
    _assess(case, status)
    source = {p: p.read_bytes() for p in case["root"].rglob("*") if p.is_file()}
    target = tmp_path / "resolved"
    assert cli.main(_finalize_args(case, target, include_report=status != "not_assessed")) == 0
    assert source == {p: p.read_bytes() for p in source}
    manifest = json.loads((target / "finalization.json").read_text())
    quality = manifest["collection_quality"]
    assert quality["status"] == status
    assert quality["accuracy"] == "not_assessed"
    if status == "not_assessed":
        assert quality == {"status": "not_assessed", "accuracy": "not_assessed"}
    else:
        assert quality["report"] == validation_workflow._identity(case["report"])
        assert quality["policy"] == validation_workflow._identity(case["report"].parent / "policy.json")
        assert set(quality["gates"]) == {"point_validity", "execution", "repeatability", "interpolation"}
    assert "cell_observations" not in quality
    request = SupportRequest.from_yaml(target / "request.yaml")
    assert json.loads(request.fpm_profile.provenance)["collection_quality"] == quality
    check_plan(request, target)
    for config in (
        CorePredictionConfig.from_yaml(target / "predict/pilot.yaml"),
        CoreRecommendationConfig.from_yaml(target / "recommend/pilot.yaml"),
    ):
        assert json.loads(config.engine.fpm_profile.provenance)["collection_quality"] == quality
        assert config.engine.fpm_profile == request.fpm_profile


@pytest.mark.parametrize("changed", ["status", "policy", "sample", "request", "directory"])
def test_finalization_rejects_stale_or_unrelated_quality(assessed_collection, tmp_path, changed):
    case = assessed_collection
    _assess(case, "passed")
    report = json.loads(case["report"].read_text())
    if changed == "status":
        report["status"] = "failed"
    elif changed == "policy":
        path = case["report"].parent / "policy.json"
        path.write_text(path.read_text() + "\n")
    elif changed == "sample":
        path = next((case["report"].parent / "repeatability").rglob("benchmark-dp0.json"))
        path.write_text(path.read_text() + "\n")
    elif changed == "request":
        payload = case["request"].model_dump(mode="json")
        payload["workload"]["request_count"] += 1
        alternate = tmp_path / "different-request.yaml"
        alternate.write_text(json.dumps(payload))
        report["inputs"]["request"] = validation_workflow._identity(alternate)
    else:
        report["inputs"]["collection_directory"] = str(tmp_path / "another-collection")
    _write(case["report"], report)
    target = tmp_path / "rejected"
    with pytest.raises(SystemExit, match="2"):
        cli.main(_finalize_args(case, target))
    assert not target.exists()


@pytest.mark.parametrize("changed", ["report", "policy", "sample", "source"])
def test_saved_plan_rechecks_bound_quality_evidence(assessed_collection, tmp_path, changed):
    case = assessed_collection
    _assess(case, "passed")
    target = tmp_path / "resolved"
    finalization.finalize(case["request"], case["root"], target, collection_report=case["report"])
    request = SupportRequest.from_yaml(target / "request.yaml")
    path = {
        "report": case["report"],
        "policy": case["report"].parent / "policy.json",
        "source": case["root"] / "request.yaml",
    }.get(changed)
    if changed == "sample":
        path = next((case["report"].parent / "repeatability").rglob("benchmark-dp0.json"))
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        check_plan(request, target)


def test_finalization_uses_the_reports_frozen_nondefault_policy(assessed_collection, tmp_path):
    case = assessed_collection
    policy = tmp_path / "reviewed-policy.json"
    _write(policy, {"repeatability": {"samples": 2, "max_cv": 0.025}})
    assert cli.main([*case["validation_args"], "--execute", "--policy", str(policy)]) == 0
    target = tmp_path / "resolved"
    assert cli.main(_finalize_args(case, target)) == 0
    quality = json.loads((target / "finalization.json").read_text())["collection_quality"]
    assert quality["status"] == "passed"
    assert quality["policy"] == validation_workflow._identity(case["report"].parent / "policy.json")
    assert len(case["calls"]) == 4


@pytest.mark.parametrize("target_status", ["passed", "failed", "incomplete"])
def test_missing_report_cannot_be_relabelled_as_assessed(tmp_path, target_status):
    original, root = build_completed_collection(tmp_path)
    target = tmp_path / "resolved"
    finalization.finalize(original, root, target)
    payload = SupportRequest.from_yaml(target / "request.yaml").model_dump(mode="json")
    resource = payload["fpm_profile"]["deployments"][0]["resources"]
    manifest = json.loads(resource["runtime_memory"]["provenance"])
    manifest["collection_quality"]["status"] = target_status
    resource["runtime_memory"]["provenance"] = json.dumps(manifest)
    profile = json.loads(payload["fpm_profile"]["provenance"])
    profile["collection_quality"] = manifest["collection_quality"]
    payload["fpm_profile"]["provenance"] = json.dumps(profile)
    altered = SupportRequest.model_validate(payload)
    with pytest.raises(ValueError, match="collection quality differs"):
        finalization.verify_finalized_data(altered, target)


def test_historical_finalization_remains_unassessed_without_rewriting_its_manifest(tmp_path):
    original, root = build_completed_collection(tmp_path)
    observations, manifest, formal, _ = finalization._verify_collection(original, root)
    assert "collection_quality" not in manifest
    payload = original.model_dump(mode="json")
    payload["fpm_profile"]["deployments"][0]["resources"] = finalization._merge_resources(observations, manifest)
    historical = SupportRequest.model_validate(payload)
    target = tmp_path / "historical"
    create_plan(historical, target)
    for relative, content in formal.items():
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    before = (target / "request.yaml").read_bytes()
    check_plan(historical, target)
    assert finalization.verify_finalized_quality(finalization.finalization_manifest(historical)) is None
    assert (target / "request.yaml").read_bytes() == before


def test_quality_report_stays_bound_to_original_collection_with_accepted_memory_revision(tmp_path, capsys):
    from aisimulate.support.plan import request_id
    from aisimulate.support.runtime import verify_runtime_profile

    from .test_onboard_runtime import _reviewed_capacity_revision

    checkpoint, original, _, root, memory_config = _reviewed_capacity_revision(tmp_path, capsys)
    revised = SupportRequest.from_yaml(memory_config)
    assessment = tmp_path / "quality"
    assert (
        cli.main(
            [
                "onboard",
                "validate-collection",
                "--config",
                str(root / "request.yaml"),
                "--output-dir",
                str(root),
                "--validation-output-dir",
                str(assessment),
            ]
        )
        == 0
    )
    report = assessment / "collection-validation.json"
    assert json.loads(report.read_text())["status"] == "incomplete"
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    accepted = checkpoint.read_bytes()
    target = tmp_path / "resolved"
    finalization.finalize(original, root, target, memory_config=memory_config, collection_report=report)
    resolved = SupportRequest.from_yaml(target / "request.yaml")
    manifest = finalization.finalization_manifest(resolved)
    assert manifest["collection_quality"]["status"] == "incomplete"
    assert manifest["source_request_id"] == request_id(original)
    assert manifest["memory_revision"]["request_id"] == request_id(revised)
    assert (
        resolved.profile_deployment().resources.runtime_memory.kv_cache_bytes
        == revised.profile_deployment().resources.runtime_memory.kv_cache_bytes
    )
    check_plan(resolved, target)
    verify_runtime_profile(resolved)
    assert before == {p: p.read_bytes() for p in before}
    assert checkpoint.read_bytes() == accepted
