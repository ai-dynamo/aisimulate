# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic integrity tests only; no fixture is GPU evidence or publishable data."""

import copy
import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from collector.fpm_forward.glm53flash_publication import partition_table
from tools.glm53flash_hf import glm53flash as policy
from tools.glm53flash_hf import import_glm53flash as integration

pytestmark = pytest.mark.unit


def report_fixture():
    cells = []
    for backend, quant, tp, phase in sorted(policy.KEYS):
        cells.append(
            {
                "backend": backend,
                "weight_quantization": quant,
                "tp": tp,
                "phase": phase,
                "acceptance": "PASSED",
                "errors": [],
                "calibration_evidence": {
                    "test_fixture": True,
                    "backend_version": "0.30.0+test.CANDIDATE" if backend == "vllm" else "0.5.20",
                },
                "holdout_evidence": {
                    "test_fixture": True,
                    "backend_version": "0.30.0+test.CANDIDATE" if backend == "vllm" else "0.5.20",
                },
                "points": [
                    {
                        "status": "MEASURED_AND_PREDICTED",
                        "context_group": group,
                        "measured_ms": 1.0,
                        "prediction_ms": 1.01,
                    }
                    for group in ("1K-32K", "64K", "128K")
                ],
                "metrics": {
                    "mape_pct": 1.0,
                    "coverage": 1,
                    "requested": 3,
                    "compared": 3,
                },
            }
        )
    return {
        "schema": "glm53flash_independent_holdout_v1",
        "mode": "fpm",
        "acceptance": "PASSED",
        "threshold_mape_pct": 10,
        "errors": [],
        "consumer": {
            "distribution": "aisimulate",
            "api": "RustForwardPassPerfModel.best_available",
            "payload_sha256": "1" * 64,
        },
        "cells": cells,
        "coverage": {
            "required_configurations": 8,
            "required_phase_cells": 16,
            "passed_phase_cells": 16,
            "requested_points": 48,
            "compared_points": 48,
        },
    }


@pytest.fixture
def staged(tmp_path, request):
    """Build ephemeral, explicitly synthetic inputs; never invoke HF APIs."""
    stage = tmp_path / "TEST_ONLY_STAGE"
    stage.mkdir()
    parts, sources = [], []
    for backend, version in (("vllm", "0.30.0+test.CANDIDATE"), ("sglang", "0.5.20")):
        source = tmp_path / backend
        source.mkdir()
        rows = [
            {
                "model_path": model,
                "system": "gb300",
                "backend": backend,
                "backend_version": version,
                "weight_quantization": "nvfp4" if "NVFP4" in model else "fp8_block",
                "kv_cache_dtype": getattr(request, "param", "fp8"),
                "parallel_strategy": "pure_tp",
                "tp": tp,
                "moe_tp": tp,
                "pp": 1,
                "dp": 1,
                "moe_ep": 1,
                "cp": 1,
                "input_tokenizer_revision": revision,
                "workload_kind": phase,
                "latency_ms": 1.0,
                "source_plan_sha256": "2" * 64,
                "collector_attempt_id": "TEST_ONLY",
                "runtime_run_id": "TEST_ONLY",
                "runtime_grid_digest": "3" * 64,
            }
            for model, revision in policy.MODELS.items()
            for tp in (2, 4)
            for phase in ("prefill", "decode")
        ]
        data = source / "fpm_forward_perf.parquet"
        pq.write_table(pa.Table.from_pylist(rows), data)
        metadata = data.with_suffix(".metadata.json")
        integration.write(
            metadata,
            {
                "schema_name": "aic_fpm_forward_perf",
                "schema_version": 7,
                "parquet_sha256": policy.sha(data),
                "row_count": len(rows),
                "system": "gb300",
                "backend": backend,
                "backend_version": version,
                "aic_revision": "1" * 40,
            },
        )
        parts.extend(partition_table(data, metadata, stage))
        # Keep this historical no-external-controller fixture on its original
        # legacy partition metadata contract. Current split-host identity has
        # separate tests with complete v2 execution attachments.
        for part in parts:
            path = stage / part["metadata"]["path"]
            info = policy.read(path)
            for field in (
                "aic_revision",
                "planner_revision",
                "revision_identity_schema",
                "producer_revision_semantics",
            ):
                info.pop(field, None)
            integration.write(path, info)
            part["metadata"]["sha256"] = policy.sha(path)
        for path in (data, metadata):
            rel = "sources/" + policy.sha(path) + path.suffix
            (stage / rel).parent.mkdir(exist_ok=True)
            (stage / rel).write_bytes(path.read_bytes())
            sources.append({"path": rel, "sha256": policy.sha(path)})
    original_manifest = {"schema": "glm53flash_independent_holdout_v1", "mode": "fpm", "test_fixture": "合成"}
    integration.write(stage / "validation/input-manifest.json", original_manifest)
    acceptance = report_fixture()
    acceptance["input_manifest_sha256"] = hashlib.sha256(
        json.dumps(
            original_manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()
    integration.write(stage / "validation/acceptance.json", acceptance)
    integration.write(
        stage / "stage.json",
        {
            "schema": "glm53flash_fpm_publication_stage_v1",
            "status": "STAGED_NOT_PUBLISHED",
            "repo_id": "nvidia/aisimulate-fpm-dataset",
            "input_manifest": {
                "path": "validation/input-manifest.json",
                "sha256": policy.sha(stage / "validation/input-manifest.json"),
            },
            "input_manifest_sha256": policy.sha(stage / "validation/input-manifest.json"),
            "acceptance": {
                "path": "validation/acceptance.json",
                "sha256": policy.sha(stage / "validation/acceptance.json"),
            },
            "sources": sources,
            "configurations": parts,
        },
    )
    external = tmp_path / "TEST_ONLY_EXTERNAL.json"
    integration.write(
        external,
        [
            dict(
                zip(("backend", "weight_quantization", "tp", "phase"), key, strict=True),
                role=role,
                uri="https://example.invalid/TEST_ONLY",
                sha256="4" * 64,
                bytes=1,
            )
            for key in sorted(policy.KEYS)
            for role in ("calibration", "holdout")
        ],
    )
    return stage, external


def test_stage_integrity(staged):
    assert len(policy.validate_stage(staged[0])["configurations"]) == 8


def test_checkpoint_identity_cannot_disagree_with_precision(staged):
    root = staged[0]
    stage = policy.read(root / "stage.json")
    part = stage["configurations"][0]
    other_model = next(model for model in policy.MODELS if model != part["model_id"])
    part.update(model_id=other_model, model_revision=policy.MODELS[other_model])
    integration.write(root / "stage.json", stage)
    with pytest.raises(ValueError, match="wrong model revision"):
        policy.validate_stage(root)


@pytest.mark.parametrize("staged", ["bf16", "auto"], indirect=True)
def test_published_cache_precision_must_match_pinned_fp8(staged):
    with pytest.raises(ValueError, match="execution identity mismatch"):
        policy.validate_stage(staged[0])


def test_routing_rejects_uninspected_validator_without_changes(tmp_path):
    path = tmp_path / "validator.py"
    source = "def validate_dataset(root):\n    return {}\n"
    path.write_text(source)
    with pytest.raises(ValueError, match="API drift"):
        integration.route_policy(path)
    assert path.read_text() == source


def test_prepare_checks_api_pin_before_importing_external_code(staged, tmp_path, monkeypatch):
    from tests.unit.tools.test_glm53flash_raw_campaign import build_bound_fixture

    _, _, root, _ = build_bound_fixture(staged[0], tmp_path, monkeypatch)
    staged = staged[0], root / "external-raw-evidence.json"
    base = tmp_path / "UNINSPECTED_BASE"
    (base / "scripts").mkdir(parents=True)
    (base / "scripts/manage_dataset.py").write_text('raise RuntimeError("must not execute")\n')
    destination = tmp_path / "UNCREATED_DESTINATION"
    with pytest.raises(ValueError, match="API drift"):
        integration.prepare(base, staged[0], destination, staged[1], "1" * 40, "2026-09-23")
    assert not destination.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_phase",
        "duplicate_phase",
        "metric",
        "uncompared",
        "threshold",
        "consumer",
    ],
)
def test_acceptance_rejects_misrepresentation(mutation):
    report = report_fixture()
    if mutation == "missing_phase":
        report["cells"].pop()
    elif mutation == "duplicate_phase":
        report["cells"][-1] = copy.deepcopy(report["cells"][0])
    elif mutation == "metric":
        report["cells"][0]["metrics"]["mape_pct"] = 0
    elif mutation == "uncompared":
        report["cells"][0]["points"][0]["status"] = "NOT_EVALUATED"
    elif mutation == "threshold":
        report["threshold_mape_pct"] = 20
    else:
        report["consumer"] = {}
    with pytest.raises(ValueError):
        policy.validate_acceptance(report)


def test_unbound_external_receipts_require_explicit_stage_and_evidence_roots(staged):
    with pytest.raises(ValueError, match="requires stage and evidence roots"):
        policy.validate_external_receipts(policy.read(staged[1]))


def test_source_mutation_rejected(staged):
    stage = staged[0]
    receipt = policy.read(stage / "stage.json")["sources"][0]
    with (stage / receipt["path"]).open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="digest mismatch"):
        policy.validate_stage(stage)


def test_runtime_identity_cannot_be_changed_with_report_rehash(staged):
    root = staged[0]
    stage = policy.read(root / "stage.json")
    path = root / stage["acceptance"]["path"]
    report = policy.read(path)
    report["cells"][0]["calibration_evidence"]["backend_version"] = "spoofed"
    integration.write(path, report)
    stage["acceptance"]["sha256"] = policy.sha(path)
    integration.write(root / "stage.json", stage)
    with pytest.raises(ValueError, match="runtime differs"):
        policy.validate_stage(root)


@pytest.mark.parametrize("mutation", ["original_bytes", "canonical_digest", "missing_manifest"])
def test_original_input_manifest_remains_bound_to_acceptance(staged, mutation):
    root = staged[0]
    stage = policy.read(root / "stage.json")
    if mutation == "original_bytes":
        with (root / stage["input_manifest"]["path"]).open("a") as stream:
            stream.write("\n")
    elif mutation == "canonical_digest":
        report_path = root / stage["acceptance"]["path"]
        report = policy.read(report_path)
        report["input_manifest_sha256"] = stage["input_manifest_sha256"]
        integration.write(report_path, report)
        stage["acceptance"]["sha256"] = policy.sha(report_path)
    else:
        stage.pop("input_manifest")
    integration.write(root / "stage.json", stage)
    with pytest.raises(ValueError):
        policy.validate_stage(root)


def test_symlink_escape_rejected(tmp_path):
    outer = tmp_path / "outer"
    outer.write_text("x")
    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to(outer)
    with pytest.raises(ValueError, match="escaping"):
        policy.checked(root, {"path": "link", "sha256": policy.sha(outer)})


@pytest.mark.skipif(not os.environ.get("GLM_TEST_BASE"), reason="requires immutable local dataset base")
@pytest.mark.timeout(300)
def test_canonical_integration_keeps_baseline_and_rejects_tampering(staged, tmp_path, monkeypatch):
    from tests.unit.tools.test_glm53flash_raw_campaign import build_bound_fixture

    _, _, root, _ = build_bound_fixture(staged[0], tmp_path, monkeypatch)
    staged = staged[0], root / "external-raw-evidence.json"
    base = Path(os.environ["GLM_TEST_BASE"])
    before = {
        str(p.relative_to(base)): policy.sha(p) for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    }
    output = tmp_path / "TEST_ONLY_CANONICAL_DO_NOT_UPLOAD"
    result = integration.prepare(base, staged[0], output, staged[1], "1" * 40, "2026-09-23")
    assert result["status"] == "CANONICAL_LOCAL_NOT_PUBLISHED"
    assert result["counts"]["configurations"] == 32
    assert result["counts"]["fpm_files"] == 61
    assert result["counts"]["source_fpm_rows"] == 301330
    assert before == {
        str(p.relative_to(base)): policy.sha(p) for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    }
    assert any("0.30.0-test.candidate" in p for p in result["added_manifests"])
    # Exact backend version remains intact despite canonical path normalization.
    manifest = next(policy.read(output / p) for p in result["added_manifests"] if "vllm" in p)
    assert manifest["framework_version"] == "0.30.0+test.CANDIDATE"
    manager = integration.load_manager(output)
    records = policy.read(output / "catalog/fpm.json")["records"]
    record = next(r for r in records if r["source_campaign_id"] == policy.CAMPAIGN)
    record["framework_version"] = "spoofed"
    with pytest.raises(ValueError, match="identity mismatch"):
        policy.validate_catalog_record(output, record)
    assert manager.validate_dataset(output, write_report=False)["configurations"] == 32
