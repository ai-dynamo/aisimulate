# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aisimulate_core.sdk.glm53flash import MODEL_REVISIONS
from collector.fpm_forward import glm53flash_publication as publication

pytestmark = pytest.mark.unit


def source(tmp_path):
    """Test-only rows; no native or accuracy acceptance is implied."""
    rows = [
        {
            "model_path": model,
            "backend": "sglang",
            "backend_version": "0.5.20",
            "system": "gb300",
            "tp": tp,
            "pp": 1,
            "dp": 1,
            "cp": 1,
            "moe_ep": 1,
            "moe_tp": tp,
            "weight_quantization": "nvfp4" if "NVFP4" in model else "fp8_block",
            "input_tokenizer_revision": revision,
            "workload_kind": phase,
            "source_plan_sha256": f"plan-{tp}-{phase}",
            "collector_attempt_id": f"attempt-{tp}-{phase}",
            "runtime_run_id": f"runtime-{tp}-{phase}",
            "runtime_grid_digest": f"grid-{tp}-{phase}",
            "latency_ms": 1.0 + tp / 100,
            "extra_nullable_column": None if phase == "decode" else "preserve",
        }
        for model, revision in MODEL_REVISIONS.items()
        for tp in (2, 4)
        for phase in ("prefill", "decode")
    ]
    path = tmp_path / "fpm_forward_perf.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    metadata = path.with_suffix(".metadata.json")
    metadata.write_text(
        json.dumps(
            {
                "schema_name": "aic_fpm_forward_perf",
                "schema_version": 7,
                "parquet_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "row_count": len(rows),
                "aic_revision": "a" * 40,
                "model_paths": list(MODEL_REVISIONS),
                "backend": "sglang",
                "backend_version": "0.5.20",
                "system": "gb300",
            }
        )
    )
    return path, metadata


def test_partition_retains_exact_arrow_values_schema_and_complete_source_union(tmp_path):
    path, metadata = source(tmp_path)
    original = pq.read_table(path)
    output = tmp_path / "stage"
    records = publication.partition_table(path, metadata, output)
    assert len(records) == 4
    covered = []
    for record in records:
        indices = record["source_partition"]["row_indices"]
        covered.extend(indices)
        subset = pq.read_table(output / record["parquet"]["path"])
        assert subset.equals(original.take(pa.array(indices, type=pa.int64())))
        meta = json.loads((output / record["metadata"]["path"]).read_text())
        assert "aic_revision" not in meta
        assert meta["producer_revision"] == "a" * 40
        assert meta["source_partition"]["parquet_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert len(meta["collector_attempt_ids"]) == 2
        assert meta["row_count"] == record["rows"] == 2
        assert record["phases"] == ["decode", "prefill"]
    assert sorted(covered) == list(range(original.num_rows))
    assert len(set(covered)) == len(covered)
    with pytest.raises(FileExistsError):
        publication.partition_table(path, metadata, output)


@pytest.mark.parametrize("corruption", ["hash", "metadata_backend", "phase", "checkpoint", "extra_model"])
def test_partition_refuses_incomplete_or_misidentified_sources(tmp_path, corruption):
    path, metadata = source(tmp_path)
    info = json.loads(metadata.read_text())
    if corruption == "hash":
        info["parquet_sha256"] = "0" * 64
    elif corruption == "metadata_backend":
        info["backend"] = "vllm"
    else:
        rows = pq.read_table(path).to_pylist()
        if corruption == "phase":
            rows = [row for row in rows if row["workload_kind"] == "prefill"]
        elif corruption == "checkpoint":
            rows[0]["input_tokenizer_revision"] = "0" * 40
        else:
            rows[0]["model_path"] = "unrelated/model"
        pq.write_table(pa.Table.from_pylist(rows), path)
        info.update(row_count=len(rows), parquet_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    metadata.write_text(json.dumps(info))
    with pytest.raises(ValueError):
        publication.partition_table(path, metadata, tmp_path / "stage")


@pytest.mark.parametrize("status", ["NOT_EVALUATED", "FAILED"])
def test_incomplete_native_acceptance_never_writes_publishable_rows(tmp_path, monkeypatch, status):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"mode":"fpm","entries":[]}')
    called = []

    def evaluate(value, base):
        called.append((value, base))
        return {"acceptance": status, "cells": []}

    monkeypatch.setattr(publication.validation, "evaluate", evaluate)
    with pytest.raises(ValueError, match="sixteen phase cells"):
        publication.stage(manifest, tmp_path / "stage")
    assert len(called) == 1
    assert not list((tmp_path / "stage").rglob("*.parquet"))
    assert json.loads((tmp_path / "stage/stage.json").read_text())["status"] == "NOT_QUALIFIED"
    assert json.loads((tmp_path / "stage/validation/acceptance.json").read_text())["acceptance"] == status


def test_full_matrix_stages_each_original_once_without_claiming_hub_publication(tmp_path, monkeypatch):
    receipts = []
    for backend, version in (("sglang", "0.5.20"), ("vllm", "0.30.0")):
        directory = tmp_path / backend
        directory.mkdir()
        path, metadata = source(directory)
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            row.update(backend=backend, backend_version=version)
        pq.write_table(pa.Table.from_pylist(rows), path)
        info = json.loads(metadata.read_text())
        info.update(
            backend=backend, backend_version=version, parquet_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
        metadata.write_text(json.dumps(info))
        system = directory / "gb300.yaml"
        system.write_text("# identical test-only system in two consumer roots\ndata_dir: data/gb300\n")
        receipts.extend(
            {"path": str(item.relative_to(tmp_path)), "sha256": hashlib.sha256(item.read_bytes()).hexdigest()}
            for item in (path, metadata, system)
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"mode": "fpm", "entries": [{"consumer_data": receipts}] * 16}))
    monkeypatch.setattr(publication.validation, "evaluate", lambda *_: {"acceptance": "PASSED", "test_only": True})
    output = publication.stage(manifest, tmp_path / "stage")
    assert output["status"] == "STAGED_NOT_PUBLISHED"
    assert "revision" not in output
    assert len(output["sources"]) == 5
    yaml_receipts = [receipt for receipt in output["sources"] if receipt["path"].endswith(".yaml")]
    assert len(yaml_receipts) == 1
    assert yaml_receipts[0]["original_consumer_paths"] == ["sglang/gb300.yaml", "vllm/gb300.yaml"]
    assert len(output["configurations"]) == 8
    assert (tmp_path / "stage" / output["input_manifest"]["path"]).read_bytes() == manifest.read_bytes()
    assert output["input_manifest"]["sha256"] == output["input_manifest_sha256"]
    for receipt in output["sources"]:
        assert hashlib.sha256((tmp_path / "stage" / receipt["path"]).read_bytes()).hexdigest() == receipt["sha256"]
        assert receipt["original_consumer_paths"]
    assert "Hub immutable commit" in output["remaining"]
