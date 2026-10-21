# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from jsonschema import Draft202012Validator

from aisimulate_core import _native
from aisimulate_core.datasets import (
    Artifact,
    CategoryIndex,
    DatasetIndex,
    Identity,
    Manifest,
    check_dataset,
    check_manifest,
    configuration_id,
    prepare_execution_profile,
)
from aisimulate_core.sdk import RustForwardPassPerfModel, utils
from aisimulate_core.sdk import engine as engine_module
from aisimulate_core.sdk.errors import PerfDataNotAvailableError

pytestmark = pytest.mark.unit


def write(root, path, payload):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload))


def artifact(root, path, role, **kwargs):
    data = (root / path).read_bytes()
    return dict(
        artifact_id=path,
        path=path,
        role=role,
        format="json",
        schema="test/v1",
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
        **kwargs,
    )


@pytest.fixture
def dataset(tmp_path, monkeypatch, request):
    def no_download(*args, **kwargs):
        pytest.fail("local profile preparation must not download model metadata")

    monkeypatch.setattr(utils, "_download_hf_json", no_download)
    root = tmp_path / "dataset"
    systems = tmp_path / "producer-systems"
    systems.mkdir()
    hardware = systems / "h200_sxm.yaml"
    hardware.write_bytes((Path(engine_module.__file__).parents[1] / "systems/h200_sxm.yaml").read_bytes())
    runtime_path = "data/h200_sxm/gemm/vllm/1.0/table.parquet"
    (systems / runtime_path).parent.mkdir(parents=True)
    # A real Parquet artifact also retains the runtime version directory. The
    # replacement measured graph consumes no baseline GEMM/attention tables.
    pq.write_table(pa.table({"latency": [2.0, 5.0]}), systems / runtime_path)
    model = "meta-llama/Meta-Llama-3.1-8B"
    if getattr(request, "param", "bundled") == "local":
        local = tmp_path / "local-model"
        local.mkdir()
        config = Path(engine_module.__file__).parents[1] / "model_configs" / f"{model.replace('/', '--')}_config.json"
        shutil.copyfile(config, local / "config.json")
        model = str(local)
    basis = json.loads(
        _native.engine_spec_json_from_bincode(
            engine_module.compile_engine(
                model,
                "h200_sxm",
                "vllm",
                "1.0",
                systems_path=str(systems),
                database_mode="SILICON",
                shared_layer=False,
                forward_model="op_level",
            )
        )
    )
    table_path = "data/op-latency/h200_sxm/gemm/vllm/1.0/table.parquet"
    (root / table_path).parent.mkdir(parents=True)
    shutil.copyfile(systems / runtime_path, root / table_path)
    table = artifact(root, table_path, "operation-measurements", rows=2, runtime_path=runtime_path)
    table["format"] = "parquet"
    (root / "metadata").mkdir()
    shutil.copyfile(hardware, root / "metadata/h200_sxm.yaml")
    hardware_ref = artifact(root, "metadata/h200_sxm.yaml", "hardware", runtime_path=hardware.name)
    profile = {
        "schema_version": 1,
        "basis": basis,
        "worker_type": "prefill",
        "ops": [
            {
                "MeasuredComposite": {
                    "name": "context_measured_scope",
                    "system": "h200_sxm",
                    "backend": "vllm",
                    "version": "1.0",
                    "is_context": True,
                    "weight_bytes": sum(_native.weights_ops_json(json.dumps(basis["context_ops"]))),
                    "scale_factor": 3.0,
                    "cells": [{"shape": [1, 4, 0], "latency_ms": 2.0}, {"shape": [1, 4, 4], "latency_ms": 5.0}],
                }
            }
        ],
        "retained_files": [
            {"path": a["runtime_path"], "sha256": a["sha256"], "size_bytes": a["bytes"]} for a in [table, hardware_ref]
        ],
    }
    write(root, "metadata/profile.json", profile)
    profile_ref = artifact(root, "metadata/profile.json", "execution-profile")
    config = {
        "model": model,
        "system": "h200_sxm",
        "backend": "vllm",
        "backend_version": "1.0",
        "worker_type": "prefill",
        "estimation_mode": "op_level",
        "fallback_policy": "deny",
        "database_mode": "SILICON",
        "enable_shared_layer": False,
        "systems_paths": [str(systems)],
        "estimator_config": {
            "correction": {"enabled": False},
            "op_level": {"execution_profile": {"path": "profile.json", "sha256": profile_ref["sha256"]}},
        },
    }
    write(root, "metadata/config.json", config)
    identity = Identity(
        hardware="h200_sxm",
        framework="vllm",
        framework_version="1.0",
        operation_family=None,
        model="meta-llama/Meta-Llama-3.1-8B",
        effective_settings={"tp": 1},
        unresolved_identity=None,
    )
    manifest = dict(
        schema_version="aisim-artifact-manifest/v1",
        category="op-latency",
        purpose="execution-profile",
        configuration_id=configuration_id("op-latency", "execution-profile", identity),
        identity=identity.model_dump(),
        requested_settings=None,
        artifacts=[table, profile_ref, artifact(root, "metadata/config.json", "predictor-config"), hardware_ref],
        provenance={"producer_revision": None},
        evidence_gaps=[],
    )
    path = "metadata/manifest.json"
    write(root, path, manifest)
    return root, path, manifest


def test_prepare_preserves_bytes_and_pin_and_refuses_overwrite(dataset):
    root, path, manifest = dataset
    output = root / "prepared"
    config = prepare_execution_profile(root, path, output)
    assert (output / "profile.json").read_bytes() == (root / "metadata/profile.json").read_bytes()
    assert (output / "systems" / manifest["artifacts"][0]["runtime_path"]).read_bytes() == (
        root / manifest["artifacts"][0]["path"]
    ).read_bytes()
    assert config["estimator_config"]["op_level"]["execution_profile"]["sha256"] == manifest["artifacts"][1]["sha256"]
    assert config["systems_paths"] == [str(output / "systems")]
    with pytest.raises(FileExistsError):
        prepare_execution_profile(root, path, output)


@pytest.mark.parametrize("dataset", ["bundled", "local"], indirect=True)
def test_prepared_profile_predicts_and_reloads_without_producer_inputs(dataset):
    root, path, _ = dataset
    output = root.parent / "prepared"
    shutil.rmtree(root.parent / "producer-systems")
    config = prepare_execution_profile(root, path, output)
    shutil.rmtree(root)
    model = RustForwardPassPerfModel.best_available(config)
    # Independent synthetic oracle: three repetitions of 2 ms or 5 ms.
    assert model.predict_prefill_latency(1, 4, 0) == 6.0
    assert model.predict_prefill_latency(1, 8, 4) == 15.0
    for saved in [json.loads((output / "config.json").read_bytes()), model.diagnostics()["provenance"]["config"]]:
        restored = RustForwardPassPerfModel.best_available(saved)
        assert restored.predict_prefill_latency(1, 4, 0) == 6.0
        assert restored.predict_prefill_latency(1, 8, 4) == 15.0


@pytest.mark.parametrize(
    "change",
    ["missing_hardware", "missing_version", "malformed_hardware", "missing_op", "schema", "basis", "remote_model"],
)
def test_preparation_rejects_unusable_runtime_inventory(dataset, change):
    root, path, manifest = dataset
    profile = json.loads((root / "metadata/profile.json").read_bytes())
    config = json.loads((root / "metadata/config.json").read_bytes())
    if change in {"missing_hardware", "missing_version"}:
        missing = "h200_sxm.yaml" if change == "missing_hardware" else manifest["artifacts"][0]["runtime_path"]
        manifest["artifacts"] = [a for a in manifest["artifacts"] if a.get("runtime_path") != missing]
        profile["retained_files"] = [f for f in profile["retained_files"] if f["path"] != missing]
    elif change == "malformed_hardware":
        (root / "metadata/h200_sxm.yaml").write_text("not a system specification")
        record = next(a for a in manifest["artifacts"] if a["role"] == "hardware")
        record.update(artifact(root, record["path"], "hardware", runtime_path="h200_sxm.yaml"))
        retained = next(f for f in profile["retained_files"] if f["path"] == "h200_sxm.yaml")
        retained.update(sha256=record["sha256"], size_bytes=record["bytes"])
    elif change == "missing_op":
        # The original graph's GEMM table is required only when this op survives
        # replacement. The fully measured fixture above legitimately omits it.
        gemm = next(op for op in profile["basis"]["context_ops"] if "Gemm" in op)
        profile["ops"][0]["MeasuredComposite"]["weight_bytes"] -= sum(_native.weights_ops_json(json.dumps([gemm])))
        profile["ops"].append(gemm)
    elif change == "schema":
        profile["schema_version"] = 999
    elif change == "basis":
        profile["basis"]["engine"]["model_name"] = "changed/model"
    else:
        config["model"] = "unbundled/model-must-not-download"
    write(root, "metadata/profile.json", profile)
    record = next(a for a in manifest["artifacts"] if a["role"] == "execution-profile")
    record.update(artifact(root, record["path"], record["role"]))
    config["estimator_config"]["op_level"]["execution_profile"]["sha256"] = record["sha256"]
    write(root, "metadata/config.json", config)
    record = next(a for a in manifest["artifacts"] if a["role"] == "predictor-config")
    record.update(artifact(root, record["path"], record["role"]))
    write(root, path, manifest)
    output = root.parent / "rejected"
    error = PerfDataNotAvailableError if change == "missing_op" else ValueError
    with pytest.raises(error):
        prepare_execution_profile(root, path, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "change", ["hash", "rows", "identity", "duplicate", "traversal", "symlink", "inventory", "schema"]
)
def test_reject_invalid_inputs_before_materialization(dataset, change):
    root, path, manifest = dataset
    first = manifest["artifacts"][0]
    if change == "hash":
        (root / first["path"]).write_bytes(b"changed")
    elif change == "rows":
        first["rows"] += 1
    elif change == "identity":
        manifest["identity"]["effective_settings"]["tp"] = 2
    elif change == "duplicate":
        manifest["artifacts"].append(first.copy())
    elif change == "traversal":
        first["runtime_path"] = "../escape"
    elif change == "symlink":
        target = root / first["path"]
        moved = root / "outside"
        target.rename(moved)
        target.symlink_to(moved)
    elif change == "inventory":
        first["runtime_path"] = "different/path"
    else:
        manifest["schema_version"] = "future/v100"
    write(root, path, manifest)
    with pytest.raises(ValueError):
        prepare_execution_profile(root, path, root / "prepared")
    assert not (root / "prepared").exists()


@pytest.mark.parametrize("field", ["hardware", "framework", "framework_version"])
def test_preparation_rejects_manifest_predictor_identity_mismatch(dataset, field):
    root, path, manifest = dataset
    manifest["identity"][field] = "another-runtime"
    manifest["configuration_id"] = configuration_id(
        manifest["category"], manifest["purpose"], Identity(**manifest["identity"])
    )
    write(root, path, manifest)
    # The manifest is self-consistent, but labels a different runtime from the
    # predictor config and the pinned profile it actually contains.
    check_manifest(root, path)
    output = root.parent / "rejected"
    with pytest.raises(ValueError, match="manifest identity differs from predictor configuration"):
        prepare_execution_profile(root, path, output)
    assert not output.exists()


@pytest.mark.parametrize("dataset_id", ["public/example", "internal/example"])
def test_identical_schema_for_public_and_private(dataset, dataset_id):
    root, path, manifest = dataset
    index_path = "catalog/op-latency/index.json"
    entry = {k: manifest[k] for k in ("configuration_id", "identity", "purpose")}
    entry["manifest"] = artifact(root, path, "configuration-manifest")
    write(root, index_path, {"schema_version": "aisim-category-index/v1", "category": "op-latency", "entries": [entry]})
    catalog = {
        "schema_version": "aisim-dataset-index/v1",
        "dataset_id": dataset_id,
        "categories": {
            "op-latency": {
                "root": "data/op-latency",
                "index": artifact(root, index_path, "category-index"),
                "layout": "category",
            }
        },
    }
    write(root, "catalog/dataset.json", catalog)
    assert check_dataset(root).dataset_id == dataset_id
    # The same generated JSON Schema validates both visibility choices.
    for cls, value in [
        (DatasetIndex, catalog),
        (CategoryIndex, json.loads((root / index_path).read_bytes())),
        (Manifest, manifest),
    ]:
        Draft202012Validator(cls.model_json_schema(by_alias=True)).validate(value)
    manifest["identity"]["hardware"] = "changed"
    write(root, path, manifest)
    with pytest.raises(ValueError, match="integrity"):
        check_dataset(root)


def test_table_identity_is_model_independent(dataset):
    root, path, manifest = dataset
    manifest["purpose"] = "operation-tables"
    manifest["configuration_id"] = configuration_id("op-latency", "operation-tables", Identity(**manifest["identity"]))
    write(root, path, manifest)
    with pytest.raises(ValueError, match="model-independent"):
        check_manifest(root, path)


def test_paths_are_canonical():
    for value in ("/absolute", "a//b", "a/./b", "../b", "a\\b"):
        with pytest.raises(ValueError, match="repository-relative"):
            Artifact(
                artifact_id="a", path=value, role="test", format="json", schema="test/v1", sha256="a" * 64, bytes=0
            )
