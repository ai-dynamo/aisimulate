# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from jsonschema import Draft202012Validator

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
def dataset(tmp_path):
    table_path = "data/op-latency/test/gemm/runtime/1/table.parquet"
    (tmp_path / table_path).parent.mkdir(parents=True)
    pq.write_table(pa.table({"latency": [2.0, 5.0]}), tmp_path / table_path)
    table = artifact(tmp_path, table_path, "operation-measurements", rows=2, runtime_path="data/test/table.parquet")
    table["format"] = "parquet"
    profile = {
        "retained_files": [{"path": table["runtime_path"], "sha256": table["sha256"], "size_bytes": table["bytes"]}]
    }
    write(tmp_path, "metadata/profile.json", profile)
    profile_ref = artifact(tmp_path, "metadata/profile.json", "execution-profile")
    config = {
        "systems_paths": ["relative"],
        "estimator_config": {
            "op_level": {"execution_profile": {"path": "profile.json", "sha256": profile_ref["sha256"]}}
        },
    }
    write(tmp_path, "metadata/config.json", config)
    identity = Identity(
        hardware="test",
        framework="runtime",
        framework_version="1",
        operation_family=None,
        model="test/model",
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
        artifacts=[table, profile_ref, artifact(tmp_path, "metadata/config.json", "predictor-config")],
        provenance={"producer_revision": None},
        evidence_gaps=[],
    )
    path = "metadata/manifest.json"
    write(tmp_path, path, manifest)
    return tmp_path, path, manifest


def test_prepare_preserves_bytes_and_pin_and_refuses_overwrite(dataset):
    root, path, manifest = dataset
    output = root / "prepared"
    config = prepare_execution_profile(root, path, output)
    assert (output / "profile.json").read_bytes() == (root / "metadata/profile.json").read_bytes()
    assert (output / "systems/data/test/table.parquet").read_bytes() == (
        root / manifest["artifacts"][0]["path"]
    ).read_bytes()
    assert config["estimator_config"]["op_level"]["execution_profile"]["sha256"] == manifest["artifacts"][1]["sha256"]
    assert config["systems_paths"] == [str(output / "systems")]
    with pytest.raises(FileExistsError):
        prepare_execution_profile(root, path, output)


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
