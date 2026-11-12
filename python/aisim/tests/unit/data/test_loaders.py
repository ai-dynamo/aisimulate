# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loader reference tests: outputs equal the existing code paths for one configuration."""

import hashlib
import json
import shutil

import pyarrow as pa
import pytest

from aisim_core.data import Identity, configuration_id, prepare_execution_profile
from aisim_core.data.fetch import Selection, SelectionError
from aisim_core.data.loaders import (
    LOADERS,
    E2eLoader,
    FpmLoader,
    LoaderError,
    OpLatencyLoader,
)
from aisim_core.data.parquet import write_parquet
from aisim_core.sdk import RustForwardPassPerfModel
from tests.unit.sdk.test_dataset_contracts import artifact, dataset, write  # noqa: F401

pytestmark = pytest.mark.unit

CONFIG = {"system": "h200_sxm", "backend": "vllm", "backend_version": "1.0", "model": "meta-llama/Meta-Llama-3.1-8B"}


def _tree(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def _op_manifest(root, manifest, **identity_overrides):
    """Operation-tables manifest over the fixture's table + hardware spec."""
    identity = Identity(
        hardware="h200_sxm",
        framework="vllm",
        framework_version="1.0",
        operation_family=None,
        model=None,
        effective_settings={},
        unresolved_identity=None,
    ).model_copy(update=identity_overrides)
    doc = dict(
        manifest,
        purpose="operation-tables",
        configuration_id=configuration_id("op-latency", "operation-tables", identity),
        identity=identity.model_dump(),
        artifacts=[a for a in manifest["artifacts"] if a.get("runtime_path")],
    )
    write(root, "metadata/op.json", doc)
    return "metadata/op.json"


# ------------------------------------------------------------------ select


def test_select_builds_catalog_selections():
    op = OpLatencyLoader().select(CONFIG)
    assert op == Selection(
        category="op-latency",
        hardware="h200_sxm",
        framework="vllm",
        framework_version="1.0",
        purpose="operation-tables",
    )  # model-independent
    fpm = FpmLoader().select(CONFIG)
    assert fpm.model == CONFIG["model"] and fpm.category == "fpm"
    assert E2eLoader().select(CONFIG).category == "e2e"
    with pytest.raises(SelectionError):
        FpmLoader().select({"system": "h200_sxm"})  # fpm needs a model
    assert set(LOADERS) == {"op-latency", "fpm", "e2e"}


# ------------------------------------------------------------------ op-latency


def test_op_latency_reproduces_systems_root(dataset, tmp_path):  # noqa: F811
    root, _, manifest = dataset
    path = _op_manifest(root, manifest)
    out = OpLatencyLoader().load(root, path, output_dir=tmp_path / "out", config=CONFIG)
    # Reference: the systems root the existing execution-profile preparation produces.
    prepared = prepare_execution_profile(root, "metadata/manifest.json", tmp_path / "prepared")
    assert _tree(out.systems_root) == _tree(tmp_path / "prepared" / "systems")
    assert out.files == tuple(sorted(a["runtime_path"] for a in manifest["artifacts"] if a.get("runtime_path")))
    assert prepared["systems_paths"] == [str(tmp_path / "prepared" / "systems")]
    # And it is discoverable like any systems root.
    from aisim_core.sdk import perf_database as pdb

    assert [r[:3] for r in pdb._discover_database_refs([str(out.systems_root)])] == [
        r[:3] for r in pdb._discover_database_refs([str(tmp_path / "prepared" / "systems")])
    ]


def test_op_latency_rejections(dataset, tmp_path):  # noqa: F811
    root, _, manifest = dataset
    path = _op_manifest(root, manifest)
    with pytest.raises(LoaderError, match="differs from configuration"):
        OpLatencyLoader().load(root, path, output_dir=tmp_path / "a", config={**CONFIG, "backend_version": "2"})
    assert not (tmp_path / "a").exists()
    with pytest.raises(LoaderError, match="model-independent"):
        OpLatencyLoader().load(root, _op_manifest(root, manifest, model="m"), output_dir=tmp_path / "b")
    with pytest.raises(LoaderError, match="cannot load purpose"):
        OpLatencyLoader().load(root, "metadata/manifest.json", output_dir=tmp_path / "c")
    table = root / manifest["artifacts"][0]["path"]
    table.write_bytes(table.read_bytes() + b"x")
    with pytest.raises(ValueError, match="integrity"):
        OpLatencyLoader().load(root, path, output_dir=tmp_path / "d")
    assert not (tmp_path / "d").exists()


def test_op_latency_refuses_existing_output(dataset, tmp_path):  # noqa: F811
    root, _, manifest = dataset
    path = _op_manifest(root, manifest)
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        OpLatencyLoader().load(root, path, output_dir=tmp_path / "out")


# ------------------------------------------------------------------ fpm


def test_fpm_loader_matches_prepare_execution_profile_and_predicts(dataset, tmp_path):  # noqa: F811
    root, path, _ = dataset
    shutil.rmtree(root.parent / "producer-systems")
    loaded = FpmLoader().load(root, path, output_dir=tmp_path / "fpm", config=CONFIG)
    reference = prepare_execution_profile(root, path, tmp_path / "ref")
    strip = lambda c: {k: v for k, v in c.items() if k not in ("systems_paths",) and k != "estimator_config"}
    assert strip(loaded.config) == strip(reference)
    assert _tree(tmp_path / "fpm" / "systems") == _tree(tmp_path / "ref" / "systems")
    assert (tmp_path / "fpm" / "profile.json").read_bytes() == (tmp_path / "ref" / "profile.json").read_bytes()
    a = RustForwardPassPerfModel.best_available(loaded.config)
    b = RustForwardPassPerfModel.best_available(reference)
    assert a.predict_prefill_latency(1, 4, 0) == b.predict_prefill_latency(1, 4, 0) == 6.0
    assert a.predict_prefill_latency(1, 8, 4) == b.predict_prefill_latency(1, 8, 4) == 15.0


def test_fpm_loader_requires_output_dir_and_checks_identity(dataset, tmp_path):  # noqa: F811
    root, path, _ = dataset
    with pytest.raises(LoaderError, match="output_dir"):
        FpmLoader().load(root, path)
    with pytest.raises(LoaderError, match="differs from configuration"):
        FpmLoader().load(root, path, output_dir=tmp_path / "x", config={**CONFIG, "system": "b200_sxm"})


def test_fpm_loader_routes_fpm_profile_artifact_through_load_fpm_profile(dataset, monkeypatch):  # noqa: F811
    # Wiring test: building a valid FpmModelProfile needs a probe-sized fixture, and
    # load_fpm_profile itself is covered elsewhere. Check the verified bytes reach it.
    import aisim_core.fpm_profile as fpm_profile

    seen = []
    monkeypatch.setattr(fpm_profile, "load_fpm_profile", lambda text: seen.append(text) or "PROFILE")
    root, _, manifest = dataset
    (root / "metadata/fpm.json").write_text('{"schema_version": 1}')
    art = artifact(root, "metadata/fpm.json", "fpm-profile")
    write(root, "metadata/fpm-manifest.json", dict(manifest, artifacts=[art]))
    loaded = FpmLoader().load(root, "metadata/fpm-manifest.json")
    assert loaded.profile == "PROFILE" and seen == ['{"schema_version": 1}'] and loaded.config is None
    (root / "metadata/fpm.json").write_text('{"schema_version": 2}')
    with pytest.raises(ValueError, match="integrity"):
        FpmLoader().load(root, "metadata/fpm-manifest.json")


# ------------------------------------------------------------------ e2e


def _e2e_dataset(tmp_path, schema="aisim.test-e2e/v1"):
    root = tmp_path / "e2e"
    table = pa.table({"request": [1, 2, 3], "ttft_ms": [10.0, 11.0, 12.5]})
    path = "e2e/m/requests.parquet"
    (root / path).parent.mkdir(parents=True)
    write_parquet(table, root / path, schema)
    data = (root / path).read_bytes()
    identity = Identity(
        hardware="h200_sxm",
        framework="vllm",
        framework_version="1.0",
        operation_family=None,
        model="m",
        effective_settings={},
        unresolved_identity=None,
    )
    art = {
        "artifact_id": "requests",
        "path": path,
        "role": "e2e-table",
        "format": "parquet",
        "schema": schema,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "rows": 3,
    }
    doc = {
        "schema_version": "aisim-artifact-manifest/v1",
        "category": "e2e",
        "purpose": "e2e-configuration",
        "configuration_id": configuration_id("e2e", "e2e-configuration", identity),
        "identity": identity.model_dump(),
        "requested_settings": None,
        "artifacts": [art],
        "provenance": {},
        "evidence_gaps": [],
    }
    (root / "e2e/m/manifest.json").write_text(json.dumps(doc))
    return root, table


def test_e2e_returns_table_views_equal_to_source(tmp_path):
    root, table = _e2e_dataset(tmp_path)
    cfg = {"system": "h200_sxm", "backend": "vllm", "backend_version": "1.0", "model": "m"}
    loaded = E2eLoader().load(root, "e2e/m/manifest.json", config=cfg)
    assert loaded["requests"].equals(table)
    projected = E2eLoader().load(root, "e2e/m/manifest.json", columns={"requests": ["ttft_ms"]})
    assert projected["requests"].column_names == ["ttft_ms"]


def test_e2e_rejects_unaccepted_schema_and_wrong_model(tmp_path):
    root, _ = _e2e_dataset(tmp_path)
    with pytest.raises(LoaderError, match="unsupported table schema"):
        E2eLoader(accepted_schemas=("aisim.other/v1",)).load(root, "e2e/m/manifest.json")
    with pytest.raises(LoaderError, match="manifest model"):
        E2eLoader().load(
            root,
            "e2e/m/manifest.json",
            config={"system": "h200_sxm", "backend": "vllm", "backend_version": "1.0", "model": "other"},
        )


def test_e2e_rejects_corrupt_and_path_escape(tmp_path):
    root, _ = _e2e_dataset(tmp_path)
    with pytest.raises(ValueError, match="canonical"):
        E2eLoader().load(root, "../manifest.json")
    p = root / "e2e/m/requests.parquet"
    p.write_bytes(p.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="integrity"):
        E2eLoader().load(root, "e2e/m/manifest.json")
