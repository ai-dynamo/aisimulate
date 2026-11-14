# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil

import pytest

from aisim_core.data.systems import (
    PACKAGED_SYSTEMS_DIR,
    SystemSpecError,
    check_published_systems,
    copy_filename,
    packaged_system_specs,
    publish_system_spec,
    read_manifest,
    spec_file_sha256,
    spec_sha256,
)

pytestmark = pytest.mark.unit

SPEC = "# comment\ngpu:\n  mem_bw: 1\n  sm_version: 100\nnode:\n  num_gpus_per_node: 8\n"


def _packaged(tmp_path, text=SPEC, name="x1"):
    d = tmp_path / "pkg"
    d.mkdir(exist_ok=True)
    (d / f"{name}.yaml").write_text(text)
    return d


def test_hash_ignores_comments_and_key_order():
    a = {"gpu": {"a": 1, "b": 2}, "node": {}}
    b = {"node": {}, "gpu": {"b": 2, "a": 1}}
    assert spec_sha256(a) == spec_sha256(b)
    assert spec_sha256(a) != spec_sha256({"gpu": {"a": 1, "b": 3}, "node": {}})


def test_publish_names_copy_by_hash_and_is_idempotent(tmp_path):
    pkg = _packaged(tmp_path)
    root = tmp_path / "data"
    copy = publish_system_spec(pkg / "x1.yaml", root)
    digest = spec_file_sha256(pkg / "x1.yaml")
    assert copy.name == copy_filename("x1", digest) == f"x1.{digest[:12]}.yaml"
    assert "#" not in copy.read_text()
    assert spec_file_sha256(copy) == digest
    publish_system_spec(pkg / "x1.yaml", root)
    assert [h.spec_sha256 for h in read_manifest(copy.parent / "manifest.yaml").history] == [digest]
    assert check_published_systems(root, pkg) == []


def test_drift_fails_and_republish_keeps_history(tmp_path):
    pkg = _packaged(tmp_path)
    root = tmp_path / "data"
    publish_system_spec(pkg / "x1.yaml", root)
    old = spec_file_sha256(pkg / "x1.yaml")
    _packaged(tmp_path, SPEC.replace("mem_bw: 1", "mem_bw: 2"))
    problems = check_published_systems(root, pkg)
    assert len(problems) == 1 and "!= packaged" in problems[0]
    publish_system_spec(pkg / "x1.yaml", root, note="recalibrated")
    manifest = read_manifest(root / "systems/x1/manifest.yaml")
    assert [h.spec_sha256 for h in manifest.history][0] == old
    assert manifest.current == spec_file_sha256(pkg / "x1.yaml")
    assert check_published_systems(root, pkg) == []


def test_tampered_copy_and_missing_manifest_detected(tmp_path):
    pkg = _packaged(tmp_path)
    root = tmp_path / "data"
    copy = publish_system_spec(pkg / "x1.yaml", root)
    copy.write_text(copy.read_text().replace("100", "101"))
    assert any("does not hash" in p for p in check_published_systems(root, pkg))
    shutil.rmtree(root / "systems/x1")
    assert any("missing" in p for p in check_published_systems(root, pkg))


def test_calibrated_field_requires_evidence_and_known_path(tmp_path):
    pkg = _packaged(tmp_path)
    root = tmp_path / "data"
    with pytest.raises(ValueError, match="evidence"):
        publish_system_spec(pkg / "x1.yaml", root, fields={"gpu.mem_bw": {"source": "calibrated"}})
    with pytest.raises(SystemSpecError, match="not present"):
        publish_system_spec(pkg / "x1.yaml", root, fields={"gpu.nope": {"source": "vendor_spec"}})
    publish_system_spec(pkg / "x1.yaml", root, fields={"gpu.mem_bw": {"source": "calibrated", "evidence": "eval-123"}})
    assert read_manifest(root / "systems/x1/manifest.yaml").fields["gpu.mem_bw"].evidence == "eval-123"


def test_private_system_not_packaged_and_declared(tmp_path):
    pkg = _packaged(tmp_path)
    root = tmp_path / "data"
    publish_system_spec(pkg / "x1.yaml", root)
    (root / "systems/secret").mkdir()
    assert any("secret" in p for p in check_published_systems(root, pkg))
    assert check_published_systems(root, pkg, private_systems=frozenset({"secret"})) == []
    assert any(
        "must not be packaged" in p for p in check_published_systems(root, pkg, private_systems=frozenset({"x1"}))
    )


def test_every_packaged_system_publishes_and_checks_clean(tmp_path):
    specs = packaged_system_specs(PACKAGED_SYSTEMS_DIR)
    assert {"b200_sxm", "h200_sxm"} <= set(specs)
    assert "query_versions" not in specs
    for path in specs.values():
        publish_system_spec(path, tmp_path)
    assert check_published_systems(tmp_path) == []
