# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AISIM-2337: byte-identical move of the FPM dataset into data/fpm/ with envelopes and a catalog."""

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aisim_core.data import check_dataset, check_manifest
from aisim_core.data import fpm_migration as fm
from aisim_core.data.catalog import Catalog
from aisim_core.data.fetch import IntegrityError, PlannedFile, _verify_file

pytestmark = pytest.mark.unit

REVISION = "a" * 40


def _leaf_manifest(directory: str, *, tp: int, status: str = "current", snapshot: str = "aisim-commit-unknown-1"):
    return {
        "aisim_commit": None,
        "aisim_commit_status": "unknown",
        "configuration_path": directory,
        "cp": 1,
        "dp": 1,
        "evidence_date": "2026-09-18",
        "fpm": [],
        "framework": "sglang",
        "framework_version": "1.0",
        "gpu_family": "B300_SXM",
        "kv_cache_dtype": "fp8",
        "manifest_version": 4,
        "measurements": {},
        "model_id": "org/Model",
        "model_revision": None,
        "moe_ep": 1,
        "moe_tp": tp,
        "parallel_strategy": "pure_tp",
        "parallelism": f"pure-tp{tp}",
        "pp": 1,
        "provenance": {"producer_identity_missing": True},
        "snapshot_id": snapshot,
        "snapshot_status": status,
        "system": "b300_sxm",
        "tp": tp,
        "weight_quantization": "fp8_block",
    }


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _parquet(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"x": list(range(rows))}), path)  # no aisim.schema metadata, like real FPM files


def _tree(root: Path) -> list[dict]:
    """A legacy-layout repository: one current leaf, one historical leaf, a legacy catalog, .gitattributes."""
    records = []
    for index, (status, base) in enumerate(
        [
            ("current", "data/org--Model/b300-sxm/sglang/1.0/pure-tp2"),
            ("historical", "data/history/org--Model/b300-sxm/sglang/1.0/pure-tp2/aisim-commit-unknown-9"),
        ]
    ):
        manifest = _leaf_manifest(
            base if status == "current" else "data/org--Model/b300-sxm/sglang/1.0/pure-tp2",
            tp=2,
            status=status,
            snapshot=f"aisim-commit-unknown-{index}",
        )
        _write(root / base / "manifest.json", json.dumps(manifest).encode())
        _parquet(root / base / "fpm" / "fpm_forward_perf.parquet", 5 + index)
        _write(root / base / "fpm" / "fpm_forward_perf.metadata.json", b'{"path": "' + base.encode() + b'"}')
        _write(root / base / "fpm" / "provenance" / "LICENSE", b"license")
        _write(root / base / "measurements" / "manifest.json", json.dumps({"manifest_version": 5}).encode())
        _parquet(root / base / "measurements" / "iterations-01.parquet", 3)
        records.append(
            {"path": f"{base}/fpm/fpm_forward_perf.parquet", "schema_name": "aic_fpm_forward_perf", "schema_version": 7}
        )
    _write(root / "catalog" / "fpm.json", json.dumps({"schema_version": 1, "records": records}).encode())
    _write(
        root / "catalog" / "index.json",
        json.dumps(
            {
                "configuration_manifests": ["data/org--Model/b300-sxm/sglang/1.0/pure-tp2/manifest.json"],
                "history_manifests": [
                    "data/history/org--Model/b300-sxm/sglang/1.0/pure-tp2/aisim-commit-unknown-9/manifest.json"
                ],
            }
        ).encode(),
    )
    _write(root / "README.md", b"readme")
    _write(root / ".gitattributes", b"*.parquet filter=lfs\ndata/org--Model/*/x.json filter=lfs\n")
    return records


@pytest.fixture
def migrated(tmp_path):
    source = tmp_path / "src"
    _tree(source)
    output = tmp_path / "out"
    result = fm.migrate(source, output, revision=REVISION, repo="org/repo")
    return source, output, result


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_target_path_is_a_pure_prefix_rule():
    assert fm.target_path("data/a/b") == "data/fpm/a/b"
    assert fm.target_path("data/history/a") == "data/fpm/history/a"
    with pytest.raises(fm.MigrationError):
        fm.target_path("catalog/x")
    with pytest.raises(fm.MigrationError):
        fm.target_path("data/fpm/a")


def test_every_data_file_moves_with_identical_bytes(migrated):
    source, output, result = migrated
    for relative in fm.list_files(source):
        if relative.startswith("data/"):
            assert _digest(source / relative) == _digest(output / fm.target_path(relative))
            assert not (output / relative).exists()
        elif relative != ".gitattributes":
            assert _digest(source / relative) == _digest(output / relative)
    assert result.moved == 12 and result.leaves == 2 and result.indexed == 1 and result.history == 1


def test_embedded_legacy_paths_are_not_rewritten(migrated):
    source, output, _ = migrated
    legacy = "data/org--Model/b300-sxm/sglang/1.0/pure-tp2/fpm/fpm_forward_perf.metadata.json"
    assert b"data/org--Model" in (output / fm.target_path(legacy)).read_bytes()


def test_gitattributes_follow_the_move_and_nothing_else():
    text = "*.parquet filter=lfs\ndata/x/*.json filter=lfs\n# data/comment\n"
    assert fm.rewrite_gitattributes(text) == "*.parquet filter=lfs\ndata/fpm/x/*.json filter=lfs\n# data/comment\n"


def test_envelopes_validate_and_declare_every_file(migrated):
    _, output, _ = migrated
    path = "data/fpm/org--Model/b300-sxm/sglang/1.0/pure-tp2/aisim-manifest.json"
    manifest = check_manifest(output, path)
    assert manifest.category == "fpm" and manifest.purpose == "execution-profile"
    assert manifest.identity.parallelism == {"aggregated": {"moe_tp": 2, "tp": 2}}
    assert manifest.identity.parallelism_complete is True
    assert manifest.identity.precision == {"weights": "fp8_block", "kv_cache": "fp8"}
    assert manifest.identity.deployment is None  # unknown stays null, with a recorded gap
    assert {g["field"] for g in manifest.evidence_gaps} >= {"deployment", "model_revision"}
    assert len(manifest.artifacts) == 6
    library = next(a for a in manifest.artifacts if a.role == "fpm-library")
    assert library.rows == 5 and library.artifact_schema == "aic_fpm_forward_perf/v7"
    assert manifest.provenance["path_mapping"] == {"from": "data/", "to": "data/fpm/"}


def test_dataset_index_catalog_and_history_policy(migrated):
    _, output, _ = migrated
    index = check_dataset(output)
    assert index.categories["fpm"].root == "data/fpm"
    rows = Catalog(output).rows(layer="fpm", hardware="b300_sxm", status="have")
    assert [(r.model, r.framework, r.n_configs) for r in rows] == [("org/Model", "sglang", 1)]
    # the historical snapshot has an envelope but is not in the default index
    assert (
        output / "data/fpm/history/org--Model/b300-sxm/sglang/1.0/pure-tp2/aisim-commit-unknown-9/aisim-manifest.json"
    ).is_file()


def test_index_history_option_lists_both_with_distinct_ids(tmp_path):
    source = tmp_path / "src"
    _tree(source)
    fm.migrate(source, tmp_path / "out", revision=REVISION, index_history=True)
    assert Catalog(tmp_path / "out").rows(layer="fpm", hardware="b300_sxm", status="have")[0].n_configs == 2


def test_migrate_is_deterministic(tmp_path):
    source = tmp_path / "src"
    _tree(source)
    fm.migrate(source, tmp_path / "a", revision=REVISION)
    fm.migrate(source, tmp_path / "b", revision=REVISION)
    for relative in fm.list_files(tmp_path / "a"):
        if not relative.endswith(".parquet") or relative.startswith("data/"):
            assert _digest(tmp_path / "a" / relative) == _digest(tmp_path / "b" / relative)


def test_audit_passes_and_catches_tampering(migrated):
    source, output, _ = migrated
    report = fm.audit(source, output, revision=REVISION)
    assert report["passed"], report["failures"]
    assert report["data_files_moved"] == report["data_files_hash_unchanged"] == 12
    assert report["catalog_configurations"] == report["coverage_catalog_configurations"] == 1
    assert report["other_files_modified"] == [".gitattributes"]
    assert report["moved_files_mentioning_legacy_data_paths"] == 4  # 2 leaf manifests + 2 metadata files

    victim = output / "data/fpm/org--Model/b300-sxm/sglang/1.0/pure-tp2/fpm/provenance/LICENSE"
    victim.write_bytes(b"licensE")
    tampered = fm.audit(source, output, revision=REVISION)
    assert not tampered["passed"]
    assert any("BYTES DIFFER" in f for f in tampered["failures"])


def test_audit_catches_a_missing_file_and_an_unindexed_legacy_configuration(migrated):
    source, output, _ = migrated
    (output / "data/fpm/org--Model/b300-sxm/sglang/1.0/pure-tp2/aisim-manifest.json").unlink()
    report = fm.audit(source, output, revision=REVISION)
    assert not report["passed"]


def test_orphan_data_file_is_refused(tmp_path):
    source = tmp_path / "src"
    _tree(source)
    _write(source / "data" / "stray.txt", b"x")
    with pytest.raises(fm.MigrationError, match="not in exactly one leaf"):
        fm.migrate(source, tmp_path / "out", revision=REVISION)
    assert not (tmp_path / "out").exists()


def test_identity_collision_is_refused(tmp_path):
    source = tmp_path / "src"
    _tree(source)
    # a second current leaf with an identical effective identity
    base = "data/org--Model/b300-sxm/sglang/1.0/pure-tp2-copy"
    _write(
        source / base / "manifest.json",
        (source / "data/org--Model/b300-sxm/sglang/1.0/pure-tp2/manifest.json").read_bytes(),
    )
    _parquet(source / base / "fpm" / "fpm_forward_perf.parquet", 1)
    records = json.loads((source / "catalog/fpm.json").read_text())
    records["records"].append({"path": f"{base}/fpm/fpm_forward_perf.parquet", "schema_name": "n", "schema_version": 1})
    (source / "catalog/fpm.json").write_text(json.dumps(records))
    with pytest.raises(fm.MigrationError, match="identity collision"):
        fm.migrate(source, tmp_path / "out", revision=REVISION)


def test_existing_output_is_refused(tmp_path):
    source = tmp_path / "src"
    _tree(source)
    (tmp_path / "out").mkdir()
    with pytest.raises(fm.MigrationError, match="already exists"):
        fm.migrate(source, tmp_path / "out", revision=REVISION)


def test_listing_verification_detects_changed_bytes(tmp_path):
    source = tmp_path / "src"
    _tree(source)
    listing = {
        relative: {"size": (source / relative).stat().st_size, "lfs": None, "blob": fm.git_blob_id(source / relative)}
        for relative in fm.list_files(source)
    }
    assert fm.verify_source(source, listing) == []
    (source / "README.md").write_bytes(b"README")
    assert any("README.md" in p for p in fm.verify_source(source, listing))


def test_upload_plan_copies_lfs_adds_the_rest_and_deletes_old_paths(migrated):
    source, output, _ = migrated
    lfs = {p for p in fm.list_files(source) if p.endswith("/fpm/fpm_forward_perf.parquet")}
    listing = {
        relative: {
            "size": (source / relative).stat().st_size,
            "lfs": _digest(source / relative) if relative in lfs else None,
            "blob": fm.git_blob_id(source / relative),
        }
        for relative in fm.list_files(source)
    }
    plan = fm.plan_upload(listing, output, source)
    ops = plan["operations"]
    assert {o["dst"] for o in ops if o["op"] == "copy"} == {fm.target_path(p) for p in lfs}
    assert {o["src"] for o in ops if o["op"] == "delete"} == {p for p in listing if p.startswith("data/")}
    added = {o["dst"] for o in ops if o["op"] == "add"}
    assert ".gitattributes" in added and "catalog/dataset.json" in added and "README.md" not in added
    assert all(o["dst"].startswith("data/fpm/") or o["op"] == "delete" or not o["dst"].startswith("data/") for o in ops)


def test_unstamped_external_schema_parquet_is_accepted_but_aisim_schema_is_not(tmp_path):
    path = tmp_path / "f.parquet"
    _parquet(path, 3)
    digest = _digest(path)
    base = dict(
        path="f.parquet",
        bytes=path.stat().st_size,
        sha256=digest,
        role="artifact",
        cached=False,
        format="parquet",
        rows=3,
    )
    _verify_file(path, "f.parquet", PlannedFile(schema="aic_fpm_forward_perf/v7", **base))
    with pytest.raises(IntegrityError, match="schema ID"):
        _verify_file(path, "f.parquet", PlannedFile(schema="aisim-catalog/v1", **base))


def test_upload_plan_table_round_trips(migrated, tmp_path):
    source, output, _ = migrated
    listing = {
        r: {"size": (source / r).stat().st_size, "lfs": None, "blob": fm.git_blob_id(source / r)}
        for r in fm.list_files(source)
    }
    plan = fm.plan_upload(listing, output, source)
    parquet, summary = fm.write_upload_plan(plan, tmp_path / "plan")
    assert pq.read_table(parquet).num_rows == len(plan["operations"])
    assert json.loads(summary.read_text())["counts"] == plan["counts"]
