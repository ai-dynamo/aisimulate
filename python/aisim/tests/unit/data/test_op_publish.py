# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Op-latency publish tree: faithful conversion, hash audit, resolvable reuse (AISIM-2340)."""

import hashlib
import importlib.util
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from aisim_core.data import check_dataset
from aisim_core.data.loaders import OpLatencyLoader
from aisim_core.data.op_schema import (
    DATA_PREFIX,
    DECLARED_REUSE_PATH,
    OP_DECLARED_REUSE_SCHEMA_ID,
    OP_QUERY_VERSIONS_SCHEMA_ID,
    OP_REUSE_MANIFEST_SCHEMA_ID,
    OP_TABLE_SCHEMA_ID,
    QUERY_VERSIONS_NAME,
    REUSE_MANIFEST_NAME,
    OpDeclaredReuseRow,
    OpQueryVersionsRow,
    OpReuseManifestRow,
    OpSourceMapRow,
    table_content_sha256,
)
from aisim_core.data.parquet import read_parquet, read_parquet_schema_id

pytestmark = pytest.mark.unit

# The converter is a tool script (tools/perf_database), not library code; load it by path.
_TOOL = Path(__file__).resolve().parents[3] / "tools" / "perf_database" / "op_latency_publish.py"
_spec = importlib.util.spec_from_file_location("op_latency_publish", _TOOL)
op = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = op  # dataclasses resolve annotations through sys.modules
_spec.loader.exec_module(op)
PublishError, audit_tree, build_publish_tree = op.PublishError, op.audit_tree, op.build_publish_tree

REUSE_MANIFEST = """# header comment
groups:
  - op_file: gemm_perf.parquet
    kernel_source: 'torch_flow'
    tier: shared
    systems: [h200_sxm]
    frameworks: [vllm, trtllm]
    total_raw_rows: 5
    within_framework_dedup_rows: 0
  - op_file: moe_perf.parquet
    kernel_source: 'x'
    tier: shared_fallback
    systems: [h200_sxm]
    frameworks: [vllm]
    total_raw_rows: 2
    within_framework_dedup_rows: 1
    median_pct_divergence: 3.5
"""
QUERY_VERSIONS = """schema_version: 1
defaults:
  vllm: {current: "1.0", previous: null}
overrides:
  h200_sxm:
    vllm: {current: "2.0", previous: "1.0"}
"""


def _table(rows=3):
    return pa.table({"b": list(range(rows)), "s": [f"k{i}" if i % 2 else None for i in range(rows)], "f": [0.5] * rows})


@pytest.fixture
def tree(tmp_path):
    systems = tmp_path / "systems"
    data = systems / "data"
    v1 = data / "h200_sxm/gemm/vllm/1.0"
    v2 = data / "h200_sxm/gemm/vllm/2.0"
    for d in (v1, v2):
        d.mkdir(parents=True)
    pq.write_table(_table(), v1 / "gemm_perf.parquet", compression="snappy")  # unstamped, not zstd
    (v1 / "collection_meta.yaml").write_text("schema_version: 1\nprovenance: collected\nruntime: {framework: vllm}\n")
    pq.write_table(_table(2), v2 / "other_perf.parquet")
    (v2 / "reuse.yaml").write_text(
        "schema_version: 1\nreuse:\n"
        "- table: gemm_perf\n  from_version: '1.0'\n  reason: same kernels\n  approved_by: me\n"
        "- table: ghost_perf\n  from_version: '0.9'\n  reason: missing donor\n  approved_by: me\n"
    )
    (data / "h200_sxm/README.md").write_text("hardware notes\n")
    (data / "l4").mkdir()
    (data / "l4/.gitkeep").write_bytes(b"")
    (systems / "perf_data_reuse_manifest.yaml").write_text(REUSE_MANIFEST)
    (systems / "query_versions.yaml").write_text(QUERY_VERSIONS)
    return systems


def _build(systems, out):
    return build_publish_tree(
        systems / "data", systems / "perf_data_reuse_manifest.yaml", systems / "query_versions.yaml", out
    )


def _audit(systems, out):
    return audit_tree(
        out,
        systems / "data",
        reuse_manifest=systems / "perf_data_reuse_manifest.yaml",
        query_versions=systems / "query_versions.yaml",
    )


def test_row_hash_ignores_compression_and_metadata(tmp_path):
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    pq.write_table(_table(), a, compression="snappy")
    pq.write_table(_table().replace_schema_metadata({"x": "y"}), b, compression="zstd", row_group_size=2)
    assert table_content_sha256(pq.read_table(a)) == table_content_sha256(pq.read_table(b))
    assert table_content_sha256(_table(3)) != table_content_sha256(_table(4))


def test_build_converts_copies_and_audits(tree, tmp_path):
    out = tmp_path / "out"
    result = _build(tree, out)
    assert (result.n_manifests, result.n_skipped) == (2, 1)
    assert result.declared_reuse == {"resolved": 1, "donor-missing": 1}

    published = out / "data/op-latency/h200_sxm/gemm/vllm/1.0/gemm_perf.parquet"
    assert read_parquet_schema_id(published) == OP_TABLE_SCHEMA_ID
    assert pq.ParquetFile(published).metadata.row_group(0).column(0).compression == "ZSTD"
    assert pq.read_table(published).replace_schema_metadata(None).equals(_table())
    for name in ("collection_meta.yaml",):  # sidecars are byte-identical
        src = tree / "data/h200_sxm/gemm/vllm/1.0" / name
        assert (published.parent / name).read_bytes() == src.read_bytes()
    assert (out / "data/op-latency/h200_sxm/README.md").read_bytes() == b"hardware notes\n"
    assert not (out / "data/op-latency/l4").exists()  # .gitkeep placeholders are recorded, not published

    report = _audit(tree, out)
    assert report.ok, report.to_dict()
    assert report.by_basis == {"file-bytes": 3, "arrow-rows": 2, "embedded-source": 2}
    check_dataset(out)


def test_yaml_metadata_round_trips_to_source_bytes(tree, tmp_path):
    out = tmp_path / "out"
    _build(tree, out)
    for name, src in (
        ("perf_data_reuse_manifest", tree / "perf_data_reuse_manifest.yaml"),
        ("query_versions", tree / "query_versions.yaml"),
    ):
        meta = pq.read_metadata(out / f"data/op-latency/{name}.parquet").metadata
        assert meta[b"aisim.source_yaml"] == src.read_bytes()
        assert meta[b"aisim.source_sha256"].decode() == hashlib.sha256(src.read_bytes()).hexdigest()
    groups = pq.read_table(out / "data/op-latency/perf_data_reuse_manifest.parquet").to_pylist()
    assert groups[0]["frameworks"] == ["vllm", "trtllm"] and groups[0]["median_pct_divergence"] is None
    assert yaml.safe_load(REUSE_MANIFEST)["groups"][1]["median_pct_divergence"] == groups[1]["median_pct_divergence"]


def test_reuse_resolvable_from_catalog_files_only(tree, tmp_path):
    out = tmp_path / "out"
    _build(tree, out)
    rows = read_parquet(out / DECLARED_REUSE_PATH, OP_DECLARED_REUSE_SCHEMA_ID).to_pylist()
    (row,) = (r for r in rows if r["version"] == "2.0" and r["table"] == "gemm_perf")
    assert row["status"] == "resolved"
    assert row["donor_table_path"] == "data/op-latency/h200_sxm/gemm/vllm/1.0/gemm_perf.parquet"
    assert row["donor_manifest"] == "data/op-latency/h200_sxm/gemm/vllm/1.0/manifest.json"
    assert [r["table"] for r in rows if r["status"] != "resolved"] == ["ghost_perf"]
    groups = read_parquet(out / DATA_PREFIX / f"{REUSE_MANIFEST_NAME}.parquet", OP_REUSE_MANIFEST_SCHEMA_ID)
    assert [g["kernel_source"] for g in groups.to_pylist() if g["op_file"] == "gemm_perf.parquet"] == ["torch_flow"]
    slots = read_parquet(out / DATA_PREFIX / f"{QUERY_VERSIONS_NAME}.parquet", OP_QUERY_VERSIONS_SCHEMA_ID)
    override = [r for r in slots.to_pylist() if r["scope"] == "overrides" and r["system"] == "h200_sxm"]
    assert [(r["current"], r["previous"]) for r in override] == [("2.0", "1.0")]


def test_loader_materializes_published_tree(tree, tmp_path):
    out = tmp_path / "out"
    _build(tree, out)
    manifest = "data/op-latency/h200_sxm/gemm/vllm/1.0/manifest.json"
    cfg = {"system": "h200_sxm", "backend": "vllm", "backend_version": "1.0"}
    loaded = OpLatencyLoader().load(out, manifest, output_dir=tmp_path / "mat", config=cfg)
    assert loaded.files == (
        "data/h200_sxm/gemm/vllm/1.0/collection_meta.yaml",
        "data/h200_sxm/gemm/vllm/1.0/gemm_perf.parquet",
    )
    table = pq.read_table(loaded.systems_root / "data/h200_sxm/gemm/vllm/1.0/gemm_perf.parquet")
    assert table.replace_schema_metadata(None).equals(_table())


@pytest.mark.parametrize("tamper", ["destination", "source", "stray", "unmapped-source"])
def test_audit_detects_divergence(tree, tmp_path, tamper):
    out = tmp_path / "out"
    _build(tree, out)
    if tamper == "destination":
        pq.write_table(_table(4), out / "data/op-latency/h200_sxm/gemm/vllm/1.0/gemm_perf.parquet")
    elif tamper == "source":
        pq.write_table(_table(5), tree / "data/h200_sxm/gemm/vllm/1.0/gemm_perf.parquet")
    elif tamper == "stray":
        (out / "data/op-latency/h200_sxm/extra.txt").write_text("x")
    else:
        (tree / "data/h200_sxm/new.txt").write_text("x")
    report = _audit(tree, out)
    assert not report.ok
    if tamper == "stray":
        assert report.unmapped_destinations == ["data/op-latency/h200_sxm/extra.txt"]
    if tamper == "unmapped-source":
        assert report.unmapped_sources == ["h200_sxm/new.txt"]


def test_build_refuses_existing_or_nested_output(tree, tmp_path):
    (tmp_path / "exists").mkdir()
    with pytest.raises(PublishError, match="already exists"):
        _build(tree, tmp_path / "exists")
    with pytest.raises(PublishError, match="inside the source"):
        _build(tree, tree / "data/nested")


def test_row_models_match_table_columns():
    doc = {
        "groups": [
            {
                "op_file": "f",
                "kernel_source": "k",
                "tier": "t",
                "systems": ["s"],
                "frameworks": ["x"],
                "total_raw_rows": 1,
                "within_framework_dedup_rows": 1,
            }
        ]
    }
    assert set(op.reuse_manifest_table(doc).column_names) == set(OpReuseManifestRow.model_fields)
    qv = {"schema_version": 1, "defaults": {"x": {"current": "1"}}, "overrides": {}}
    assert set(op.query_versions_table(qv).column_names) == set(OpQueryVersionsRow.model_fields)
    assert set(op._MAP_COLUMNS) == set(OpSourceMapRow.model_fields)
    assert set(op._declared_reuse_table([]).column_names) == set(OpDeclaredReuseRow.model_fields)
