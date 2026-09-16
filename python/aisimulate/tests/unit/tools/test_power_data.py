# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytestmark = pytest.mark.unit

POWER_DATA = Path(__file__).resolve().parents[3] / "tools" / "perf_database" / "power_data.py"
SHIPPED_DATA_ROOT = Path(__file__).resolve().parents[3] / "src" / "aiconfigurator_core" / "systems" / "data"


@pytest.fixture
def power_data_module():
    spec = importlib.util.spec_from_file_location("power_data", POWER_DATA)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_power_table(path: Path, *, shapes: tuple[int, int] = (1, 2)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "shape": list(shapes),
                "latency": [1.0, 2.0],
                "power": pa.array([500.0, 0.0], type=pa.float64()),
                "power_limit": pa.array([1000.0, 0.0], type=pa.float64()),
            }
        ),
        path,
    )


def _write_manifest(root: Path, table: Path, *, upstream_rows: int = 2) -> Path:
    digest = _sha256(table)
    packaged_rows = pq.read_metadata(table).num_rows
    retained_rows = packaged_rows - upstream_rows
    import_mode = "identity-merge" if retained_rows else "exact-copy"
    anomalies = []
    if retained_rows:
        anomalies.append(
            {
                "kind": "aisimulate-only-identities",
                "rows": retained_rows,
                "treatment": "paired-zero-sentinel",
                "tables": [{"path": table.relative_to(root).as_posix(), "rows": retained_rows}],
            }
        )
    shape_values = pq.read_table(table, columns=["shape"]).column("shape").to_pylist()
    manifest = {
        "schema_version": 1,
        "source": {
            "repository": "https://example.test/upstream",
            "commit": "a" * 40,
            "license": "Apache-2.0",
        },
        "dataset": {
            "system": root.name,
            "backend": "trtllm",
            "version": "1.0.0",
            "units": {"latency": "ms", "power": "W", "power_limit": "W"},
            "unavailable_sentinel": {"power": 0.0, "power_limit": 0.0},
            "anomalies": anomalies,
        },
        "tables": [
            {
                "path": table.relative_to(root).as_posix(),
                "identity_columns": ["shape"],
                "shape_bounds": {"shape": {"min": min(shape_values), "max": max(shape_values)}},
                "import_mode": import_mode,
                "upstream_sha256": digest,
                "packaged_sha256": digest,
                "upstream_rows": upstream_rows,
                "packaged_rows": packaged_rows,
                "measured_rows": 1,
                "zero_sentinel_rows": 1,
            }
        ],
        "totals": {
            "upstream_rows": upstream_rows,
            "packaged_rows": packaged_rows,
            "measured_rows": 1,
            "zero_sentinel_rows": 1,
        },
    }
    if retained_rows:
        upstream_path = root / "power_upstream" / "source.parquet"
        upstream_path.parent.mkdir(exist_ok=True)
        pq.write_table(pq.read_table(table).slice(0, upstream_rows), upstream_path)
        manifest["tables"][0]["upstream_evidence_path"] = upstream_path.relative_to(root).as_posix()
        manifest["tables"][0]["upstream_sha256"] = _sha256(upstream_path)
    path = root / "power_data_provenance.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_manifest_validates_hash_counts_and_power_contract(power_data_module, tmp_path):
    table_path = tmp_path / "gemm" / "trtllm" / "1.0.0" / "gemm_perf.parquet"
    _write_power_table(table_path)
    manifest = _write_manifest(tmp_path, table_path)

    assert power_data_module.validate_manifest(manifest) == []
    assert power_data_module.main([str(manifest)]) == 0


def test_manifest_accepts_positive_pairs_without_ratio_threshold(power_data_module, tmp_path):
    table_path = tmp_path / "gemm" / "trtllm" / "1.0.0" / "gemm_perf.parquet"
    _write_power_table(table_path)
    table = pq.read_table(table_path)
    table = table.set_column(table.schema.get_field_index("power"), "power", pa.array([1060.0, 0.0]))
    pq.write_table(table, table_path)
    manifest = _write_manifest(tmp_path, table_path)

    # Import integrity verifies the recorded measurements without inventing
    # a hardware-quality threshold for their ratio to the reported limit.
    assert power_data_module.validate_manifest(manifest) == []
    assert power_data_module.main([str(manifest)]) == 0


def test_manifest_rejects_checksum_drift(power_data_module, tmp_path):
    table_path = tmp_path / "gemm" / "trtllm" / "1.0.0" / "gemm_perf.parquet"
    _write_power_table(table_path)
    manifest = _write_manifest(tmp_path, table_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["tables"][0]["packaged_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    assert any("packaged_sha256 mismatch" in issue for issue in power_data_module.validate_manifest(manifest))


def test_manifest_requires_source_repository(power_data_module, tmp_path):
    table_path = tmp_path / "gemm" / "trtllm" / "1.0.0" / "gemm_perf.parquet"
    _write_power_table(table_path)
    manifest = _write_manifest(tmp_path, table_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    del payload["source"]["repository"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    assert any(
        "source.repository must be a non-empty string" in issue
        for issue in power_data_module.validate_manifest(manifest)
    )


def test_manifest_rejects_identity_evidence_drift(power_data_module, tmp_path):
    table_path = tmp_path / "gemm" / "trtllm" / "1.0.0" / "gemm_perf.parquet"
    _write_power_table(table_path)
    manifest = _write_manifest(tmp_path, table_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["tables"][0]["shape_bounds"]["shape"]["max"] = 3
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    assert any("shape_bounds do not match" in issue for issue in power_data_module.validate_manifest(manifest))


def test_identity_merge_manifest_validates_retained_anomaly(power_data_module, tmp_path):
    table_path = tmp_path / "gemm" / "trtllm" / "1.0.0" / "gemm_perf.parquet"
    _write_power_table(table_path)
    manifest = _write_manifest(tmp_path, table_path, upstream_rows=1)

    assert power_data_module.validate_manifest(manifest) == []


@pytest.mark.parametrize("column, dtype", [("shape", pa.float64()), ("shape", pa.int32()), ("latency", pa.float32())])
def test_identity_merge_rejects_schema_drift_with_equal_values(power_data_module, tmp_path, column, dtype):
    table_path = tmp_path / "gemm" / "trtllm" / "1.0.0" / "gemm_perf.parquet"
    _write_power_table(table_path)
    manifest = _write_manifest(tmp_path, table_path, upstream_rows=1)
    payload = json.loads(manifest.read_text())
    table = pq.read_table(table_path)
    index = table.schema.get_field_index(column)
    table = table.set_column(index, column, table.column(column).cast(dtype))
    pq.write_table(table, table_path)
    payload["tables"][0]["packaged_sha256"] = _sha256(table_path)
    manifest.write_text(json.dumps(payload))

    assert any("upstream evidence schema" in issue for issue in power_data_module.validate_manifest(manifest))


def test_identity_merge_accepts_metadata_only_differences(power_data_module, tmp_path):
    table_path = tmp_path / "gemm" / "trtllm" / "1.0.0" / "gemm_perf.parquet"
    _write_power_table(table_path)
    manifest = _write_manifest(tmp_path, table_path, upstream_rows=1)
    payload = json.loads(manifest.read_text())
    table = pq.read_table(table_path).replace_schema_metadata({b"producer": b"local"})
    field = table.schema.field("shape").with_metadata({b"description": b"local annotation"})
    table = table.set_column(0, field, table.column("shape"))
    pq.write_table(table, table_path)
    payload["tables"][0]["packaged_sha256"] = _sha256(table_path)
    manifest.write_text(json.dumps(payload))

    assert power_data_module.validate_manifest(manifest) == []


def test_manifest_accepts_legacy_table_layout(power_data_module, tmp_path):
    table_path = tmp_path / "trtllm" / "1.0.0" / "gemm_perf.parquet"
    _write_power_table(table_path)
    manifest = _write_manifest(tmp_path, table_path)

    assert power_data_module.validate_manifest(manifest) == []


def test_manifest_accepts_explicit_legacy_power_table(power_data_module, tmp_path):
    table_path = tmp_path / "gemm" / "trtllm" / "1.0.0" / "gemm_perf.parquet"
    legacy_path = tmp_path / "attention" / "vllm" / "0.22.0" / "context_attention_perf.parquet"
    _write_power_table(table_path)
    _write_power_table(legacy_path)
    manifest = _write_manifest(tmp_path, table_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["legacy_power_tables"] = [legacy_path.relative_to(tmp_path).as_posix()]
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    assert power_data_module.validate_manifest(manifest) == []


def test_manifest_rejects_undeclared_power_table_in_another_version(power_data_module, tmp_path):
    table_path = tmp_path / "gemm" / "trtllm" / "1.0.0" / "gemm_perf.parquet"
    second_version = tmp_path / "gemm" / "trtllm" / "2.0.0" / "gemm_perf.parquet"
    _write_power_table(table_path)
    _write_power_table(second_version)
    manifest = _write_manifest(tmp_path, table_path)

    assert any(
        "manifest omits packaged power tables" in issue and "trtllm/2.0.0" in issue
        for issue in power_data_module.validate_manifest(manifest)
    )


def test_manifest_rejects_non_object_root(power_data_module, tmp_path):
    manifest = tmp_path / "power_data_provenance.json"
    manifest.write_text("[]", encoding="utf-8")

    assert power_data_module.validate_manifest(manifest) == [f"{manifest}: manifest root must be an object"]


def test_manifest_rejects_non_string_anomaly_path(power_data_module, tmp_path):
    table_path = tmp_path / "gemm" / "trtllm" / "1.0.0" / "gemm_perf.parquet"
    _write_power_table(table_path)
    manifest = _write_manifest(tmp_path, table_path, upstream_rows=1)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["dataset"]["anomalies"][0]["tables"][0]["path"] = []
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    assert any("path must be a string" in issue for issue in power_data_module.validate_manifest(manifest))


def test_manifest_rejects_duplicate_table_identities(power_data_module, tmp_path):
    table_path = tmp_path / "gemm" / "trtllm" / "1.0.0" / "gemm_perf.parquet"
    _write_power_table(table_path, shapes=(1, 1))
    manifest = _write_manifest(tmp_path, table_path)

    assert any(
        "identity_columns do not uniquely identify every row" in issue
        for issue in power_data_module.validate_manifest(manifest)
    )


def test_manifest_rejects_omitted_and_outside_dataset_tables(power_data_module, tmp_path):
    table_path = tmp_path / "gemm" / "trtllm" / "1.0.0" / "gemm_perf.parquet"
    outside_path = tmp_path / "gemm" / "trtllm" / "2.0.0" / "gemm_perf.parquet"
    _write_power_table(table_path)
    _write_power_table(outside_path)
    manifest = _write_manifest(tmp_path, table_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["tables"][0]["path"] = outside_path.relative_to(tmp_path).as_posix()
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    issues = power_data_module.validate_manifest(manifest)
    assert any("manifest omits packaged power tables" in issue for issue in issues)
    assert any("path does not match the declared backend/version" in issue for issue in issues)


def test_manifest_rejects_identity_merge_that_drops_upstream_rows(power_data_module, tmp_path):
    table_path = tmp_path / "gemm" / "trtllm" / "1.0.0" / "gemm_perf.parquet"
    _write_power_table(table_path)
    manifest = _write_manifest(tmp_path, table_path, upstream_rows=1)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["tables"][0]["upstream_rows"] = 3
    payload["totals"]["upstream_rows"] = 3
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    assert any(
        "identity-merge dropped upstream rows" in issue for issue in power_data_module.validate_manifest(manifest)
    )


def test_manifest_rejects_anomaly_count_and_table_split_drift(power_data_module, tmp_path):
    table_path = tmp_path / "gemm" / "trtllm" / "1.0.0" / "gemm_perf.parquet"
    _write_power_table(table_path)
    manifest = _write_manifest(tmp_path, table_path, upstream_rows=1)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["dataset"]["anomalies"][0]["rows"] = 2
    payload["dataset"]["anomalies"][0]["tables"][0]["rows"] = 2
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    issues = power_data_module.validate_manifest(manifest)
    assert any("expected 1 retained identities" in issue for issue in issues)
    assert any("anomaly rows are 2, expected 1 retained identities" in issue for issue in issues)


def test_manifest_rejects_incomplete_totals(power_data_module, tmp_path):
    table_path = tmp_path / "gemm" / "trtllm" / "1.0.0" / "gemm_perf.parquet"
    _write_power_table(table_path)
    manifest = _write_manifest(tmp_path, table_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["totals"]["measured_rows"] = 0
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    assert any(
        "totals do not cover every packaged row" in issue for issue in power_data_module.validate_manifest(manifest)
    )


def test_shipped_power_manifests_are_current(power_data_module):
    manifests = sorted(SHIPPED_DATA_ROOT.rglob(power_data_module.MANIFEST_BASENAME))

    assert manifests
    assert [issue for manifest in manifests for issue in power_data_module.validate_manifest(manifest)] == []


@pytest.mark.parametrize("column_type, values", [(pa.string(), ["500", "0"]), (pa.bool_(), [True, False])])
def test_manifest_reports_non_numeric_metrics_without_traceback(power_data_module, tmp_path, column_type, values):
    table_path = tmp_path / "gemm" / "trtllm" / "1.0.0" / "gemm_perf.parquet"
    _write_power_table(table_path)
    manifest = _write_manifest(tmp_path, table_path)
    table = pq.read_table(table_path)
    table = table.set_column(table.schema.get_field_index("power"), "power", pa.array(values, type=column_type))
    pq.write_table(table, table_path)
    issues = power_data_module.validate_manifest(manifest)
    assert any("power must be double" in issue for issue in issues)
    assert power_data_module.main([str(manifest)]) == 1


@pytest.mark.parametrize("change", ["measurement", "identity", "local_power", "upstream_hash"])
def test_identity_merge_checks_pinned_source_rows(power_data_module, tmp_path, change):
    table_path = tmp_path / "gemm" / "trtllm" / "1.0.0" / "gemm_perf.parquet"
    _write_power_table(table_path)
    manifest = _write_manifest(tmp_path, table_path, upstream_rows=1)
    payload = json.loads(manifest.read_text())
    table = pq.read_table(table_path)
    if change == "measurement":
        table = table.set_column(2, "power", pa.array([501.0, 0.0]))
    elif change == "identity":
        table = table.set_column(0, "shape", pa.array([3, 2]))
    elif change == "local_power":
        table = table.set_column(2, "power", pa.array([500.0, 500.0]))
        table = table.set_column(3, "power_limit", pa.array([1000.0, 1000.0]))
    else:
        payload["tables"][0]["upstream_sha256"] = "0" * 64
    pq.write_table(table, table_path)
    payload["tables"][0]["packaged_sha256"] = _sha256(table_path)
    manifest.write_text(json.dumps(payload))
    issues = power_data_module.validate_manifest(manifest)
    expected = {
        "measurement": "changed 1 upstream measurement",
        "identity": "dropped 1 upstream identities",
        "local_power": "local-only identities must use",
        "upstream_hash": "upstream evidence checksum mismatch",
    }
    assert any(expected[change] in issue for issue in issues)
