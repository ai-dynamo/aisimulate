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
SHIPPED_DATA_ROOT = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "aiconfigurator_core"
    / "systems"
    / "data"
)


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
    path = root / "power_data_provenance.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_manifest_validates_hash_counts_and_power_contract(power_data_module, tmp_path):
    table_path = tmp_path / "gemm" / "trtllm" / "1.0.0" / "gemm_perf.parquet"
    _write_power_table(table_path)
    manifest = _write_manifest(tmp_path, table_path)

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


def test_manifest_accepts_legacy_table_layout(power_data_module, tmp_path):
    table_path = tmp_path / "trtllm" / "1.0.0" / "gemm_perf.parquet"
    _write_power_table(table_path)
    manifest = _write_manifest(tmp_path, table_path)

    assert power_data_module.validate_manifest(manifest) == []


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
    assert any("manifest omits packaged tables" in issue for issue in issues)
    assert any("manifest lists tables outside the dataset" in issue for issue in issues)


def test_manifest_rejects_identity_merge_that_drops_upstream_rows(power_data_module, tmp_path):
    table_path = tmp_path / "gemm" / "trtllm" / "1.0.0" / "gemm_perf.parquet"
    _write_power_table(table_path)
    manifest = _write_manifest(tmp_path, table_path, upstream_rows=1)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["tables"][0]["upstream_rows"] = 3
    payload["totals"]["upstream_rows"] = 3
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    assert any(
        "identity-merge dropped upstream rows" in issue
        for issue in power_data_module.validate_manifest(manifest)
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
        "totals do not cover every packaged row" in issue
        for issue in power_data_module.validate_manifest(manifest)
    )


def test_shipped_power_manifests_are_current(power_data_module):
    manifests = sorted(SHIPPED_DATA_ROOT.rglob(power_data_module.MANIFEST_BASENAME))

    assert manifests
    assert [issue for manifest in manifests for issue in power_data_module.validate_manifest(manifest)] == []
