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


def _write_manifest(root: Path, table: Path) -> Path:
    digest = _sha256(table)
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
            "anomalies": [],
        },
        "tables": [
            {
                "path": table.relative_to(root).as_posix(),
                "identity_columns": ["shape"],
                "shape_bounds": {"shape": {"min": 1, "max": 2}},
                "import_mode": "exact-copy",
                "upstream_sha256": digest,
                "packaged_sha256": digest,
                "upstream_rows": 2,
                "packaged_rows": 2,
                "measured_rows": 1,
                "zero_sentinel_rows": 1,
            }
        ],
        "totals": {
            "upstream_rows": 2,
            "packaged_rows": 2,
            "measured_rows": 1,
            "zero_sentinel_rows": 1,
        },
    }
    path = root / "power_data_provenance.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_manifest_validates_hash_counts_and_power_contract(power_data_module, tmp_path):
    table_path = tmp_path / "gemm" / "trtllm" / "1.0.0" / "gemm_perf.parquet"
    table_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "shape": [1, 2],
                "latency": [1.0, 2.0],
                "power": pa.array([500.0, 0.0], type=pa.float64()),
                "power_limit": pa.array([1000.0, 0.0], type=pa.float64()),
            }
        ),
        table_path,
    )
    manifest = _write_manifest(tmp_path, table_path)

    assert power_data_module.validate_manifest(manifest) == []
    assert power_data_module.main([str(manifest)]) == 0


def test_manifest_rejects_checksum_drift(power_data_module, tmp_path):
    table_path = tmp_path / "gemm" / "trtllm" / "1.0.0" / "gemm_perf.parquet"
    table_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "shape": [1, 2],
                "latency": [1.0, 2.0],
                "power": pa.array([500.0, 0.0], type=pa.float64()),
                "power_limit": pa.array([1000.0, 0.0], type=pa.float64()),
            }
        ),
        table_path,
    )
    manifest = _write_manifest(tmp_path, table_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["tables"][0]["packaged_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    assert any("packaged_sha256 mismatch" in issue for issue in power_data_module.validate_manifest(manifest))


def test_manifest_rejects_identity_evidence_drift(power_data_module, tmp_path):
    table_path = tmp_path / "gemm" / "trtllm" / "1.0.0" / "gemm_perf.parquet"
    table_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "shape": [1, 2],
                "latency": [1.0, 2.0],
                "power": pa.array([500.0, 0.0], type=pa.float64()),
                "power_limit": pa.array([1000.0, 0.0], type=pa.float64()),
            }
        ),
        table_path,
    )
    manifest = _write_manifest(tmp_path, table_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["tables"][0]["shape_bounds"]["shape"]["max"] = 3
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    assert any("shape_bounds do not match" in issue for issue in power_data_module.validate_manifest(manifest))


def test_shipped_power_manifests_are_current(power_data_module):
    manifests = sorted(SHIPPED_DATA_ROOT.rglob(power_data_module.MANIFEST_BASENAME))

    assert manifests
    assert [issue for manifest in manifests for issue in power_data_module.validate_manifest(manifest)] == []
