# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the public loader, not just the more permissive producer writer."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from aisimulate.sdk.perf_database import _database_version_dir_is_declared, _load_collection_meta_yaml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "src/aisimulate_core/systems/data/b200_sxm"


def test_published_025_metadata_loads_and_is_discoverable():
    report = json.loads((DATA / "vllm-0.25.0-collection-report.json").read_text())
    assert report["runtime"]["source_commit"] == "dd10e03f95f94edbea1975c67ace3a35ec9a8a40"
    sidecars = sorted(DATA.glob("*/vllm/0.25.0/collection_meta.yaml"))
    assert len(sidecars) == 10
    total_rows = 0
    for path in sidecars:
        meta = _load_collection_meta_yaml(str(path))
        assert _database_version_dir_is_declared(str(path.parent), data_dir=str(DATA))
        assert meta["runtime"]["version"] == "0.25.0"
        assert "source_commit" not in meta["runtime"]
        for table, entry in meta["tables"].items():
            evidence = report["tables"][table + ".parquet"]
            assert entry["rows"] == evidence["rows"]
            assert hashlib.sha256((path.parent / (table + ".parquet")).read_bytes()).hexdigest() == evidence["sha256"]
            total_rows += entry["rows"]
            assert len(entry["collections"]) == len(evidence["source_jobs"])
            for event, source in zip(entry["collections"], evidence["source_jobs"], strict=True):
                assert event["collector_ref"] == source["collector_ref"]
                assert event["rows"] == source["source_rows"]
                assert "classified_failures" not in event
                assert isinstance(source["classified_failures"], int) and source["classified_failures"] >= 0
    assert total_rows == report["total_published_rows"] == 419538


@pytest.mark.parametrize("field", ["source_commit", "classified_failures"])
def test_public_schema_remains_fail_closed_for_relocated_fields(tmp_path, field):
    source = next(DATA.glob("*/vllm/0.25.0/collection_meta.yaml"))
    meta = yaml.safe_load(source.read_text())
    if field == "source_commit":
        meta["runtime"][field] = "a" * 40
    else:
        next(iter(meta["tables"].values()))["collections"][0][field] = 1
    path = tmp_path / "collection_meta.yaml"
    path.write_text(yaml.safe_dump(meta))
    with pytest.raises(ValueError, match="unsupported key"):
        _load_collection_meta_yaml(str(path))
