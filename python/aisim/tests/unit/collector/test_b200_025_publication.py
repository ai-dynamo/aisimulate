# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise published B200 metadata through the public loader."""

from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml

from aisim.sdk.perf_database import _database_version_dir_is_declared, _load_collection_meta_yaml
from aisim_core.data.default_systems import default_systems_path

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
DATA = Path(default_systems_path()) / "data/b200_sxm"


def test_published_025_metadata_loads_and_is_discoverable():
    # The collection report and raw-file digests are not runtime inputs and are not part of the pinned
    # op-latency dataset; the sidecars and the tables they describe are.
    sidecars = sorted(DATA.glob("*/vllm/0.25.0/collection_meta.yaml"))
    assert len(sidecars) == 10
    for path in sidecars:
        meta = _load_collection_meta_yaml(str(path))
        assert _database_version_dir_is_declared(str(path.parent), data_dir=str(DATA))
        assert meta["runtime"]["version"] == "0.25.0"
        assert "source_commit" not in meta["runtime"]
        for table, entry in meta["tables"].items():
            assert entry["rows"] == pq.ParquetFile(path.parent / (table + ".parquet")).metadata.num_rows
            for event in entry["collections"]:
                assert "classified_failures" not in event


@pytest.mark.parametrize("field", ["unexpected_runtime_field", "classified_failures"])
def test_public_schema_remains_fail_closed_for_unknown_fields(tmp_path, field):
    source = next(DATA.glob("*/vllm/0.25.0/collection_meta.yaml"))
    meta = yaml.safe_load(source.read_text())
    if field == "unexpected_runtime_field":
        meta["runtime"][field] = "a" * 40
    else:
        next(iter(meta["tables"].values()))["collections"][0][field] = 1
    path = tmp_path / "collection_meta.yaml"
    path.write_text(yaml.safe_dump(meta))
    with pytest.raises(ValueError, match="unsupported key"):
        _load_collection_meta_yaml(str(path))
