# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Renaming a publication preserves measurements and rejects altered evidence."""

import hashlib
import json
import tarfile
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from collector.sglang_rubin import publish_prefill_graph as publisher

pytestmark = pytest.mark.unit


@pytest.fixture
def original(tmp_path):
    root = tmp_path / "original"
    fixture = Path(__file__).parents[3] / "fixtures/vr_nvl72/original-pilot.tar.gz"
    with tarfile.open(fixture) as archive:
        archive.extractall(root, filter="data")
    return root


def test_migration_preserves_measurements_and_matches_shipped_bundle(original, tmp_path):
    output = tmp_path / "migrated"
    result = publisher.migrate_vr_nvl72(base=original, output=output)
    assert result["status"] == "MIGRATED"
    assert not (output / "vr200_hecate.yaml").exists()
    assert not (output / "data/vr200_hecate").exists()
    receipt_path = f"prefill_graph_publications/{publisher.RENAMED_PROFILE}.json"
    receipt = json.loads((output / receipt_path).read_text())
    assert receipt["migration"]["measurements_changed"] is False
    assert receipt["migration"]["original_profile_id"] == publisher.ORIGINAL_PROFILE_ID
    original_receipt = json.loads((original / f"prefill_graph_publications/{publisher.PROFILE}.json").read_text())
    assert receipt["source_publication"] == original_receipt
    for name, pin in receipt["published_files"].items():
        assert publisher._record(output / name) == pin
    for source in original.rglob("*.parquet"):
        target = output / str(source.relative_to(original)).replace("vr200_hecate", "vr_nvl72")
        table = pq.read_table(source)
        migrated = pq.read_table(target)
        if "profile_id" in table.column_names:
            assert table.drop(["profile_id"]).equals(migrated.drop(["profile_id"]))
            assert table.schema == migrated.schema
            assert set(migrated["profile_id"].to_pylist()) == {result["profile_id"]}
        else:
            assert target.read_bytes() == source.read_bytes()
    for source in original.rglob("collection_meta.yaml"):
        target = output / str(source.relative_to(original)).replace("vr200_hecate", "vr_nvl72")
        assert target.read_bytes() == source.read_bytes()
    profiles = list(output.rglob("*.profile.json"))
    assert len(profiles) == 2
    assert profiles[0].read_bytes() == profiles[1].read_bytes()
    assert hashlib.sha256(profiles[0].read_bytes()).hexdigest() == result["profile_id"]
    shipped = Path(__file__).parents[4] / "src/aisimulate_core/systems"
    shipped_receipt = json.loads((shipped / receipt_path).read_text())
    assert shipped_receipt["profile_id"] == result["profile_id"]
    assert shipped_receipt["source_publication"] == original_receipt
    assert shipped_receipt["migration_tool"] == {
        "module": publisher.MODULE,
        **publisher._record(Path(publisher.__file__)),
    }
    for name in receipt["published_files"]:
        assert publisher._record(shipped / name) == shipped_receipt["published_files"][name]
        if name.endswith(".parquet"):
            # Parquet container bytes may differ between supported Arrow versions.
            assert pq.read_table(output / name).equals(pq.read_table(shipped / name)), name
        else:
            assert (output / name).read_bytes() == (shipped / name).read_bytes(), name
    second = tmp_path / "second"
    assert publisher.migrate_vr_nvl72(base=original, output=second) == result
    assert publisher._inventory(second) == publisher._inventory(output)


@pytest.mark.parametrize(
    "filename",
    [
        "vr200_hecate.yaml",
        "profile-evidence.json",
        "gemm_perf.parquet",
        "moe_perf.parquet",
        "collection_meta.yaml",
        "sglang_prefill_attention_sequence_perf.parquet",
        f"{publisher.PROFILE}.profile.json",
        f"{publisher.PROFILE}.json",
    ],
)
def test_migration_rejects_changed_inputs(original, tmp_path, filename):
    path = next(original.rglob(filename))
    path.write_bytes(path.read_bytes() + b"\n")
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="Input changed|Unapproved source publication"):
        publisher.migrate_vr_nvl72(base=original, output=output)
    assert not output.exists()


def test_migration_requires_fresh_output_and_rejects_symlinks(original, tmp_path):
    with pytest.raises(ValueError, match="fresh output"):
        publisher.migrate_vr_nvl72(base=original, output=original)
    with pytest.raises(ValueError, match="fresh output"):
        publisher.migrate_vr_nvl72(base=original, output=original / "child")
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(ValueError, match="fresh output"):
        publisher.migrate_vr_nvl72(base=original, output=occupied)
    hardware = original / "vr200_hecate.yaml"
    moved = tmp_path / "hardware.yaml"
    hardware.rename(moved)
    hardware.symlink_to(moved)
    with pytest.raises(ValueError, match="Symlink"):
        publisher.migrate_vr_nvl72(base=original, output=tmp_path / "rejected")
