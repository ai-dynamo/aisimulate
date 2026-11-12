# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""materialize_systems_root: a self-contained systems root for non-Python consumers."""

import pytest

from aisim_core.data import materialize as mat
from aisim_core.data.ensure import DataPin, EnsuredData, MissingDataError

pytestmark = pytest.mark.unit

CONFIG = {"system": "h200_sxm", "backend": "vllm", "backend_version": "1.0"}
PROVENANCE = {"repo": "o/n", "revision": "a" * 40, "source": "hf-cache", "configuration_id": "cfg-x"}


@pytest.fixture
def layout(tmp_path):
    packaged = tmp_path / "packaged"
    (packaged / "data" / "h200_sxm" / "gemm" / "vllm" / "0.9").mkdir(parents=True)
    (packaged / "data" / "h200_sxm" / "gemm" / "vllm" / "0.9" / "packaged.parquet").write_bytes(b"old")
    (packaged / "profiles").mkdir()
    (packaged / "profiles" / "p.yaml").write_text("p: 1\n")
    (packaged / "h200_sxm.yaml").write_text("spec: 1\n")
    loaded = tmp_path / "cache" / "systems"
    table = loaded / "data" / "h200_sxm" / "gemm" / "vllm" / "1.0" / "gemm_perf.parquet"
    table.parent.mkdir(parents=True)
    table.write_bytes(b"pinned-table")
    return packaged, loaded


@pytest.fixture
def stub_ensure(monkeypatch, layout):
    _, loaded = layout
    calls = []

    def ensure(config, pin, *, transport=None):
        calls.append(pin)
        return EnsuredData(config={**config, "systems_paths": [str(loaded)]}, datasets={"op-latency": PROVENANCE})

    monkeypatch.setattr(mat, "ensure_data", ensure)
    return calls


def test_composes_specs_and_pinned_data_without_packaged_tables(tmp_path, layout, stub_ensure):
    packaged, _ = layout
    out = mat.materialize_systems_root(
        CONFIG, DataPin(local_root=tmp_path / "x"), tmp_path / "root", packaged_systems=packaged
    )
    root = out.root
    assert (root / "h200_sxm.yaml").read_text() == "spec: 1\n"
    assert (root / "profiles" / "p.yaml").is_file()
    assert (root / "data/h200_sxm/gemm/vllm/1.0/gemm_perf.parquet").read_bytes() == b"pinned-table"
    assert not (root / "data/h200_sxm/gemm/vllm/0.9").exists()  # packaged op data is never mixed in
    assert (out.data_files, out.data_bytes) == (1, len(b"pinned-table"))
    assert out.env == {"AICONFIGURATOR_SYSTEMS_PATH": str(root)}
    assert out.to_dict()["dataset"]["revision"] == "a" * 40
    assert (root / mat.MARKER).is_file()
    assert stub_ensure[0].categories == ("op-latency",)  # forced regardless of the pin


def test_refuses_existing_destination(tmp_path, layout, stub_ensure):
    (tmp_path / "root").mkdir()
    with pytest.raises(FileExistsError):
        mat.materialize_systems_root(
            CONFIG, DataPin(local_root=tmp_path), tmp_path / "root", packaged_systems=layout[0]
        )
    assert stub_ensure == []  # nothing fetched


def test_missing_data_propagates_and_leaves_no_destination(tmp_path, layout, monkeypatch):
    def ensure(config, pin, *, transport=None):
        raise MissingDataError("offline: not fully cached")

    monkeypatch.setattr(mat, "ensure_data", ensure)
    with pytest.raises(MissingDataError):
        mat.materialize_systems_root(
            CONFIG, DataPin(local_root=tmp_path), tmp_path / "root", packaged_systems=layout[0]
        )
    assert not (tmp_path / "root").exists()


def test_copy_failure_cleans_up(tmp_path, layout, stub_ensure, monkeypatch):
    def boom(source, target):
        raise OSError("disk full")

    monkeypatch.setattr(mat, "_link_or_copy", boom)
    with pytest.raises(OSError, match="disk full"):
        mat.materialize_systems_root(
            CONFIG, DataPin(local_root=tmp_path), tmp_path / "root", packaged_systems=layout[0]
        )
    assert not (tmp_path / "root").exists()


def test_root_has_specs_and_data_at_packaged_relative_paths(tmp_path, layout, stub_ensure):
    """The composed root keeps the relative layout Rust and Python readers walk."""
    out = mat.materialize_systems_root(
        CONFIG, DataPin(local_root=tmp_path), tmp_path / "root", packaged_systems=layout[0]
    )
    rels = sorted(str(p.relative_to(out.root)) for p in out.root.rglob("*") if p.is_file())
    assert "data/h200_sxm/gemm/vllm/1.0/gemm_perf.parquet" in rels and "h200_sxm.yaml" in rels
